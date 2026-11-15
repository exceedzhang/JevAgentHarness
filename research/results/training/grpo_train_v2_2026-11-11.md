# train_v2 / train_v3 决策级 GRPO 实验记录（2026-11-11 ～ 2026-11-14）

> 结论先行：第一轮在已发布的 `AgentDecision-4B` LoRA adapter 上，用 `data/train_v2` 做 1 epoch 决策级 GRPO，dev accuracy 从 **10.81%** 提升到 **35.14%**（+24.3 pp），但只比 dev 多数类先验（32.43%）高 **2.7 pp**；NLL / Brier / ECE 同时变差。训练奖励在前 20–40 步快速上升，之后基本平台化，并伴随明显的 entropy collapse 与 zero-variance group 增多。最终 checkpoint 不是最优部署点。
>
> 第二轮补上了逐 checkpoint dev 评估与早停（G=16、lr=3e-6、beta=0.05、最多 80 步）。entropy / zero-variance 明显改善，早停条件未触发；dev top-1 在 step 40 达到 **29.73%**，低于第一轮最终的 35.14%，但 ECE 从 0.484 降到 **0.317**。第一轮更高的 top-1 伴随更差的校准；在 37 条 dev 上，两轮差异仍不足以下强结论。第二轮最可靠的产出是建立了“按 dev 选 checkpoint”的流程，并确认 **40 步后继续训练主要让策略变尖，而不是提升 dev**。
>
> 2026-11-14 又在 `train_v3`（train 2481 / dev 81）上从第二轮 step40 继续训练。v3 final step104 与 v2 第一轮 final 在 train_v3 dev 上同为 **53.09%**；v3 的 Brier / ECE 更好，NLL 略差。相对 v3 的起点仍是“更准但更自信”。详见 §12。

- 日期：2026-11-11（初版）；2026-11-14 增补 train_v3 对比
- 类型：探索性实验（exploratory），不是预注册结果
- 代码：`src/jevharness/training/grpo_trl.py --impl torch`
- 数据：`data/train_v2/{train,dev}.jsonl`
- 起始模型：`Qwen/Qwen3-4B` + `AgentDecision-4B` PEFT adapter
- 产物：`checkpoints/grpo_train_v2/`（第一轮）、`checkpoints/grpo_train_v2_r2/`（第二轮）、`checkpoints/grpo_train_v3/`（train_v3），均为本地产物，未作为代码提交

## 1. 实验目标

验证两件事：

1. 能否直接在已发布的 `AgentDecision-4B` adapter 上继续做决策级 GRPO；
2. 用生产分布数据 `train_v2` 做 verifiable-reward GRPO，是否能提升 dev 上的选项选择准确率。

这里的奖励只有 `correct`：采样出的 option code 是否等于该行 gold label。`train_v2` 是硬标签、没有 soft target / q-value；受限 readout 只采样 option code，所以 format reward 恒为 0，没有加入训练。

## 2. 环境与起始模型

| 项 | 值 |
|---|---|
| GPU | 1 × RTX 4090 24 GB（GPU1） |
| base model | `/data1/apps/huggingface/Qwen/Qwen3-4B` |
| init adapter | `/tmp/opencode/AgentDecision-4B-grpo`，指向已发布 `AgentDecision-4B` |
| Python | 3.10.19（`research/.venv`，`--system-site-packages`） |
| torch | 2.10.0+cu128 |
| transformers | 4.57.6 |
| peft | 0.21.1 |
| 网络 | 离线训练；`TRANSFORMERS_OFFLINE=1`、`HF_HUB_OFFLINE=1` |

### 2.1 为什么需要临时 adapter 目录

`AgentDecision-4B/tokenizer_config.json` 里的 `extra_special_tokens` 是 list，transformers 4.57.6 会报：

```text
AttributeError: 'list' object has no attribute 'keys'
```

因此创建 `/tmp/opencode/AgentDecision-4B-grpo`：

- `adapter_config.json`、`adapter_model.safetensors`（symlink 到原始 adapter）
- `tokenizer.json`、`chat_template.jinja`（symlink）
- 重写后的 `tokenizer_config.json`：`extra_special_tokens: {}`

这只修复 tokenizer 兼容性，**不改变 adapter 权重**。训练时 `--init` 指向这个目录；`static_reference` 也使用同一份 adapter 的冻结副本。

### 2.2 GPU 腾挪

GPU1 原本运行 `vllm-openai-2`（Qwen3-Embedding-4B）。训练前临时停止：

```bash
sudo -n docker stop vllm-openai-2
```

训练结束后已恢复：

```bash
sudo -n docker start vllm-openai-2
```

恢复后容器 Up，内部 `/health` 返回 200，GPU1 约 15 GB 显存占用（vLLM `gpu_memory_utilization=0.6`，约 0.6 × 24 GB 显存池）。

## 3. 数据检查

`data/train_v2` 为生产 `tool_select` 决策数据，按 session 分组切分：

| split | rows | 说明 |
|---|---:|---|
| train | 1,539 | 生产日志中的 tool_select 决策点 |
| dev | 37 | 同分布留出集 |

使用**实际训练 tokenizer / chat template** 统计 prompt 长度：

| 指标 | train | dev |
|---|---:|---:|
| min tokens | 105 | — |
| p50 | 1,955 | — |
| p90 | 2,700 | — |
| p95 | 2,887 | — |
| p99 | 3,056 | — |
| max tokens | **3,745** | 2,261 |

其他统计：

- prompt 字符数：p50 6,227，max 8,641
- options 数：2–52，mean 15.46
- 所有 row 都有非零 target

**关于 `--max-len 4096`：** 最大真实 prompt 只有 3,745 tokens，`0` 行超过 4,096，`0` 行超过 8,192。`fit_prompt()` 不会裁 state，`DecisionCollator.encode()` 保留完整 token。因此 4,096 不会截断任何训练样本，本次不需要为了长度重开训练；加长上下文只会增加显存和计算，不会改变数据覆盖。

## 4. 训练命令与配置

在 `/data1/apps/JevAgentHarness/research` 下运行：

```bash
CUDA_VISIBLE_DEVICES=1 \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
nohup .venv/bin/python -m jevharness.training.grpo_trl \
  --impl torch \
  --model /data1/apps/huggingface/Qwen/Qwen3-4B \
  --init /tmp/opencode/AgentDecision-4B-grpo \
  --train data/train_v2/train.jsonl \
  --dev data/train_v2/dev.jsonl \
  --out checkpoints/grpo_train_v2 \
  --rewards correct \
  --num-generations 8 \
  --batch-size 8 \
  --grad-accum 8 \
  --epochs 1 \
  --lr 1e-5 \
  --beta 0.02 \
  --max-len 4096 \
  --gradient-checkpointing \
  --logging-steps 5 \
  --seed 0 > /tmp/opencode/grpo_train_v2_full.log 2>&1 &
```

关键配置：

| 参数 | 值 | 含义 |
|---|---:|---|
| P | 8 | prompts / optimizer step |
| G | 8 | generations / prompt |
| micro-batch | 1 prompt | 每次 forward/backward |
| 每步采样 | 64 | 8 × 8 |
| epochs | 1 | 1,539 行约 193 步 |
| lr | 1e-5 | LoRA 学习率 |
| beta | 0.02 | KL penalty 系数 |
| max_len | 4,096 | 实际最长 3,745，无截断 |
| num_iterations | 1 | 每次 rollout 只更新一次 |

## 5. 训练结果

### 5.1 总览

- 完成全部 **193 steps**，约 **29 分钟**（约 9 秒/step）。
- 无 NaN / Inf / OOM / 异常退出。
- 仅有一个无害的 PEFT warning：base model 目录缺少 config 的提示。
- `metrics.jsonl` 193 行，`eval.json` 已生成。

### 5.2 全局指标（193 步）

| 指标 | mean | std | p50 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| reward / acc_sampled | 0.4497 | 0.1634 | 0.4531 | 0.6375 | 0.0625 | 0.8750 |
| entropy | 0.4159 | 0.3305 | 0.3282 | 0.6973 | 0.0396 | 1.8539 |
| frac_zero_std_groups | 0.7448 | 0.2117 | 0.7500 | 1.0000 | 0 | 1 |
| grad_norm | 5.7742 | 3.7712 | 5.7766 | 10.5382 | 0.0250 | 22.1246 |
| KL | 1.3564 | 0.4298 | 1.4167 | 1.7931 | ~0 | 2.4067 |
| loss | 0.0271 | 0.0086 | 0.0283 | 0.0359 | 0 | 0.0481 |

说明：

- 这里的 `loss` 是 GRPO surrogate loss，不是交叉熵，不能按普通 SFT loss 解读。
- 第 1 步 loss / KL 为 0，因为 policy 与 reference 完全相同。
- `clipped = 0` 全部 193 步。原因不是实现 bug，而是 `num_iterations=1`：每个 rollout 只做一次更新，PPO ratio 恒为 1，clip 永远不会触发。

### 5.3 分阶段变化

每 20 步窗口：

| steps | reward | entropy | zero_std | grad_norm |
|---|---:|---:|---:|---:|
| 1–20 | 0.3070 | 1.2443 | 0.3937 | 5.68 |
| 21–40 | 0.4883 | 0.5401 | 0.6188 | 5.44 |
| 41–60 | 0.4148 | 0.4017 | 0.7750 | 4.96 |
| 61–80 | 0.4875 | 0.2965 | 0.8125 | 5.41 |
| 81–100 | 0.4813 | 0.2778 | 0.8187 | 4.50 |
| 101–120 | 0.4461 | 0.3198 | 0.7500 | 6.49 |
| 121–140 | 0.4656 | 0.3168 | 0.7625 | 6.47 |
| 141–160 | 0.4305 | 0.2759 | 0.8000 | 7.09 |
| 161–180 | 0.4773 | 0.2372 | 0.8625 | 6.81 |
| 181–193 | 0.5252 | 0.1596 | 0.9135 | 4.42 |

### 5.4 趋势与有效信号

- reward 全段 OLS slope：`+0.000567/step`，193 步累计约 `+0.109`，`p ≈ 0.0065`，但 `R² = 0.037`。
- 只看第 21–193 步：slope `+0.000137/step`，`p ≈ 0.571`，没有持续上升趋势。
- 41–193 步平均 reward 约 `0.463`；最后 40 步 `0.4820`，前 40 步 `0.3977`。
- 前 20 步 `0.3070`；最后 13 步 `0.5252`；最好的 10 步窗口是 180–189，均值 `0.5938`。
- 单步 reward std 为 `0.163`；若 64 个样本独立且 p≈0.45，二项分布参考 std 约 `0.062`。实际波动更大，说明每个 batch 的题目难度差异明显。

### 5.5 entropy collapse 与 zero-variance group

这是本次训练最重要的现象：

- entropy 从第 1 步 `1.687`、第 2 步 `1.854`，降到 21–40 步均值 `0.540`，最后 40 步 `0.212`，最低 `0.0396`，最终 `0.1166`。
- `frac_zero_std_groups` 从第一窗口 `0.394` 升到全段均值 `0.745`，最后 13 步 `0.914`。
- 全段平均有效 prompt 数只有 `8 × (1 - 0.7448) ≈ 2.04`；最后 13 步约 `0.69`。
- 193 步中有 **33 步（17.1%）所有 group 都是 zero-variance**，这些步的 `grad_norm` 全部小于 0.5，几乎不提供学习信号。

全零方差步：

```text
44, 47, 50, 52, 63, 67, 69, 70, 83, 94, 97, 100,
112, 114, 115, 118, 121, 127, 129, 136, 147, 148,
157, 158, 162, 167, 171, 180, 183, 185, 186, 187, 193
```

解释：G=8 时，如果 8 个采样全部答对或全部答错，该 prompt 的 advantage 为 0。训练越到后面，策略越尖锐，采样越一致，zero-variance group 越多；有效 batch 从 8 个 prompt 缩到不足 1 个。这比“奖励不再上涨”更值得关注。

### 5.6 KL 与梯度

- KL 从 0 升到约 1.3–1.8，最高 `2.4067`，均值 `1.3564`。
- beta=0.02 时，KL penalty 约 `0.027`，与平均 loss `0.0271` 同量级；但即使有这个 penalty，entropy 仍然 collapse。
- `max_grad_norm=1.0`；大多数有效步的 pre-clip grad norm 大于 1，33 个零信号步则小于 0.5。
- reward 与 entropy 的相关为 `-0.388`，与 step 的相关为 `-0.664`（entropy 随时间下降）；reward 与 KL / loss 的相关都约 `+0.308`。

## 6. dev 评估

`eval.json`（n=37）与初始 adapter 的同口径受限 readout 对比：

| 指标 | 初始 AgentDecision-4B | GRPO 后 | 变化 |
|---|---:|---:|---:|
| accuracy | 10.81% | **35.14%** | **+24.32 pp** |
| NLL | 2.7985 | 3.6125 | +0.8140（变差） |
| Brier top-1 | 0.2370 | 0.4440 | +0.2070（变差） |
| ECE | 0.3701 | 0.4839 | +0.1139（变差） |

解读：

- argmax 准确率明显提升：4/37 → 13/37。
- 但 dev 多数类先验是 `reply_without_tool` 12/37 = **32.43%**；35.14% 只比“永远猜多数类”高 **2.7 pp**。这次更像是从“接近随机”拉到了“类别先验水平”，还没有证明它真的按状态区分工具。
- 概率质量变得更集中，但校准明显变差：模型更“自信”，NLL / Brier / ECE 都恶化。
- 37 条 dev 很小，而且只来自 **3 个 session**（其中一个 session 占 26 条）。准确率的 95% Wilson 区间大约为：
  - 初始：4.3%–24.7%
  - GRPO 后：21.8%–51.2%
  - 两个区间仍有重叠，不能把 +24.3 pp 当作精确效应量。

因此这次结果支持“GRPO 能把选项选择从很低起点拉起来”，但不支持“最终 checkpoint 的概率可直接部署”。如果下游使用概率或阈值，必须重新做 temperature scaling / calibration。

## 7. 经验总结

1. **4,096 不是瓶颈。** 真实最长 prompt 3,745 tokens，没有截断；不要为了“放开长度”盲目加到 8k，先看数据分布。
2. **主要收益在前 20–40 步。** 之后 reward 基本平台化；继续训练主要是在让策略变尖，而不是继续提升准确率。
3. **entropy collapse 是核心风险。** 最后 13 步 91% 的 group 零方差，平均有效 prompt 不足 1 个；这时训练日志还在跑，但有效学习已经接近停止。
4. **zero-variance group 要单独监控。** 只看 mean reward 会误以为训练仍在继续；应把 `frac_zero_std_groups`、有效 prompt 数、entropy 和 grad_norm 一起看。
5. **`clipped=0` 不一定是异常。** `num_iterations=1` 时 PPO ratio 恒为 1，clip 本来就不会触发；不要用 clipped 判断训练是否正常。
6. **KL penalty 没有阻止 collapse。** beta=0.02 的 penalty 与 loss 同量级，但 entropy 仍从 1.7 降到 0.12；需要更小的 LR、更大的 beta 或早停。
7. **准确率和校准是两件事。** 本次 accuracy 大幅提升，但 NLL / Brier / ECE 全面变差；最终模型更适合“取 argmax”，不适合直接输出概率。
8. **最终 checkpoint 不一定是最好 checkpoint。** 第一轮只保存了最终 adapter；第二轮补上中间点后，step 40 确实优于 step 80（见 §11）。
9. **dev 只有 37 条，结论要保守。** 需要更大的 dev 或交叉验证，才能把 +24 pp 的效应量估准。
10. **`--rewards correct` 足够。** train_v2 是硬标签，受限 readout 只采样 option code，format reward 没有信息量；不要为了“奖励更丰富”加入恒为常数的项。

## 8. 下一次训练建议

> 以下建议是第一轮后提出的；其中第 1–4 项已在第二轮执行，结果与修正见 §11。

优先级从高到低：

1. **每 10–20 步保存 checkpoint**，每个点跑 dev；按 dev accuracy（同时看 NLL/ECE）选点，不要默认用最后一个。
2. **早停**：在 reward 平台化、entropy 降到约 0.4–0.6、`frac_zero_std_groups` 明显上升时停止。按本次曲线，大约在 20–40 步附近就应开始认真评估。
3. **降低学习率**：从 `1e-5` 降到 `3e-6`–`5e-6`，减缓策略变尖。
4. **提高 KL 系数**：`beta` 从 `0.02` 提到 `0.05`–`0.1`，观察 entropy 是否能维持在更有信号的范围。
5. **处理零方差 group**：可以跳过全对/全错的 group，或增大 G（如 16，若显存允许）以增加组内差异；也可以按难度分层采样。
6. **校准**：如果部署需要概率，对选中的 checkpoint 重新拟合 temperature；不要直接使用 GRPO 后的原始分布。
7. **扩大 dev**：37 条太少；至少几百条，或做按 session 分组的交叉验证。
8. **长度保持 4,096**：本次数据没有超过 3,745 tokens，4,096 足够；除非未来 state 变长，否则不需要 8k。

一个建议的下一轮配置：

```text
lr = 3e-6 ~ 5e-6
beta = 0.05 ~ 0.1
epochs = 1（但每 10–20 步存点 + 早停）
G = 8（显存允许可试 16）
max_len = 4096
save_steps = 10 ~ 20
```

## 9. 复现要点

### 9.1 环境

```bash
cd /data1/apps/JevAgentHarness/research
# 本次使用 Python 3.10.19 的 vllm 环境作为 base
/data1/apps/miniconda3/envs/vllm/bin/python -m venv --system-site-packages .venv
.venv/bin/pip install --no-deps peft==0.21.1
```

验证：

```bash
.venv/bin/python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_available())"
```

### 9.2 tokenizer 兼容目录

如果 `/tmp/opencode/AgentDecision-4B-grpo` 不存在，需要重新创建：复制/链接 adapter 权重与 tokenizer 文件，并把 `tokenizer_config.json` 中的 `extra_special_tokens` 改为 `{}`。否则 transformers 4.57.6 会在加载 tokenizer 时因 list 类型报错。

### 9.3 训练

按第 4 节命令运行即可。若要换 GPU，修改 `CUDA_VISIBLE_DEVICES`；若要释放 GPU1，先停止 `vllm-openai-2`，训练结束后务必恢复。

### 9.4 产物

```text
checkpoints/grpo_train_v2/
├── adapter_model.safetensors   # 253 MB
├── adapter_config.json
├── eval.json
├── metrics.jsonl               # 193 行
├── decision_config.json
├── tokenizer files
└── README.md
```

`decision_config.json` 记录：base_model `Qwen/Qwen3-4B`、loss `grpo_torch`、lora true、max_len 4096、init 临时 adapter、rewards `correct`、steps 193。

第二轮产物额外包含逐 checkpoint 目录和 `checkpoints.jsonl`：

```text
checkpoints/grpo_train_v2_r2/
├── step-000010/ ... step-000070/   # 每个含 adapter、tokenizer、eval.json
├── checkpoints.jsonl               # 各 step 的 dev 指标
├── adapter_model.safetensors       # 最终 step 80（不是最优点）
├── eval.json
├── metrics.jsonl                   # 80 行
└── decision_config.json            # steps=80, early_stopped=false
```

最优点为 `checkpoints/grpo_train_v2_r2/step-000040/`，详见 §11.3。

## 10. 补充：数据结构性评估（2026-11-11）

> 这一节是事后直接统计 `data/train_v2` 得到的，用来回答“1,576 条是否够”。结论是：绝对量够跑一次 4B LoRA 单 epoch，但**长尾工具覆盖**和**验证集结构**确实会挡住“达到 95% 一致率 / 可部署”的目标。

### 10.1 训练/验证集的真实分布

先澄清三个容易混的数字：

- `train_v2` 总行数是 **1,576** = train **1,539** + dev **37**；实际训练只用了 1,539 条。
- train 里有 **15 个** gold 工具，dev 里有 **8 个**，并集 **17 个**。
- dev 的 `web_search` 和 `ask_user_question` 在 train 中出现 **0 次**，所以这两个类在 dev 上不可能被学到。

train 的标签分布：

| 工具 | train 条数 | train 占比 | dev 条数 |
|---|---:|---:|---:|
| shell | 592 | 38.47% | 5 |
| read | 256 | 16.63% | 4 |
| edit | 251 | 16.31% | 7 |
| reply_without_tool | 196 | 12.74% | 12 |
| compress | 76 | 4.94% | 0 |
| grep | 57 | 3.70% | 0 |
| write | 41 | 2.66% | 0 |
| bash | 17 | 1.10% | 0 |
| execute | 15 | 0.97% | 6 |
| webfetch | 13 | 0.84% | 0 |
| glob | 12 | 0.78% | 1 |
| websearch | 4 | 0.26% | 0 |
| question | 4 | 0.26% | 0 |
| skill | 4 | 0.26% | 0 |
| present | 1 | 0.06% | 0 |
| web_search | 0 | 0 | 1 |
| ask_user_question | 0 | 0 | 1 |

前 4 个工具合计 `592 + 256 + 251 + 196 = 1,295`，占 train 的 **84.1%**。

需要修正一个说法：长尾不能笼统说成“13 个工具每个 <1%”。`compress`（4.94%）、`grep`（3.70%）、`write`（2.66%）属于**中尾**，不是 <1%；真正 <1% 的是 `execute`、`webfetch`、`glob`、`websearch`、`question`、`skill`、`present`，以及 dev 独有的两个类。把 train + dev 合并后，`execute` 是 21 条、`glob` 13 条、`bash` 17 条、`webfetch` 13 条；这些数字没错，但分母是 1,576，而不是训练集 1,539。

这组分布的含义：

- GRPO 的可验证奖励对 1–4 条样本的工具几乎学不到可靠信号；这些组要么 8 个采样全错（零方差），要么靠先验猜中。
- 长尾工具不是“再跑几轮就能学会”的问题，而是数据里没有足够对比样本。
- `web_search` / `ask_user_question` 在 train 中完全缺失，dev 上出现只能算泛化/先验，不能算学习。

### 10.2 dev=37 的类别先验问题

dev 的多数类先验：

- `reply_without_tool`：12/37 = **32.43%**
- 模型：13/37 = **35.14%**
- 只高 **2.7 pp**

所以 +24.3 pp 的准确率提升，主要是从初始 4/37（10.81%，远低于先验）拉到了接近先验；还没有达到“显著优于平凡策略”的程度。再叠加 NLL / Brier / ECE 全部变差、entropy 从 1.69 塌到 0.12，模型更像是在“高置信地输出类别先验”，而不是“按状态挑工具”。

dev 只有 3 个 session（其中一个 26 条）；train 有 34 个 session，但其中一个 session 占 975 条（63%）。这意味着：

- 单纯把 dev 行数扩大，如果不增加独立 session，验证方差仍然很大；
- “按 session 分组交叉验证”方向正确，但当前 session 数太少，且训练集被单一长会话主导；需要更多独立会话，而不只是更多行。

### 10.3 有效样本量

全段 `frac_zero_std_groups` 均值 0.7448。G=8 时，一个 prompt 的 8 个采样全对或全错，advantage 为 0，不产生梯度。

- 名义训练量：1,539 prompts / epoch；
- 粗略有效量：`1,539 × (1 - 0.7448) ≈ 393` prompts；
- 最后 13 步的有效 prompt 数不足 1。

因此“1,576 条”不能按 1,576 个独立学习信号来理解；本次真正贡献梯度的量级约 **400**。这个估算是 step-level 平均的近似，不是逐 prompt 精确计数，但足以说明有效训练量远小于名义值。

### 10.4 修复顺序

1. **先修训练（免费，优先级最高）**
   - 每 10–20 步保存 checkpoint，并在每个点跑 dev；
   - entropy < 0.5 或 `frac_zero_std_groups` 明显上升时早停；
   - lr 降到 `3e-6`–`5e-6`，beta 提到 `0.05`–`0.1`；
   - 显存允许时把 G 提到 16，或跳过全对/全错的 group；
   - 选点后单独做 temperature calibration，再回生产 shadow A/B。

2. **再让数据自然长大**
   - 采集速率约 390–400 条/天；两周约 5k、一个月约 10k；
   - 长尾工具会随流量自然填上，但需要 2–4 周，不是一次训练能解决；
   - 同时扩大 dev 的**独立 session 数**，否则采再多行也无法可靠验证。

3. **关于“全量 tool_select 标签”**
   - 只有带 `state` 的行才能拼出与线上一致的 prompt；旧的 355 条无 state 标签不能直接进 prompt 级训练/验证；
   - 当前 `train_v2` 的 1,539 + 37 已经是可用集合；扩大 dev 应从后续生产日志中按 session 留出，而不是重复使用训练行。

### 10.5 第二轮前的代码限制（已修复）

`grpo_trl.py` 的 argparse 里确实有 `--save-steps`，但它原先只传给 TRL trainer；**`--impl torch` 分支不使用它**。torch 循环只在全部 step 结束后调用一次 `_save()`，也没有 early stopping 或逐 checkpoint 的 dev 评估。

这个限制已在 2026-11-11 修复：

- 每 N 步调用 `_save()` 到 `out/step-XXXXXX`；
- 每个 checkpoint 跑同一套 restricted-readout dev eval，写 `eval.json` 并追加到 `out/checkpoints.jsonl`；
- 支持按 entropy / `frac_zero_std_groups` 早停（`--early-stop-*`）；
- 最终 root adapter 记录 `steps`、`planned_steps`、`early_stopped`。

实现细节和第二轮结果见 §11。需要注意：当前早停只看 entropy / zero-std，**还没有按 dev 指标早停**；如果以后要“dev 连续 N 次不升就停”，还需要再加一层逻辑。

## 11. 第二轮：checkpoint + 早停验证（2026-11-11）

### 11.1 配置

第二轮不改数据，只改训练控制和超参，目标是验证“中间 checkpoint + dev 选点”能否避免第一轮的 entropy collapse 和最终点过尖。

| 项 | 值 |
|---|---|
| base model | `/data1/apps/huggingface/Qwen/Qwen3-4B` |
| init adapter | `/tmp/opencode/AgentDecision-4B-grpo` |
| train / dev | `data/train_v2/train.jsonl` / `data/train_v2/dev.jsonl` |
| G | 16 |
| batch size | 16 |
| grad accumulation | 8（P = 16×8/16 = 8 prompts/step） |
| max steps | 80 |
| save steps | 10 |
| early stop | entropy < 0.5 或 zero-std > 0.9，连续 5 步，至少 20 步 |
| lr | 3e-6 |
| beta | 0.05 |
| max len | 4096 |
| 其他 | bf16、gradient checkpointing、seed 0 |
| 输出 | `checkpoints/grpo_train_v2_r2/` |
| 日志 | `/tmp/opencode/grpo_train_v2_r2.log` |

关键命令参数：

```bash
--num-generations 16 --batch-size 16 --grad-accum 8 \
--max-steps 80 --save-steps 10 \
--early-stop-entropy 0.5 --early-stop-zero-std 0.9 \
--early-stop-patience 5 --early-stop-min-steps 20 \
--lr 3e-6 --beta 0.05
```

### 11.2 训练动态

80 步跑完，**早停未触发**；`decision_config.json` 记录 `steps: 80`、`planned_steps: 80`、`early_stopped: false`。没有 NaN / Inf / OOM。

| 指标（80 步） | mean | min | max | last |
|---|---:|---:|---:|---:|
| reward / acc_sampled | 0.3417 | 0.0234 | 0.6484 | 0.5234 |
| entropy | 1.1085 | 0.4854 | 1.8548 | 0.5829 |
| frac_zero_std_groups | 0.3625 | 0 | 0.7500 | 0.2500 |
| grad_norm | 5.9554 | 1.8415 | 24.2876 | 7.3101 |
| KL | 0.6224 | ~0 | 1.6841 | 1.6841 |
| loss | 0.0311 | ~0 | 0.0842 | 0.0842 |

每 20 步窗口：

| steps | reward | entropy | zero_std |
|---|---:|---:|---:|
| 1–20 | 0.2094 | 1.5646 | 0.3250 |
| 21–40 | 0.3609 | 1.2628 | 0.2625 |
| 41–60 | 0.3527 | 0.9464 | 0.3875 |
| 61–80 | 0.4437 | 0.6604 | 0.4750 |

与第一轮相比，两个 collapse 指标明显改善：

- entropy 最终 `0.583`，第一轮最终 `0.117`；全段均值 `1.109` vs 第一轮 `0.416`；
- `frac_zero_std_groups` 均值 `0.363` vs 第一轮 `0.745`；
- 80 步中**没有任何一步所有 group 都零方差**；第一轮有 33/193 步；
- 但 reward 均值 `0.342` 低于第一轮 `0.450`。这不是同条件对比：G 从 8 提到 16、lr 从 1e-5 降到 3e-6、beta 从 0.02 提到 0.05，前期上升自然更慢。

更重要的是，最后 20 步训练 reward 仍在上升（`0.4437`），但 dev accuracy 已从 step 40 的 `29.73%` 跌到 step 80 的 `24.32%`。这说明后 40 步主要是在让策略变尖，而不是提升泛化。

### 11.3 逐 checkpoint dev 结果

`checkpoints.jsonl` 记录了 step 10–70 的 dev 指标；最终 step 80 的指标在根目录 `eval.json`。

| step | accuracy | NLL | Brier top-1 | ECE |
|---:|---:|---:|---:|---:|
| 10 | 13.51% | 2.8163 | 0.2666 | 0.3694 |
| 20 | 21.62% | **2.8035** | **0.2876** | 0.3365 |
| 30 | 27.03% | 2.8135 | 0.3136 | 0.3465 |
| **40** | **29.73%** | 2.8366 | 0.3233 | **0.3172** |
| 50 | 29.73% | 2.9234 | 0.3395 | 0.3351 |
| 60 | 29.73% | 3.0344 | 0.3706 | 0.3686 |
| 70 | 27.03% | 3.1261 | 0.3879 | 0.4101 |
| 80（最终） | 24.32% | 3.1745 | 0.4021 | 0.4597 |

选点结论：

- **step 40 是最佳 top-1 / ECE 平衡点**：`29.73%`、ECE `0.3172`；它在 accuracy、NLL、Brier、ECE 上都优于 step 50/60。
- step 20 的 NLL / Brier 最好，但 accuracy 只有 `21.62%`；如果下游只看 argmax，step 40 更合适。
- 40 步之后 accuracy 不再提升，NLL / Brier / ECE 持续变差；最终 step 80 比 step 40 低 **5.4 pp**。
- 因此第二轮应部署 `checkpoints/grpo_train_v2_r2/step-000040/`，不要默认使用根目录的 step 80 adapter。

与初始 adapter、第一轮最终点的对比：

| 模型 / checkpoint | accuracy | NLL | Brier top-1 | ECE |
|---|---:|---:|---:|---:|
| 初始 AgentDecision-4B | 10.81% | 2.7985 | **0.2370** | 0.3701 |
| 第一轮最终（193 步） | **35.14%** | 3.6125 | 0.4440 | 0.4839 |
| 第二轮 step 40 | 29.73% | 2.8366 | 0.3233 | **0.3172** |
| 第二轮最终（80 步） | 24.32% | 3.1745 | 0.4021 | 0.4597 |

解读：

- 第二轮没有复现第一轮最终的 35.14% top-1；但第一轮的 35.14% 伴随 ECE `0.484`，概率明显过度自信。第二轮 step 40 牺牲了约 5.4 pp top-1，换来了更好的校准。
- dev 只有 37 条、3 个 session；初始 / 第一轮 / 第二轮的 95% Wilson 区间分别为 4.3%–24.7%、21.8%–51.2%、17.5%–45.8%，仍相互重叠。不能把 35.14% 和 29.73% 的差异当成稳定效应。
- 如果线上只用 argmax，第一轮最终点在这 37 条上更高；如果要用概率、阈值或置信度，第二轮 step 40 更合理，或者对第一轮最终点单独做 temperature calibration。
- 更可靠的结论需要更大的、按独立 session 留出的 dev；当前数据不足以判定哪一轮的泛化更好。

### 11.4 第二轮的结论

1. **checkpoint 流程有效，而且是必需的。** 最终点不是最优点；step 40 比 step 80 高 5.4 pp。以后不应默认保存/部署最后一个 step。
2. **G=16 + 更低 lr + 更大 beta 确实缓解了 entropy collapse。** entropy 和 zero-std 都显著好于第一轮，且没有全零方差步。
3. **但没有自动转化为更高 top-1。** 训练 reward 在 61–80 仍上升，dev 反而下降；这更像策略锐化/过拟合，而不是继续学习。
4. **早停条件设得太保守。** 当前 entropy < 0.5 / zero-std > 0.9 没有触发；若按 dev 选点，step 40–50 就应停止。后续应增加“dev 连续 N 次不提升”早停，或把 entropy 阈值提高到 0.6–0.7。
5. **当前最优部署候选是 step 40，而不是最终 adapter。** 若只看 top-1，第一轮最终点仍更高；两者取舍取决于是否需要概率校准。

### 11.5 下一轮建议

- 数据侧优先：增加独立 session 的 dev，至少几百条；否则 37 条上的 5 pp 差异无法解释。
- 训练侧：以 step 40 为参考，`max_steps` 降到 40–50；或保留 80 步但加 dev-based early stopping。
- 超参：`lr=3e-6`、`beta=0.05` 这组比第一轮更稳；若要追求 top-1，可在这个基础上小范围试 `lr=5e-6`，但必须每 10 步看 dev，不能跑满。
- 概率侧：对候选 checkpoint 做 temperature scaling，再比较 ECE / Brier；不要直接使用 GRPO 原始分布。
- 长尾工具：`web_search` / `ask_user_question` 在 train 中仍为 0，`execute` / `glob` / `webfetch` 等仍只有十几条；这部分不是调参能解决的，需要继续积累生产数据。

## 12. train_v3 对比：继续训练是否优于 train_v2 模型（2026-11-14）

### 12.1 数据与配置

`data/train_v3` 是 train_v2 之后的全量生产数据：

- train `2481` 条，dev `81` 条；其中 train `tool_select 2433` / `stop 48`，dev `tool_select 78` / `stop 3`。
- train 有 128 个 session，其中一个 session 占 975 条；dev 有 14 个 session，其中一个占 38 条。
- dev 的工具分布和 train 不完全一致：dev 里 `shell_run` 24 条，而 train 只有 5 条。
- 用 v2 第二轮 tokenizer 统计，train 最长 **4144 token**；因此本次 `--max-len 4608`，没有截断。

本次训练从 **v2 第二轮 step40** 继续（该 checkpoint 与线上 `AgentDecision-4B-1111` 字节一致），配置为：

```text
G=16, batch=16, grad-accum=8 (P=8)
max-steps=120, save-steps=10
lr=3e-6, beta=0.05, max-len=4608
early-stop: entropy<0.5 或 zero-std>0.9，连续 5 步，最早 step 40
```

训练在 **step 104** 触发早停（entropy `0.1912`、zero-std `1.0`），没有跑满 120 步。

### 12.2 同一 dev、同一评估口径的对比

训练时的 checkpoint eval 用 `batch_size=16`；复评发现 batch 大小会让 81 条 dev 中 1–2 条预测翻转。例如 v3 最终点在训练日志里是 `51.85%`，`batch_size=1` 复评是 `53.09%`。下表统一使用 **train_v3 dev + batch_size=1**，避免把 batch 噪声当成模型差异。

| 模型 | all acc | tool_select | stop | NLL | Brier top-1 | ECE |
|---|---:|---:|---:|---:|---:|---:|
| v2 第一轮 final（193 步） | **53.09%** (43/81) | **51.28%** (40/78) | 3/3 | **1.9339** | 0.3410 | 0.3491 |
| v2 第二轮 step40（起始/线上） | 49.38% (40/81) | 47.44% (37/78) | 3/3 | **1.6011** | **0.2446** | **0.2253** |
| v2 第二轮 final（80 步） | 50.62% (41/81) | 48.72% (38/78) | 3/3 | 1.7777 | 0.2916 | 0.3056 |
| v3 step20 | 50.62% (41/81) | 48.72% (38/78) | 3/3 | 1.7223 | 0.2671 | 0.2529 |
| v3 step80 | 51.85% (42/81) | 50.00% (39/78) | 3/3 | 1.8404 | 0.2983 | 0.3021 |
| v3 final step104 | **53.09%** (43/81) | **51.28%** (40/78) | 3/3 | 1.9613 | 0.3112 | 0.3177 |

注意：v2 第一轮原来的 `35.14%` 是在旧的 37 条 dev 上；这里是在 train_v3 的 81 条 dev 上重新评估的 `53.09%`，两个数字不能直接比较。

### 12.3 解读

1. **v3 final 与 v2 第一轮 final 的 top-1 完全打平**（all 43/81，tool_select 40/78）。v3 final 的 Brier / ECE 更好（`0.311 / 0.318` vs `0.341 / 0.349`），但 NLL 略差（`1.961` vs `1.934`）。也就是说，v3 用更好的概率质量达到了同样的 argmax 准确率。
2. **相对自己的起点 v2 step40，v3 final 确实是“更准但更自信”**：all acc `+3.70 pp`、tool_select `+3.85 pp`，但 NLL `1.601 → 1.961`、Brier `0.245 → 0.311`、ECE `0.225 → 0.318`。如果下游使用概率阈值，这个代价不能忽略。
3. **v3 step20 是更保守的折中点**：相对起点 `+1.23 pp`（all）和 `+1.28 pp`（tool_select），ECE 只从 `0.225` 升到 `0.253`，明显好于 final104 的 `0.318`。
4. **对比 v2 第二轮 final80**：v3 final104 的 all acc 高 `2.47 pp`，但 NLL / Brier / ECE 都更差；v3 step80 比 v2 final80 高 `1.23 pp`，ECE 略好（`0.302` vs `0.306`），NLL / Brier 仍更差。
5. **没有任何 v3 checkpoint 在 NLL / Brier / ECE 上超过起始的 v2 step40。** 这次继续训练的主要收益是 top-1 的小幅提升，不是概率校准；它没有复现 v2 第一轮那种“准确率大涨但严重过度自信”的幅度，但方向相同。
6. **早停发生在 step104，而 dev 在 step70 后基本平台化。** 训练 reward 在最后阶段仍上升，dev 没有同步提升；这再次说明“跑得更久”主要是在锐化策略，而不是继续学习。若按 dev 选点，step20 / step80 比 final104 更稳。

### 12.4 部署建议

- **只取 argmax、且能接受后续校准**：v3 final104 或 v2 第一轮 final；两者在这 81 条上打平。若必须在两者之间选，v3 final104 的 Brier / ECE 更好，优先 v3 final104。
- **需要概率 / 置信度阈值**：优先保留 v2 第二轮 step40（校准最好）；若必须使用 v3，选 **step20**，不要直接用 final104。
- **不建议**把 v3 final104 当成“全面优于 v2”的结论。它只是 top-1 打平、校准优于 v2 第一轮，但仍明显差于自己的起点。
- dev 只有 81 条 / 14 个 session，很多差异只对应 1–3 条样本；在扩大独立 session 的 dev 之前，不应据此做最终上线决策。

> **后续更新（2026-11-14）**：`train_v3` 的旧 81 条 dev 已被 `data/train_v3_clean/`
> 的 session-level 重划分取代（主 dev 264 条 / 11 sessions）；旧 dev 受单个 alt session
> 主导、缺少 main shell/edit，并含 16 条 v2 train 泄漏。§12 的 v2/v3 对比只作为历史记录，
> 新模型比较应在 clean split 上重训。详见 `data/train_v3_clean/SPLIT.md`。

## 13. 相关文档

- train_v3 训练报告：`results/training/grpo_train_v3_2026-11-14.md`
- 数据说明：`docs/DECISION_DATA_V2.md`
- 训练配方：`docs/TRAINING_RECIPES.md`
- 脚本：`scripts/train/60_grpo_trl.sh`
- 依赖约束：`configs/train/constraints-gpu.txt`

---

**一句话带走：** train_v2 的 GRPO 值得继续，但不要把“跑满一个 epoch”当成默认策略。第一轮的信号主要在前 20–40 步，后面是 entropy collapse；第二轮用 G=16 + 低 lr + 中间 checkpoint 缓解了 collapse，但 dev top-1 没有超过第一轮，最优点是 step 40 而不是 step 80。下一步优先补独立 session 的 dev，再谈调参和部署。