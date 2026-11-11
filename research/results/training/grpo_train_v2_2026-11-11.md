# train_v2 决策级 GRPO 实验记录（2026-11-11）

> 结论先行：在已发布的 `AgentDecision-4B` LoRA adapter 上，用 `data/train_v2` 做 1 epoch 决策级 GRPO，dev accuracy 从 **10.81%** 提升到 **35.14%**（+24.3 pp），但 NLL / Brier / ECE 同时变差。训练奖励在前 20–40 步快速上升，之后基本平台化，并伴随明显的 entropy collapse 与 zero-variance group 增多。最终 checkpoint 不是最优部署点；下一次应保存中间 checkpoint，按 dev 选点并早停。

- 日期：2026-11-11
- 类型：探索性实验（exploratory），不是预注册结果
- 代码：`src/jevharness/training/grpo_trl.py --impl torch`
- 数据：`data/train_v2/{train,dev}.jsonl`
- 起始模型：`Qwen/Qwen3-4B` + `AgentDecision-4B` PEFT adapter
- 产物：`checkpoints/grpo_train_v2/`（本地，未作为代码提交）

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

恢复后容器 Up，内部 `/health` 返回 200，GPU1 约 8.2 GB 显存占用。

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
- 概率质量变得更集中，但校准明显变差：模型更“自信”，NLL / Brier / ECE 都恶化。
- 37 条 dev 很小。准确率的 95% Wilson 区间大约为：
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
8. **最终 checkpoint 不一定是最好 checkpoint。** 这次只保存了最终 adapter，无法按 dev 回选；下一次必须保存中间点。
9. **dev 只有 37 条，结论要保守。** 需要更大的 dev 或交叉验证，才能把 +24 pp 的效应量估准。
10. **`--rewards correct` 足够。** train_v2 是硬标签，受限 readout 只采样 option code，format reward 没有信息量；不要为了“奖励更丰富”加入恒为常数的项。

## 8. 下一次训练建议

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

## 10. 相关文档

- 数据说明：`docs/DECISION_DATA_V2.md`
- 训练配方：`docs/TRAINING_RECIPES.md`
- 脚本：`scripts/train/60_grpo_trl.sh`
- 依赖约束：`configs/train/constraints-gpu.txt`

---

**一句话带走：** train_v2 的 GRPO 值得继续，但不要把“跑满一个 epoch”当成默认策略；这次真正的信号在前 20–40 步，后面主要是 entropy collapse 和过拟合式变尖。下一次请按 dev 选中间 checkpoint，并把校准单独处理。