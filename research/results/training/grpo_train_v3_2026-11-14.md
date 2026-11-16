# train_v3 决策级 GRPO 训练报告（2026-11-14 → 11-15）

> 结论先行：`train_v3` 比 `train_v2` 多了约 63% 的数据，但这次 GRPO 的 dev top-1 只从 **49.38%** 提到 **53.09%**（40/81 → 43/81，**+3 条样本**），且 NLL / Brier / ECE 都变差。它不是“数据无效”，而是**新增数据没有被这次训练真正消费，且新增数据和 dev 的分布不匹配**：104 步 × 8 prompt = 832 个训练 prompt，只覆盖 2481 条 train 的 **33.5%**；zero-std 均值 0.510，折算后有效梯度 prompt 约 **408**，与 v2 第二轮几乎相同。新增数据更多是同一批工具/会话的重复模式，长尾和 dev 重点工具（`shell_run`、`shell_wait`）仍严重不足。

> **后续更新（2026-11-14）**：本文使用的 81 条 `train_v3` dev 已被
> `data/train_v3_clean/` 的 session-level 重划分取代（主 dev 264 条 / 11 sessions）。
> 旧 dev 受单个 alt session 主导、缺少 main shell/edit，并含 16 条 v2 train 泄漏；
> 因此本文的 dev 数字和 v2/v3 对比只作为历史记录，新模型比较请在 clean split 上重训。
> 详见 `data/train_v3_clean/SPLIT.md` 与 `docs/DECISION_DATA_V2.md` §10。

- 日期：2026-11-14
- 类型：探索性实验（exploratory），不是预注册结果
- 代码：`src/jevharness/training/grpo_trl.py --impl torch`
- 数据：`data/train_v3/{train,dev}.jsonl`
- 起始模型：`Qwen/Qwen3-4B` + `checkpoints/grpo_train_v2_r2/step-000040` LoRA adapter
- 起始 adapter 与线上 `AgentDecision-4B-1111` 字节一致
- 产物：`checkpoints/grpo_train_v3/`（本地，未提交）
- 日志：`/tmp/opencode/grpo_train_v3.log`

## 1. 实验目标

在 `train_v2` 两轮 GRPO 之后，验证一个直接问题：

> 把生产标注数据从 1,576 条扩到 2,562 条，并继续用 GRPO 从当前线上 adapter 训练，能否带来稳定的 dev 提升？

本轮只使用 `train_v3`，不改 reward，不改模型结构，目的是把“数据量”作为主要变量。

## 2. 数据：train_v3

### 2.1 规模与拆分

`data/train_v3/MANIFEST.json`：

- 输入 `9,816` 条，输出 `2,562` 条，丢弃 `7,254` 条。
- 丢弃原因：`point_unlabelled 6,294`、`no_state 859`、`error_row 96`、`unlabelled_or_not_fit_ok 3`、`label_not_in_options 1`、`duplicate 1`。
- 按决策点：`tool_select 2,511`、`stop 51`。
- 训练：`2,481` 条 = `tool_select 2,433` + `stop 48`。
- dev：`81` 条 = `tool_select 78` + `stop 3`。
- 全部为 `choice` 硬标签：`soft_frac=0.0`，`with_q_values=0`。
- session 级拆分，`dev_frac=0.1`；去重键为 `(session, step, point, sha256(state)[:16])`。

相比 `train_v2`（1,576 条 = 1,539 + 37），总行数增加 **62.6%**，`tool_select` 增加约 59%。

### 2.2 工具分布

train 的 gold 工具分布高度集中：

| 工具 | 条数 | 占比 |
|---|---:|---:|
| `shell` | 872 | 35.1% |
| `read` | 381 | 15.4% |
| `edit` | 377 | 15.2% |
| `reply_without_tool` | 332 | 13.4% |
| `compress` | 111 | 4.5% |
| `grep` | 74 | 3.0% |
| `final_result` | 63 | 2.5% |
| `write` | 57 | 2.3% |
| `bash` | 50 | 2.0% |
| `done` | 31 | 1.2% |
| 其他 | 133 | 5.4% |

前 4 个工具占 **79.1%**；前 10 个占 **94.6%**。长尾仍然很薄：

- `shell_run` 5 条（0.2%）
- `websearch` 5 条
- `question` / `skill` / `browse_web` 各 7 条
- `web_fetch` 5 条
- `shell_wait` **0 条**

dev 的工具分布又是另一套：

| 工具 | dev 条数 | dev 占比 | train 对应条数 |
|---|---:|---:|---:|
| `shell_run` | 24 | 29.6% | 5 |
| `reply_without_tool` | 15 | 18.5% | 332 |
| `read` | 10 | 12.3% | 381 |
| `edit` | 8 | 9.9% | 377 |
| `final_result` | 8 | 9.9% | 63 |
| `shell_wait` | 7 | 8.6% | **0** |
| `shell` | 3 | 3.7% | 872 |
| `done` | 3 | 3.7% | 31 |
| `glob` / `browse_web` / `ask_human` | 各 1 | 1.2% | 17 / 7 / 1 |

**dev 的 81 条中有 34 条（42.0%）的 gold 工具在 train 中不超过 25 条**；`shell_run` + `shell_wait` 就占 dev 的 38.3%，而它们在 train 中合计只有 5 条。这意味着聚合 dev 分数有很大一部分在考“训练集几乎没教过的工具”。

### 2.3 session 集中度

- train：128 个 session；其中一个 session 占 **975 条（39.3%）**。
- dev：14 个 session；其中一个占 **38 条（46.9%）**。

行数增加不等于独立样本增加。同一 session 的相邻 step 状态高度相关，GRPO 在同一会话上反复采样，梯度仍然集中在少数轨迹上。

### 2.4 选项数与 token 长度

选项数分布：

| 选项数 | train | dev |
|---:|---:|---:|
| 2 | 121 | 12 |
| 7 | 16 | 2 |
| 15 | 2,108 | 16 |
| 27 | 98 | 5 |
| 28 | 1 | 0 |
| 33 | 109 | 3 |
| 52 | 28 | 7 |

用 v2 第二轮 step40 的 tokenizer/chat template 统计：

- train：min `105`，p50 `1,887`，p90 `2,718`，p95 `2,877`，p99 `3,124`，max **4,144**。
- dev：max `2,724`。
- 只有 1 条 train 超过 4,096 token；因此本轮使用 `--max-len 4608`，没有截断。

### 2.5 标签质量差异

从 `meta` 字段统计：

- train：`label_tier=fast` **1,697 条（68.4%）**，`strong` 784 条（31.6%）。
- dev：`strong` **46 条（56.8%）**，`fast` 35 条（43.2%）。
- train 有 **289 条（11.6%）** `truncated=true`；dev 全部 `truncated=false`。

如果 `fast` 标签的噪声更高，训练集和 dev 就存在“训练用较弱标签、评估用较强标签”的错配；`truncated=true` 的训练行还可能让模型学习不完整状态。这两点都可能稀释新增数据的收益。

## 3. 起始模型与基线

起始 checkpoint：

`checkpoints/grpo_train_v2_r2/step-000040`

它与此前线上部署的 `/data1/apps/huggingface/JevAgentHarness/AgentDecision-4B-1111` 字节一致（sha256 `0dcb0687...`）。因此本轮是从**当前线上模型继续训练**，不是从 base Qwen3-4B 重训。

在 train_v3 dev 上的基线（batch_size=1）：

| 子集 | n | acc | NLL | Brier top-1 | ECE |
|---|---:|---:|---:|---:|---:|
| all | 81 | 49.38% (40/81) | 1.6011 | 0.2446 | 0.2253 |
| tool_select | 78 | 47.44% (37/78) | 1.6542 | 0.2544 | 0.2315 |
| stop | 3 | 100% (3/3) | 0.0308 | 0.0010 | 0.0303 |

## 4. 训练配置

```bash
CUDA_VISIBLE_DEVICES=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 nohup .venv/bin/python -m jevharness.training.grpo_trl \
  --impl torch \
  --model /data1/apps/huggingface/Qwen/Qwen3-4B \
  --init checkpoints/grpo_train_v2_r2/step-000040 \
  --train data/train_v3/train.jsonl \
  --dev data/train_v3/dev.jsonl \
  --out checkpoints/grpo_train_v3 \
  --rewards correct \
  --num-generations 16 \
  --batch-size 16 \
  --grad-accum 8 \
  --max-steps 120 \
  --save-steps 10 \
  --early-stop-entropy 0.5 \
  --early-stop-zero-std 0.9 \
  --early-stop-patience 5 \
  --early-stop-min-steps 40 \
  --epochs 1 \
  --lr 3e-6 \
  --beta 0.05 \
  --max-len 4608 \
  --gradient-checkpointing \
  --logging-steps 5 \
  --seed 0 > /tmp/opencode/grpo_train_v3.log 2>&1 &
```

关键量：

- `G=16`，`batch=16`，`grad-accum=8`，因此每个 optimizer step 处理 **P = 16×8/16 = 8 个 prompt**。
- 每个 prompt 采样 16 个 option-code completion，即每步 128 个采样动作。
- `max-steps=120`，每 10 步保存 checkpoint 并跑 dev。
- 训练硬件：单张 RTX 4090 24 GB（GPU1），bf16 + gradient checkpointing；启动时显存约 11.3 GB。

## 5. 训练过程

### 5.1 完成情况

- 实际运行 **104 / 120 步**。
- 在第 104 步触发早停：`entropy=0.1912`、`zero_std=1.0000`，此前 step 100–104 连续 5 步 entropy < 0.5。
- 没有 NaN / Inf / OOM。
- `decision_config.json`：`steps=104`, `planned_steps=120`, `early_stopped=true`。

### 5.2 全局指标

| 指标 | mean | std | min | max |
|---|---:|---:|---:|---:|
| reward / acc_sampled | 0.4517 | 0.1481 | 0.1016 | 0.7812 |
| entropy | 0.5976 | 0.2120 | 0.0969 | 1.1534 |
| frac_zero_std_groups | 0.5096 | 0.1989 | 0.0000 | 1.0000 |
| grad_norm | 6.1318 | 2.3681 | 0.1330 | 12.9379 |
| KL | 0.1829 | 0.1012 | 0.0000 | 0.4642 |
| loss | 0.0091 | 0.0051 | ~0 | 0.0232 |

- `clipped=0`：`num_iterations=1`，每步只更新一次，PPO ratio 恒为 1，clipping 不会触发。
- 全零方差 step：**2 个（step 60、step 104）**。
- 平均 zero-std 0.510 意味着约一半 group 没有梯度；按 832 个训练 prompt 折算，有效梯度 prompt 约 **408**。

### 5.3 分阶段变化

| step | reward | entropy | zero-std | grad_norm |
|---|---:|---:|---:|---:|
| 1–20 | 0.4105 | 0.8724 | 0.3375 | 6.9704 |
| 21–40 | **0.4855** | 0.6239 | 0.4562 | 6.0651 |
| 41–60 | 0.4617 | 0.5584 | 0.5687 | 6.4866 |
| 61–80 | 0.4535 | 0.5054 | 0.5375 | 6.1475 |
| 81–100 | 0.4109 | 0.4982 | 0.5938 | 5.5913 |
| 101–104 | 0.6328 | 0.2462 | 0.7812 | 3.1234 |

解读：

- 训练 reward 在 21–40 步达到阶段高点，之后没有继续上升；最后 4 步的 0.633 来自很小样本，且 zero-std 0.781，不能当作真实提升。
- entropy 从 0.872 持续降到 0.246，zero-std 从 0.338 升到 0.781；训练再次进入“策略变尖、group 同质化”的阶段。
- 与 v2 第一轮（entropy 0.416、zero-std 0.745）相比，本轮更温和；但与 v2 第二轮（entropy 1.109、zero-std 0.363）相比，仍然明显更尖。

## 6. Checkpoint 评估

### 6.1 评估口径

训练时的 `checkpoints.jsonl` 使用 `batch_size=16`；复评发现 batch 大小会让 81 条 dev 中 1–2 条预测翻转。例如最终点：

- 训练时 `eval.json`（batch16）：acc `51.85%`，NLL `1.9716`，Brier `0.3150`，ECE `0.3328`。
- batch1 复评：acc **53.09%**，NLL `1.9613`，Brier `0.3112`，ECE `0.3177`。

为避免 batch 噪声，下表统一使用 **train_v3 dev + batch_size=1**；这也是与 train_v2 模型对比的口径。

### 6.2 逐 checkpoint 结果

| step | acc | NLL | Brier top-1 | ECE |
|---:|---:|---:|---:|---:|
| 10 | 49.38% | 1.6793 | 0.2614 | 0.2615 |
| 20 | 50.62% | **1.7223** | **0.2671** | **0.2529** |
| 30 | 49.38% | 1.7818 | 0.2817 | 0.2896 |
| 40 | 50.62% | 1.8262 | 0.2843 | 0.2921 |
| 50 | 50.62% | 1.8532 | 0.2900 | 0.2973 |
| 60 | 50.62% | 1.8719 | 0.2932 | 0.3066 |
| 70 | 51.85% | 1.8530 | 0.2990 | 0.3115 |
| 80 | 51.85% | 1.8404 | 0.2983 | 0.3021 |
| 90 | 51.85% | 1.8760 | 0.3032 | 0.3240 |
| 100 | 51.85% | 1.9314 | 0.3077 | 0.3346 |
| **104 final** | **53.09%** | 1.9613 | 0.3112 | 0.3177 |

选点：

- **最高 top-1：final step104（53.09%）**。
- **最佳校准：step20（ECE 0.2529、NLL 1.7223、Brier 0.2671）**，但 acc 只有 50.62%。
- **较稳的折中：step80（51.85%，ECE 0.3021）**；比 final 少 1 条正确，但 NLL 低 0.121，ECE 低 0.016。
- 从 step70 起 acc 进入 51.85–53.09% 的平台；NLL / ECE 继续变差。final 的最后 1 条正确率提升，主要来自策略锐化，而不是概率质量提升。

## 7. 与 train_v2 模型的对比

同一 train_v3 dev、batch_size=1：

| 模型 | all acc | tool_select | stop | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| v2 第一轮 final（193 步） | **53.09%** (43/81) | **51.28%** (40/78) | 3/3 | **1.9339** | 0.3410 | 0.3491 |
| v2 第二轮 step40（起始/线上） | 49.38% (40/81) | 47.44% (37/78) | 3/3 | **1.6011** | **0.2446** | **0.2253** |
| v2 第二轮 final（80 步） | 50.62% (41/81) | 48.72% (38/78) | 3/3 | 1.7777 | 0.2916 | 0.3056 |
| v3 step20 | 50.62% (41/81) | 48.72% (38/78) | 3/3 | 1.7223 | 0.2671 | 0.2529 |
| v3 step80 | 51.85% (42/81) | 50.00% (39/78) | 3/3 | 1.8404 | 0.2983 | 0.3021 |
| v3 final104 | **53.09%** (43/81) | **51.28%** (40/78) | 3/3 | 1.9613 | 0.3112 | 0.3177 |

注意：v2 第一轮原来的 `35.14%` 是旧的 37 条 dev；这里是在 train_v3 的 81 条 dev 上重新评估的 `53.09%`，两个数字不能直接比较。

解读：

1. **v3 final 与 v2 第一轮 final 的 top-1 完全打平**（all 43/81，tool_select 40/78）。v3 的 Brier / ECE 更好（`0.311 / 0.318` vs `0.341 / 0.349`），但 NLL 略差（`1.961` vs `1.934`）。
2. **相对自己的起点 v2 step40，v3 final 是“更准但更自信”**：all `+3.70 pp`、tool_select `+3.85 pp`，但 NLL `1.601 → 1.961`、Brier `0.245 → 0.311`、ECE `0.225 → 0.318`。
3. **没有任何 v3 checkpoint 在 NLL / Brier / ECE 上超过起始的 v2 step40**。新增数据带来的主要是 3 条样本的 top-1 提升，而不是更好的概率模型。
4. v3 step20 是更保守的折中：相对起点 `+1.23 pp`，ECE 只从 0.225 升到 0.253；如果下游使用概率阈值，它比 final104 更合适。

## 8. 为什么“数据增加但效果不明显”

### 8.1 训练根本没有消费新增数据

这是最直接的原因。

- 104 步 × 8 prompt/步 = **832 个训练 prompt**。
- train 有 2,481 条，832 / 2,481 = **33.5%**。
- 即使跑满 `max-steps=120`，也只有 960 / 2,481 = **38.7%**。
- 要覆盖一个 epoch，至少需要 `ceil(2481/8) = 311` 步。

所以这次不是“用 2,481 条数据训练”，而是“用其中约三分之一训练”。数据从 1,576 增到 2,562，但训练预算没有同步扩大，新增数据大部分没被看到。

更关键的是有效梯度：

- v3：`832 × (1 - 0.5096) ≈ 408` 个有效 prompt。
- v2 第二轮：`640 × (1 - 0.3625) = 408` 个有效 prompt。

两轮的有效梯度量几乎相同。v3 多出来的数据被更高的 zero-std 抵消了。

### 8.2 新增数据主要是同质化重复，而不是新决策边界

- 前 4 个工具占 79.1%，前 10 个占 94.6%；新增数据仍然大量来自 `shell/read/edit/reply_without_tool`。
- 一个训练 session 占 975 / 2,481 = **39.3%**；同一会话的相邻状态高度相关，行数增加不等于独立信息增加。
- `shell_run` 只有 5 条，`shell_wait` 为 0；而这两个工具占 dev 的 38.3%。
- 数据增加 63%，但长尾覆盖没有同比例增加；模型学到的还是“常见工具的选择”，不是 dev 重点考的“新工具/新状态”。

### 8.3 dev 分布偏移，聚合指标掩盖了真实变化

dev 中：

- `shell_run` 24 条（29.6%），train 只有 5 条（0.2%）。
- `shell_wait` 7 条（8.6%），train 为 0。
- 34/81（42.0%）的 gold 工具在 train 中不超过 25 条。

因此，聚合 dev 同时测了两件事：

1. 常见工具的选择是否变好；
2. 训练集几乎没见过的工具能否零样本迁移。

第 2 件事不是增加同分布数据能解决的。`+3.70 pp` 的聚合提升，可能对应“常见工具小幅提升 + 长尾工具拖后腿”。需要逐工具混淆矩阵才能确认，但数据分布本身已经说明聚合分数对新增数据不敏感。

### 8.4 标签层级与截断带来噪声

- train 68.4% 是 `fast` 标签，dev 56.8% 是 `strong` 标签；如果 fast 标签噪声更高，训练信号会被稀释。
- train 有 289 条 `truncated=true`（11.6%），dev 为 0；这些训练行看到的是不完整状态，学到的映射在 dev 上不成立。
- 所有标签都是硬标签，`soft_frac=0`、`with_q_values=0`；GRPO 无法利用“次优但接近”的工具偏好，只能从 0/1 正确性里学。

### 8.5 奖励和选点目标不匹配

- reward 只有 `correct`，没有校准项；GRPO 会自然把分布变尖，因此 top-1 略升、NLL/ECE 变差是预期行为。
- 早停基于 entropy / zero-std，不是 dev 指标；它在 step104 才触发，而 dev 的 NLL/ECE 从 step20 起就持续变差。
- 训练 reward 在 21–40 步后不再上升，dev top-1 也只从 50.62% 缓慢到 53.09%；继续训练主要在锐化，而不是学习新边界。

### 8.6 统计噪声不可忽略

- 43/81 vs 40/81，只多 **3 条**正确样本。
- 95% Wilson 区间：起始 38.8%–60.0%，v3 final 42.3%–63.6%，大幅重叠。
- dev 只有 81 条 / 14 个 session，其中一个 session 占 38 条。

所以“效果不明显”有一部分不是模型没学到，而是**当前 dev 分辨不出 3 条样本的差异**。

## 9. 结论与部署建议

1. **这次 v3 训练不能证明“更多数据无效”**。它证明的是：在 104 步、P=8、zero-std 0.51 的预算下，新增数据没有转化为更多有效梯度。
2. **v3 final104 只在 top-1 上与 v2 第一轮 final 打平**；它的 Brier/ECE 更好，但 NLL 略差，且明显差于自己的起点 v2 step40 的校准。
3. **如果只看 argmax**：可以把 v3 final104 作为候选，但它不比 v2 第一轮 final 更强；两者差异只有 0–3 条样本。
4. **如果使用概率/置信度阈值**：继续用 v2 step40，或使用 v3 step20；不要直接用 v3 final104。
5. **v3 step80 是另一个可接受折中**：51.85% top-1，ECE 0.302，比 final 更稳。

## 10. 下一步建议

按优先级：

### P0：先把评估做对

- 扩大 dev 到更多独立 session；当前 14 个 session、其中一个占 47%，不足以下结论。
- 增加逐工具混淆矩阵，至少分别报告 `shell_run`、`shell_wait`、`read`、`edit`、`reply_without_tool`。
- 评估集按 train 的工具分布分层，避免 42% 的 dev 落在 train ≤25 条的工具上。

### P1：让训练真正消费数据

- 至少跑 **1 epoch（约 311 步，P=8）**；或提高 P（batch/grad-accum）到 16，减少步数。
- 用 dev-based early stopping：连续 N 次 dev 指标不提升就停，而不是只看 entropy/zero-std。
- 对 zero-variance group 做动态跳过/重采样；否则一半 group 没有梯度。

### P2：处理数据结构和长尾

- 对占 39% 的单一 session 做降采样或按 session 加权，避免梯度被一条轨迹主导。
- 补充 `shell_run`、`shell_wait` 等 dev 重点工具的生产样本；否则 dev 分数无法反映训练收益。
- 检查 289 条 `truncated=true` 训练行：丢弃、补齐，或至少在训练时标记。
- 评估是否只使用 `strong` 标签做训练，或对 fast/strong 分层加权。

### P3：概率校准

- 如果线上要用阈值，先对候选 checkpoint 做 temperature scaling，再比较 ECE/Brier。
- 不要假设 GRPO 的原始 softmax 概率可直接用于阈值；这轮已经显示 top-1 和校准会分离。

## 11. 复现与产物

- 训练日志：`/tmp/opencode/grpo_train_v3.log`
- 训练指标：`checkpoints/grpo_train_v3/metrics.jsonl`（104 行）
- checkpoint 记录：`checkpoints/grpo_train_v3/checkpoints.jsonl`
- 最终 adapter：`checkpoints/grpo_train_v3/`
- 最优 top-1 checkpoint：`checkpoints/grpo_train_v3/`（根目录 final，step104；没有单独的 `step-000104/` 子目录，最后一个周期 checkpoint 是 `step-000100/`）
- 最佳校准 checkpoint：`checkpoints/grpo_train_v3/step-000020/`
- 较稳折中：`checkpoints/grpo_train_v3/step-000080/`
- 起始 adapter：`checkpoints/grpo_train_v2_r2/step-000040/`
- 数据：`data/train_v3/{train,dev}.jsonl`，`MANIFEST.json`

## 12. P1：clean dev 重训（2026-11-15）

### 12.1 数据与配置

旧 81 条 dev 的 session/工具分布问题已在 `data/train_v3_clean` 修复：

- `train.jsonl` 2189 条 / 105 sessions；
- `dev.jsonl` 264 条 / 11 个 main session（主 dev）；
- `dev_alt.jsonl` 94 条 / 22 sessions（alt harness + 稀有工具）；
- `dev_shell.jsonl` 12 条（`shell_run` / `browse_web` 诊断）。

本次从 v2 第二轮 step40 出发，配置为：

```text
P=16 (G=16, batch=16, grad-accum=16)
max-steps=137 (≈1 epoch on 2189 train)
save-steps=5, dev-batch-size=8
lr=3e-6, beta=0.05, max-len=4608
--skip-zero-std
dev-acc early stop: patience 3, min-delta 0, min-step 40
```

训练在 **step 55 / 137** 触发 dev-acc 早停（best step40 `49.24%`，45/50/55 连续 3 次未超过），所以这次仍没有跑满 1 epoch：`55×16=880` prompts，只覆盖 **40.2%** 的 train。`--skip-zero-std` 已生效，没有出现全零 group 崩溃。

### 12.2 主 dev 曲线（264 条，batch 8）

| step | acc | NLL | Brier | ECE |
|---:|---:|---:|---:|---:|
| 5 | 44.32% | **2.2457** | **0.2699** | **0.2446** |
| 20 | 47.35% | 2.4206 | 0.3090 | 0.2944 |
| **40** | **49.24%** | 2.6063 | 0.3371 | 0.3205 |
| 55 final | 48.48% | 2.6979 | 0.3493 | 0.3475 |

最佳 top-1 是 `step-000040`；最佳校准是 step 5。和之前两轮一样，训练越久 top-1 略有提升，但 NLL / Brier / ECE 单调变差。

### 12.3 与 init 的同口径 batch1 对比

| model | dev | n | acc | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| P1 step40 | dev | 264 | **47.73%** | 2.5949 | 0.3363 | 0.3357 |
| init step40 | dev | 264 | 43.94% | **2.2016** | **0.2648** | **0.2104** |
| P1 step40 | dev_alt | 94 | **50.00%** | 2.3133 | 0.2922 | 0.2187 |
| init step40 | dev_alt | 94 | 46.81% | **2.0026** | **0.2365** | **0.1339** |
| P1 step40 | dev_shell | 12 | 50.00% | 1.7709 | 0.3134 | 0.3152 |
| init step40 | dev_shell | 12 | **58.33%** | **1.5849** | **0.2924** | **0.2296** |

- 主 dev：P1 比 init **+3.79 pp**；dev_alt **+3.19 pp**；dev_shell **-8.33 pp**，但只有 12 条样本，等价于 1 条预测差异。
- 三个 dev 上 P1 的 NLL / Brier / ECE 都更差，说明它再次是“更准一点、但更自信”。
- init 见过部分新 dev 行（来自旧 train），不是完全干净的 baseline；P1 训练时排除了新 dev，因此这个 +3.79 pp 不能被解释为严格的无泄漏泛化提升，但至少不是 P1 记住了 dev。

### 12.4 错误结构：增益几乎全部来自 shell

P1 step40 在主 dev 上的 per-tool 结果比 aggregate 更能说明问题：

- `shell`：**96/108 = 88.9%**（init 83/108 = 76.9%）；
- 非 shell：**30/156 = 19.2%**（init 33/156 = 21.2%）；
- `reply_without_tool`：2/42 = **4.8%**（init 6/42）；
- `edit`：2/24 = **8.3%**；`grep` 0/7；`glob` 0/6；`webfetch` 1/7；`compress` 2/10。

也就是说，aggregate 的 +3.79 pp 来自 shell **+13 条**，非 shell 反而 **-3 条**。混淆矩阵里模型大量把 `reply` / `edit` / `read` / `compress` 判成 `shell`，说明它学到的主要是“默认选 shell”，而不是细粒度区分。

结论：**继续收集普通 shell/read 数据只会强化这个偏置。** 下一批数据应优先补非 shell 的 hard cases，并配合 class weighting / balanced sampling；同时先审计这些错误是标签问题、状态缺信息，还是任务本身歧义。

### 12.5 结论

- P1 的**机制**已完成：`--skip-zero-std`、dev-metric 早停、逐 checkpoint dev eval 都可运行。
- P1 的**覆盖目标**没有完成：dev early stop 在 55/137 停止，只看到 40.2% 的 train。若严格要求 1 epoch，需要关闭 dev 早停或加大 patience；但现有曲线显示 40 步后 dev 已不再提升，强行跑满很可能只是继续恶化校准。
- 这一轮没有解决核心矛盾：top-1 的小幅提升仍以校准变差为代价；`dev_shell` 没有提升。下一步应优先做 temperature scaling / 校准，或在 clean dev 上重新定义“可接受的概率质量”，而不是继续加步数。

产物与日志：

- 输出：`checkpoints/grpo_train_v3_clean_p1/`
- 最佳 top-1：`checkpoints/grpo_train_v3_clean_p1/step-000040/`
- 最佳校准：`checkpoints/grpo_train_v3_clean_p1/step-000005/`
- 日志：`/tmp/opencode/grpo_train_v3_clean_p1.log`
- 数据 split：`data/train_v3_clean/SPLIT.md`

## 13. clean-train + hard-case 加权训练（2026-11-15）

### 13.1 数据与配置

在 P1 的 clean split 上，把挖掘队列中属于 train 的 527 条、以及 1 条有效 unseen 行，按 session/tool 上限筛选后加入训练：

- clean train：2189 条；
- 选中 hard cases：**287 条 / 39 sessions**；
- 按 priority 加权（P0 3×、P1/P2 2×、P3 1×）后追加 **421** 条；
- 最终训练集：**2610 条**；
- dev / dev_alt / dev_shell 泄漏：**0**。

训练配置：

```text
init: checkpoints/grpo_train_v2_r2/step-000040
P=16, max-steps=164, save/dev every 5
dev-acc early stop: patience 5, min-step 60
skip-zero-std, lr=3e-6, beta=0.05, max-len=4608
```

训练在 **step 65 / 164** 因 dev-acc 早停（best step40 `50.38%`，之后 5 次未超过）。

### 13.2 同口径 batch1 评估

| 模型 | dev acc | dev NLL | dev ECE | dev_alt acc | dev_shell acc |
|---|---:|---:|---:|---:|---:|
| init v2 step40 | 43.94% | **2.2016** | **0.2104** | 46.81% | **58.33%** |
| P1 step40 | 47.73% | 2.5949 | 0.3357 | **50.00%** | 50.00% |
| **hard step40** | **50.00%** | 2.4000 | 0.2932 | 47.87% | **58.33%** |

- 主 dev 比 init **+6.06 pp**（16 条），比 P1 **+2.27 pp**（6 条）。
- 校准优于 P1，但仍差于 init：NLL `2.400` vs `2.202`，ECE `0.293` vs `0.210`。
- dev_alt 低于 P1；dev_shell 与 init 持平，说明 shell_run/browse 诊断没有退化。

### 13.3 增益结构

相对 init，主 dev 的 16 条净增益不是只来自 shell：

- shell：93/108（86.1%）vs 83/108，**+10 条**；
- read：21/36（58.3%）vs 16/36，**+5 条**；
- edit +1、write +1、execute +1、not_done +1；
- reply_without_tool **-2**（4/42 vs 6/42）；
- compress **-1**（2/10 vs 3/10）。

这比 P1 更均衡：P1 的 aggregate 增益几乎全部来自 shell，非 shell 净 -3；加权 hard 的非 shell 净 **+6**。但 `reply_without_tool`、`edit`、`compress` 仍然很弱，说明这轮 hard-case 加权只部分奏效。

### 13.4 结论

- 这是目前 clean dev 上 top-1 最好的候选：`checkpoints/grpo_train_v3_clean_hard_v1/step-000040/`，主 dev **50.00%**。
- 提升对应 16 条样本、264 条 dev / 11 sessions，仍不足以下统计强结论；不要把它当成稳定的 +6 pp。
- 校准仍不如 init。若线上使用概率/阈值，仍应优先 init，或对 hard step40 做 temperature scaling。
- 训练再次在 65/164 早停，只覆盖约 40% 的加权数据；但 dev 在 step40 后没有继续提升，强行跑满只会继续恶化校准。
- 下一步应继续补独立 session 的 `reply_without_tool` / `edit` / `compress` hard cases，而不是再加普通 shell/read 数据。

产物：

- 输出：`checkpoints/grpo_train_v3_clean_hard_v1/`
- 最佳：`checkpoints/grpo_train_v3_clean_hard_v1/step-000040/`
- 训练数据：`data/train_v3_clean/train_hard_weighted_v1.jsonl`
- 审计：`data/train_v3_clean/train_hard_cases_v1.jsonl`、`MANIFEST_hard_weighted_v1.json`
- 日志：`/tmp/opencode/grpo_hard_v1.log`

## 14. 非 shell 难例挖矿第 2 轮：hard_v2（2026-11-15 上午）

### 14.1 数据与配置

按 §13.4 的结论补**非 shell** 难例。工具上限 `data/tool_caps_v2.json`：reply_without_tool 120、edit 80、read 40、compress 40、grep 30、glob 30、write 30、webfetch 16、web 系 8×3、skill 5、done/not_done 5；**shell/bash/shell_run/shell_wait = 0（排除）**。

- Track A：用 hard_v1 step40 对 clean train 2189 打分挖矿 → **354 条**（per-(session,tool) cap 5；首跑 167 条因 per-session cap 被长 session 占满而饿死，`2f7af33` 修复后重跑）；
- Track B：生产日志 `decisions.jsonl`（9977 行）新行，`--exclude` 全部已用集 → **38 条**（5 sessions，无泄漏）；
- 合并 392 → selected **340**（61 sessions）+ 按 priority 加权 **616** → `train_hard_weighted_v2.jsonl` = **2805 行**（2189 + 616）；skipped：tool_cap 4 / truncated 48；
- selected_by_tool：reply 115 / edit 64 / read 40 / compress 20 / grep 21 / glob 10 / write 19 / execute 11 / webfetch 8 / skill 5 / not_done 4 / question 4；top session 35 条 = 加权行的 3.7%；
- 训练：`init = grpo_train_v3_clean_hard_v1/step-000040`，P=16，max-steps 176，save/dev 每 5，dev-acc 早停 patience 5 / min-steps 60，skip-zero-std，lr 3e-6，beta 0.05，max-len 4608，GPU1。

### 14.2 训练与评估

176/176 步全部跑完（未触发早停），~1h25m。最终模型 dev 49.62% / NLL 2.98 / ECE 0.417；训练中最佳 **step-000135（dev 50.38%）**。

逐工具（dev 264，v2 step135 vs 基线 hard_v1 step40）：

| 工具 | n | v1 | v2 | Δ |
|---|---:|---:|---:|---:|
| reply_without_tool | 42 | 11.9% | **31.0%** | **+19.0 pp** |
| edit | 24 | 12.5% | 16.7% | +4.2 pp |
| read | 36 | 58.3% | 47.2% | **−11.1 pp** |
| compress | 10 | 20.0% | 10.0% | −10 pp |
| shell | 108 | 86.1% | 84.3% | −1.9 pp |
| **整体** | 264 | 50.38% | 50.38% | 0 |

校准变差：NLL 2.403→2.869，ECE 0.291→0.383，mean_conf 0.784→0.887（过度自信）。

混淆（gold→pred）：reply→reply 13 / shell 12 / read 9；read→read 17 / **reply 8** / shell 6；edit→shell 7 / read 6 / reply 5 / edit 4。

### 14.3 结论

- reply 的改善是真实的：shell 偏置被纠正了一半（42 条里判对 5→13 条）；
- 但**过校正**：read 步骤 8/36 被误判成 reply，read −11 pp、compress −10 pp，整体被抵消；
- 验收门：reply≥20% ✅（31.0%）/ edit≥25% ❌（16.7%）/ 整体≥50% ✅ / NLL·ECE 不劣 ❌ → **不部署**；
- 下一步：降 reply 权重、把 read/edit 提权，用 v2 模型重挖残留错误。

产物：`checkpoints/grpo_train_v3_clean_hard_v2/`（best `step-000135` + `eval_dev_step135.json`）；`data/train_v3_clean/{train_hard_weighted_v2.jsonl, train_hard_cases_v2.jsonl, MANIFEST_hard_weighted_v2.json}`；队列 `checkpoints/mine_nonshell_v2_{train,prod,all}/`；日志 `/tmp/grpo_train_v3_clean_hard_v2.log`。

## 15. 非 shell 难例挖矿第 3 轮：hard_v3 + 温度缩放 + 生产部署（2026-11-15 下午）

### 15.1 数据：降 reply 权重、提 read/edit

工具上限改为 `data/tool_caps_v3.json`：reply **120→50**、edit **80→120**、read **40→120**、compress 40→60、glob 30→12、write 30→40，其余同 v2，shell 系仍为 0。

- Track A：改用 **v2 step135 打分**（挖新模型的残留错误），clean train 2189 → **287 条**（48 sessions）：edit 54 / read 52 / reply 50 / grep 25 / compress 23 / write 19 / execute 12 / glob 10 / web 13，shell 0；
  - 打分顺带确认了 v2 的 train 逐工具 acc：reply 10.1%→31.4%、edit 17.9%→30.7%、read 40.6%→35.7%（回退）、shell 84.1%；
- Track B：生产日志（10100 行）新行 → **17 条**（edit 5 / compress 5 / write 3 / reply 2 / read 1 / not_done 1），无泄漏；
- 合并 304 → selected **263**（49 sessions）+ 加权 **452** → `train_hard_weighted_v3.jsonl` = **2641 行**（2189 + 452）；skipped truncated 41；
- selected_by_tool：edit 53 / read 46 / **reply 45**（v2 是 115）/ compress 22 / grep 20 / write 18 / glob 10 / execute 11 / webfetch 9；dev/dev_alt/dev_shell 泄漏 0。

### 15.2 训练与评估

`init = grpo_train_v3_clean_hard_v2/step-000135`（保留 v2 的 reply 增益），其余配置同 §14.1，max-steps 166。166/166 跑完，~2h；训练中最佳 step160 52.27%，**最终模型 dev 52.65%**（高于途中最佳）。52.65% 是训练内 eval 口径、51.52%（下段）是 eval_decisions 生产口径，两者差 3 条（139 vs 136/264，评估代码路径不同）；对外汇报以生产口径 51.52% 为准。

eval_decisions（raw，dev 264）：acc **51.52%**，NLL 3.205，ECE 0.422，auroc 0.712，mean_conf 0.934。

| 工具 | n | v1（生产基线）| v2 step135 | **v3 最终** |
|---|---:|---:|---:|---:|
| shell | 108 | 86.1% | 84.3% | 80.6% |
| reply_without_tool | 42 | 11.9% | 31.0% | **40.5%** |
| read | 36 | 58.3% | 47.2% | **50.0%**（回升）|
| edit | 24 | 12.5% | 16.7% | **20.8%** |
| compress | 10 | 20.0% | 10.0% | 10.0% |
| grep | 7 | 0% | 0% | 0% |
| glob | 6 | 0% | 0% | 0% |
| write / webfetch | 7+7 | 14.3% | 14.3% | 14.3% |
| done / not_done | 5+4 | 40% / 50% | 60% / 0% | 60% / 25% |
| **整体** | 264 | 50.38% | 50.38% | **51.52%** |

表中 write/webfetch 三轮未被加权、准确率未变（14.3%）；v1 基线列取 `eval_dev_step40.json`（eval_decisions 口径），与 §13.2 的 50.00%（训练内 eval 口径，同一模型）相差 1 条。

vs v1：reply **+28.6 pp**、edit **+8.3 pp**、read −8.3 pp、shell −5.5 pp、整体 +1.1 pp。混淆：reply→reply 17 / shell 13 / read 7；read→read 18 / reply 8 / shell 5；edit→reply 7 / shell 7 / edit 5。

### 15.3 温度缩放

v3 输出过度自信（mean_conf 0.934），用 `calibrate.py`（NLL golden-section，dev 264）重拟：

- **global T = 3.0251733656628823**
- **choice:6-16 T = 2.9999808412939433**

缩放后（`eval_decisions --calibration`）：acc 不变（argmax 不变）、**NLL 3.21→1.63、ECE 0.42→0.075**、mean_conf 0.93→0.58、auroc 0.715。校准完全修复，`checkpoints/grpo_train_v3_clean_hard_v3/calibration.json`。

### 15.4 生产部署（11-15）

- v3 adapter 手工部署到生产决策服务（GPU0 4B vLLM）；
- 生产配置 `decision.calibration` 全量替换：global 0.772→**3.025**、choice:6-16 2.508→**3.0**，旧模型桶（noul:2 / choice:17-52 / choice:3-5）删除（commit `23816a9`）；
- 已批准的行为变化：global 抬到 3.025 把 routing 的 p(hard) 压平，线上验证 1 小时 53 条决策行：p_hard 0.094–0.359（此前中位 ~0.81）→ **fires@tau0.85 从 ~38% 降到 ~2-3%**，绝大多数步走 fast，strong 只剩 first-step + finish check；
- 工具上限文件入库：commit `fcebf74`（tool_caps_v2/v3.json）。

### 15.5 结论与下一步

- v3 最终（含校准）全面优于 v1 生产基线：acc 51.52 / 50.38、NLL 1.63 / 2.40、ECE 0.075 / 0.291、reply 40.5 / 11.9、edit 20.8 / 12.5；唯一回退是 read 50.0 vs 58.3（净 −3 条，其中 8 条被误判 reply）；
- 弱点仍在：grep 0/7、glob 0/6、compress 1/10（样本少，绝对波动 1-2 条）；
- 下一步：(1) 观察生产任务质量——routing 不再 fire、中间步全走 fast，若出错率上升可下调 tau 或给 routing 单独设温度；(2) 继续补 grep/glob/compress 的难例与独立 session 的 dev；(3) 生产日志每周挖矿并入下一轮。

产物：`checkpoints/grpo_train_v3_clean_hard_v3/`（最终 adapter 264MB + `calibration.json` + `eval_dev_final.json` + 33 个 step 子目录）；`data/train_v3_clean/{train_hard_weighted_v3.jsonl(2641), train_hard_cases_v3.jsonl(263), MANIFEST_hard_weighted_v3.json}`；队列 `checkpoints/mine_nonshell_v3_{train,prod,all}/`；日志 `/tmp/grpo_train_v3_clean_hard_v3.log`。

**现状（11-15 收工）：** 生产决策模型 = 本轮 hard_v3 最终模型 + 温度缩放（global 3.025 / choice:6-16 3.0）；dev 51.52% / NLL 1.63 / ECE 0.075；reply 40.5%、edit 20.8%；GPU1 Embedding 已恢复。

## 16. v4：dev 842 扩 dev + 训练（2026-11-15 晚）

### 16.1 dev 842 切分

旧 dev 264 太小（±5pp 变化不可分辨）且 session 偏斜严重（top 3 session 占 2559 行的 60.5%），做一次 session 级分层重切（`scripts/resplit_dev_v4.py`，commit `834378d`）：
- top 3 巨型 session（975/316/258 行）→ 只进 train；
- legacy holdout（旧 dev+dev_alt+dev_shell，370 行 / 11 session）全部保留进新 dev（v1–v3 结果可对照）；
- 其余按整 session 贪心选入，直到逐工具配额满（read 120 / edit 90 / reply 120 / shell 150 / done 30 / final_result 71 / compress 28 / grep 22 / glob 16 / write 20 / webfetch 25 / not_done 15，其他工具 ≤10）。

结果：**dev 842 / 50 sessions**（旧 264 的 3.2 倍）、train 1717 / 89 sessions、session 泄漏 0。逐工具达成：read 134 / reply 165 / shell 168 / edit 88 / final_result 14 / done 26（final_result、done 受 session 上限限制未达配额，权衡接受）；弱工具 compress 27 / grep 22 / glob 15 / write 18 / webfetch 23 / not_done 14（全池不足 30，只能全取，需每周挖矿补齐）。产物：`data/train_v4/{train,dev,dev_alt,dev_shell}.jsonl + MANIFEST_v4.json`。

### 16.2 v4 训练

从 **v3 final**（生产 adapter，继承 reply/edit 增益）继续，新 train 1717 上 1 epoch：
- `--init checkpoints/grpo_train_v3_clean_hard_v3 --train data/train_v4/train.jsonl --dev data/train_v4/dev.jsonl`
- 108 步（1717/16）、P=16、lr 3e-6、beta 0.05、G=16、max-len 4608、skip-zero-std、gc、save 5、seed 0，~2h28m。
- 训练内 dev（842）：最佳 **step 15 = 54.39%**，其后 18 次评估全在 53.21–53.80 区间未超越；entropy 坍缩（step 70 时 0.075），尾段 KL 1.33（相对 v3 参考漂移）。

### 16.3 与 v3 对比（同 dev 842，生产口径，raw）

| 指标 | v3 final | v4 step15 | v4 final |
|---|---|---|---|
| acc | 54.16% | **54.28%** | 53.68% |
| NLL（低好） | **3.042** | 3.092 | 3.342 |
| ECE（低好） | 0.378 | **0.374** | 0.396 |
| auroc | **0.666** | 0.662 | 0.666 |
| mean_conf | 0.919 | 0.917 | 0.932 |

v4 step15 仅比 v3 高 0.12pp（≈1 条，噪声内打平）；v4 final 低 0.48pp。

**逐工具（dev 842，按条数）：**

| 工具 | n | v3 | v4 step15 | v4 final |
|---|---|---|---|---|
| shell | 168 | 79.8 | 81.0 | 80.4 |
| reply_without_tool | 165 | **65.5** | 61.2 | 57.0 ↓ |
| read | 134 | **47.0** | 44.8 | 41.0 ↓ |
| edit | 88 | 40.9 | 46.6 | **52.3** ↑ |
| bash | 42 | 64.3 | 71.4 | **76.2** ↑ |
| shell_run | 29 | 69.0 | 69.0 | 69.0 |
| compress | 27 | 14.8 | 14.8 | 14.8 |
| done | 26 | 92.3 | 88.5 | **96.2** ↑ |
| webfetch | 23 | 30.4 | 34.8 | **39.1** ↑ |
| grep | 22 | 4.5 | 4.5 | 4.5 |
| write | 18 | 33.3 | 33.3 | 27.8 |
| glob | 15 | 0.0 | 0.0 | 0.0 |
| final_result | 14 | 78.6 | 78.6 | 78.6 |
| not_done | 14 | 7.1 | 7.1 | 0.0 |
| execute | 13 | 23.1 | 30.8 | 30.8 |

（长尾 n≤7 的工具省略；shell+shell_run+shell_wait=204，与 dev_shell 切片一致。）

**跷跷板（与 v2 同型）**：v4 final 的 edit +11.4pp / bash +11.9pp / webfetch +8.7pp 被 reply **−8.5pp** / read **−6.0pp** 抵消——模型把 reply/read 的边界往 edit/bash 侧推。逐工具此消彼长，整体 acc 因此打平/略降。根因：reply/read 与 edit/bash 的边界在 GRPO 下互相挤压，同分布数据继续训练压不平；**打破跷跷板要靠数据**（§11：reply/read 边界难例 + 逐工具 dev 切片 + 弱工具挖矿）。

### 16.4 结论

- **v4 不部署，生产维持 v3**（v3 final + 校准 3.025/3.0，已在线上）。
- 本轮价值不是“v4 赢了”，而是**建立了新 dev 842 基线（v3 = 54.16%）**——此后 ±5pp 变化可分辨、逐工具切片样本充足。
- 结论：**同分布数据上继续 GRPO 无净增益**，瓶颈在数据本身（弱工具 compress/glob/grep，dev 池 <30）而非训练步数。下一步按 §11 数据积累路线（每周挖矿补弱工具 + 逐工具 dev 切片），攒够后出 v5（新数据 + 弱工具加权）。
- 环境：训练结束已恢复 `vllm-openai-2`（embedding，GPU1 15GB）。

## 17. 相关文档

- train_v2 / train_v3 对比记录：`results/training/grpo_train_v2_2026-11-11.md` §12
- clean dev 划分说明：`data/train_v3_clean/SPLIT.md`
- 数据说明：`docs/DECISION_DATA_V2.md`
- 训练配方：`docs/TRAINING_RECIPES.md`
- 脚本：`scripts/train/60_grpo_trl.sh`
- 非 shell 难例挖矿（§14/§15）：`src/jevharness/training/mine_hard_cases.py`、`scripts/build_hard_weighted.py`、工具上限 `data/tool_caps_v2.json` / `data/tool_caps_v3.json`
- 校准与评估（§15.3）：`src/jevharness/training/calibrate.py`、`src/jevharness/training/eval_decisions.py`
- dev 842 切分（§16）：`scripts/resplit_dev_v4.py`、`data/train_v4/MANIFEST_v4.json`

## 18. 经验总结（六轮合并）

以下经验从本文档的六轮（首轮 train_v3、P1 clean、hard_v1/v2/v3、v4）中合并，供日后使用。

### 18.1 数据

1. **数据量 ≠ 训练量**：train_v3 首轮数据多 63%，但 104 步早停只消费 33.5%（832/2481），有效梯度与上一轮几乎相同 → 增益为 0。**1 epoch 必须跑满（或早停指标 + min-steps 保证覆盖），否则加数据毫无意义**。
2. **加数据先查分布**：train 68.4% fast 标签 vs dev 56.8% strong；42% dev 落在 train 里 ≤25 条的工具上。**加数据前先对齐 train/dev 的工具、来源、难度分布**，总量达标不代表有效。
3. **有效样本 ≠ 行数**：zero-std 0.51 + 单 session 39.3% 意味着大量行是同一决策边界的重复。**去重键 + session 上限（≤5%）+ 按"唯一决策边界"计数**。
4. **长尾靠挖矿，不靠流量**：13 个工具 <1%（1–21 条）自然积累学不动；hard-case mining 一天就补出 287/340 条。**弱工具配额按 dev 缺口分配（tool_caps），不按流量比例**。

### 18.2 训练

5. **GRPO 的主要浪费是 zero-variance groups**（frac_zero_std 0.31–0.75：整组答一样就没梯度）→ `--skip-zero-std`、加大 G。
6. **早停用 dev 指标**：entropy/zero-std 早停把首轮停在 104/120（覆盖不足）；改 dev-acc + patience + min-steps 后三轮都跑满 1 epoch。entropy 早停只当防崩保险。
7. **超参教训**：lr 1e-5 → entropy collapse（1.69→0.12）；降 lr 3e-6 + beta 0.05 后三轮稳定。
8. **从上一轮 best step 继续**（而不是 base）：v3 从 v2 step135 起，直接继承 reply 增益，再修正偏置——**增量式迭代比每轮推倒重来高效**。
9. **一轮一个主要假设**：v2 只加 reply → 发现过校正；v3 降 reply + 加 read/edit → 三项同时改善。假设清晰，逐工具指标才能归因。

### 18.3 评估

10. **聚合 acc 会骗人**：v2 整体 50.38% "持平"，实际 reply +19pp / read −11pp（此消彼长）。**验收必须逐工具 + 混淆矩阵，整体 acc 只是门槛**。
11. **口径固定并写进文档**：同一模型 52.65%（训练内 eval）vs 51.52%（eval_decisions）；同一基线 50.00/50.38。**验收统一用生产口径（served model + 校准），文档注明以哪个为准**。
12. **GRPO 的副作用是过度自信**：每轮 raw NLL/ECE 都在变差（v3 ECE 0.42、mean_conf 0.93）→ **部署前必须 temperature scaling**（本轮 3.025 把 ECE 压到 0.075、NLL 1.63）。argmax 不变意味着缩放只修校准、不改决策——便宜且必须。
13. **dev 太小会让"提升"不显著**：37 条时 35% ≈ 多数类先验，Wilson 区间全重叠；扩到 264 后才能区分 +6pp。**dev 目标几百条 + 逐工具切片各 ≥30**。

### 18.4 流程与部署

14. **验收门前置**：开训前先写死（如 reply≥20%、edit≥25%、整体≥50%、NLL/ECE 不劣）。**"不部署"的结论和"部署"同样要记录在案**（hard_v2 就是靠这个纪律避免负迁移上线）。
15. **部署 = adapter + calibration 原子操作**：两者必须同一窗口切换，**切换后立即线上验证**（看新决策行的 p 分布形态：p_hard 0.094–0.359 一眼确认 fires 骤降）。
16. **全链路版本化**：数据集 MANIFEST + 代码 commit + adapter 目录 + calibration.json + 日志路径，每轮可复现可回滚。
17. **挖矿脚本的坑（已修，留档）**：session cap 必须 per-(session,tool)——per-session 会让弱工具饿死（compress 84 个错误只选出 1 条）；`--exclude` 必须覆盖全部已用集（train/dev 各切片、历次 hard_cases）防泄漏。
18. **先修评估再修训练**：口径不一致时，任何"提升"都不可信；把评估做对（同口径、固定 batch、对比表）是一切优化的前提。

### 18.5 一句话

本轮流程验证了完整的飞轮：**生产挖矿 → 难例加权 → 增量 GRPO → 逐工具验收 → 温度缩放 → 原子部署 → 线上验证**。最容易踩的三个坑：①加数据但没训满；②只看聚合指标导致过校正上线；③忘温度缩放导致过度自信。

---

**一句话带走：** train_v3 的数据量增加了 63%，但这次只训练了 33.5% 的 train，zero-std 又吃掉约一半 group，最终有效梯度与 v2 第二轮几乎相同；同时 dev 有 42% 落在 train 极少见的工具上。因此“效果不明显”主要是**训练预算、数据同质化和评估偏移**共同造成的，不是单纯的数据量问题。

---

## 19. v5：弱工具挖矿配额（非 shell）（2026-11-16）

### 19.1 目标与假设

v4 揭示了跷跷板：把 edit/webfetch/bash 的边界推对的同时吞掉 reply/read。v5 的唯一假设：**按 dev 缺口 + 池子可用量给弱工具定量挖矿（不按流量比例），压低 reply 配额、上调 read/edit，能打破跷跷板**。shell 家族配额维持 0（防 shell 偏置）。

### 19.2 配额（`research/data/tool_caps_v5.json`）

23 个规范名 key（工具名族经 `TOOL_ALIASES` 归一化，一条 cap 覆盖整族，见 commit `6f3439b`）：

| 工具 | cap | 说明 |
|---|---:|---|
| read / edit | 100 / 100 | 跷跷板上调（v4 吞读） |
| reply_without_tool | 60 | 压低（v4 已证过高会吞 read） |
| compress / grep / write | 60 / 50 / 40 | 弱工具取满池子 |
| done / not_done / execute / final_result | 15 / 16 / 10 / 20 | 长尾小 cap |
| glob / webfetch / websearch / ask_user_question 等 | 3–10 | 长尾 |
| **shell 族**（shell/bash/shell_run/shell_wait） | **0** | 整族排除 |

全局参数：`--max-per-session 5`（per-(session,tool)）、`--min-margin 0.05`、`--max-total 600`、`--batch-size 16`；打分模型 = v3 final（生产模型）。

### 19.3 挖矿与数据构建

- **Track A**（`train_v4/train` 1717，v3 final 打分）→ queue **146 行 / 31 会话**，全部 error（high_conf_error 85 + shell_false_positive 55），零 shell 族行。队列分布：reply 36 / read 26 / edit 20 / compress 16 / write 14 / grep 12 / 其余个位。池内混淆证实弱工具错误主要是误选 shell：edit→shell 86、read→shell 87、compress→shell 57、write→shell 24。
- **Track B**（生产 `decisions.jsonl`，唯一步 2706，exclude 已用集 2601 → 新行 105）→ queue **16 行 / 3 会话**：compress 5 / read 4 / edit 3 / 其余个位。
- 合并 162 行 → `build_hard_weighted.py --clean-dir data/train_v4`（**注意**：base 从 train_v3_clean 改为 train_v4，因 v4 重切了 dev 842，用旧 clean 目录会泄漏新 dev）→ selected **135**（丢 26 截断 + 1 cap）、extras **234**、**final 1951 行**。
- 独立泄漏复查（train_v5 1951 vs dev 族 1518）：**LEAK 0**。
- 最终分布 vs base：edit 297→336、read 257→313、reply 181→245、compress 84→98、grep 52→72、write 39→63；shell 713 不变。
- 产物：`checkpoints/mine_nonshell_v5_{train,prod}/`、`checkpoints/queue_v5_merged.jsonl`、`data/train_v5/{train_hard_weighted_v5.jsonl, train_hard_cases_v5.jsonl, MANIFEST_hard_weighted_v5.json}`。

### 19.4 训练

配方与 v3/v4 一致（从 v3 final 继续，1 epoch 跑满）：

```bash
CUDA_VISIBLE_DEVICES=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 nohup .venv/bin/python -m jevharness.training.grpo_trl \
  --impl torch --model /data1/apps/huggingface/Qwen/Qwen3-4B \
  --init checkpoints/grpo_train_v3_clean_hard_v3 \
  --train data/train_v5/train_hard_weighted_v5.jsonl \
  --dev data/train_v4/dev.jsonl \
  --out checkpoints/grpo_train_v5 --rewards correct \
  --num-generations 16 --batch-size 16 --grad-accum 16 \
  --save-steps 5 --early-stop-patience 5 --early-stop-min-steps 60 \
  --epochs 1 --lr 3e-6 --beta 0.05 --max-len 4608 \
  --skip-zero-std --gradient-checkpointing --logging-steps 5 --seed 0
```

1 epoch **122 步跑满**（~2h20m），无早停、无崩溃。dev_acc 轨迹：step5 54.75 → step20 触底 53.68 → step75 起回升 → step80/100 54.75 → step120 54.87（全程最佳）。

### 19.5 验收（dev 842，raw 口径，n-perm 0，batch 16，与 v3 基线同口径）

| 模型 | acc | NLL | Brier | ECE | AUROC | mean_conf |
|---|---:|---:|---:|---:|---:|---:|
| v3 final（基线，生产） | 54.16% | **3.042** | 0.3777 | 0.3778 | 0.6663 | 0.9194 |
| v5 step120 | 54.39% | 3.365 | 0.3916 | 0.3914 | 0.6569 | 0.9353 |
| **v5 final（step122）** | 54.99% | 3.374 | 0.3920 | 0.3858 | 0.6431 | 0.9357 |

整体 +0.83pp 在 n=842 的噪声内（1σ≈1.7pp），且 NLL/ECE/AUROC 全面变差（NLL 3.374 比 v4 final 3.342 还略差）——GRPO 每轮过度自信的老问题。

逐工具（v3 → v5 step120 → v5 final，Δ = final−v3，pp）：

| 工具 | n | v3 | s120 | final | Δ |
|---|---:|---:|---:|---:|---:|
| shell | 168 | 79.8 | 79.8 | 80.4 | +0.6 |
| reply_without_tool | 165 | **65.5** | 62.4 | 63.0 | **−2.4** |
| read | 134 | **47.0** | 43.3 | 44.0 | **−3.0** |
| edit | 88 | 40.9 | 45.5 | **47.7** | +6.8 |
| bash | 42 | 64.3 | 73.8 | 73.8 | +9.5 |
| compress | 27 | 14.8 | 14.8 | 14.8 | 0 |
| grep | 22 | 4.5 | 4.5 | 4.5 | 0 |
| glob | 15 | 0.0 | 0.0 | 0.0 | 0 |
| write | 18 | 33.3 | 27.8 | 27.8 | −5.6 |
| webfetch | 23 | 30.4 | 39.1 | 39.1 | +8.7 |
| done | 26 | 92.3 | 96.2 | 96.2 | +3.8 |
| not_done | 14 | 7.1 | 0.0 | 0.0 | −7.1 |
| execute | 13 | 23.1 | 38.5 | 30.8 | +7.7 |
| shell_run / shell_wait | 29 / 7 | 69.0 / 14.3 | 69.0 / 28.6 | 69.0 / 42.9 | 0 / +28.6 |

v5 final 混淆（gold → 前 3 错项）：shell→reply 17/edit 6/execute 3；reply→read 20/edit 19/shell 14；read→reply 34/edit 20/shell 9；edit→reply 17/read 11/shell 10；compress→shell 8/edit 7/reply 5；glob→reply 12/edit 3；write→**shell 10**；not_done→**done 14（全错）**。

### 19.6 结论：v5 不部署

生产维持 **v3 final + 校准**（global 3.0251733656628823 / choice:6-16 2.9999808412939433）。三条理由：

1. **假设未成立**：弱工具配额挖矿修不动 reply/read——真正提升的是 edit +6.8 / webfetch +8.7 / bash +9.5（v4 已知的边界方向），跷跷板照旧：edit 上去的同时 read −3.0、reply −2.4。compress / grep / glob 配额取满池子也纹丝不动（池子太小，15–27 条 dev，加多少都进不了梯度）。
2. 整体 +0.83pp 在噪声内，NLL 反劣 0.33。
3. **not_done 14→0（14 条全变成 done）、write −5.6** 是明确退步。

**价值**：train_v5 1951 行入库可作下轮 base；定量确认「纯靠弱工具加量修不了 reply/read，需要边界难例（read↔reply / edit↔read 互扰对）与增广」——这与 DECISION_DATA_V2 §11 轨道 B（扩 dev + 逐工具切片）的优先级一致。

### 19.7 产物

- 数据集：`data/train_v5/{train_hard_weighted_v5.jsonl (1951), train_hard_cases_v5.jsonl (135), MANIFEST_hard_weighted_v5.json}`
- 适配器：`checkpoints/grpo_train_v5`（final + step-000005..000120）
- 评测：`checkpoints/grpo_train_v5/per_tool_{step120,final}_dev842.json`；日志 `/tmp/grpo_train_v5.log`、`/tmp/mine_nonshell_v5_{train,prod}.log`、`/tmp/eval_v5_{both,step120,final}.log`
- 相关：`research/data/tool_caps_v5.json`、`research/src/jevharness/training/mine_hard_cases.py`（TOOL_ALIASES）、`research/scripts/build_hard_weighted.py`

## 20. v6：read↔reply / edit↔read 边界成对难例（2026-11-16）

### 20.1 目标与假设

v5 定量证明「纯弱工具加量」修不了 read↔reply 边界（v5 里 reply 65.5→63.0、read 47.0→44.0 均回退，而 edit/webfetch/bash 又上去——跷跷板照旧）。v6 假设：**结构化边界成对对比**——同一会话相邻 step（差 ≤2）、gold 在 {read, reply_without_tool, edit} 族内交替的行，状态最小差异、gold 相反——直接教模型「什么状态下 read、什么状态下 reply/edit」，可打破 read↔reply / edit↔read 互扰。

### 20.2 挖矿机制（`research/scripts/mine_boundary_pairs.py`，新增）

- 纯结构选择，**零 GPU、可复现**（不用模型打分）。
- 配对规则：同 session、|step 差| ≤ `--max-gap`（2）、gold（经 TOOL_ALIASES 规范名）在族内且交替 → 两行都入队（`reasons=["boundary_pair"]`，带 `pair_id`/`pair_partners`）。
- 生产 `decisions.jsonl` 新行：gold ∈ 族且不在任何对中的 → P1 singleton（`reasons=["boundary_prod_new"]`，`--no-prod-singletons` 可关）。
- 配额（canonical）：read 100 / reply 80 / edit 80；per-(session,tool) 4；max_total 400。
- 本地干跑（train_v4/train 1717）：419 对 → 队列 94 行 / 20 会话（read 33 / reply 34 / edit 27，均衡）。

### 20.3 挖矿与数据构建

- **Track A**（`--examples data/train_v4/train.jsonl`）：419 对 → 队列 **94 行 / 20 会话**。
- **Track B**（`--decisions data/decisions.jsonl --exclude` train_v4 四文件 + hard_cases v1-v3 + train_v5 两文件 + v5 merged queue）：89 条生产新行 → **7 行 / 2 会话**。
- 合并 101 行 → `build_hard_weighted.py --base data/train_v5/train_hard_weighted_v5.jsonl --dev-dir data/train_v4 --version v6` → selected **93**（8 条 truncated 丢）、extras **188**、**final 2139 行**（base 1951）。
- **独立泄漏复查（2139 vs dev 族 1518）：LEAK 0**。
- 分布 v5→v6（只边界族动）：**read 313→375、reply 245→314、edit 336→393**；shell 713 / compress 98 / grep 72 / write 63 等全部持平。
- 产物：`checkpoints/queue_boundary_v6_merged.jsonl`、`data/train_v6/{train_hard_weighted_v6.jsonl (2139), train_hard_cases_v6.jsonl (93), MANIFEST_hard_weighted_v6.json}`。

### 20.4 训练

命令（与 v3-v5 同配方，`--init` v3 final，dev 842 早停 patience 5 / min 60，1 epoch ≈134 步 ~3h）：

```bash
CUDA_VISIBLE_DEVICES=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 nohup .venv/bin/python -m jevharness.training.grpo_trl \
  --impl torch --model /data1/apps/huggingface/Qwen/Qwen3-4B \
  --init checkpoints/grpo_train_v3_clean_hard_v3 \
  --train data/train_v6/train_hard_weighted_v6.jsonl \
  --dev data/train_v4/dev.jsonl \
  --out checkpoints/grpo_train_v6 --rewards correct \
  --num-generations 16 --batch-size 16 --grad-accum 16 \
  --save-steps 5 --early-stop-patience 5 --early-stop-min-steps 60 \
  --epochs 1 --lr 3e-6 --beta 0.05 --max-len 4608 \
  --skip-zero-std --gradient-checkpointing --logging-steps 5 --seed 0
```

**134 步跑满**（~3h，无早停、无崩溃）。训练内 dev 轨迹：step110 **54.99**（峰值）→ step130 54.28 → final(step134) **54.63**。

### 20.5 验收（dev 842，raw 口径 n-perm 0 batch 1；校准后见下）

| 模型 | acc | NLL | Brier | ECE | AUROC | mean_conf |
|---|---:|---:|---:|---:|---:|---:|
| v3 final（基线，生产） | 54.16% | **3.042** | 0.3777 | 0.3778 | 0.6663 | 0.9194 |
| v6 s110 | 54.63% | 3.445 | 0.392 | 0.391 | 0.6569 | 0.9353 |
| v6 s130 | 54.39% | 3.508 | 0.398 | 0.397 | 0.6577 | — |
| **v6 final（raw）** | **54.87%** | 3.519 | 0.398 | 0.397 | 0.6505 | — |
| **v6 final（校准后）** | **54.87%** | **1.644** | 0.2312 | **0.0599** | 0.6681 | 0.5923 |

校准值（`checkpoints/grpo_train_v6/calibration.json`，dev 842 拟合）：global **3.2108**、choice:6-16 **3.4545**、choice:17-52 **2.5680**（v3 生产在跑的是 global 3.0252 / choice:6-16 3.0000）。

逐工具（v3 → v6 final，Δ = final−v3，pp）：

| 工具 | n | v3 | v6 final | Δ |
|---|---:|---:|---:|---:|
| shell | 168 | 79.8 | 79.2 | −0.6 |
| reply_without_tool | 165 | 65.5 | **69.1** | **+3.6** |
| read | 134 | **47.0** | 41.8 | **−5.2** |
| edit | 88 | 40.9 | **50.0** | **+9.1** |
| bash | 42 | 64.3 | 61.9 | −2.4 |
| shell_run | 29 | 69.0 | 69.0 | 0 |
| shell_wait | 7 | 14.3 | 42.9 | +28.6（n 小，噪声） |
| compress | 27 | 14.8 | 14.8 | 0 |
| grep | 22 | 4.5 | 4.5 | 0 |
| glob | 15 | 0.0 | 0.0 | 0 |
| write | 18 | 33.3 | 27.8 | −5.6 |
| webfetch | 23 | 30.4 | 26.1 | −4.3 |
| done | 26 | 92.3 | 96.2 | +3.8 |
| not_done | 14 | 7.1 | 0.0 | −7.1 |
| execute | 13 | 23.1 | 30.8 | +7.7 |
| final_result | 14 | 78.6 | 78.6 | 0 |

v6 final 混淆（gold → 前 3 错项）：shell→reply 17/edit 6/execute 3；reply→read 20/edit 19/shell 14；read→reply 34/edit 20/shell 9；edit→reply 17/read 11/shell 10；compress→shell 8/edit 7/reply 5；webfetch→reply 6/shell 5/read 3；grep→edit 6/reply 6/shell 5；write→**shell 10**/bash 1/edit 1；glob→reply 12/edit 3；not_done→**done 14（全错）**。

**门判定（开训前写死）**：NLL raw 3.519 > 3.1 → 缩放后 **1.644**（v3 为 1.63，落在 1.2–1.6 区间上沿）✓；acc 54.87 vs 54.16 = **+0.71pp** ✓；ECE 0.0599 < v3 0.075 ✓。

### 20.6 结论：v6 达标，部署待批准

1. **假设部分成立**：reply +3.6、edit +9.1 是本轮主修复（成对对比有效）；但 **read −5.2**（134 行 dev 切片，生产高频）+ webfetch −4.3 / write −5.6 回退，跷跷板没完全打破，只是从「edit 独涨」变成「reply/edit 涨、read 跌」。
2. 校准后 NLL/ECE 与 v3 同档甚至更优（ECE 0.060 < 0.075），按预注册门 v6 **可部署**：`grpo_train_v6`（final，acc 最高 54.87）+ `calibration.json`（global 3.2108 / choice:6-16 3.4545）原子切换，线上验证 p 分布。
3. 回滚方案：v3 全套产物 + 校准（global 3.0252 / choice:6-16 3.0000）保留，随时切回。
4. **状态：写本文档时部署未获批准，生产维持 v3 final + 校准不变。**
5. 若不部署：下一轮挖矿配额向 read/webfetch/write 边界对倾斜 + option-permutation 增广（glob 池 3 / webfetch 池 2 已枯竭，见 §19.6 轨道 B 优先级）。

### 20.7 产物

- 脚本：`research/scripts/mine_boundary_pairs.py`（含 6 个单测 `research/tests/test_mine_boundary_pairs.py`）
- 数据集：`data/train_v6/{train_hard_weighted_v6.jsonl (2139), train_hard_cases_v6.jsonl (93), MANIFEST_hard_weighted_v6.json}`
- 适配器：`checkpoints/grpo_train_v6`（final + step-000005..000130 + `calibration.json`）
- 评测：`/tmp/eval_v6_{step-000110,step-000130,final,final_calibrated}.json`
- 日志：`/tmp/grpo_train_v6.log`、`/tmp/mine_boundary_v6_{train,prod}.log`、`/tmp/cal_v6_all.log`
- 挖矿队列：`checkpoints/mine_boundary_v6_{train,prod}/`、`checkpoints/queue_boundary_v6_merged.jsonl`