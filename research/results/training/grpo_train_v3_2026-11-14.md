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

### 20.6 结论：v6 达标；用户裁决维持 v3，不部署

1. **假设部分成立**：reply +3.6、edit +9.1 是本轮主修复（成对对比有效）；但 **read −5.2**（134 行 dev 切片，生产高频）+ webfetch −4.3 / write −5.6 回退，跷跷板没完全打破，只是从「edit 独涨」变成「reply/edit 涨、read 跌」。
2. 校准后 NLL/ECE 与 v3 同档甚至更优（ECE 0.060 < 0.075），按预注册门 v6 **可部署**：`grpo_train_v6`（final，acc 最高 54.87）+ `calibration.json`（global 3.2108 / choice:6-16 3.4545）原子切换，线上验证 p 分布。
3. 回滚方案：v3 全套产物 + 校准（global 3.0252 / choice:6-16 3.0000）保留，随时切回。
4. **状态（11-16 用户裁决）：维持 v3，v6 不部署**；`checkpoints/grpo_train_v6`（final）+ `calibration.json` 入库待命，随时可切换上线或回滚。
5. 若不部署：下一轮挖矿配额向 read/webfetch/write 边界对倾斜 + option-permutation 增广（glob 池 3 / webfetch 池 2 已枯竭，见 §19.6 轨道 B 优先级）。

### 20.7 产物

- 脚本：`research/scripts/mine_boundary_pairs.py`（含 6 个单测 `research/tests/test_mine_boundary_pairs.py`）
- 数据集：`data/train_v6/{train_hard_weighted_v6.jsonl (2139), train_hard_cases_v6.jsonl (93), MANIFEST_hard_weighted_v6.json}`
- 适配器：`checkpoints/grpo_train_v6`（final + step-000005..000130 + `calibration.json`）
- 评测：`/tmp/eval_v6_{step-000110,step-000130,final,final_calibrated}.json`
- 日志：`/tmp/grpo_train_v6.log`、`/tmp/mine_boundary_v6_{train,prod}.log`、`/tmp/cal_v6_all.log`
- 挖矿队列：`checkpoints/mine_boundary_v6_{train,prod}/`、`checkpoints/queue_boundary_v6_merged.jsonl`

## 21. v7：read / write / webfetch 边界成对 + option-permutation 增广（2026-11-16）

### 21.1 目标与假设

v5 证明「纯加量」修不了弱工具边界，v6 证明「结构化成对对比」对 reply/edit 有效（+3.6/+9.1）但 read 又回退（−5.2）。v7 假设：把成对机制**换靶心**到 v5 证明有效、v6 又回退的 **read / write / webfetch** 族，并叠加 **option-permutation 增广**（`grpo_trl --n-permutations 2`，每例选项顺序 shuffle 2 次）提升对选项顺序的鲁棒性。init 仍为 v3 final。

### 21.2 配额与挖矿（`mine_boundary_pairs.py --family read,write,webfetch`）

- 配额（canonical 上限，受池子约束）：read 100 / write 40 / webfetch 20；`--max-gap 2`；`--max-per-session-tool 8`（放宽以向靶心倾斜）；`--max-total 400`。
- 池子实况（train_v4/train 1717）：read 257 / write 39 / **webfetch 仅 2（枯竭，无对可挖）**；`read↔write` 恰好 **53 对 / 83 成员行**（本轮主体）。
- Track A：53 对 → per-session-tool 8 → **31 行（read 16 / write 15）**。Track B（生产新鲜行）：82 条全非族 → **0 行**（read/write/webfetch 新鲜行已耗尽）。

### 21.3 数据构建（base = v6 2139）

31 行队列 → **selected 25**（丢 6 truncated）+ 50 加权 extras → **train_v7 = 2189 行**。**独立 LEAK 复查：0**。分布 v6→v7：**read 375→401 (+26)、write 63→87 (+24)**，webfetch 4 / shell 713 / edit 393 / reply 314 / compress 98 / grep 72 全持平。

### 21.4 训练

与 v3–v6 同配方 + `--n-permutations 2`（日志确认 **4378 rows = 2189×2**）。`--init checkpoints/grpo_train_v3_clean_hard_v3`，dev 842 早停（patience 5 / min 60）。**274 步跑满（1 epoch，无早停）**，final in-train dev_acc **0.5594**（NLL 3.754），in-train 峰 step245=0.5606。

### 21.5 验收（dev 842，n-perm 0 batch 1）

| 模型 | acc | NLL | Brier | ECE | AUROC |
|---|---:|---:|---:|---:|---:|
| v3 final（基线，当时生产） | 54.16% | 3.042 | 0.378 | 0.378 | 0.666 |
| v6 final（校准后） | 54.87% | 1.644 | 0.231 | 0.060 | 0.668 |
| v7 final（raw） | 55.94% | 3.743 | 0.396 | 0.397 | — |
| **v7 final（校准后）** | **55.94%** | **1.642** | 0.226 | **0.0486** | **0.682** |

校准值（`checkpoints/grpo_train_v7/calibration.json`）：global **3.3813**、choice:6-16 **3.6644**、choice:17-52 **2.6715**。

逐工具（v3 → v7 校准后，Δ pp）：

| 工具 | n | v3 | v7 | Δ |
|---|---:|---:|---:|---:|
| **write** | 18 | 33.3 | **44.4** | **+11.1**（靶心 ✓）|
| **webfetch** | 23 | 30.4 | **34.8** | **+4.3**（靶心 ✓）|
| **read** | 134 | 47.0 | **50.0** | **+3.0**（靶心 ✓）|
| edit | 88 | 40.9 | 53.4 | +12.5 |
| bash | 42 | 64.3 | 73.8 | +9.5 |
| execute | 13 | 23.1 | 46.2 | +23.1 |
| **reply** | 165 | 65.5 | 58.8 | **−6.7** ⚠️ |
| shell | 168 | 79.8 | 78.6 | −1.2 |
| done / not_done / compress / grep / glob | — | 92.3 / 7.1 / 14.8 / 4.5 / 0 | 持平 | — |

**v7 是历轮综合最强**：acc 55.94（+1.78 vs v3、+1.07 vs v6）、ECE 0.0486（历轮最低）、AUROC 0.682（历轮最高）。靶心三项（read/write/webfetch）全部命中、edit/bash/execute 顺带大涨；**代价是 reply −6.7**（跷跷板再现，方向从 v6 的 read 换到了 reply）。

### 21.6 结论：v7 达标；作为 v8 的 init，未单独部署

1. 假设成立：成对机制 + option-permutation 对 read/write/webfetch 靶心有效（+11.1/+4.3/+3.0）。
2. 但 reply −6.7 说明跷跷板未根治——v7 是「靶心命中 + 顺带修 edit/bash + 牺牲 reply」的混合体。
3. 状态：v7 作为下一轮（v8）的 init，**未单独部署**（当时生产维持 v3）。

### 21.7 产物

- 数据集：`data/train_v7/{train_hard_weighted_v7.jsonl (2189), train_hard_cases_v7.jsonl (25), MANIFEST_hard_weighted_v7.json}`
- 适配器：`checkpoints/grpo_train_v7`（final + step-000005..000275 + `calibration.json`）
- 评测：`/tmp/eval_v7_{final,final_calibrated}.json`；日志 `/tmp/grpo_train_v7.log`、`/tmp/v7_watch_eval.log`、`/tmp/v7_pipeline.log`
- 队列：`checkpoints/queue_boundary_v7_merged.jsonl`（31 行）

## 22. v8：reply↔read 平衡轮（init = v7，2026-11-17）

### 22.1 目标与假设

v7 靶心命中但 reply −6.7（跷跷板再现）。v8 假设：把成对机制**换到 reply↔read 族**（train_v4/train 里 reply↔read 恰好 161 对），把**两端同时**抬到 v3 之上（v7 只抬了 read、砸了 reply），叠加 option-permutation。**关键改动：init = v7 final**（在最强模型上继续，而非回 v3）。

### 22.2 配额与挖矿（`--family reply_without_tool,read`）

- 配额：reply 140 / read 140；`--max-gap 2`；`--max-per-session-tool 20`。
- 池子：reply↔read **161 对 / 253 成员行**（read 139 / reply 114，18 session）。
- Track A：**126 行（read 64 / reply 62）**。Track B：82 条生产新鲜全非族 → **0 行**（reply/read 已枯竭）。

### 22.3 数据构建（base = v7 2189）

126 行队列 → **selected 113**（read 55 / reply 58）+ 226 加权 extras → **train_v8 = 2415 行**（13 truncated 丢）。**LEAK 0**。分布 v7→v8：**read 401→511 (+110)、reply 314→430 (+116)**，其余全持平。**风险**：top-2 session（ses_e54598… 36 行 / ses_e3fa… 35 行）占 selected 113 里 71 行（63%），session 集中度偏高。

### 22.4 训练

同配方 + `--n-permutations 2`（**4830 rows = 2415×2**）。**`--init checkpoints/grpo_train_v7`**（v7 final，关键改动）。dev 842 早停。**302 步跑满（1 epoch，无早停，~6.5h）**，in-train dev 峰 **0.5713（step 230，历轮最高）**。

### 22.5 验收（dev 842，n-perm 0 batch 1）

| 模型 | acc | NLL | Brier | ECE | AUROC |
|---|---:|---:|---:|---:|---:|
| v3 final（基线，原生产） | 54.16% | 3.042 | 0.378 | 0.378 | 0.666 |
| v7 final（校准后） | 55.94% | 1.642 | 0.226 | 0.0486 | 0.682 |
| v8 step-230（raw，in-train 峰） | 56.89% | 4.265 | — | — | — |
| **v8 final（校准后）** | **56.41%** | **1.659** | 0.227 | 0.0578 | 0.679 |

校准值（`checkpoints/grpo_train_v8/calibration.json`）：global **3.8770**、choice:6-16 **4.2028**、choice:17-52 **3.0691**。global 温度逐轮抬升（v3 3.03 → v6 3.21 → v7 3.38 → v8 3.88），因 raw 模型越来越过度自信；校准后 NLL 均落 1.63–1.66 健康区。

逐工具（v3 → v7 校准 → v8 校准，acc%）：

| 工具 | n | v3 | v7 | v8 | v8 vs v3 |
|---|---:|---:|---:|---:|---:|
| **reply** | 165 | 65.5 | 58.8 | **69.1** | **+3.6（超 v3）** ✓ |
| **read** | 134 | 47.0 | 50.0 | **48.5** | +1.5（≥v3，<v7）✓ |
| edit | 88 | 40.9 | 53.4 | 50.0 | +9.1 |
| bash | 42 | 64.3 | 73.8 | 69.0 | +4.7 |
| final_result | 14 | 78.6 | 78.6 | 100.0 | +21.4 |
| execute | 13 | 23.1 | 46.2 | 30.8 | +7.7 |
| write | 18 | 33.3 | 44.4 | 33.3 | 0（−11.1 vs v7）|
| webfetch | 23 | 30.4 | 34.8 | 30.4 | 0（−4.3 vs v7）|
| shell | 168 | 79.8 | 78.6 | 76.8 | −3.0 |
| done / not_done / compress / grep / glob / shell_run | — | 92.3 / 7.1 / 14.8 / 4.5 / 0 / 69.0 | 持平 | 持平 | — |

**门判定（开训前写死）**：reply 回升 ≥+3 vs v7 → 58.8→**69.1（+10.3，且超 v3 65.5）** ✓；read ≥ v3 47.0 → **48.5** ✓（未达 v7 50.0 偏好）；整体 acc ≥ v7 → **56.41** ✓；校准后 NLL/ECE → 1.659 / 0.0578（与 v7 同档、略差 0.009）✓。

### 22.6 结论：v8 达标，已部署（客户端 JAH 决策）

1. **假设成立**：reply↔read 两端**同时**抬到 v3 之上（reply 69.1 > 65.5、read 48.5 > 47.0）——v7 没做到的，v8 做到了。整体 acc **56.41 历轮最高**（+2.25 vs v3、+0.47 vs v7）。
2. 代价：write −11.1 / webfetch −4.3 / execute −15.4 vs v7（非本轮靶心）、shell −3.0 vs v3——跷跷板把「牺牲」从 reply 转到了 write/webfetch。
3. **部署（v8 = 现生产）**：服务端 vLLM adapter 切 v8（served 目录 `AgentDecision-4B-1117`，md5 与 `grpo_train_v8` 一致）+ 本地 litellm-jah 镜像重建为 v8 校准（global 3.877 / choice:6-16 4.203 / 17-52 3.069）+ 客户端侧 `docker compose up -d --force-recreate` 生效。**客户端 JAH 决策 + 校准 = v8，全链路 v8**（架构：生产 JAH 决策跑在本地 litellm-jah + ssh-tunnel，服务器 vllm-openai-1 供 v8 adapter、纯代理不带校准）。
4. **p_hard 验证（初步）**：decisions.jsonl 切点前最近 150 条 routing 行 p_hard 中位 **0.541 / >0.85 占 7.3%**（更早 bulk 0.619 / 20.8%）→ 明显更平、更少触发 strong；但数据止于切点（~02:00 UTC），**尚无干净 post-切点 routing 样本**（v8 校准 T=3.877 尚未隔离验证），待积累 post-切点流量后复核。
5. 回滚方案：v3 / v7 全套 adapter + calibration 保留（v3 global 3.025 / choice:6-16 3.000；v7 global 3.381 / choice:6-16 3.664），随时可切回。

### 22.7 产物

- 数据集：`data/train_v8/{train_hard_weighted_v8.jsonl (2415), train_hard_cases_v8.jsonl (113), MANIFEST_hard_weighted_v8.json}`
- 适配器：`checkpoints/grpo_train_v8`（final + step-000005..000300 + `calibration.json`；部署版 served 目录 `AgentDecision-4B-1117`）
- 评测：`/tmp/eval_v8_{final,final_calibrated,step-000230,step-000285}.json`；日志 `/tmp/grpo_train_v8.log`、`/tmp/v8_watch_eval.log`
- 配置：`litellm_jah/docker/jah-config.yaml`（v8 校准，commit `fe495e5`）；本地镜像 `litellm-jah:latest` = `litellm-jah:v8`（a6bcfc0e）
- 队列：`checkpoints/queue_boundary_v8_merged.jsonl`（126 行）

## 23. v9：read/edit/write 边界成对（init=v8）（2026-17）

### 23.1 目标与假设
v7/v8 已把 reply/webfetch 挖空（v8 后池内 read/reply/edit 挖 ~80%+）。v9 靶心 = **read+edit+write 边界**（剩余可挖），加权 write（v8 最弱 33.3%、write→shell 混淆 9/18）。假设：成对边界同时抬 read/edit/write 三端、不破跷跷板。

### 23.2 挖矿与数据构建
- family=read+edit+write+webfetch（webfetch 池 2 枯竭）；caps read 180/edit 180/write 62，per-session-tool 20，max-total 600。
- 生产 fresh 1467 行（较 v8 时 82 大幅积累）；Track A 池 1717 → n_pairs 442 → queue **397**（edit 162/read 180/write 55）；build（base v8 2415）→ selected **318**（read 145/edit 131/write 42，41 会话 top-3 38%）/ extras 731 / **final 3146 行**（`data/train_v9/train_hard_weighted_v9.jsonl`）。
- 独立 LEAK 复查（train_v9 3146 vs dev 842，键 (session,step,sha256(state)[:16])）：**0**。

### 23.3 训练
init = **v8 final**（`checkpoints/grpo_train_v8`），train_v9 3146 + `--n-permutations 2`（= 6292 行），其余同 v3-v8 配方（P16/lr 3e-6/beta 0.05/max-len 4608/dev 842 早停）。**394 步跑满**（~6-7h，无早停），in-train dev 走平（峰值 0.5713 早期 = v8 峰值；后段 0.5629–0.5701）→ read/edit/write 挖矿**边际递减**（与数据天花板分析一致）。

### 23.4 验收（dev 842，raw + 校准，n-perm 0 batch 1）
| 模型 | acc | NLL | ECE | AUROC |
|---|---:|---:|---:|---:|
| v8 final（生产，基线） | 56.41 | 1.659 | 0.0578 | 0.6792 |
| v9 final（raw） | 56.89 | 4.775 | 0.4077 | 0.6639 |
| **v9 final（校准后）** | **56.89** | 1.648 | **0.0425** | 0.6827 |

校准值（`checkpoints/grpo_train_v9/calibration.json`）：global **4.2206** / choice:6-16 **4.5747** / choice:17-52 **3.3336**（raw NLL 4.78，需缩放）。

逐工具（v8 → v9 校准，Δ，pp）：edit 50.0→**64.8（+14.8，最大）**；read 48.5→**50.7（+2.2）**；write 33.3→**38.9（+5.6）**；bash 69.0→71.4（+2.4）；shell_run 69.0→72.4（+3.4）；not_done 7.1→14.3（+7.1）；shell 76.8→76.2（−0.6）；**reply 69.1→61.8（−7.3，违反门 4）**；**final_result 100.0→78.6（−21.4，n=14 小样本）**；compress/grep/glob/webfetch/execute 持平。

**门判定**：1) read/edit/write 三靶心不降 → 全部上升 ✓；2) acc ≥ v8 56.41 → 56.89 ✓；3) 校准后 NLL/ECE 达标 → 1.648/0.0425（ECE 历轮最低）✓；4) reply > −3pp → **−7.3 ✗ 违反**。

### 23.5 结论：v9 整体最优但 reply 破门，用户裁决维持 v8、v9 不部署
- v9 校准后**整体历轮最高**（acc 56.89 +0.48 vs v8、ECE 0.0425 最低），三靶心全升（edit +14.8 最大）；但 **reply −7.3**（v8 靶心回退）+ final_result −21.4 → 违反预注册门 4。
- **深层**：v8/v9 都是「整体最佳但各砸一个高流量工具」（v8 砸 write/webfetch、v9 砸 reply）——**边界挖矿的跷跷板在池子挖 ~80% 后边际递减**，与 90%/86% 天花板分析一致（继续挖数据 ROI 低）。
- **状态（11-17 用户裁决）：维持 v8，v9 不部署**；`checkpoints/grpo_train_v9`（final）+ `calibration.json`（global 4.2206/choice:6-16 4.5747/17-52 3.3336）入库待命，可随时切换/回滚。
- 下一步建议：停止纯边界挖矿（池子近枯竭）；转（a）生产积累真实 glob/webfetch 流量、（b）换机制（工具 schema 感知 prompt / 两段式决策）。

### 23.6 产物
- 数据集：`data/train_v9/{train_hard_weighted_v9.jsonl (3146), train_hard_cases_v9.jsonl (318), MANIFEST_hard_weighted_v9.json}`
- 适配器：`checkpoints/grpo_train_v9`（final + `calibration.json`）
- 评测：`/tmp/eval_v9_{final,final_calibrated}.json`；日志 `/tmp/grpo_train_v9.log`、`/tmp/v9_watch.log`

## 24. v10：生产真实死池数据挖矿（glob/grep/compress/webfetch，init=v9）（2026-11-18）

### 24.1 目标与假设

v5-v9 把 `data/train_v4/train` 池挖到 ~80% 后边际递减，v9 证实 reply/edit/read 池近枯竭；死池 **glob / webfetch / grep / compress** 长期纹丝不动（glob 0%、grep 4.5%、compress 14.8%、webfetch 30.4%），此前的根因假设是"池子枯竭（train_v4 只有 glob 3 / webfetch 2）"。v10 假设：**改用生产新累积的真实死池样本挖边界对**，能喂进训练池里没有的 glob/grep/compress/webfetch 边界，修死池 → 突破前几轮的天花板。这是复盘 §10"等生产积累"路线的第一次落地（死池累积追踪 `decisions_daily.py` 确认生产已累计 glob 51 / webfetch 33 / grep 137 / compress 227 真实样本）。

### 24.2 挖矿与数据构建

- **数据源**：`decisions.jsonl`（本地 JAH 容器，13319 行，Nov1→Nov18），**含 `state` 字段（8700/13319）+ `label`（5012）** → 死池带 state 可挖：glob 47 / webfetch 30 / grep 127 / compress 216 / websearch 7 = 427。
- **挖矿**（`mine_boundary_pairs.py --decisions`，family=glob,webfetch,grep,compress,reply,shell,read,edit，shell cap 0，exclude 全集=train_v3_clean v1-3 + train_v4 四文件 + train_v5-v9 + 四 merged queues）：pool_rows 0（train_v4 池无死池对）/ **prod_fresh 1609** / n_pairs 483 / **queue 221**（compress 57 / edit 40 / glob 29 / grep 53 / read 40 / webfetch 2）。
- **构建**（`build_hard_weighted.py --base data/train_v9/train_hard_weighted_v9.jsonl --dev-dir data/train_v4 --queue 死池队列 --version v10 --tool-caps {glob30,webfetch10,grep60,compress70,read50,edit50}`）：selected **136**（compress 43 / edit 20 / glob 26 / grep 39 / read 7 / webfetch 1；21 会话；top ses_e17744 41 行=30%）/ extras **390** / **final 3536 行**（base v9 3146 + 390）。truncated 85。
- **独立 LEAK 复查 = 0**（dev 842 keys vs train_v10 2037 unique keys）。
- 分布 v9→v10：compress/grep/glob 加权后上升（死池），其余持平。

### 24.3 训练

init = **v9 final**（`checkpoints/grpo_train_v9`，历轮最强 56.89）；train = train_v10 3536 + `--n-permutations 2`（7072 行，已确认）；P16 配方、dev 842 早停 patience 5/min 60、~442 步 ~8h。

in-train dev 轨迹：早期峰 0.5713（step 20-30）→ 中段振荡 0.5606-0.5748（**峰 0.5748**，step ~255）→ final 步 ~0.5677。**跑满 442 步（~8h），无早停**（振荡重置 patience）。大量 `frac_zero_std=1.0` no-op（v9 强 init 已解对多数，死池增广产不出梯度）。

### 24.4 验收（dev 842，n-perm 0 batch 1）

| 模型 | acc | NLL（校准后） | ECE | AUROC |
|---|---:|---:|---:|---:|
| v8（生产） | 56.41 | 1.659 | 0.0578 | 0.6792 |
| v9（历轮最优） | **56.89** | 1.648 | **0.0425** | 0.6827 |
| **v10** | 56.41 | 1.658 | 0.0636 | 0.6779 |

校准值（`checkpoints/grpo_train_v10/calibration.json`）：global **4.5067** / choice:6-16 **4.8445** / choice:17-52 **3.5937**（温度继续升：v8 3.88 → v9 4.22 → v10 4.51）。

逐工具（v10 vs v9 vs v8，n≥13，acc%）：

| 工具 | n | v10 | v9 | v8 |
|---|---:|---:|---:|---:|
| read | 134 | **54.5** | 50.7 | 48.5 |
| edit | 88 | 60.2 | 64.8 | 50.0 |
| shell | 168 | 76.2 | 76.2 | 76.8 |
| reply | 165 | 62.4 | 61.8 | 69.1 |
| bash | 42 | 61.9 | 71.4 | 69.0 |
| **compress** | 27 | **18.5** | 14.8 | 14.8 |
| **grep** | 22 | 4.5 | 4.5 | 4.5 |
| **glob** | 15 | 0.0 | 0.0 | 0.0 |
| **webfetch** | 23 | 30.4 | 30.4 | 30.4 |
| write | 18 | 38.9 | 38.9 | 33.3 |
| execute | 13 | 38.5 | 30.8 | 30.8 |

**门判定（开训前写死）**：1) 死池 glob/grep/compress/webfetch 净提升 → ✗（4 个里只有 compress +3.7 微升，glob/grep/webfetch 全 0）；2) reply 不大幅回退 → ✓（vs v9 +0.6）；3) 整体 acc ≥ v9 56.89 → ✗（56.41 < 56.89）；4) 校准后 NLL/ECE 达标 → ✓（1.658/0.064）。→ **2✓ 2✗**。

### 24.5 结论：v10 不部署，数据天花板第三次确认

- **假设否决**：换用生产真实死池数据（glob 26/grep 39/compress 43 边界对），死池工具**几乎全没动** —— glob 仍 0%、grep 仍 4.5%、webfetch 仍 30.4%，只有 compress +3.7（27 样本 ≈ 1 条，噪声级）。整体 56.41 = v8、< v9 56.89。
- **天花板第三次确认**：v9（挖 train_v4 池）没动死池 + v10（挖生产真实死池数据）也没动 → **铁证这 4 个死池工具不是"缺数据"，是结构性的**（模型没建立 glob/grep/webfetch 的边界判别，或 dev 样本太小 ±10pp 全噪声）。**换数据源同样突破不了 ~72-78% 硬顶**。
- **状态（11-18）：v10 不部署，生产维持 v8**；`checkpoints/grpo_train_v10`（final）+ `calibration.json`（global 4.5067）入库待命。
- **至此所有路线已试完**：边界挖矿（v5-v9 枯竭）+ 换真实数据（v10 死池没动）+ 三个推断期 trick（schema −4.87 / 两段式 −5.11 / 分组读出 no-op）= 现有机制+数据下 **v8 即局部最优**。再往前的唯一路径：① 持续生产积累慢慢喂（glob 今日 19 在涨）+ ② 换根本机制（schema 感知需重训、或更强 base）。

### 24.6 产物

- 数据集：`data/train_v10/{train_hard_weighted_v10.jsonl (3536), train_hard_cases_v10.jsonl (136), MANIFEST_hard_weighted_v10.json}`
- 挖矿队列：`checkpoints/mine_deadpool_v10/queue.jsonl`（221 行）
- 适配器：`checkpoints/grpo_train_v10`（final + `calibration.json`）
- 评测：`/tmp/eval_v10_{final,final_calibrated}.json`；日志 `/tmp/grpo_train_v10.log`、`/tmp/v10_watch.log`
- 挖矿脚本：`research/scripts/mine_boundary_pairs.py --decisions`；死池追踪：`research/scripts/decisions_daily.py`

## 25. v11：schema 感知重训（name -- desc，init=AgentDecision-4B Nov1 原版）（2026-11-18/19）

### 25.1 目标与假设
v7-v10 确认数据天花板 ~72-78%（挖矿枯竭 + 换真实死池数据都没动 glob/grep/webfetch 死池），三个推断期 trick 全废。v11 换**根本机制 = 表征**：把 prompt options 从裸名改成 `name -- desc`（真实工具语义），假设模型混淆 reply↔shell / 死池 glob/grep/webfetch 的根因之一是"选项只是一个名字、缺乏语义边界"，加描述能建立更锐利的工具判别。init 用 **AgentDecision-4B（Nov 1 原版，v3-v10 GRPO 前基线，md5 661b34）** 而非 v8（用户明确"不从 v8 开始"，避免继承 v8 裸名偏见，干净归因格式变量）。base 维持 Qwen3-4B 不动。

### 25.2 格式（方案 A）
- 27 个 core 工具（= dev 842 全部 gold 工具）→ `name -- desc`（描述源 `data/tool_desc.json`，真实 JAH 语义）；200 个 MCP 工具（全 distractor、gold=0）保持裸名。
- 生产侧同用共享 `data/tool_desc.json` 只给这 27 个描述 → **训练/生产逐字节一致**（方案 A，最小改动）。
- 数据 `data/train_v11/{train_hard_weighted_v11.jsonl (3536), dev.jsonl (842), ...}`（options + target 键同步转 `name -- desc`），**LEAK 0**（transform 不改 state/session/step，dedup key 不变，同 v10）。

### 25.3 训练
init = AgentDecision-4B（Nov1 原版）；train_v11 3536 + `--n-permutations 2`（7072 行）；P16 配方（lr3e-6 / beta0.05 / max-len4608）、dev 842 早停 patience 5/min 60、1 epoch 442 步、GPU1、`--rewards correct`。

in-train dev 轨迹：起步 ~0.477（Nov1 基线比 v8/v9 弱，符合预期）→ 缓慢爬升 → **峰 0.5119（step 440）= final**（跑满 442 步，无早停）。全程低于 v8/v9 的 ~0.565；`frac_zero_std` 末段升到 0.94（已收敛，继续训不涨）。

### 25.4 验收（dev 842，n-perm 0 batch 1，name--desc）
| 模型 | acc | NLL（校准后） | ECE（校准后） | AUROC |
|---|---:|---:|---:|---:|
| v8（生产，裸名） | 56.41 | 1.659 | 0.0578 | 0.6792 |
| v9（历轮最优，裸名） | 56.89 | 1.648 | 0.0425 | 0.6827 |
| **v11（name--desc）** | **51.19** | 1.725 | 0.052 | 0.6779 |

> 口径：v11 在 name--desc dev 评、v8/v9 在裸名 dev 评，但**同 842 个决策**（同 state / 同 gold，仅 option 渲染不同），逐工具可比。

raw（未校准）：acc 51.19 / NLL 3.416 / Brier 0.405 / ECE 0.410 / mean_conf **0.920**（过度自信）。校准 global **3.079**（**首次下降**：v8 3.877 → v9 4.22 → v10 4.51 → **v11 3.079**，schema 让 raw 概率更分散）→ 校准后 ECE 0.052（达标）、NLL 1.725、AUROC 0.6779。

逐工具（v11 vs v8，n≥13，acc%）：

| 工具 | n | v8 | v11 | Δ |
|---|---:|---:|---:|---:|
| read | 134 | 48.5 | **67.9** | **+19.4** |
| webfetch | 23 | 30.4 | **47.8** | **+17.4** |
| compress | 27 | 14.8 | 25.9 | +11.1 |
| edit | 88 | 50.0 | 56.8 | +6.8 |
| shell | 168 | 76.8 | 71.4 | −5.4 |
| write | 18 | 33.3 | 22.2 | −11.1 |
| bash | 42 | 69.0 | 40.5 | **−28.5** |
| reply | 165 | 69.1 | **31.5** | **−37.6** |
| grep | 22 | 4.5 | 4.5 | 0（死池没动）|
| glob | 15 | 0.0 | 0.0 | 0（死池没动）|

**门判定（开训前写死）**：1) schema 让整体 acc ≥ v8 56.41 → ✗（51.19，−5.2pp）；2) reply 不大幅回退 → ✗（reply 69.1→31.5，−37.6）；3) 校准后 NLL/ECE 达标 → ✓（1.725 / 0.052）。→ **主门 ✗**。

### 25.5 结论：v11 不部署，schema 引入系统性"过度动作"偏置
- **假设部分成立但净负**：schema 确实让"行动类"工具判别更锐利 —— read **+19.4** / webfetch **+17.4** / compress +11.1 / edit +6.8 全升（描述帮模型建立"该用哪个行动工具"的边界）。**但**给工具加描述让 act 类工具变"更抢手"，模型系统性**过度动作**：reply 该停手时疯狂发 read(50) / edit(34)，reply 崩塌 **−37.6**、bash −28.5。净 **−44 个正确**（v8 475 → v11 431）。
- **新发现（重要，改写候选清单 E3 方向判断）**：v8/v9/v10（裸名）主误差 = **欠动作**（gold=shell→reply）；v11（schema）主误差翻转为**过度动作**（gold=reply→tool，reply→read 50 + reply→edit 34 = 84 条）。**主误差方向随"格式"翻转**——方向性惩罚 E3 不能一刀切，必须跟随目标模型误差方向。用户 @431@ 原提案（gold=reply→tool 罚 −1）在 v8 时代罚的是少数派，**但在 schema 模型上恰好命中主误差**。
- **死池仍没动**（glob 0 / grep 4.5）→ **数据天花板第四次确认**：换机制（schema 表征）同样破不了，且 schema 净负（比 v8 差 5.2pp）。
- **状态（11-19）：v11 不部署，生产维持 v8**；`checkpoints/grpo_train_v11`（final，51.19）+ `calibration.json`（global 3.079）入库待命。

### 25.6 产物
- 数据集：`data/train_v11/{train_hard_weighted_v11.jsonl (3536), dev.jsonl (842), ...}`；描述源 `data/tool_desc.json`（27 core，训练/生产共享）
- 数据变换：`research/scripts/transform_schema_v11.py`；工具清单：`research/scripts/list_tools_v11.py`
- 适配器：`checkpoints/grpo_train_v11`（final + `calibration.json` global 3.079）
- 评测：`/tmp/eval_v11_{final,final_calibrated}.json`；日志 `/tmp/grpo_train_v11.log`、`/tmp/v11_watch.log`
- 部署前提（若采纳 schema）：生产 render 同步改 `option_text(label, desc)` + 重校准，两侧原子改；且须先解决"过度动作"偏置（见 E3 方向修正）。
- 挖矿：`checkpoints/mine_boundary_v9/queue.jsonl`（397 行）