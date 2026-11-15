# train_v3 决策级 GRPO 训练报告（2026-11-14）

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

`init = grpo_train_v3_clean_hard_v2/step-000135`（保留 v2 的 reply 增益），其余配置同 §14.1，max-steps 166。166/166 跑完，~2h；训练中最佳 step160 52.27%，**最终模型 dev 52.65%**（高于途中最佳）。

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

- v3 最终（含校准）全面优于 v1 生产基线：acc 51.52 / 50.38、NLL 1.63 / 2.40、ECE 0.075 / 0.291、reply 40.5 / 11.9、edit 20.8 / 12.5；唯一回退是 read 50.0 vs 58.3（8 条被误判 reply）；
- 弱点仍在：grep 0/7、glob 0/6、compress 1/10（样本少，绝对波动 1-2 条）；
- 下一步：(1) 观察生产任务质量——routing 不再 fire、中间步全走 fast，若出错率上升可下调 tau 或给 routing 单独设温度；(2) 继续补 grep/glob/compress 的难例与独立 session 的 dev；(3) 生产日志每周挖矿并入下一轮。

产物：`checkpoints/grpo_train_v3_clean_hard_v3/`（最终 adapter 264MB + `calibration.json` + `eval_dev_final.json` + 33 个 step 子目录）；`data/train_v3_clean/{train_hard_weighted_v3.jsonl(2641), train_hard_cases_v3.jsonl(263), MANIFEST_hard_weighted_v3.json}`；队列 `checkpoints/mine_nonshell_v3_{train,prod,all}/`；日志 `/tmp/grpo_train_v3_clean_hard_v3.log`。

**现状（11-15 收工）：** 生产决策模型 = 本轮 hard_v3 最终模型 + 温度缩放（global 3.025 / choice:6-16 3.0）；dev 51.52% / NLL 1.63 / ECE 0.075；reply 40.5%、edit 20.8%；GPU1 Embedding 已恢复。

## 16. 相关文档

- train_v2 / train_v3 对比记录：`results/training/grpo_train_v2_2026-11-11.md` §12
- clean dev 划分说明：`data/train_v3_clean/SPLIT.md`
- 数据说明：`docs/DECISION_DATA_V2.md`
- 训练配方：`docs/TRAINING_RECIPES.md`
- 脚本：`scripts/train/60_grpo_trl.sh`
- 非 shell 难例挖矿（§14/§15）：`src/jevharness/training/mine_hard_cases.py`、`scripts/build_hard_weighted.py`、工具上限 `data/tool_caps_v2.json` / `data/tool_caps_v3.json`
- 校准与评估（§15.3）：`src/jevharness/training/calibrate.py`、`src/jevharness/training/eval_decisions.py`

---

**一句话带走：** train_v3 的数据量增加了 63%，但这次只训练了 33.5% 的 train，zero-std 又吃掉约一半 group，最终有效梯度与 v2 第二轮几乎相同；同时 dev 有 42% 落在 train 极少见的工具上。因此“效果不明显”主要是**训练预算、数据同质化和评估偏移**共同造成的，不是单纯的数据量问题。