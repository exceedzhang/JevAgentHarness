# 决策模型 GRPO 训练数据（v2，生产数据源）

> 2026-11-07。配合 11-07 上线的 `log_decision_state` 与 `training/data_v2.py` 使用。
> 规范与配方：`BENCHMARK_SPEC.md`（指标）、`TRAINING_RECIPES.md`（R6b 决策级 GRPO、
> `data.py` 的示例格式是唯一数据格式）。本文只讲生产侧数据。

## 1. 现状盘点（11-07，`~/.jah/litellm-logs/decisions.jsonl`，4509 行）

| 决策点 | 行数 | 带标签 | 带 state（11-07 后） | 说明 |
|---|---|---|---|---|
| routing | 3642 | 0（无重跑，无金标） | 11-07 起 | 只可做 soft 目标（模型自身 p），`--include-routing` 可选 |
| tool_select | 355 标签（≤11-04） | 355 | 11-07 起重新采集 | 标签 = agent 实际调用的工具（System-2 金标） |
| stop | 148 标签 | 148 | 否（保持 off） | 标签需 strong 全上下文重跑；fit 负结论已定案，不重开 |
| 错误行 | 5 | — | — | 全部在 11-05 前（旧模型/隧道瞬断） |

关键缺口：**11-07 之前所有决策行都没有 state**——只有 state + question + options 才能拼出
与服务端 byte 级一致的训练 prompt（`data.render`），旧标签行因此不可用，这是 v2 与旧日志的
分界线，不是缺陷。

## 2. 11-07 的两处改动（已部署验证）

1. **`log_decision_state: true`**（`jah-config.yaml`）：决策行（routing + 各决策点）写入
   `state` 字段——即**截断后发给决策模型的精确字符串**（`Decision.state`，含 TRUNC_NOTE）。
   探针验证：`grpo-state-check-1` 会话的 routing 与 tool_select 行均带 71 字符 state，
   格式 `TASK: … STEPS SO FAR: … [n] ACTION: … RESULT: …`。
2. **`tool_select: {mode: shadow}` 重新开启**：仅为采标签（shadow 不干预请求；fit 负结论
   仍然有效——该点禁止进 guard/trigger 模式）。代价 = 每步一次免费决策调用（~0.26 s），
   按 11-03 流量约 200 标签/天。

## 3. 构建 v2 数据集

```bash
# research/ 目录
PYTHONPATH=src uv run python -m jevharness.training.data_v2 \
    ~/.jah/litellm-logs/decisions.jsonl --out data/train_v2 [--since 2026-11-07]
```

- 输出 `train.jsonl` / `dev.jsonl`（按 session 分组切分，同一会话不跨集）+ `MANIFEST.json`
  （每条 skip 原因计数：`no_state` / `label_not_in_options` / `duplicate` / `error_row` …，
  静默丢数据会显形）。
- 规则：`label` 非空 + `fit_ok` + `state` 非空；标签必须出现在 options 里（工具改名
  bash→shell 的跨代行自动跳过）；(session, step, point) 去重，与 `jah fit` 同一口径。
- 11-07 当天实测：4509 行 → 3 条（全部来自部署后的新行），`no_state` 859、
  `point_unlabelled` 3642（routing）、`error_row` 5。

规模估计：tool_select ~200/天，**两周 ≈ 2–3k 条**（加上 routing soft 行可更多，但 routing
soft 行只建议用于消融，不进主目标）。

## 4. 训练路径（与 R6b 配方的对应）

当前部署的 4B 输出 **JSON 概率**（`/v1/systemone` 的 card 接口），而 `grpo_trl.py` 的 TRL
实现要求初始模型直接输出 **option-code token**（A/B/C…，`data.render` 格式）。因此重训是
两段式：

1. **SFT**（`scripts/train/12_sft_v1.sh`，L4/L40S preset）：在 `mix(train_v1, train_v2)` 上
   把 4B 从 JSON 概率头迁到 option-code 头（v1 的 15.7k 条为主干，v2 提供生产分布增量）。
2. **R6b 决策级 GRPO**（`60_grpo_trl.sh`，G=8，rewards `correct,format`；或 `IMPL=torch`
   的受限 softmax 实现，无需生成、单进程）：在 SFT adapter 之上，用 v2 的金标行做可验证
   奖励（argmax==gold）+ 格式奖励 +（可选）soft/qvalue 奖励。

之后回生产：新模型上线时换 `decision.model` 与 calibration（新模型重新标定；
`jah fit --decision-model <name>` 按 era 重拟合 tau；decisions 行带 model×calibrated×truncated，
era 天然可切，见 `6909f1d`）。

## 5. 注意事项

- **不要混代**：训练/评测一律按 era（model × calibrated × truncated）切分；旧 355 条
  tool_select 标签（无 state）若将来需要，只能用于统计，不能进 prompt 级训练。
- **state 截断是忠实的**：`max_state_tokens: 2800` 截尾正是服务端看到的字节，训练用截断态
  等于线上真实输入，不是近似。
- **tool_select 标签语义**：标签是 agent *实际*调用的工具（含模型可能答错的情形）——这正是
  GRPO 要的"可验证奖励"：答对=agent 真实选择。它不要求决策模型在训练前就答得对
  （11-04 一致率 13.4% 不改变数据的有效性，只说明起点低）。
- **stop 不采**：标签要付 strong 全上下文重跑的代价且 fit 已证伪，维持 off。
- **成本**：shadow 采集只消耗免费决策模型；SFT/GRPO 的 GPU 成本见 `TRAINING_RECIPES.md`
  预算表（T6b 单卡 1–8 h）。

## 6. 首次 train_v2 GRPO 实验记录

2026-11-11 在已发布的 `AgentDecision-4B` adapter 上，用 `data/train_v2` 做了 1 epoch
决策级 GRPO（`--impl torch`，reward=`correct`）。dev accuracy 10.81% → 35.14%，但
NLL/Brier/ECE 变差，且训练后段出现明显 entropy collapse。完整命令、指标、问题分析与
下一次训练建议见
[`../results/training/grpo_train_v2_2026-11-11.md`](../results/training/grpo_train_v2_2026-11-11.md)。

## 7. routing 金标与 choice:2 拟合（11-12，负结论）

新 GRPO 决策模型于 11-11 18:00 上线（同一逻辑名 `AgentDecision-4B`、同一 ssh-tunnel 端点）。
tool_select 已单独重标定（`choice:6-16: 2.508`），但 routing 头（2 选项 hard/routine）仍沿用旧
模型 `global` 温度（0.772）且无自身金标。这里生成 routing 金标并拟合专用 `choice:2`。

**金标构造**（`scripts/routing_gold_judge.py`）：取新 era（≥11-11 18:00）全部带 state 的 routing 行
（292 条），用 strong 模型（deepseek-v4.1-flash，即生产 strong 层）作 reference judge。judge 只看
state（不看结果，与决策模型输入一致），逐条判 ROUTINE/HARD。结果 **47.3% HARD / 52.7% ROUTINE**
（judge 判断均衡、无明显偏向）。金标 + state 存 `data/routing_gold_v2.jsonl`（292 行，自包含）。

**拟合**（`scripts/routing_gold_fit.py`）：反演 logged p(hard)→raw（global 0.772），
`fit_temperature` → **T 撞搜索上界 20**（坍缩到先验：median 0.756→0.511，fires@0.85 36%→0%）。

**结论（负）**：routing 头没有可用的难度信号。
- AUC(logged p(hard) vs judge) = **0.462**（SE 0.034，z=−1.13，p≈0.26）——**与 0.5 无显著差异**
  （raw AUC 0.446）。并非强反相关，信号基本≈随机。
- 对 judge 的 routing 准确率 @tau 0.85 = **49.7% < "永远发 fast" 基线 52.7%**；全 tau 最优仅 54.8%
  （+2.1pp，边际）。
- T=20 坍缩后 NLL/Brier/ECE "变好"是坍缩到先验的假象，routing 并未改善。

**处理**：**不给 routing 加 `choice:2`**（温度只能重标定单调信号；对无信息信号无意义，只会坍缩
成抛硬币）。维持当前配置（global 0.772 + tau 0.85，fires ~36%，*率*合理）。这是真实发现：GRPO 提升
了 tool_select（10.8%→35.1%）但 routing 头基本失效。若 routing 质量真正重要，有两条确定性路径：
(a) **A/B 金标**：对一批步骤实跑 fast vs strong，量实际失败率（"hard=fast 会失败"的权威定义）；
(b) **routing 回退**：routing 用 GRPO 前旧模型、tool_select 用新模型（需同时服务两个决策模型）。

注：judge 是 strong 模型的判断（代理，非 ground truth）；即便按 judge 字面，routing 也低于
always-fast 基线，故"不信任新 routing / 不为其标定"的结论成立。

## 8. A/B 金标：fast 模型真实失败率（11-12/13，含数据告警）

§7 的 judge 是代理。这里用一个**客观** A/B 金标直接回答"这步是否真的 hard"：把 fast 模型
（Qwen3.8-27B，仅给 15 个工具名、无 schema）跑在每个 routing state 上，让它预测下一个工具，
与 agent 实际执行的动作（`tool_select` 标签）对比。**fast 未能复现实际动作 = hard**。
脚本：`scripts/ab_fast_run.py`（远端 27B 上跑）+ `scripts/ab_score.py`（本地计分）。

**数据告警（重要）**：292 条新 era routing 行其实只有 **120 个唯一 (session,step) 步**（客户端重试
把同一步记多次；同一步的 `actual` 与 `tools` 一致，但 `state` 文本每次渲染不同）。`ab_out.jsonl`
（fast 预测）以完成顺序写入、与 `ab_input.jsonl` 不按行对齐，故 86 个重复步的两条预测无法绑回
具体某次重试，其中 42 个两步预测不一致——约 35% 唯一步的 A/B 金标有歧义。因此**下文精确数字依赖
去重/对齐方法**（`ab_score.py` 另有一处 judge 计数小 bug）。稳健结论不依赖方法。干净重建
（按步取最早 state 去重 + 重跑 fast）是后续。

**结果（干净重建，权威）**：120 个唯一步各取最早一条 canonical state 重跑 fast（1:1，无配对歧义）：
- A/B 金标 hard 率（fast 失败）= **63.3%**；judge 金标 hard 率 = **55.0%**；两者一致率仅 **53.3%**
  （A/B gold 被 fast 残血污染，"fast 失败" ≠ "这步本身难"）。
- **AUC(p_hard, A/B 金标) = 0.459**（SE 0.039，≈随机）；**AUC(p_hard, judge 金标) = 0.548**
  （SE 0.037，略高于随机）。
- 早先"78 无歧义步"的 0.646 是用多条一致预测的高估；单条干净预测下回落到随机。

**结论（最终）**：routing 头基本无有用信号（对两种 gold 的 AUC 都≈随机）；决策模型价值在
tool_select（GRPO 后 10.8%→35.1%）。维持 routing 现状（global 0.772 + tau 0.85），不为它标定；
要让 routing 更强需重训 routing 头或用旧模型。

## 9. train_v2 GRPO 实验记录（两轮）

2026-11-11 在已发布的 `AgentDecision-4B` adapter 上，用 `data/train_v2` 做了两轮
决策级 GRPO（`--impl torch`，reward=`correct`）。第一轮 1 epoch：dev accuracy
10.81% → 35.14%，但 NLL/Brier/ECE 变差，且训练后段出现明显 entropy collapse。
第二轮加了逐 checkpoint dev 评估与早停（G=16、lr=3e-6、beta=0.05）：collapse 明显缓解，
但 dev top-1 最高为 step 40 的 29.73%，低于第一轮最终点；最佳部署候选不是最终 step 80。
完整命令、指标、两轮对比与下一步建议见
[`../results/training/grpo_train_v2_2026-11-11.md`](../results/training/grpo_train_v2_2026-11-11.md)。

## 10. train_v3 重新划分 dev（11-14）

旧 `data/train_v3/dev.jsonl` 只有 81 条 / 14 个 session：其中一个 alt session 占 38 条，
且 dev 几乎没有 main harness 的 shell/edit 主体；另外有 16 条来自 v2 train。用它做 v2/v3
模型对比会同时受到分布偏移和泄漏影响。

新的 session-level 划分在 `data/train_v3_clean/`，原始 `data/train_v3` 保持不变：

| split | rows | sessions | 用途 |
|---|---:|---:|---|
| `train.jsonl` | 2189 | 105 | 训练（975 条 main session 和 49ff 都留在 train） |
| `dev.jsonl` | 264 | 11 | **主 dev**，main harness |
| `dev_alt.jsonl` | 94 | 22 | alt harness + 稀有工具，单独报告 |
| `dev_shell.jsonl` | 12 | 1 | shell_run / browse_web 诊断，不参与主指标 |

关键规则：

- 不要把 `dev.jsonl` / `dev_alt.jsonl` / `dev_shell.jsonl` 拼成一个 aggregate；
- `dev.jsonl` 只作为模型选择主指标，alt 和 shell 分开看 per-tool accuracy / confusion；
- `shell_wait` 没有 held-out 行（7 条都在 train 的 49ff 里），这是当前数据缺口；
- 旧 v2/v3 checkpoint 都见过这些 dev 行，不能在新 dev 上直接比较，必须在
  `train_v3_clean/train.jsonl` 上重训。

重建脚本：`scripts/resplit_v3_dev.py --force`；详细说明见
`data/train_v3_clean/SPLIT.md`。评估脚本已增加 `by_tool` 与 `confusion`。

2026-11-15 已按该 split 做 P1 重训：`dev-acc` 早停在 step 55/137（未跑满 1 epoch），
最佳 top-1 为 `step-000040`；主 dev 相对 init +3.79 pp，但 NLL/Brier/ECE 都更差。
结果与对比见 [`../results/training/grpo_train_v3_2026-11-14.md`](../results/training/grpo_train_v3_2026-11-14.md) §12。

## 11. 数据积累路线（11-15，hard_v2/v3 之后）

现状基线：生产 = hard_v3 模型 + 新校准（dev 51.52%、NLL 1.63、ECE 0.075）；
tool_select 标签采集 ~300-500 条/天（shadow）；dev 264 条。

### 11.1 轨道 A：每周生产挖矿（主力）
- 命令：`mine_hard_cases --decisions data/decisions.jsonl --exclude <历次 hard_cases + 全部 dev 切片>`（服务器）；
  模型/adapter 用当前生产 adapter 打分。
- caps 向弱工具倾斜：grep / glob / compress / webfetch / write 提配额；reply 继续压低
  （已 40.5%，防 v2 式过校正）；shell 系 = 0。
- 净增 ~500 条/月（标签增速 ~300-500/天，去重 + 排除已用后）；gold = agent 实际动作
  （自动入库），抽查 10% 保证标签质量。
- 新样本：本机 `~/.jah/litellm-logs/decisions.jsonl` 定期 scp 上服务器
  `research/data/decisions.jsonl`（11-15 时 10100 行）。

### 11.2 轨道 B：扩 dev（最优先，结构性短板）
- 当前 dev 264 条太小：单工具 ±4pp ≈ ±10 条，区分度低（v2 的 read 回退就靠它才看见）。
- rolling holdout：最近 3-5 天的 session 只进 dev、不进 train（时间切分天然防泄漏，
  与 session 切分二选一或混合）。
- 逐工具切片（仿 dev_alt/dev_shell）：dev_grep / dev_compress / dev_edit / dev_web，
  各 10-30 条，用于逐工具验收。
- 目标：3 个月内 dev 264 → 800-1200，每个弱工具 ≥30 条。

### 11.3 轨道 C（低优先）
- **stop**：shadow 继续攒新 era 标签，≥30 条后重拟合 tau —— 唯一有直接省钱收益的点
  （fast 终答不再被 strong 重跑）。
- **routing**：现状 = 校准后几乎不 fire（p_hard 0.09-0.36 < tau 0.85），first-step +
  finish check 兜底。若要翻案，需要 A/B 金标连续 4 周（~150 唯一步/周，攒 600+）且
  AUC 稳定 >0.65；否则维持现状。

### 11.4 纪律
1. 去重键 = `(session, step, sha256(state)[:16])`，保留最早一条（与 data_v2 一致）。
2. 防泄漏：任何新样本入 train 前先 `--exclude` 对照 dev / dev_alt / dev_shell + 历次
   hard_cases（v2/v3 均为 0 泄漏，保持）。
3. 版本化：`train_v4` / `hard_weighted_v4` + MANIFEST 提交仓库；不覆盖旧版本。
4. session 失衡：单 session ≤5% 入 train 池（v3 头部 session 占 39%，需降采样）。
5. 验收门 = 逐工具混淆矩阵 + NLL/ECE；整体 acc 不算数（v2 教训：整体 50% 不变，
   但 reply +19pp 与 read −11pp 此消彼长）。
6. truncated 样本：本轮 build 丢了 41 条（state 超 2800 token 截断），需查占比与原因，
   决定是否提高 max_state_tokens 或单独处理。

### 11.5 节奏
- 每周：生产挖矿（轨道 A）+ 时间切分进 dev（轨道 B）。
- 每 2-4 周：合并构建 `hard_weighted_vN` → 训练（P=16、skip-zero-std、dev-acc 早停）
  → 逐工具验收 → 温度缩放 → 部署（adapter + calibration 一起，见 grpo_train_v3 记录
  §14/§15 的流程）。
- 每月：审 session 失衡与 truncated 比例；看 stop 标签是否到 30 条可拟合。
