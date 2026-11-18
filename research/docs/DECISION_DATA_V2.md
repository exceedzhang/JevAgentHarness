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

### 12. v5/v6 轮进展与部署决策（11-16）

- **v5（弱工具配额挖矿，train_v5 1951 行）**：假设不成立，不部署（理由见 grpo 文档 §19）。
- **v6（read↔reply / edit↔read 边界成对，train_v6 2139 行）**：校准后 dev 842 = acc **54.87** / NLL **1.644** / ECE **0.060**（v3 基线 54.16 / 1.63 / 0.075）；逐工具 reply +3.6、edit +9.1，read −5.2、webfetch −4.3、write −5.6。预注册门达标 → 用户裁决（11-16）：**维持 v3，v6 不部署**；v6 adapter + calibration 入库待命，随时可切换/回滚。
- 纪律保持：`--exclude` 覆盖全部已用集（train_v4 四文件 + hard_cases v1-v3 + train_v5 两文件 + 历次 merged queue）；去重键 (session, step, sha256(state)[:16])；v5/v6 两轮独立 LEAK 复查均为 0。
- 下一步：部署决定后——若上 v6，盯 read 切片线上 p 分布（可回滚 v3）；若不上，配额向 read/webfetch/write 边界对倾斜 + option-permutation 增广（glob/webfetch 池已枯竭）。

### 13. v7/v8 轮进展与 v8 部署（11-16 / 11-17）

- **v7（read/write/webfetch 边界成对 + option-permutation n=2，train_v7 2189 行，init=v3 final）**：dev 842 校准后 acc **55.94** / NLL 1.642 / ECE **0.0486** / AUROC 0.682（历轮最强）。靶心三项全部命中：read 47.0→50.0(+3.0)、write 33.3→44.4(+11.1)、webfetch 30.4→34.8(+4.3)；但 reply 65.5→58.8（−6.7）。作为 v8 init，未单独部署。详见 grpo 文档 §21。
- **v8（reply↔read 平衡轮，train_v8 2415 行，init=**v7 final**）**：dev 842 校准后 acc **56.41**（历轮最高，+2.25 vs v3、+0.47 vs v7）/ NLL 1.659 / ECE 0.0578。reply↔read **两端同时**抬到 v3 之上：reply 58.8→**69.1**（+10.3，超 v3 65.5）、read 50.0→48.5（≥v3 47.0）。代价：write −11.1 / webfetch −4.3 / execute −15.4 vs v7。预注册门四条全达标 → **已部署**。详见 grpo 文档 §22。
- **v8 部署（客户端 JAH 架构）**：生产 JAH 决策跑在本地 litellm-jah + ssh-tunnel（非服务器纯 litellm 代理）。服务端 vLLM adapter = v8（served `AgentDecision-4B-1117`，md5 与 `grpo_train_v8` 一致）；本地 litellm-jah 镜像重建为 v8 校准（`litellm-jah:v8`，`jah-config.yaml` global 3.877 / choice:6-16 4.203 / 17-52 3.069，commit `fe495e5`）；`docker compose up -d --force-recreate` 生效。p_hard 初步：切点前最近 150 条 routing 中位 0.541、>0.85 占 7.3%（更早 bulk 0.619 / 20.8%）→ 更平；但尚无干净 post-切点样本，待积累后复核。
- 纪律保持：`--exclude` 覆盖全部已用集；去重键 (session, step, sha256(state)[:16])；v7/v8 两轮独立 LEAK 复查均为 0。v8 selected 113 难例里 top-2 session 占 63%（集中度偏高，需观察是否过拟合这两会话）。
- 回滚：v3（global 3.025 / choice:6-16 3.000）、v7（global 3.381 / choice:6-16 3.664）adapter+calibration 均保留，可随时切回。

### 13.1 v8 上线后 fit 报告 + routing 触发率（11-17，客户端 decisions.jsonl）

数据：本地 `litellm-jah` 容器 `~/.jah/litellm-logs/decisions.jsonl`（12678 行，Nov 1→17 01:48 UTC）。`jah fit`（`jevagentharness.cli fit`）在 `--target 0.95 --min-count 30 --z 1.645` 下拟合 `tool_select`/`stop` 阈值；`--since 2026-11-17 01:18` 隔离 v8 时代。

- **全量 pooled（Nov 1–17，主体 v3/preview 时代）**：tool_select 4128 标签 / stop 373，**均无阈值达 95% 一致**（≥30 条）。逐工具最佳一致率（/ 下界 / tau / n）：shell 90.6/78.7/0.8225/32、read 87.5/68.4/0.6016/16、compress 79.0/60.5/0.8365/19、bash 90.0/65.2/0.5104/10、edit 43.8/30.4/0.3854/32、write 36.4/17.5/0.7627/11、webfetch **16.3**/9.4/0.2107/49（最差）。routing:cascade 7631 calls，**fires at logged tau 28.2%**。→ tool_select/stop 均保持 LLM 决策（shadow/strong），只收集标签。
- **v8 时代（`--since 2026-11-17 01:18`，~30min，269 tool_select 标签）**：routing **fires at logged tau 10.7%**（vs pooled 28.2%）；tool_select 269 标签**仍无 95% 阈值**（shell 最佳 70.3% / 37 条，read 100% 仅 1 条）；stop 42 无阈值。配置片段同 pooled：tool_select → off/shadow、stop → strong。
- **解读**：v8 校准（global T=3.877）压平 p_hard → routing 触发从 28.2% 降到 **10.7%**（~89% 步走 fast、省 ~2/3 strong 调用，方向对/省钱）；但存在**路由不足**风险（若 T 压太狠，本应走 strong 的难步被压到 fast，需盯 10.7% 那批 strong 步 vs fast 步的下游成功率）。tool_select 在 v8 时代一致率（shell 70.3%）仍 <95% 且样本仅 30min，**与 dev 逐工具（webfetch/write/edit 弱、shell/read 相对好）交叉一致** → tool_select 维持 shadow。
- **caveat + 下一步**：v8 时代样本仅 ~30min（269 tool_select / 263 routing），属**方向性结论**；建议每天用 `--since 2026-11-17T01:18Z` 重跑 fit，盯（1）routing 10.7% 是否路由不足（若难例被 fast 漏 → 单独调 routing tau 或 T）、（2）tool_select 逐工具一致率是否收敛到某工具可单独提 act（shell 最有希望）。

### 13.2 v9 轮（read/edit/write 边界，init=v8）+ 裁决（11-17）
- **v9（read/edit/write 边界成对，train_v9 3146 行，init=v8 final）**：校准后 acc **56.89**（历轮最高，+0.48 vs v8）/ NLL 1.648 / **ECE 0.0425**（历轮最低）/ AUROC 0.6827。三靶心全升：edit 50.0→64.8（**+14.8 最大**）、read 48.5→50.7（+2.2）、write 33.3→38.9（+5.6）；但 **reply 69.1→61.8（−7.3，违反预注册门 4 的 reply>−3pp）**、final_result 100.0→78.6（−21.4，n=14）。校准 global 4.2206 / choice:6-16 4.5747 / 17-52 3.3336。详见 grpo 文档 §23。
- **裁决（11-17 用户）：维持 v8，v9 不部署**（3/4 门通过、门 4 reply 回退违反；v9 整体最优但砸 reply 高流量工具）。v9 adapter + calibration 入库待命，可切换/回滚。
- **深层 + 下一步**：v8/v9 都是「整体最佳但各砸一个高流量工具」（v8 砸 write/webfetch、v9 砸 reply）——**边界挖矿的跷跷板在池子挖 ~80% 后边际递减**（与 90%/86% 天花板分析一致，继续挖数据 ROI 低）。建议停止纯边界挖矿，转（a）生产积累真实 glob/webfetch 流量、（b）换机制（工具 schema 感知 prompt / 两段式决策）。

### 13.3 shell 族混淆分析（11-17，客户端 decisions.jsonl，tool_select）

数据：本地 `litellm-jah` 容器 `decisions.jsonl`（13214 行，Nov 1→17 15:43 UTC）的 tool_select 行（全量 4480 标签 / 今天 Nov17 522 标签）。`choice`=决策模型 top-1 预测、`label`=agent 实际用的工具、agree=两者一致。注：这是 tool_select（shadow 收集）的 top-1 口径，与 dev 逐工具 acc（不同决策点/样本）不直接比大小。

**核心结论：shell 和 bash 不互相混淆（双向交叉 = 0）**。混淆矩阵（gold label 行 × 模型 choice 列，全量）：

| gold ↓ \ choice → | shell | bash | shell_run | shell_wait | other | 总 |
|---|---:|---:|---:|---:|---:|---:|
| shell (1556) | **473** | 0 | 0 | 0 | 1083 | 1556 |
| bash (246) | 0 | **22** | 0 | 0 | 224 | 246 |
| shell_run (30) | 0 | 0 | 18 | 0 | 12 | 30 |
| shell_wait (7) | 0 | 0 | 4 | 2 | 1 | 7 |

- **shell↔bash 交叉双向 0**：模型从不误把该用 shell 判成 bash 或反之。`TOOL_ALIASES` 把它们归 `shell` 一族是**挖矿/配额**用，不是给模型消歧——模型自己分得清。

**真正的主误差是 shell → reply（该跑命令时误判成该回复）**：

- 全量：gold=shell 的 1556 行里模型选 shell 仅 **30.4%**，**选 reply_without_tool 高达 750（48%）**；gold=bash 的 246 行里选 bash 仅 **8.9%**、选 reply 138（56%）。
- 今天（v8 时代）：gold=shell 146 行 → 选 shell **43.2%**、选 reply 23（16%）；gold=bash 12 行 → 选 bash **75%**。
- **over-pick（模型选 shell 族、实际不是）**：全量 296（reply 73/edit 72/read 59/compress 41/write 26）、今天 50（read 13/compress 10/reply 9/write 7）。

**解读**：(1) shell 族的短板不是 shell vs bash，而是 **shell vs reply**——模型把该执行命令误判成该直接回复，正是 GRPO 训练里 reply↔动作跷跷板的体现；(2) **v8 已改善**（shell top-1 43.2% vs 全量 30.4%、bash 75% vs 8.9%，方向与 dev 上 shell 76.8% 一致）；(3) 要修 shell 族，重点仍是 reply↔shell 边界，而非 shell/bash 区分。

### 13.4 v10 轮：生产真实死池数据挖矿 + 天花板第三次确认（11-18）

- **v10（挖生产累积的真实死池样本 glob/grep/compress/webfetch 边界对，init=v9）**：`decisions.jsonl`（13319 行）含 `state`（8700 行）+ `label`（5012）→ 死池带 state 可挖（glob 47/webfetch 30/grep 127/compress 216/websearch 7）；挖矿 prod_fresh 1609 / queue 221；构建 base v9 3146 → **train_v10 3536 行**（selected 136 / extras 390），**LEAK 0**。
- **训练 + 验收**：442 步跑满（in-train 峰 0.5748）；校准后 dev 842 = acc **56.41** / NLL 1.658 / ECE 0.0636 / AUROC 0.6779（v9 56.89/1.648/0.0425、v8 56.41/1.659/0.0578）；校准 global **4.5067**（温度继续升）。
- **逐工具（v10 vs v9）**：死池几乎全没动 —— glob 0（=0）、grep 4.5（=4.5）、webfetch 30.4（=30.4），仅 compress 14.8→18.5（+3.7，27 样本≈1 条，噪声级）；read +3.7、edit −4.5、bash −9.5、reply +0.6（守门）。
- **门判定 2✓2✗**（死池净提升 ✗、acc≥v9 ✗；reply 守门 ✓、NLL/ECE ✓）→ **v10 不部署，生产维持 v8**；v10 adapter + 校准入库待命。
- **天花板第三次确认（关键）**：v9（挖 train_v4 池）没动死池 + v10（挖生产真实死池数据）也没动 → 证明 **glob/grep/webfetch 不是"缺数据"，是结构性的**（模型没建立边界判别，或 dev 样本太小 ±10pp 全噪声）；**换数据源同样破不了 ~72-78% 硬顶**。
- **至此所有路线已试完**：边界挖矿（v5-v9 枯竭）+ 换真实数据（v10 死池没动）+ 三个推断期 trick（schema −4.87 / 两段式 −5.11 / 分组读出 no-op）= 现有机制+数据下 **v8 即局部最优**。再往前：① 持续生产积累慢慢喂（glob 今日 19 在涨）+ ② 换根本机制（schema 感知需重训、或更强 base）。

### 13.5 v11 轮：schema 感知重训（进行中）+ 之后的候选实验清单

- **v11（schema 感知重训）**：prompt options 从裸名改 `name -- desc`（真实工具语义），init = **AgentDecision-4B（Nov 1 原版，v3-v10 GRPO 前基线）**，不用 v8（用户明确）。格式 **方案 A**：27 个 core 工具 name--desc + 200 个 MCP 工具保持裸名；生产侧同用共享 `data/tool_desc.json` 只给这 27 个描述 → 训练/生产逐字节一致。数据 `data/train_v11/`（train 3536 / dev 842，LEAK 0），配方不变（correct 二元、P16、lr3e-6、beta 0.05、dev842 早停），GPU1、~8h、out `checkpoints/grpo_train_v11`。
- **推理侧前提**：生产 render 必须同步改 `option_text(label, desc)`（否则训 name--desc、推裸名 = OOD 反向伤害）；部署 = 训练+推理两侧原子改 + 重校准。
- **v11 之后的候选实验清单**（详细定义/风险/裁决口径见 `GRPO_RETROSPECTIVE_v5_to_v9.md` §13）：
  1. **E1 `soft` 奖励**（reward=target[k]，`grpo_trl.py` 已实现未启用）：对"选到次优"给部分分；仅当 target 非 one-hot 才有区分度，否则退化= correct。
  2. **E2 KL reference 锚点**（v11 特有微妙点，**优先排查**）：v11 的 reference = AgentDecision-4B（裸名时代），beta 0.05 会把 schema 模型往裸名行为拉，与新格式方向轻微冲突；若 v11 schema 没吃透，先查这里（杠杆：降 beta / 换 reference / 关 KL）。
  3. **E3 方向性惩罚**（用户提案：gold=reply 却选 tool → −1）：**方向需先校准**——dev/生产主误差是反方向的**欠动作**（gold=shell 却选 reply，生产 48%），原提案罚的是少数派；修正为反向罚（gold=tool→reply −1）或双向非对称。
  4. 顺序：v11 → E2 → E1 → E3，各单跑、独立归因，禁止混跑。
