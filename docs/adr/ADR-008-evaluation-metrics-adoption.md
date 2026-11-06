# ADR-008：采纳 research bench 指标规范为唯一口径 + 四项增补

- 状态：Accepted
- 日期：2026-11-07
- 相关：`research/docs/BENCHMARK_SPEC.md`（指标规范）、`research/docs/PREREGISTRATION.md`（余量与统计协议）、
  `research/docs/METRICS_A6.md`（非劣判定程序）、`litellm_jah/docs/ppt-outline.md`、`src/jevagentharness/{fit,report}.py`

## 背景

2026-11-07 起草了评测指标草案（暂用编号 ADR-007，三层 14 指标 + 固定报告版式 + n 分阶段）。对账发现：
research bench 已经**定义并实现**了完整指标体系（`BENCHMARK_SPEC.md` §5–§9，代码在
`research/src/jevharness/bench/{metrics,scorecard,report}.py`），草案的多数指标是重复发明，且草案里
"n=500、非劣余量 −1pp"与**预注册冲突**——`PREREGISTRATION.md` 已定余量 JevGym/ALFWorld/L1 取 3pp、
τ² 取 5pp，并给出功效要求（3pp 在 5% 分歧率下需 ≈340 任务，Holm 校正后 ≈650）。余量是预注册的一部分，
改动必须先重做功效计算，不能事后挑数字。

另外编号 007 已被插件目录占用（`litellm_jah/docs/decisions/ADR-007`，Withdrawn）。`docs/adr/` 是仓库级
ADR 目录，本决定从 008 起号；草案文件删除，其内容按本 ADR 重定位。

## 决策

### 1. 规范唯一化

Auto/hybrid 路由的评测指标以 `BENCHMARK_SPEC.md` 为唯一规范（连同 PREREGISTRATION 的检验协议与余量、
METRICS_A6 的判定程序）。不再并行维护第二套指标表。草案 14 指标到现有实现的映射：

| 草案 | 现有口径（代码） |
|---|---|
| T1 pass rate | success + 任务级 bootstrap CI、McNemar、paired bootstrap（`metrics.py`） |
| T4 oracle 上界差 | `iso_quality_savings` / vs `llm_only` 基线的 Δ（A6 verdict：NI / inconclusive / not NI） |
| T3 一致率、T5 判噪 | §5.6 decision-level（S1 accuracy、deferral、harmful-delegation、guard FAR/FRR）+ §5.7 校准（ECE/AURC）——**仅作校准与阈值拟合指标**（见增补 D1） |
| C1 ¥/任务、C4 决策开销 | `cost_usd` / `s1_cost_usd`、`cost_of_pass` |
| C2 预算达成分 | `effectiveness_under_budget`（scorecard `--budget`，默认 2× 基线） |
| C5 反事实节省归因 | `vs_baseline`：同任务配对的 save % / Δcost / Δlat / ΔU |
| C6 缓存修正成本 | `cost_usd_cache_modelled` / `cost_usd_no_cache` / `cached_frac` 列 |
| R1 误升/误降 | guard FAR/FRR + over/under（§5.6，两向分列、禁止合并 accuracy） |
| R2 校准误差 | §5.7 ECE/ACE/AURC + 可靠性分桶 |
| R4 级联成本 | delegation/deferral、`s1_calls`、`llm_free_steps` |
| C3、R3 尾部 | **规范缺口，由本 ADR 增补 D2/D3 落地** |

### 2. 增补（随本 ADR 实现）

- **D1（口径规则，进 reporting rules）**：对外"不降准确率"的宣称必须附非劣判定结果（预注册余量 + Holm
  校正的 pass 指标）；oracle/LLM 一致率、ECE 类校准指标不得单独作为质量宣称——一致率再高也要报 pass，
  因为 oracle 本身会错、其分布也与任务真实难度无关。节省数字必须带基线与口径（§9 已有，此处重申并
  同步修订 PPT 相应页面）。
- **D2（bench 实现）**：`episode_table` 增加每 episode 成本 **p90/max** 与关键路径延迟 **p99/max**；
  scorecard markdown 增加"Cost and latency tails"小节。均值掩盖肥尾：11-06 生产日志里 368 步、¥37 的
  单会话对日均值几乎无影响，但那就是真实的事故形态。
- **D3（bench 实现）**：`decision_block` 聚合 `DecisionLog.s1_latency_ms`（measurement 臂）为
  `s1_lat_ms {n, p50, p99, max}`，进 System-1 decisions 表。p50 0.16 s 曾掩盖隧道故障期 17 s+ 的尾部。
- **D4（生产侧实现）**：`jah report` 决策表延迟口径 p50 → **p50/p99/max**（`fit.describe` 增两字段）。

### 3. 样本量与宣称门槛（沿用预注册，不自造数字）

- 冒烟 ≈50 任务：只验链路，禁止引用任何数字；
- 主对比：预注册的 500 任务 E2 suite（seed 200000–200499），按 PREREGISTRATION 的余量与 Holm 流程出
  判定；n=500 以下不得发布"不降低准确率"类对外宣称；
- 回归监控 ≈200/周：只看趋势 + D2/D4 的尾部告警；
- 改余量 = 重新功效计算，走新 ADR。

## 明确不做

- 不建平行指标框架（草案 14 指标表废弃，只保留上面的映射与增补）；
- 不回补历史报告的 CI/基线/判定行——新实验起用新口径；
- 运行时预算硬顶（rollout 级 `--budget` 中断）暂不实现：现有 C2 是分析侧 EuB（按 episode 成本阈值统计），
  执行侧硬顶是独立候选 PR。

## 后果

- bench：scorecard 的 md/json 多一个小节与一列，metrics/scorecard/测试同步更新（已实现，测试全绿）；
- 生产：`jah report` 决策表口径变化（已实现）；
- PPT（S4/S7/S9/S14/S16/S18）按 D1 同步修订：一致率表述降级为校准指标，stop/tool_select 的最终负结论
  （11-05 定案：stop 81 条标签、总体一致率 61.7%、任何 tau 不过关 → off）替换 11-04 的中间状态；
- ADR-007 草案文件删除，不再存在两套口径。
