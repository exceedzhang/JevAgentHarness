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
