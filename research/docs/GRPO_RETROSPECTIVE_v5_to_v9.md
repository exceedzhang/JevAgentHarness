# JAH 决策模型 GRPO 多轮训练复盘（v3→v9，2026-11）

> 独立复盘文档。逐轮细节见 `results/training/grpo_train_v3_2026-11-14.md`（§18–§23）与 `docs/DECISION_DATA_V2.md`（§11–§13.2）。本文只放结论、数字与教训，数字均来自真实训练/评测日志（dev 842，生产口径 `--n-perm 0 --batch-size 1`）。

## 0. 摘要（TL;DR）
- 目标：提升 4B 决策模型（Qwen3-4B + LoRA）在 **routing**（fast/strong）与 **tool_select**（选哪个工具）上的准确率，用 dev 842 作唯一标尺，逐工具 + NLL/ECE + 混淆矩阵验收（聚合 acc 不算数）。
- 结果：**生产从 v3（54.16%）升到 v8（56.41%）**，+2.25pp。v9（56.89%）整体最高但未部署（违反预注册门，见 §3）。
- 核心发现：**逐工具存在「跷跷板」**（reply↔read↔edit 互相挤占），没有任何一轮能同时抬升所有弱工具；**数据挖矿已近天花板**（glob/webfetch 池枯竭、read/reply/edit 池挖 ~80%+），再挖收益递减（每轮 ≤ +0.5pp）。
- 现实天花板 **~72–78%**（90% 是结构性天花板，靠数据做不到）；突破需**生产积累真实 glob/webfetch 流量 + 换机制**（schema 感知 prompt / 两段式决策），而非更多挖矿。
- 部署架构：**客户端 JAH**（本地 litellm-jah 镜像 + ssh-tunnel → GPU 机 vLLM）；服务器 litellm 是纯代理（不应用校准）。v8 上线后 routing 走 strong 从 28.2% → **10.7%**（T=3.877 压平 p_hard，省 ~2/3 strong 调用）。

## 1. 背景
- JAH 决策点：`routing`（决定走 fast 还是 strong/COMPLEX 模型）与 `tool_select`（决定下一步用哪个工具），由 4B 决策模型出概率 p，按 tau 阈值/校准决策。
- 模型：`Qwen3-4B` + LoRA adapter（`checkpoints/grpo_train_vN`），底座固定。
- 数据：生产日志 `decisions.jsonl`（tool_select/routing/stop，含 label + 工具调用结果）+ 人工金标（routing/stop，见 DCSV2 §7/§8）。
- dev 842：固定验收集（`data/train_v4/dev.jsonl`），train/dev 严格分离，防泄漏键 `(session, step, sha256(state)[:16])`。

## 2. 通用方法（每轮相同，只有「假设 + 挖矿靶心」变）
- **训练**：GRPO（`grpo_trl`，P=16、`--skip-zero-std`、`--num-generations 16`、lr 3e-6、beta 0.05、max-len 4608、1 epoch、dev 早停 patience 5 / min 60）。v7 起加 `--n-permutations 2`（选项顺序置换增广，行数×2）。
- **挖矿**：`mine_boundary_pairs.py`——同会话相邻 step（差≤2）且 gold 在目标族内交替的「边界成对」难例（状态最小差异、gold 相反），纯结构、零 GPU、可复现。Track A = `train_v4/train` 池；Track B = 生产新鲜行（`--exclude` 覆盖全部已用集防泄漏）。
- **构建**：`build_hard_weighted.py`（base = 上一轮 + 新难例 + 加权 extras），输出 `train_vN` + MANIFEST。
- **温度缩放**：`calibrate.py`（golden-section 最小化 NLL，global + per-option-count 桶）。**每轮 raw 都过度自信 → 必做**（raw NLL 从 3.04 升到 4.77，T 从 3.03 升到 4.22）。
- **验收门**：开训前写死（逐工具不降 + acc + 缩放后 NLL/ECE + 目标工具不回退）。

## 3. 各轮一览（dev 842，校准后口径）
| 轮 | 假设 / 挖矿靶心 | 数据行数 | init | acc | NLL | ECE | 裁决 |
|---|---|---:|---|---:|---:|---:|---|
| **v3** | 基线（clean + hard_v3） | 1717 | — | 54.16 | 1.63 | 0.075 | **部署**（生产起点） |
| v4 | 早期轮 | — | — | （未留，v3 更优） | — | — | 不部署 |
| v5 | 弱工具配额挖矿（非 shell） | 1951 | v3 | 54.75* | — | — | 不部署（reply/read 回退） |
| v6 | read↔reply / edit↔read 边界成对 | 2139 | v3 | 54.87 | 1.644 | 0.060 | 不部署（read −5.2） |
| v7 | read/write/webfetch 边界成对 + n-perm 2 | 2189 | v3 | 55.94 | 1.642 | **0.0486** | 作 v8 init，未单独部署 |
| **v8** | reply↔read 平衡 | 2415 | **v7** | **56.41** | 1.659 | 0.0578 | **部署**（当前生产） |
| v9 | read/edit/write 边界成对 + n-perm 2 | 3146 | **v8** | **56.89** | 1.648 | **0.0425** | 不部署（reply −7.3 破门） |
| v10 | 生产真实死池挖矿（glob/grep/compress/webfetch） | 3536 | v9 | 56.41 | 1.658 | 0.0636 | 不部署（死池没动，天花板第三次确认）|
| v11 | **schema 感知**（name--desc） | 3536 | **AgentDecision-4B(Nov1)** | **51.19** | 1.725 | 0.052 | 不部署（过度动作，天花板第四次确认）|

\* v5 的 54.75 为 final（in-train/raw）；v5 未做完整校准复测，故 NLL/ECE 未列。
**部署线**：v3（第一版）→ v8（当前）。v9 整体 acc/ECE 历轮最高，但违反预注册门 4（reply 回退 > −3pp），按纪律维持 v8。

## 4. 逐轮要点
- **v3**：基线，54.16% / NLL 1.63 / ECE 0.075。部署后线上验证 p_hard 分布，确立「聚合 acc 不算数、逐工具 + 混淆」纪律。
- **v5**：假设「按 dev 缺口给弱工具定量挖矿能打破跷跷板」→ 证伪（纯加量无效，reply/read 反而回退）。
- **v6**：假设「结构化边界成对对比教模型区分 read/reply/edit」→ 部分成立（reply +3.6、edit +9.1）但 read −5.2，不部署。
- **v7**：靶心 read/write/webfetch，三靶心全中（read +3.0/write +11.1/webfetch +4.3）但 reply −6.7；整体 55.94 / ECE 0.0486 历轮最强，作 v8 init。
- **v8**：靶心 reply↔read 平衡，两端同时抬（reply 69.1 超 v3、read 48.5 ≥ v3），整体 56.41。**部署**（代价：write −11.1/webfetch −4.3/execute −15.4 vs v7）。
- **v9**：靶心 read/edit/write，三靶心全升（edit **+14.8** 最大、read +2.2、write +5.6），整体 56.89 / ECE 0.0425 历轮最优，但 **reply −7.3（破门）** + final_result −21.4（n=14）。维持 v8。

## 5. 校准温度演变（golden-section 拟合 global）
| 轮 | global | choice:6-16 | choice:17-52 | raw NLL | 校准后 NLL |
|---|---:|---:|---:|---:|---:|
| v3 | 3.0252 | 3.0000 | — | 3.04 | 1.63 |
| v6 | 3.2108 | 3.4545 | 2.5680 | 3.52 | 1.644 |
| v7 | 3.3813 | 3.6644 | 2.6715 | 3.74 | 1.642 |
| v8 | 3.8770 | 4.2028 | 3.0691 | 4.38 | 1.659 |
| v9 | 4.2206 | 4.5747 | 3.3336 | 4.77 | 1.648 |
| v10 | 4.5067 | 4.8445 | 3.5937 | — | 1.658 |
| v11 | **3.079** | 3.285 | 2.553 | **3.416** | 1.725 |

**规律**：raw 模型逐轮更过度自信（raw NLL 3.04→4.77）→ global 温度单调升（3.03→4.22→4.51@v10）去压平；**校准后 NLL 稳定在 1.63–1.66**（健康区），ECE 从 0.075 降到 0.0425。这说明 GRPO 在「排序/argmax」上持续进步，但「概率校准」持续恶化，必须靠 post-hoc 温度缩放兜底。**v11（schema）首次反转**：global 4.51→**3.08**、raw NLL 4.77→**3.42**（name--desc 让 raw 概率更分散、更欠自信），但 acc 反而最低（51.19）——温度反转是格式表征的副产物，不是质量提升（见 §25 过度动作）。

## 6. 逐工具跷跷板（dev 842，校准后 acc%）
| 工具 | n | v3 | v6 | v7 | v8 | v9 | v11 | 趋势 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| reply | 165 | 65.5 | 69.1 | 58.8 | 69.1 | 61.8 | **31.5** | **v11 崩塌**（schema 过度动作）|
| read | 134 | 47.0 | 41.8 | 50.0 | 48.5 | 50.7 | **67.9** | v11 大涨 +19.4 |
| edit | 88 | 40.9 | 50.0 | 53.4 | 50.0 | 64.8 | 56.8 | v11 +6.8 |
| shell | 168 | 79.8 | 79.2 | 78.6 | 76.8 | 76.2 | 71.4 | 缓↓ |
| bash | 42 | 64.3 | 61.9 | 73.8 | 69.0 | 71.4 | 40.5 | v11 跌 −28.5 |
| write | 18 | 33.3 | 27.8 | 44.4 | 33.3 | 38.9 | 22.2 | 震荡 |
| webfetch | 23 | 30.4 | 26.1 | 34.8 | 30.4 | 30.4 | 47.8 | v11 +17.4 |
| compress | 27 | 14.8 | 14.8 | 14.8 | 14.8 | 14.8 | 25.9 | v11 +11.1 |
| grep | 22 | 4.5 | 4.5 | 4.5 | 4.5 | 4.5 | 4.5 | **纹丝不动** |
| glob | 15 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **纹丝不动** |

**读法**：
- **reply↔read↔edit 是混淆三角**：v6/v7/v8/v9 每轮抬其中一个，必砸相邻的（v7 抬 read/write → 砸 reply；v8 抬 reply/read → 砸 write；v9 抬 read/edit/write → 砸 reply）。这是结构性互扰，不是数据量问题。
- **shell 是最大流量强项**（168 条，76–80%），逐轮缓降（被 reply/edit 挤占），是「净赢」的主要来源被抵消。
- **compress/grep/glob 是死池**：挖矿 5 轮 acc 纹丝不动（池枯竭 + 模型学不动），证明**纯挖矿对这三类无效**。
- **v11（schema）打破平衡**：加描述让 act 类工具"更抢手" → read(+19.4)/webfetch(+17.4)/compress(+11.1)/edit(+6.8) 全升，但 **reply 崩塌到 31.5（过度动作）+ bash 跌 28.5**；净 −44。首次"换根本机制"也净负，且暴露 schema 的系统性**过度动作**偏置（详见 §25）。

## 7. 数据天花板（为什么 90% 达不到）
dev 842，v8 = 56.41%（475 对 / 367 错）：
- **90% 需 758 对（+283 翻转）**。5 个「硬工具」（glob 0%·池3 / webfetch 30%·池2 / grep 4.5% / compress 14.8% / not_done 7.1%）= 88 错，全修到 95% 也只是天花板 89.5%；含小工具约 30 错 → **现实上限 ~86%**。
- **86% 需 +249 翻转 = 68% 全部错题**，要求边界工具（read/reply/edit）达 95–100% 准确率——不现实（最强 shell 才 76.8%）。
- **现实可达 ~72–78%**（边界工具到 75–85%、弱信号不动、shell 族按纪律不挖），需**再挖 ~800–1500 行**（read/reply/edit 池已被 v7+v8+v9 挖 ~80%+，剩几百行）。
- **结论**：90%/86% 是**结构性天花板，非数据量问题**。突破需（a）生产积累真实 glob/webfetch 流量（按月计）+（b）换机制（工具 schema 感知 prompt / 两段式「先分类再选工具」决策）。

## 8. v8 部署（客户端 JAH 架构）
- **架构**：生产 JAH 决策跑在**本地 litellm-jah 镜像**（客户端）+ ssh-tunnel → GPU 机 vLLM；**服务器 litellm 是纯代理**（基础镜像 `litellm:1.106.6`，只路由，不应用校准）。校准 = 客户端 `backend.py` 对 raw p 做 `p' ∝ p^(1/T)`。
- **部署动作**：服务端 vLLM adapter 换 v8（served `AgentDecision-4B-1117`，md5 与 `grpo_train_v8` 一致，用户换）；本地 `litellm_jah/docker/jah-config.yaml` 改 v8 校准（global 3.877/6-16 4.203/17-52 3.069，commit `fe495e5`）；`docker build` 重建 `litellm-jah:latest`（烤 v8 配置，md5 逐字节一致）+ `docker compose up -d --force-recreate` 生效。
- **上线验证**（`jah fit`，客户端 `decisions.jsonl`）：
  - **routing**：fires at logged tau（0.85）从全量 pooled **28.2%** 降到 v8 时代 **10.7%**（`--since 2026-11-17 01:18`）→ T=3.877 压平 p_hard，~89% 步走 fast、省 ~2/3 strong 调用（方向对/省钱；**风险**：若 T 压太狠可能路由不足，需盯 strong vs fast 步的下游成功率）。
  - **tool_select/stop**：pooled + v8 时代**均无阈值达 95% 一致**（tool_select pooled shell 90.6%/v8时代 70.3%；webfetch 16.3% 最差）→ **维持 shadow（只收集标签，不自动决策）**，与 dev 逐工具交叉一致。
- **回滚**：v3（global 3.025/6-16 3.000）、v7（3.381/3.664）adapter+校准均保留，随时切回。

## 9. 关键教训
1. **跷跷板是结构性的**：reply/read/edit/write 在混淆图上相邻，GRPO 每轮抬一个必砸相邻。没有单轮能「全赢」；部署决策要在「整体 +2pp」与「某高流量单点回退」间权衡（v9 整体最高但 reply −7.3 → 维持 v8）。
2. **挖矿有硬天花板**：glob/webfetch 池枯竭、read/reply/edit 挖 ~80%+，第 4–5 轮（v8/v9）每轮 ≤ +0.5pp。**继续纯挖矿 ROI 已极低**。
3. **n-permutations n=2 收益有限**：v7/v8/v9 加 n=2 后整体 +~1pp，但 n=4 边际更小且不针对语义边界（选项顺序在生产是固定的）；「一轮一假设」纪律下不应混入位置不变性。
4. **校准温度逐轮升**：raw 模型持续过度自信，post-hoc 温度缩放是必须的兜底；校准后 NLL 稳定 1.63–1.66、ECE 持续改善（0.075→0.0425），说明 GRPO 改进的是「排序」而非「概率标定」。
5. **防泄漏纪律有效**：v5–v9 每轮独立 LEAK 复查均为 0（exclude 全集 + 去重键 + 构建内部 dev-key 排除），dev 842 标尺全程可信。

## 10. 下一步（非挖矿优先）
1. **停止纯边界挖矿**（ROI 极低），转**生产积累**：让 glob/webfetch/grep/compress 在真实流量里攒 labeled 数据（按月），这些死池只有真实流量能解。
2. **换机制**：三个零成本推断期读法已全部 A/B 否决/无效（schema 感知 −4.87pp / 两段式方案 A −5.11pp / 分组读出方案 B no-op，**详见 §12**）——**"不动数据的推断期"这条路已试尽**；突破天花板得靠真重训（reply↔shell 边界，边际递减）或生产数据。
3. **tool_tau 逐工具门槛**（`b948a39` 已加）：tool_select 上把 shell/read/bash（一致率 87–90%）单独从 shadow 提为 act，webfetch/write/edit 继续 shadow——低风险先落地。
4. **routing 充分性监控**：每天 `--since 2026-11-17T01:18Z` 重跑 `jah fit`，盯 routing 10.7% 是否路由不足（若难例被 fast 漏 → 单独调 routing tau 或 T）。

## 11. 两段式决策 A/B（方案 A，2026-11-18，否决）

§7/§10 提出的「换机制」选项之一——两段式（先粗分 回复/动作，再选工具）——做了一次零风险 A/B（独立脚本 `/tmp/twostage_ab.py`，v8 adapter，dev 842，未动生产）。**结论：否决。**

**设计（方案 A）**：Stage 1 二元门 `{reply_without_tool, __ACT__}`（同 v8 模型 + 同 prompt）→ 若判"动作"，Stage 2 在工具集（不含 reply）里选具体工具；组合 `reply 若 s1=reply 否则 s2`。对照 = 扁平 v8（同 842 state、同模型，argmax）。

**结果（dev 842，argmax）**：

| 项 | 扁平 v8 | 两段式 | Δ |
|---|---:|---:|---:|
| 整体 acc | 56.41 | **51.31** | **−5.11pp** |
| done | 92.3 | **0.0** | **−92.3** |
| reply | 69.7 | 55.2 | −14.5 |
| **shell**（原靶心） | 76.8 | 76.8 | **0** |
| webfetch | 30.4 | 39.1 | +8.7 |
| write | 33.3 | 38.9 | +5.6 |
| compress | 14.8 | 18.5 | +3.7 |
| bash | 69.0 | 71.4 | +2.4 |

Stage-1 门：73% 判"动作"，reply 召回 55.2%、动作召回 79.9%（门不精准）。

**否决原因**：
1. **二元门无法表达 done/not_done/final_result**——扁平里它们是独立选项，拆成 {reply, 动作} 后全塌进 reply，done 从 92.3% 归零（最大净损失）。
2. **门本身 recall 仅 55%/80%**，收益上限被锁死。
3. **没救到目标 shell**（76.8 不变）。webfetch/write/compress/bash 的小增益抵不过 done + reply 的损失。

**结论**：方案 A 否决、不采纳。若将来再试，Stage-1 至少需 3 向 `{reply, done, 动作}` 才能保留 done，但二元已净负、收益上限受门 recall 限制。→ 换机制重心转 **schema 感知 prompt**（§10.2 首选）+ 生产积累（§10.1）。

**产物**：`/tmp/twostage_ab.py`、`/tmp/twostage_ab.json`（overall/stage1/by_tool/confusion 全量）。
5. **v9 待命**：v9 adapter（56.89%/ECE 0.0425，历轮最高）+ 校准入库，若线上 reply 切片表现可接受，可切 v9 上线（当前维持 v8）。

## 12. 三个推断期 trick 全部试完（2026-11-18）：A / schema / B 均不动 reply↔shell

§7/§10 的「换机制」里，有 3 个**不动数据、不改模型权重**的推断期读法，本周期全部做了零风险 A/B（独立 /tmp 脚本 + v8 adapter + dev 842，未动生产）：

| 方案 | 形式 | 结果（整体 acc） | 逐工具关键点 | 判定 |
|---|---|---:|---|---|
| A 两段式二元门 | `{reply, __ACT__}` 门 + stage-2 | 51.31（−5.11） | done 92.3→0、reply −14.5、**shell 76.8 不变** | ❌ 否决（§11） |
| schema 感知 | 选项改 `name -- 描述` | 51.54（−4.87） | shell 76.8→65.5、edit→29.5、reply 过选更重 | ❌ 否决 |
| B 分组读出 | reply 仅当 `p(reply)>Σ其余` 才赢，否则 argmax 其余 | **56.41（0，no-op）** | 全部逐工具 Δ=0、shell→reply 24 不变 | ⚪ 无作用 |

**为什么三者都动不了 reply↔shell（统一解释）**：
- 主误差 **shell→reply（24 条）** 是模型概率分布里"焊死"的——p 把本应给 shell 的质量错给了 reply。
- 三个 trick 都是**只重读/重排同一个 p，不改 p 的底层**：
  - 方案 A、schema 还引入了**分布外格式**（`__ACT__` 没训过 / `name -- desc` 没训过）；v8 在裸名上训，OOD 直接伤害（shell 反而更差）。
  - 方案 B 把整分布切成 {reply} vs {其余}，而「其余」= 1 − p(reply)（互补），门退化成「p(reply)>0.5」；v8 的 reply 只要是最优解 p 都 >0.5，**一条决策都没变**（no-op）。
- **结论**：reply↔shell 这个结构核心误差，**推断期重读/重排/加描述都修不了**，除非改 p 本身（= 重训 reply↔shell 边界，边际递减）或换真正的新机制/数据。

**含义**：换机制里"零成本推断期"这条路**已试尽**（A/schema/B 全负或无效）。真正能动的回到 §10.1（生产积累 + 时间切分 dev）与 §10.2 的真重训（reply↔shell 边界数据，接受边际递减/跷跷板），或接受 ~72-78% 天花板。

**产物**：`/tmp/schema_ab.py`、`/tmp/schema_ab.json`、`/tmp/planB_ab.py`、`/tmp/planB_ab.json`（均 dev 842、v8 adapter、argmax 口径）。

## 13. v11 之后的候选实验清单（奖励 / 参考分布 / 方向性惩罚，待 v11 结论后逐个单跑）

> 纪律：**一轮一假设**。下面每一项 = 同数据（train_v11，name--desc）、同配方（P16/lr3e-6/dev842 早停），只改一个变量（reward 或 KL reference 或方向性惩罚项），单独成轮、单独归因。v11 只验证「schema 表征」一个假设，reward 维持 `correct` 不动。v11 结论出来后再排这些。

### 13.0 当前基线（v3-v11 一致）
- reward = `correct`（二元，`grpo_trl.py` `make_reward_funcs` → `option_reward`）：采样选项 k，`r = 1.0 if k == gold else 0.0`，gold = target 向量 argmax（`gold = int(max(range(n), key=lambda k: tgt[k]))`）。选对 1 / 选错 0，无部分分、无格式分。
- KL 惩罚在 **loss 项**（`beta=0.05`），不在 reward。reference = `--init` 适配器（frozen copy，作 TRL ref adapter）。
- P16 每 prompt 采 16 条，组内归一化 advantage（`--scale-rewards group`）；`--skip-zero-std` 丢 16 条全同奖励（零方差）的组 → 梯度集中在不确定边界例。

### 13.1 E1：`soft` 奖励（代码已实现，未启用）
- **定义**（`grpo_trl.py` `soft`，约 L135-138）：`r = float(target[k])`（选到的选项 k 在 gold target 里的概率），k 非法则 0。即把 gold 的**软分布**当奖励，不再是 0/1。
- **假设**：对「总是选错但选到次优」的组（如 shell 例总选 bash）给非零梯度，把边界从「全错」往「次优→正」推；对 one-hot 目标无差别。
- **适用条件**：仅当 target 是**非 one-hot**（teacher-soft / MC 目标）才有区分度。若 target 基本 one-hot（当前 train 多为 one-hot），E1 退化成 correct，**收益≈0**——故 E1 需先确认/改造成软 target，否则空转。
- **风险**：部分分会把锐利边界「糊」化；我们当前要的是锐利 reply↔shell 边界，soft 可能反向削弱 → 需 A/B。
- **裁决口径**：同 v11 数据，只把 `--rewards correct` 换成 `--rewards soft`（target 改软），逐工具 + 混淆矩阵 + NLL/ECE 对比 v11。

### 13.2 E2：KL reference 锚点（v11 特有的微妙点，优先排查项）
- **微妙点**：v11 的 `--init` = reference = **AgentDecision-4B（Nov 1 原版，裸名时代训练）**。beta 0.05 的 KL 会把 schema-aware 新模型往「裸名行为」方向拉，与「学 `name -- desc` 新格式」**方向上轻微冲突**——KL 锚点本身是旧格式的代表。
- **优先级**：如果 v11 训完发现 **schema 没吃透**（模型仍偏裸名风格 / `name -- desc` 的增益不明显），**第一个该查的就是这个 KL 锚点**（而不是 reward 本身）。
- **三个可动杠杆**（各成单独实验）：
  1. **降 beta**（0.05 → 0.01/0.0）：削弱向旧格式的回拉，代价是偏离 reference 更自由。
  2. **换 reference**：reference 改成 base Qwen3-4B（无格式偏见，但无决策先验）或一个 schema 格式 SFT（与训练格式一致，最理想但需先产 SFT）。
  3. **关 KL**（beta=0）：完全无锚，看 schema 是否更快吃透 + 是否发散。
- **风险**：beta 0.05 本就温和，影响大概率有限；但这是唯一「与 v11 新格式方向不一致」的项，值得作为第一排查。
- **裁决口径**：v11 基础上只改 beta / reference，其余不动。
- **结果（E2a，β 0.05→0.01，已跑）→ REJECTED**：总 acc 49.53% < v11 51.19%（更差）；**唯一大幅变化 = reply 31.5→19.4%（过度动作加剧）**，其余工具≈持平。结论：弱锚**没有**改善 schema 适应，反而**放大了过度动作**——因为 v11 的 KL 锚（裸名基线）带"欠动作"倾向、恰是"过度动作"的部分刹车，弱化它 → 失去抑制。**KL 锚非瓶颈、非根因**；过度动作根源在 schema + correct 不罚跨边界 → 转 E3（grpo §26）。

### 13.3 E3：非对称 / 方向性惩罚（用户 @431@ 提案，方向需先校准）
- **原始提案**：`gold = reply_without_tool` 但选了某个 tool → 罚 −1（即「该回话却去动工具」加重罚）。
- **方向性核查（关键）**：提案罚的是**过度动作**（gold=reply 选 tool）。但 dev / 生产主误差方向是**欠动作**：
  - dev 主混淆 = **gold=tool（尤其 shell）→ 选 reply**（shell→reply 24+ 条，shell 召回被压低）；过度动作（gold=reply→选 tool）数量明显少。
  - 生产 `shell→reply` 48%（本应 shell 却 reply）= **欠动作**。
  - 所以 E3 若按原方向只罚「gold=reply 选 tool」，**罚的是少数派**，与主误差**方向相反**——**但这是裸名模型（v8/v9/v10）的情况**。
  - **v11 实证（关键反转）**：主误差方向**随格式翻转**——**v11（schema）的主误差是过度动作**（gold=reply→tool：reply→read 50 + reply→edit 34，reply 崩到 31.5）。即"该罚哪个方向"必须与**目标模型**匹配：裸名 → 罚欠动作；schema → 罚过度动作（= 用户 @431@ 原提案方向）。
- **修正方案**（择一，单跑）：
  1. **反向**：gold=tool 但选了 reply → 罚 −1（直击欠动作主误差，shell 召回优先）。
  2. **双向非对称**：欠动作（gold=tool→reply）罚 −1，过度动作（gold=reply→tool）罚 −0.5（或 0），按实际错误频率加权。
  3. 若坚持原方向，须先跑数据确认「gold=reply→tool」在生产/dev 的真实频率，证明该方向值得单独罚。
- **风险**：二元 correct 已能表达「选错=0」，加 −1 是把它变**三元**（对 / 可接受错 / 重罚错），改了 advantage 量纲 → 与 `--scale-rewards group` + `--skip-zero-std` 的交互需重新观察（frac_zero_std 会变）。
- **裁决口径**：同 v11 数据，reward 从 correct 换成「correct + 方向性惩罚」，逐工具重点看 shell 召回 + reply 误报双向。

### 13.4 执行顺序建议
1. 先跑完 v11（schema 假设）→ 得基线数字。
2. 若 v11 schema 增益不明显 → **先 E2**（KL reference，最可能是 schema 没吃透的元凶）。
3. 再 E1（soft，需先确认 target 是否软）。
4. 最后 E3（方向性惩罚，需先定方向）。
- 每项独立 commit + 独立结果章节 + 按 §22-24 的预注册门（逐工具 + 混淆矩阵 + NLL/ECE）裁决，禁止混跑。

**状态（11-20，E2a + E3 已收口）**：v11 已完成——dev 51.19（< v8 56.41，−5.2pp），schema 引入**过度动作**偏置（reply 69.1→31.5、bash −28.5，但 read +19.4/webfetch +17.4），**不部署、生产维持 v8**，数据天花板第四次确认。**清单执行进度**：① **E2a（β 弱化 0.05→0.01）已做 → REJECTED**（§14，非瓶颈、弱锚放大过度动作）→ ② **E3（方向性惩罚 `reply_bound`）已做 → REJECTED**（§14，reply 塌到 0）→ ③ E1（soft，target 若 one-hot 则空转，鉴于 E2a/E3 皆败，列为长 shot）。**结论：奖励 / 参考分布线基本走死**（§14 + DCSV2 §13.6）。

## 14. 候选实验 E2a / E3 实测结果（2026-11-19/20）—— 奖励 / 参考分布线走死

### 14.1 E2a：弱 KL 锚（beta 0.05→0.01）—— REJECTED
- **假设**：v11 的 KL reference = AgentDecision-4B（裸名时代）在"拽" schema 模型回裸名行为；弱锚（β 0.05→0.01）应让 name--desc 吃透。
- **结果（calibrated，唯一变量 beta）**：

  | 指标 | v11 (β0.05) | E2a (β0.01) | Δ |
  |---|---:|---:|---:|
  | acc | 51.19 | 50.11 | **−1.08** |
  | KL | 1.31 | 4.18 | +2.87（弱锚更远离 ref，符合预期）|
  | reply n=165 | 31.5% | 24.2% | **−7.3（过度动作更重）** |
  | shell n=168 | 71.4% | 71.4% | 0 |

- **verdict**：弱 KL **非瓶颈**——降低 β 让模型更远离 ref（KL↑），但没换来 acc，反而 reply 过度动作更严重（31.5→24.2，混淆 reply:40/read:78/edit:38/shell:7）。v11 的过度动作**不是 KL 锚拽的**，是 schema 格式本身使 act 类"更抢手"。→ **E2a REJECTED**。

### 14.2 E3：方向性惩罚 reply_bound（gold=reply 选 tool → −1）—— REJECTED（灾难性回退）
- **假设**：v11 主误差 = 过度动作（gold=reply→tool），罚它应修 reply 边界（= 用户 @431@ 原提案方向，对症 schema 的过度动作）。
- **结果（calibrated，唯一变量 reward：correct→reply_bound）**：

  | 指标 | v11 | E3 | Δ |
  |---|---:|---:|---:|
  | acc | 51.19 | **46.56** | **−4.63** |
  | reply n=165 | 31.5% | **0.00%** | **−31.5（塌到 0！）** |
  | 混淆 gold=reply→reply | 52 | **0** | 一次都没选 reply |
  | read n=134 | 67.9% | 73.9% | +6.0 |
  | webfetch n=23 | 47.8% | 56.5% | +8.7 |

- **失败机制（关键）**：`−1` 负奖励 + GRPO 组归一化 + `--skip-zero-std` 三者交互——reply-gold 点模型本以 tool 为主，16 采样多为 tool → 全拿 −1 → **零方差 → 被 skip（无梯度）**；真正产生梯度的只有模型本已大致对的 reply-gold 组。净效果 = 把 reply 选项概率**整体压到 ≈0**（模型彻底放弃 reply），而非学会"何时 reply"。副作用 = read/webfetch/bash 等 tool-gold 反而 +5~8pp（典型**"扫到地毯下"**：放弃 reply 换 tool 精度，净大亏）。
- **教训**：在 GRPO 的 group-normalized + skip-zero-std 框架里，**加负奖励不是"给方向信号"，而是"系统性压制该选项概率"**——当某选项（reply）本就低概率时，负奖励让它雪崩。方向惩罚在此实现下**不可用**。

### 14.3 结论：奖励 / 参考分布线走死 + 天花板第五次确认
- **5 个 post-v8 实验角度全败**：v9（teacher-soft，reply 破门）/ v10（挖矿，死池没动）/ v11（schema，过度动作）/ **E2a（弱 KL，更差）** / **E3（方向惩罚，reply 塌 0）**。无一超 v11 51.19，更无一超生产 v8 56.4。
- **根因**：reply↔tool 边界是**所有版本共同死穴**（v8 欠动作 / v11 过度 / E3 塌 0），而现有任务表示（**20+ 工具 + reply 在同一一维选项空间里竞争 + 简单标量奖励**）**表达不出这个边界**。继续在"单步选一个工具 + 标量奖励"上做数据/格式/KL/奖励微调，天花板已 5 次坐实。
- **下一步（待用户定方向）**：**A = 结构性突破**（扩 eval 基建 842→2-3k，逐工具才可信 + **两阶段分解** 先判"行动 or reply"二元 → 再判"哪个工具"条件，唯一从根上拆 reply↔tool 死穴）；**B = 收手 v8，转系统层**（路由 strong ~10.7% + 校准 + 集成，已验证有效、天花板更高）。E1（soft）鉴于 E2a/E3 皆败，列为长 shot。
