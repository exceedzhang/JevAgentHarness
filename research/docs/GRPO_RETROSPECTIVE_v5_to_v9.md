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

**规律**：raw 模型逐轮更过度自信（raw NLL 3.04→4.77）→ global 温度单调升（3.03→4.22）去压平；**校准后 NLL 稳定在 1.63–1.66**（健康区），ECE 从 0.075 降到 0.0425。这说明 GRPO 在「排序/argmax」上持续进步，但「概率校准」持续恶化，必须靠 post-hoc 温度缩放兜底。

## 6. 逐工具跷跷板（dev 842，校准后 acc%）
| 工具 | n | v3 | v6 | v7 | v8 | v9 | 趋势 |
|---|---:|---:|---:|---:|---:|---:|---|
| reply | 165 | 65.5 | 69.1 | 58.8 | 69.1 | 61.8 | **震荡**（v7 砸→v8 修→v9 又砸） |
| read | 134 | 47.0 | 41.8 | 50.0 | 48.5 | 50.7 | 稳步↑ |
| edit | 88 | 40.9 | 50.0 | 53.4 | 50.0 | 64.8 | 稳步↑（v9 大涨 +14.8） |
| shell | 168 | 79.8 | 79.2 | 78.6 | 76.8 | 76.2 | 缓↓（强项被挤） |
| bash | 42 | 64.3 | 61.9 | 73.8 | 69.0 | 71.4 | ↑ |
| write | 18 | 33.3 | 27.8 | 44.4 | 33.3 | 38.9 | 震荡 |
| webfetch | 23 | 30.4 | 26.1 | 34.8 | 30.4 | 30.4 | 震荡（池枯竭，v8/v9 无挖） |
| compress | 27 | 14.8 | 14.8 | 14.8 | 14.8 | 14.8 | **纹丝不动**（挖矿无效） |
| grep | 22 | 4.5 | 4.5 | 4.5 | 4.5 | 4.5 | **纹丝不动** |
| glob | 15 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **纹丝不动**（池 3 枯竭） |

**读法**：
- **reply↔read↔edit 是混淆三角**：v6/v7/v8/v9 每轮抬其中一个，必砸相邻的（v7 抬 read/write → 砸 reply；v8 抬 reply/read → 砸 write；v9 抬 read/edit/write → 砸 reply）。这是结构性互扰，不是数据量问题。
- **shell 是最大流量强项**（168 条，76–80%），逐轮缓降（被 reply/edit 挤占），是「净赢」的主要来源被抵消。
- **compress/grep/glob 是死池**：挖矿 5 轮 acc 纹丝不动（池枯竭 + 模型学不动），证明**纯挖矿对这三类无效**。

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
2. **换机制**（突破天花板的关键）：工具 **schema 感知 prompt**（给模型看工具签名而非仅名字，**首选**）；两段式决策（先粗分类再选工具）**方案 A 已 A/B 否决（见 §11，done 塌 + 没救 shell）**，若要再试需 ≥3 向门保留 done。
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
