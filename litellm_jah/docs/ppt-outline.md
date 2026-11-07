# litellm-jah 插件功能介绍 — PPT 大纲

> 建议篇幅：19 页左右（含 S5b 对比页）。所有数据来自真实运行日志（`~/.jah/litellm-logs`，截至 2026-11-04），
> 由 `jah report` / `jah fit` 统计，未做任何虚构或修饰。

---

## S1 封面

- 标题：**litellm-jah：把 JevAgentHarness 装进你的 LiteLLM 代理**
- 副标题：任意 harness 可用 · 降 token 成本 · 不降任务准确率 · 更快执行
- 一句话定位：一个 LiteLLM proxy 插件，用小决策模型按步路由 strong/fast 两层模型，并给每个
  关键决策点装上"安全网"

## S2 目录

1. 背景与挑战
2. 实现目标
3. 主要原理
4. 如何实现
5. 如何使用
6. 当前实验数据
7. 结论与下一步

## S3 背景与挑战

- Agent 的每一步都要调 LLM：简单步骤（查文件、拼命令）和复杂步骤（长上下文编排、收尾决策）
  混在一起，全部走最强模型 → **贵、慢**
- 全部走最快模型 → **省，但关键步骤质量掉、任务失败率上升**
- 换模型、调路由策略通常要改 harness 代码或换一套网关，**不同 harness（OpenCode、
  自研 agent、任意 OpenAI 兼容客户端）无法共享同一套路由策略**

## S4 实现目标（四个，对应四个设计支柱）

| 目标 | 由什么保证 |
|---|---|
| ① 支持各种 harness | 标准 OpenAI 兼容 API + LiteLLM 代理：任何 harness 只需改 `base_url` 和 key，无需改代码 |
| ② 降低 token 成本 | 简单步骤路由到 fast 层（价格低一个量级），按自己的计价表精确审计每步成本 |
| ③ 不降任务准确率 | hard 步强制 strong + on_error=strong（决策层故障兜底）；finish check / guard 复核为可选机制，按 fit 数据决定开关（见 S16） |
| ④ 提高执行速度 | fast 层处理占多数的 routine 步骤；决策开销仅 ~0.2 s/步（单步 LLM 延迟 ~12 s 量级） |

> 讲稿提示：④的"提速"来自 fast 层模型本身更快；当前日志记录每步 LLM 延迟
> （115.3 s/会话），与"全 strong 基线"的直接 A/B 对比是下一步工作，PPT 中如实表述。

## S5 总体架构（一张图）

```
任意 harness / 任意 OpenAI 兼容客户端
        │  model: "jah-auto"（stream 或非        ▼
   ┌─────────────── LiteLLM Proxy（Docker） ───────────────┐
   │  ① classifier（complexity router 分类器）              │
   │      问决策模型："这一步 hard 吗？" p(hard) ≥ tau → COMPLEX│
   │  ② JahPreCall  hook：tool_select / stop trigger /     │
   │     reflection，可改写请求（注记、tool_choice）          │
   │  ③ JahPostCall hook：finish check / stop / tool /     │
   │     routing / action guard，需要时 strong 重跑          │
   │  ④ JahStepLogger：steps.jsonl / decisions.jsonl        │
   │  引擎 engine：policy + backend + sessions(Redis) +     │
   │  journal + points + config                              │
   └────────────────────────────────────────────────────────┘
        ▼                    ▼
   jah-strong 层        jah-fast 层        （决策模型：独立小模型，输入免费）
```

- 要点：所有 JAH 逻辑在一个共享 engine 里；三个 hook + 一个 classifier 挂在 LiteLLM 生命周期上
- 决策模型失败**永不阻断请求**：降级到 strong 层（on_error=strong）

## S5b 与 LiteLLM 内置 Auto Routing（complexity_router, v1.94+ beta）的关系与差异

- **不是竞品，是分工**：litellm_jah 的 Wiring 1 直接把官方 auto_router 当"服务层"复用
  （两层 tier 池、模型挑选、流式都行），只把它的**分类器换成 JAH 决策引擎**，再加四个决策点 hook
- 五个关键差异：

| 维度 | 内置 auto_router | litellm-jah |
|---|---|---|
| 分类对象 | 只抽"最后一条真人提问 + system prompt"，工具结果/reminder 刻意剔除；无真人文本的步直接落 default_model | 渲染**整个 agent 步骤状态**（含工具输出与轨迹），按决策模型训练格式提问——harness 步骤的难度恰恰取决于累积状态 |
| 阈值来源 | `tier_boundaries` / tau 是**配置出来的** | **shadow 采集 → `jah fit` 用生产标签拟合**（Wilson 下界 ≥95% 才 promote），负结论也会拦下坏配置（tool_select） |
| 失败语义 | 分类器失败默认回退**启发式打分**（可能把复杂步判到 fast 档）；需显式改 `classifier_fallback: default_model` | `on_error: strong`，失败**永不降档**，只会更保守 |
| 会话模型 | `session_affinity` 把整会话钉在首轮模型（难题开场→全程付贵档）；除此之外按请求无状态 | **每步重新决策** + 状态机（first_steps / stick / fast 连续上限 / 重试复用 plan / Redis 共享）；明确警告不要开 session_affinity |
| 范围 | 只管选模型（另有四档、池化、Thompson、关键词、vision、看板） | 路由之外还有 **tool_select / stop(finish check) / reflection / action**：fast 产出、strong 复核、必要时重跑——路由错了收尾也兜得住 |

- 成本口径：auto_router 的"节省额"是对反事实基线的**推导估算**；JAH 的 `jah report` 是每步**实测**
  （token/延迟/CNY 计价/缓存命中逐条落账）
- 一句话：auto_router 回答"**这条消息难不难**"；JAH 回答"**这步该谁写、写得对不对、结束得对不对**"

## S6 原理一：按步复杂度路由（cascade）

- 每一步都问一次决策模型：`p(hard)`
  - 前 N 步（`first_steps_strong`）强制 strong——开局规划最贵
  - `p(hard) ≥ tau`（当前 0.75）→ strong 层
  - 否则 → fast 层
  - 升级后粘滞（stick）、fast 连续上限（max_fast_streak）防止"刚升级又跳回"
- 四种 routing 模式：`cascade`（默认）/ `shadow`（只记录，全 strong）/
  `guard`（fast 写，事后决策模型检查可重跑）/ `off`
- 为什么有效：routine 步骤占大多数（本部署 24h 数据 636/869 ≈ 73%），它们不需要最强模型

## S7 原理二：决策点安全网（保准确率的核心）

- 五个决策点，每个都有模式（off / shadow / guard / …），按需开启：
  - **routing**：这步该用哪层？
  - **tool_select**：下一步该调哪个工具？
  - **stop（finish check）**：这里结束回合对不对？—— fast 层的无工具终答会被 strong 重跑，
    strong 的回复才是真正结束回合的回复
  - **reflection**：要不要暂停自我复查？
  - **action**：拟执行的动作是否危险？
- 统一语义：**fast 层产出 + strong 层复核**，只有复核不过才重跑（一次）；
  重跑失败则保留原回复——hook 永远不破坏已完成的请求（fail-safe）
- 本部署当前开启：routing=cascade（tau 0.80）+ on_error=strong；stop 与 tool_select 都经两轮 fit 验证后关闭（见 S16）

## S8 原理三：shadow → fit → promote 自校准闭环

- **shadow**：决策点"只提问、只记录"，把决策模型的答案和 agent 真实行为（标签）并排写入
  `decisions.jsonl`，不改变任何请求 → 采集零成本（本部署决策模型输入免费）
- **fit**：`jah fit` 用你自己的日志拟合阈值 tau——取"采信后与 LLM 一致率 ≥ 95%"
  （单侧 Wilson 下界，样本 ≥ 30）的最小阈值；够不到 95% 就**不采信该决策点**
- **promote**：拟合出的 tau 才允许把该点提到 guard/assist 等行动模式
- 价值：**阈值不是拍脑袋，是用生产流量证明过的**；负结论同样有价值（S16 的 tool_select）

## S9 原理四：fail-safe 与成本审计

- 决策模型超时/报错 → 步骤走 strong（on_error=strong），agent 不停摆
- 重跑（finish check / guard redo）失败 → 保留 fast 原回复（代理场景部署已成功，不能把成功的请求打挂）
- 会话存储 Redis 不可达 → 进程内本地会话继续服务，请求不失败
- 计价审计：`prices` 块按计费币种（本部署 CNY/百万 token）配置，注册进 LiteLLM 计价表，
  缓存命中/缓存写入分别计价；每步、每次决策都落账到 journal；
  决策模型单独计价（本部署 `price_per_m: 0`，输入免费，决策成本不会与步骤成本混币种）

## S10 如何实现：代码构成

- 包 `litellm_jah/`，Python ≥ 3.10，约 3000 行，**不 import litellm 本体**（鸭子类型，
  引擎可完全离线单测）；218 个测试全部离线通过（脚本化假后端，无 key 无网络）

| 模块 | 职责 |
|---|---|
| `engine.py` | 共享引擎：每步决策流程、会话状态机、重跑编排 |
| `policy.py` / `state.py` | cascade 策略（first_steps / tau / stick / fast 连续上限）与会话状态 |
| `backend.py` | 决策后端：`jev`（typesafe / openrouter / local 部署）或 `rules`（无模型回退） |
| `classifier.py` | complexity router 分类器（返回 COMPLEX/SIMPLE，失败回退 default_model） |
| `precall.py` | 请求前 hook：发起 tool_select / stop trigger / reflection，可改写请求 |
| `postcall.py` | 请求后 hook：finish check、stop/tool/routing/action guard、settle 落账 |
| `logger.py` | step 日志（steps.jsonl，LiteLLM cost map 计价） |
| `journal.py` | decisions.jsonl / steps.jsonl 追加写（`jah report` / `jah fit` 直接可读） |
| `sessions.py` | 会话存储：进程内 LRU 或 Redis（多副本共享，idle TTL） |
| `points.py` / `config.py` | 五决策点模式解析、preset（route/paper/safe/shadow）、配置校验 |
| `hook.py` | finish check 的独立挂法（P2 兼容别名） |

- 会话识别：metadata 的 `jah_session` / `session_id` / `litellm_session_id`；
  没有则哈希 system prompt + 首条用户消息——**对任何 harness 的现有流量都自动生效**
- 同一步重试复用会话里的 plan，绝不重复提问；settle 按 (step, question) 去重，
  重试只落一条标签样本

## S11 如何实现：部署形态

- Docker：官方 LiteLLM 镜像 + 插件，自包含（proxy 配置、JAH 配置、entrypoint 都在镜像内，
  key 只走环境变量）
- 推荐 compose：`redis`（会话共享，多副本就绪）+ `litellm-jah`（proxy）两个服务，一条命令起
- 日志卷挂到宿主 `~/.jah/litellm-logs`，`jah report` / `jah fit` 在宿主机直接跑
- 流式请求支持（标签语义：stream 流量只采集决策标签，不替换已流式发出的回复）

## S12 如何使用：三步部署

1. **改模型**：把 `litellm_config.yaml` 里 `jah-strong` / `jah-fast` 指到你的两个模型
   （附真实模型上限：如 max_output / max_input，客户端超限时代理自动钳制，**不截断 prompt**）
2. **写配置**：挂载 `jah-config.yaml`（决策后端、tau、模式、strong 重跑凭据、prices/currency、log_dir）
3. **启动**：
   ```bash
   cd litellm_jah/docker
   LITELLM_MASTER_KEY=sk-... DASHSCOPE_API_KEY=sk-... docker compose up -d --build
   curl localhost:4000/v1/models -H "Authorization: Bearer sk-..."
   ```
- 客户端只需：`base_url` 指向代理、`model: "jah-auto"`，其余不动
- 注意：不要给 `jah-auto` 开 `session_affinity`（JAH 每步重决策，粘滞会锁死首步的部署）

## S13 如何使用：配置要点（一张表）

| 配置 | 含义 | 本部署取值 |
|---|---|---|
| `decision` | 决策后端 | `{type: jev, provider: local, model: decision-model-preview}` |
| `tau` | p(hard) ≥ tau 升级 strong | 0.80（11-05 按校准 4B 分布上调） |
| `mode` / `decisions.*` | routing 及五决策点模式 | routing=cascade；stop="off"；tool_select="off"（注意 YAML 1.1：裸 off 是布尔，要加引号） |
| `first_steps_strong` | 开局强制 strong 步数 | 1 |
| `on_error` | 决策失败时走哪层 | strong |
| `strong` + `strong_api_key/base` | 重跑模型与凭据 | deepseek-v4.1-flash（`os.environ/…`，key 不进配置文件） |
| `currency` / `prices` | 计费币种与每百万 token 价（input/output/cached_input/cache_write） | CNY：deepseek-v4.1-flash 2/8/0.2；Qwen/Qwen3.8-27B 0/0（11-07 起 fast 层，自托管零计费）；qwen3.8-flash 0.8/2.7/0.1/1.25（历史回算用） |
| `decision.price_per_m` | 决策模型计价 | 0（输入免费） |
| `session_store` | 多副本会话共享 | redis://jah-redis:6379/0 |
| `log_dir` | journal 输出 | ~/.jah/litellm-logs |

## S14 如何使用：运维工作流（report / fit）

```bash
jah report ~/.jah/litellm-logs   # 会话数、步数、tier 占比、成本、延迟、错误
jah fit    ~/.jah/litellm-logs   # 阈值拟合 + 各决策点 describe 表
```

- 日常：看 `jah report` 的 strong 占比与成本趋势，判断负载是否变化
- 调阈值：某点 shadow 标签 ≥ 30 条 → `jah fit` → 有 tau 达 95% 才 promote，否则保持/关闭
- 本部署节奏示例：tool_select 标签 54 → 344 条、两轮 fit 结论一致（不可用）→ 关；
  stop 14 条时 tau 0.70 全对（但下界不过关）→ 继续攒到 81 条 → 第二轮 fit 证伪 → 关——
  ≥30 样本 + Wilson 下界的门槛就是为拦住小样本假信号设的

## S15 实验数据：总览（截至 11-04，真实日志）

`jah report`：

| 指标 | 值 |
|---|---|
| 会话数 | 152 |
| 步数/会话 | 7.5 |
| strong 占比 | 37%（fast 63%） |
| 成本/会话 | ¥0.1649 |
| LLM 延迟/会话 | 115.3 s |
| 错误 | 11（全部已定位，见 S17，11-04 零新增） |

最近 24 小时（869 步 / 45 会话）：

- fast 636 / strong 233；步骤原因：routine 636、hard 176、first_steps 49、finish_check 8
- 成本 ¥20.91；输入缓存命中率 95%（缓存价仅为普通输入的 1/10～1/20）

建议图：按日堆叠柱状图（fast vs strong 步数）+ 成本曲线
（11-01 13 步 → 11-02 204 → 11-03 779 → 11-04 150）。

## S15b 实验数据：fast 层切换本地 Qwen3.8-27B（11-07）与成本测算

变更：fast 层由 maas `qwen3.8-flash` 换为自托管 `Qwen/Qwen3.8-27B`（与 4B 决策模型同服、
同一 ssh 隧道；工具调用实测通过，单步 ~1.4 s，计价条目 0）。
切换只动 `litellm_config.yaml` 一处 + 计价表加一条——harness 与路由代码零改动。

**成本测算**（校准 era 实测，32 会话 / ¥89.75）：

| 层 | 调用数 | 成本 | 占比 | 切换后 |
|---|---|---|---|---|
| fast（原 qwen3.8-flash @maas） | 1040 | ¥30.79 | **34.3%** | **0**（自托管） |
| strong（deepseek-v4.1-flash @maas） | 879 | ¥58.96 | 65.7% | 不变 |
| 决策模型（4B 本地） | 3500+ | 0 | 0% | 不变 |
| **合计 / 会话** | | **¥2.80** | | **≈¥1.84（−34%）** |

- 按近期 ~11 会话/天 ≈ 省 ¥10/天；按 11-03 的 45 会话/天负载 ≈ 省 ¥40/天
- **降幅下限在 strong**：路由按 p_hard≥0.80 判层、与价格无关，strong 的 65.7% 不受本次切换影响；
  再降只能动 tau（0.85→−9%、0.9→−19% 的强判比例，代价见 S16 routing 条）
- 未计入：本地 GPU 成本——27B 与决策模型共卡，是成本转移（账单→算力）而非消失；
  高峰观察 `jah report --calibrated true` 的延迟 p99 决定是否给 27B 单独开卡

## S16 实验数据：决策点与 fit 结果

`jah report` 决策表：

| 决策点:模式 | calls | 错误 | p50 延迟 |
|---|---|---|---|
| routing:cascade | 1033 | 0 | 0.195 s |
| stop:shadow（11-05 已关） | 19（14 带标签，最终累计 81 条） | 0 | 0.193 s |
| tool_select:shadow（已关） | 694（350 带标签） | 4（旧遗留） | 0.259 s |

`jah fit`（目标：与 LLM 一致率 ≥ 95%，Wilson 下界，≥30 样本）：

- **tool_select：负结论**——344 条标签，原始一致率仅 **13.4%**，任何 tau 都达不到 95%
  （最好 18% @ tau 0.32）。根因：模型 77% 的步预测"不调工具"，而 agent 实际 88% 的步调工具
  （reply_without_tool→bash 111 条等）。若当时直接上 guard，约 3/4 步骤会被错误地压成
  "直接回答"。**shadow 机制在 promote 前拦截了这个坏配置** → 已设 `mode: "off"`，
  省掉每步 ~0.26 s 决策延迟
- **stop：负结论（11-05 定案）**——第一轮 14 条标签时 tau 0.70 全对（11/11）但 Wilson 下界
  80.3% 不过关；攒到 81 条重拟合，总体一致率仅 **61.7%**，任何 tau 达不到 95%。且 shadow 的重跑
  回复本就丢弃（只采标签、不提供保护），代价是每次 fast 终答（~6% 步）一次全上下文 strong 调用
  + ~0.24 s → `mode: "off"`。小样本假信号被门槛拦住，正是闭环设计的证明
- **routing：无标签可拟合**（strong 层怎么答未知），只有 tau 一个旋钮：fires 25.4%@0.75、
  16.7%@0.85、11.7%@0.9。11-05 校准 4B 的 p(hard) 分布右移（strong 一度到 69%），tau 上调至 0.80

建议图：tool_select 混淆矩阵 top 条 + stop 的 tau-覆盖率-一致率曲线。

## S17 实验数据：错误审计（11 条，全部解释、且都已不复现）

| 时间 | 数量 | 原因 | 现状 |
|---|---|---|---|
| 11-02 | 4 | 旧 master key 认证失败 | 换 key 后不复现 |
| 11-02 | 2 | strong 重跑缺 api_key | 配置 `strong_api_key` 后不复现 |
| 11-02 | 2 | 客户端 max_tokens 超模型上限 400 | 钳制上线后不复现（只钳输出，不截 prompt） |
| 11-03 | 2 | maas 服务端瞬时故障（PreemptionPlan） | 非配置问题，未再发生 |
| 11-03 | 1 | 客户端中途断开 | 客户端取消，非代理问题 |

- 要点：**11-04 零新增错误**；routing 1033 次调用 0 错误——决策链路本身是稳的

## S18 结论与下一步

结论：

- 四个目标都落地：任意 harness 接一个 OpenAI 兼容端点即用；simple 步走 fast 层直接省钱
  （本部署 fast 占 63%、24h ¥20.91/869 步、缓存命中 95%）；hard 步强升 + on_error=strong
  守住准确率；决策开销 ~0.2 s 相对单步 ~12 s 可忽略
- 闭环机制验证有效：tool_select 与 stop 的负结论都是**机制的正确输出**——shadow 采集 → fit 证伪 →
  不 promote（前者会误杀 ~87% 步骤的工具调用；后者 81 条标签定案，任何 tau 不过关，连采标签的 strong 重跑钱也省下）
- 成本/路由漂移可观测：11-04 strong 占比升到 69% 是因为决策模型判 hard 更勤
  （routing fires 8.5% → 25.4%），report/fit 立刻定位，可经 tau 调节

下一步：

1. **strong 层替换为 Kimi K3.1 Pro**：只改 `litellm_config.yaml` 里 `jah-strong` 的模型/端点/凭据
   （及 `jah-config.yaml` 的 `strong` 重跑配置），路由、决策点、日志、计价全部不变——
   这正是"配置即策略"的价值：换模型是配置变更，不是代码变更；替换后用 `jah report`
   对比 strong 占比与成本
2. **决策模型换成本地部署的 Decision8B**：`decision` 块改为 local 部署
   （`provider: local` + 本地端点 + `price_per_m: 0`），去掉对 maas `decision-model-preview`
   的依赖，决策延迟进一步下降、数据不出内网；注意本地模型的 p(hard) 分布会与线上预览版
   不同——换后重新跑 `jah report` 看 routing fires，必要时重调 tau
3. **stop 保护重建**：决策模型路径已证伪；若要找回 fast 终答的 strong 复核，只能无条件重跑（stop:strong，~6% 步的全额 strong 调用）——先拿下一步第 5 项的 A/B pass 数据，再决定这笔钱花不花
4. routing tau 上探 0.85–0.9 仍是降本旋钮（fires 16.7%/11.7%）；用 describe 表持续监控 p(hard) 分布漂移
5. 与"全 strong 基线"做 A/B：准确率与速度的直接对比数据（当前为日志观测值）
6. 多副本压测（Redis 会话已就绪，ADR-006）

## S19 结尾 / Q&A

- 一句话收尾：**让每个 agent 步骤都花在自己值得的那一档模型上，且每一分钱的取舍都有日志可查。**
- 仓库：`github.com/exceedzhang/JevAgentHarness`（插件在 `litellm_jah/`，MIT）

---

### 附：数据口径说明（备答提问）

- 统计来源：`~/.jah/litellm-logs/{steps,decisions}.jsonl`，命令
  `PYTHONPATH=src python -m jevagentharness.cli report|fit ~/.jah/litellm-logs`（11-04 运行）
- cost/session ¥0.1649 是**全历史**均值，含计价上线前的 unpriced 行，偏低；看单日更有意义
  （11-03 ¥21.24/779 步，11-04 ¥3.86/150 步）
- 11-04 的 strong 69% 是负载相关（当天任务长上下文偏多，判 hard 多），非路由故障
- 工具集在运行中从 27 个选项缩到 15 个、`bash` 改名 `shell`，tool_select 标签跨两代口径，
  不影响其"不可用"结论
- "提速"为机制性结论（fast 层模型更快 + 决策延迟 0.2 s 量级），直接 A/B 对比待补（见 S18）
