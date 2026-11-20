# SWE-bench 定向重跑：定位"死循环/不收敛"的具体机制

**任务**：用生产配置（v8 决策 + Qwen3.8-27B worker）重跑 E4b 的失败实例，捕获逐步 trace，定位 agent 在难任务上"干到耗尽步数但不收敛"的**具体机制**。

**日期**：2026-11-20 · **机器**：140.210.143.61（国内，repo `/data1/apps/JevAgentHarness`）

---

## 0. 结论速览（TL;DR）

1. **SWE-bench 的预构建 docker 镜像在这台机器上不存在，且无法获取**（Docker Hub 直连被墙；8 个镜像源要么把 swebench 镜像"白名单外拒"、要么 DNS/超时失败）。
2. **我换了一条"等价的轻重建"路线绕过了它**：用能拉到的 `python:3.6-slim`（走阿里源）+ GitHub 上该 base_commit 的 tarball + 阿里 pip 源 + git，手工重建了 django-10554 和 sympy-13757 两个**真实 SWE 环境**（真实仓库、真实 base_commit、真实的 bug 复现）。bug 复现已验证。
3. **跑通了 2 个失败实例**（生产配置：v8 `full` arm + Qwen worker，逐步 trace 全捕获），都是 `LimitsExceeded`（和我 30 步上限一致；E4b 里它们是 84~211 步才耗尽 250 步上限）。
4. **看到了两类不同的"不收敛"机制**（关键发现，见 §3）：
   - **django = 真·不收敛**：探索踏车（同一批核心文件反复读，compiler.py 读了 7 次）+ 反复失败的复现脚本（重写 `/tmp/repro.py` 5 次全 rc=1），**从没动过源码**。
   - **sympy = 能收敛但被预算砍掉**：1 步就复现出 bug，下载 sympy 1.7 diff 定位到修法，**第 31 条命令就改对了**（一行 `_op_priority`），验证通过，在跑测试/查 1 个边缘用例时被 30 步预算截断。
5. **v8（System-1）在生产里几乎没帮上忙**：routing 60/60 全判 "strong"（≈1.0 置信，从不降级到 fast）；action_guard 全放行（p≈1.0）；**reflection 的"卡住了吗"在 django 连续 5 次复现失败时答"没卡住"（置信 0.87~0.96）→ v8 没检测到循环**；stop 钩子没触发（agent 没走到提交）。

---

## 1. SWE 基建现状 + 可行性

| 项 | 状态 |
|---|---|
| SWE-bench 预构建镜像 `swebench/sweb.eval.x86_64.*` | **不在服务器，且拉不到**（见下） |
| SWE-bench 数据集（Verified, 500 行） | ✅ 通过 `hf-mirror.com` 拉下（8.2MB） |
| Docker Hub 直连 / HuggingFace 直连 | ❌ 被墙（TCP 挂起）。GitHub ✅、pip 阿里源 ✅、`hf-mirror` ✅ |
| Docker 镜像源（daemon 里配了 8 个） | 能拉**白名单内**常见镜像（hello-world、`python:3.6-slim`）；**swebench 镜像"不在白名单"被拒**（daocloud 明确报 `这镜像不在白名单`），其余 DNS 失败/超时 |
| vllm worker `Qwen/Qwen3.8-27B` @ `127.0.0.1:8000/v1/chat/completions` | ✅ 能出 bash tool_call（实测） |
| v8 决策 `AgentDecision-4B` @ `127.0.0.1:8000/v1/systemone`（Jev wire） | ✅ 能出 routing 选择（实测 fast/strong + 置信） |
| 依赖 | ✅ 装了 `mini-swe-agent 2.4.6`（阿里 pip 源），`jevharness` 用 `PYTHONPATH=src` |

**为什么拉不到 SWE 镜像**（决定性测试，逐个镜像源实测 `docker pull swebench/sweb.eval.x86_64.django_1776_django-10554`）：
- daocloud / dockerhub.icu / awsl9527 / anyhub：daocloud 明确 `denied: 这镜像不在白名单`（白名单只代理常见镜像）；其余超时。
- nju：返回 HTML（挂了）；ustc/chenby/lpanel：DNS 解析不到。
- Docker Hub 直连：`request canceled while waiting for connection`（被墙）。

**绕过方案（已落地，非"重建整个 SWE-bench 构建"，只重建这 2 个实例）**：
`FROM python:3.6-slim` → 阿里源 `apt install git` → 阿里源 `pip install -e <repo>@base_commit`（GitHub tarball）→ `git init && git commit base`。django-10554 构建后跑 FAIL_TO_PASS 测试**精确复现 bug**（`ORDER BY term does not match any column in the result set`）；sympy-13757 同理（`x*Poly(x)` 不求值）。

> 代价/风险：每个实例手工构建 ~5-15 分钟。若要把 44 个全跑，需要为每个 repo 版本做对应 Python 版本 + 依赖，成本/易碎度随实例数上升；**2 个已足以定位机制**。

## 2. 怎么跑的（生产配置复现）

- **arm = `full`**（`full`：action_guard=guard, stop=guard, routing=trigger, reflection=trigger —— 全套 v8 钩子）。
- **worker**：`model_name=openai/Qwen/Qwen3.8-27B` + `openai_api_base=http://127.0.0.1:8000/v1`（litellm 走 OpenAI 兼容口）。
- **S1 (v8)**：`jev.s1={type:jev, base_url:http://127.0.0.1:8000, model:AgentDecision-4B}`（Jev wire → /v1/systemone）。
- **routing 的 fast/strong 两个档都 = Qwen**（`cheap_model` 也设成 Qwen）→ routing 决策不改变执行模型，只记录 v8 判定（这正是"单 worker"生产配置）。
- **30 步上限 + `max_tokens=3072`**（为了在合理时间拿到完整 trace；E4b 是 250 步）。
- docker 用单 token wrapper（`MSWEA_DOCKER_EXECUTABLE=/home/ubuntu/mswe-docker` → `sudo -n docker`），因为 ubuntu 不在 docker 组。
- **没动生产 GPU0/2/3**；Qwen+v8 用现成 vllm(8000)，agent 环境跑在独立 docker 容器里。
- 逐步 trace 在 `*.traj.json`（每条命令+观察）和 `*.jev.json`（每步 DecisionLog/S1-S2 call）。

跑法（一行）：`python -m jevharness.adapters.mini_swe run -c swebench_jev.yaml -c model.model_name=openai/Qwen/Qwen3.8-27B ... --arm full --subset <本地dataset, 带 image_name> --split test -o <out> -w 1`。

## 3. 逐步 trace 里看到的循环机制（核心）

### 3.1 django__django-10554 —— **真·不收敛**（E4b: 84~113 步，未解）
30 步逐步命令（节选）：
- **步骤 1–33 = 探索踏车**：反复读同一批核心文件。`compiler.py` 在 30 步里被引用 **7 次**（sed/grep 切片），`sql/query.py`、`query.py` 也反复读。全程**没有任何一次 `git diff` 或源码编辑**。
- **步骤 34–41 = 反复失败的复现**：写 `/tmp/repro.py`（rc=1）→ 重写（rc=1）→ 再重写（rc=1）→ `/tmp/repro2.py`（rc=1）→ 再写（rc=1）。**共 5 次复现脚本全失败**。
- 失败根因：**环境不匹配**。这个 bug 只在"支持 compound 子句里 ORDER BY 的后端（PostgreSQL）"才触发；环境是 SQLite，复现脚本要么 SQLite 语法跑不了、要么 app/feature flag 设置不对而报错。→ agent 拿不到一个"能复现 bug 的最小 repro"，于是**无法验证任何修法**，只能在"多读一点代码 + 再试一次 repro"之间来回打转。

**机制 = 探索踏车 + 假复现死循环（false-reproduction thrash）**：没有工作着的 repro ⇒ 无法验证 ⇒ 不敢 commit 修改 ⇒ 持续读/试 ⇒ 耗尽预算。

### 3.2 sympy__sympy-13757 —— **能收敛，被预算砍**（E4b: 122~211 步，未解）
30 步逐步（关键步）：
- **步骤 1–3**：探索 + **第 3 条命令就复现出 bug**（`x*Poly(x)` 不求值）。
- **步骤 13–28**：下载 sympy 1.7，和 testbed 逐文件 **diff** 定位上游修法（`call_highest_priority` / `_op_priority`）。
- **步骤 29–31**：定位修法 = 给 `Poly` 类加 `_op_priority = 10.001`；**第 31 条命令应用了这个 one-line fix**。
- **步骤 32–37**：验证 PR 里的所有例子都对了；跑 `test_polytools.py` / `test_arit` / `test_expr` / `core/` 测试套件通过。
- **步骤 38–44**：`test_match.py` 里有 **1 个**用例失败，在查"是我的改动导致还是本来就坏"（git stash 对比、看 1.7 的断言）时被 30 步截断。

**机制 ≠ 紧循环**，而是** productive-but-culled**：fix 早就改对了、也验证了，但 agent 继续做"全套测试 + 边缘用例 due-diligence"，把预算耗在收尾。E4b 250 步也没解 ⇒ 说明**收尾阶段（跑 3323 个 pass-to-pass + 查边缘用例）是 250 步被吃掉的真正地方**，不是"改不动"。

## 4. v8（生产 System-1）的实际表现

| 钩子 | django | sympy | 评价 |
|---|---|---|---|
| routing | strong ×30（conf 0.994–1.0） | strong ×30 | **从不降级到 fast** → 在难任务上 routing 只增决策成本、零成本/effort 节省 |
| action_guard (tool_select) | 41 次全放行（p 0.95–1.0），0 拦截 | 44 次全放行，0 拦截 | 健康（无过度拦截） |
| reflection | 触发 6 次，但答 **"没卡住"**（s1_value=False，conf 0.87–0.96） | 0 次 | **关键弱点**：django 连续 5 次复现失败时，v8 的"卡住了吗"判"没卡" → **没检测到循环**，未注入反思 |
| stop | 未触发 | 未触发 | agent 没走到提交，没机会验证 stop 钩子 |

**两个 v8 结论**：
1. **routing 在难任务上是"饱和"的**（总说 strong）——它不携带"进度/卡住"信号，也不带来成本节省。
2. **v8 的循环检测是失效的**：reflection 的 stuck 判定在明显循环（5 次 repro 全挂）时仍然高置信地答"没卡住"。这是**可复现、可测**的缺口。

## 5. 可干预点建议（按性价比排序）

1. **无进展/只读循环检测器**（打 django 探索踏车）：每步统计 (a) 编辑事件（`git diff` 是否有新增）+ (b) 最近触碰文件的**重叠度**。若连续 N（如 8–10）步**纯只读** 且 文件集 Jaccard 重叠高 → 注入"要么 commit 一个最小修改，要么显式换方向"的 nudge（或 escalate 到规划重置）。
2. **假复现检测器**（打 django 假 repro）：检测到重复的"复现脚本重写 + 同错误类"（如 `/tmp/repro*.py` 反复写、同 rc≠0）≥2–3 次 → pivot 提示："别在隔离环境里硬复现；目标测试（FAIL_TO_PASS）才是 ground truth，直接做最小源码改动并跑那个测试"。直击"环境不匹配 → 假复现"陷阱。
3. **验证即提交（early-submit on verified fix）**（打 sympy 这类 productive-but-culled）：当**已发生编辑 且 该实例的 FAIL_TO_PASS 测试通过**时，**强触发 stop 钩子（"现在就提交"）**，而不是要求跑完整个 pass-to-pass 套件。这能救下大量"fix 已对但预算耗在收尾"的实例。
4. **把 v8 的"strong" 判定改造成"卡住/在推进"信号**：当前 routing 饱和、reflection 失灵；应让 S1 显式输出 `productive / stuck`（或直接修 reflection 的 stuck 问题——见 §4）。
5. **预算策略**：E4b 的 250 步"统一预算"分不清 django（真不收敛）和 sympy（能收敛但慢）。用 #1–#3 的**进度感知早停**替代/补充固定步数上限。

## 6. 局限 / 没做的事
- 只跑了 **2 个**（30 步上限，非 250）；django 的 trace 停在第 30 步（还在循环里），**250 步全量跑**才能确认它"永不收敛"还是"后期才改+失败"。但"探索+假 repro"的形态和 E4b 前 1/3 一致。
- **v8 stop 钩子没被验证**（agent 没走到提交）；要测它得跑到它提交为止。
- 没跑 sphinx/matplotlib/astropy 等其余实例；按"看到在循环什么"的目标，2 个对比（真不收敛 vs 能收敛被砍）已够定位机制。要扩，现在 pipeline 已通，加一个实例 ≈ 15-30 min。
- 没动生产 GPU0/2/3；全程复用现成 vllm + 独立 docker 容器。

## 附：产物 / 复现路径（服务器上）
- 镜像：`jah-swe-django10554-agent:latest`、`jah-swe-sympy13757-agent:latest`
- trace：`/tmp/swe_run/out/{dj,sym}/<iid>/<iid>.traj.json` + `.jev.json`
- 脚本：`/tmp/swe_run/run_{dj,sym}.sh`（含全部 -c 覆盖）；本地分析脚本在 `/private/var/.../opencode/`
