# 提案：在插件侧应用 AgentDecision-4B 的校准温度（不改动生产，先评审）

状态：草案，等待评审。未应用任何代码/配置改动。
依据：模型卡 [JevAgentHarness/AgentDecision-4B](https://huggingface.co/JevAgentHarness/AgentDecision-4B) 的 `calibration.json` / `thresholds.json` / README（2026-11-05 读取），以及对现网 `ssh-tunnel:8000/v1/systemone` 的两次探针。

## 1. 事实与证据

1. **`calibration.json` 是温度，不是乘性因子**：概率 = `softmax(logits / T)`，按 `<kind>:<bucket>` 查 T，查不到用 `global`。本模型的 4 个值：
   `global 0.7720` / `choice:3-5 0.9676` / `choice:17-52 0.0500` / `noul:2 0.4231`。
   分桶：2 个选项 → `2`；3–5 → `3-5`；6–16 → `6-16`；17–52 → `17-52`。
2. **我们的服务没有应用校准**（高置信实证）：21 个选项、其中 19 个是同义改写（"ask the user a clarifying question" 的变体）的探针返回
   `t01 0.651 / t06 0.078 / t02 0.0776 / t05 0.069 / t07 0.053 / …` 的平滑分布。
   若 T=0.05 生效（logits ×20），不同措辞的同义选项必须近乎 one-hot，需要措辞间 logit 差 <0.1 nat，不可能。
   `serve_jev` 里 `--calibration` 是显式 flag——服务端没传（或等价实现未启用）。
3. **我们所有问题都是 `kind="choice"`**（`backend.Question.wire()` 硬编码 `type: "choice"`）：
   routing（routine/hard）和 stop 落在 `choice:2` —— 卡片里没有这个桶 → 用 `global 0.772`；
   tool_select（15–27 个工具）落在 `choice:6-16`/`choice:17-52`，但该点已 `mode: "off"`。
4. 卡片明确：**routing 问题与编码 agent 状态都在训练分布外**，官方指引就是"shadow 跑一段、用自己的日志 `jah fit` 重定阈值"。校准不解决这个问题，tau 重校（P-1）照旧必要。

## 2. 方案：客户端温度变换

服务端只有 T=1 的原始概率时，任意温度可以纯后处理等价复现：

```
p' ∝ p ** (1/T)   （再归一化）        因为 softmax(z/T) ∝ exp(z/T) = (softmax(z))^(1/T)
```

在 `JevBackend._parse` 归一化之后做一次幂变换；配置块名 `decision.calibration`，
**不配置 = 行为与现在完全一致**（原始空间）。非法配置在启动时 `raise`（fail-closed，与 `config._validate` 惯例一致）。
decisions 行透传 `calibrated: true/false`，让 `jah fit`/报表能区分"原始空间 era"和"校准空间 era"（与 preview→4B 的 `model` 字段切分并用）。

不推荐的替代：让远端服务补 `--calibration` —— 我们控制不了 ssh-tunnel 后面的部署参数，且插件侧变换可单测、可回滚、语义显式。

## 3. Diff（评审用，未应用）

### 3.1 `litellm_jah/backend.py`

```diff
--- a/litellm_jah/backend.py
+++ b/litellm_jah/backend.py
@@ dataclass Decision (L55)
 @dataclass
 class Decision:
     p: dict[str, float]
     model: str = ""
     cost_usd: float = 0.0
     latency_s: float = 0.0
     input_tokens: int = 0
     why: str = ""
+    calibrated: bool = False # p went through the model card's calibration temperature (else raw T=1)
```

```diff
+++ b/litellm_jah/backend.py  (helpers, right before ``class JevBackend``)
+CALIBRATION_KEYS = frozenset({f"choice:{b}" for b in ("2", "3-5", "6-16", "17-52")} | {"noul:2"})
+
+
+def _bucket(n: int) -> str:
+    """The model card's bucket of an n-option question (README's ``bucket()``)."""
+    return "2" if n == 2 else "3-5" if n <= 5 else "6-16" if n <= 16 else "17-52"
+
+
+def _check_calibration(cal: Mapping[str, Any] | None) -> dict[str, Any] | None:
+    """``decision.calibration``: the model card's calibration.json inline (softmax(logits / T), one temperature
+    per ``<kind>:<bucket>``, ``global`` fallback). Our server returns the raw T=1 probabilities, so the plugin
+    applies the temperature itself (p' ~ p**(1/T)). Fail closed: a bad block stops the proxy at startup."""
+    if cal is None:
+        return None
+    if not isinstance(cal, Mapping):
+        raise ValueError("decision.calibration must be a mapping {global: T, buckets: {key: T}}")
+    try:
+        g = float(cal["global"])
+    except (KeyError, TypeError, ValueError):
+        raise ValueError("decision.calibration.global must be a number: the global temperature") from None
+    if not math.isfinite(g) or g <= 0:
+        raise ValueError("decision.calibration.global must be a finite temperature > 0")
+    buckets: dict[str, float] = {}
+    for k, v in (cal.get("buckets") or {}).items():
+        try:
+            t = float(v)
+        except (TypeError, ValueError):
+            raise ValueError(f"calibration bucket {k} must be a number") from None
+        if k not in CALIBRATION_KEYS:
+            raise ValueError(f"unknown calibration bucket {k!r}; allowed: {sorted(CALIBRATION_KEYS)}")
+        if not math.isfinite(t) or t <= 0:
+            raise ValueError(f"calibration bucket {k} must be a finite temperature > 0")
+        buckets[k] = t
+    return {"global": g, "buckets": buckets}
```

```diff
+++ b/litellm_jah/backend.py  (JevBackend.__init__, L85)
     def __init__(self, provider: str = "openrouter", model: str | None = None, base_url: str | None = None,
                  key_env: str | None = None, timeout_s: float = 8.0, retries: int = 2,
-                 price_per_m: float | None = None) -> None:
+                 price_per_m: float | None = None, calibration: Mapping[str, Any] | None = None) -> None:
         ...
         self.retries = retries
+        self.calibration = _check_calibration(calibration)
         self._session: ClientSession | None = None
```

```diff
+++ b/litellm_jah/backend.py  (JevBackend._parse, L143-151)
         s = sum(p.values())
         if s <= 0:
             raise RuntimeError(f"decision model returned no distribution: {str(a)[:200]}")
         p = {k: v / s for k, v in p.items()}
+        calibrated = self.calibration is not None
+        if calibrated:
+            # softmax(z/T) == renormalized p**(1/T) when p is the raw softmax(z): the card's temperature,
+            # applied here because the server was started without --calibration
+            t = self.calibration["buckets"].get(f"choice:{_bucket(len(p))}", self.calibration["global"])
+            q = {k: v ** (1.0 / t) for k, v in p.items()}
+            p = {k: v / sum(q.values()) for k, v in q.items()}
         u = data.get("usage") or {}
         tokens = int(u.get("input_tokens") or 0)
         cost = float(u["cost"]) if isinstance(u.get("cost"), (int, float)) else tokens * self.price_per_m / 1e6
         return Decision(p=p, model=str(data.get("model") or self.model), cost_usd=cost, latency_s=latency,
-                        input_tokens=tokens)
+                        input_tokens=tokens, calibrated=calibrated)
```

```diff
+++ b/litellm_jah/backend.py  (describe L157, make_backend L230)
     def describe(self) -> dict[str, Any]:
-        return {"backend": self.name, "provider": self.provider, "model": self.model, "base_url": self.base_url}
+        return {"backend": self.name, "provider": self.provider, "model": self.model, "base_url": self.base_url,
+                "calibration": self.calibration}
```
```diff
         allowed = {"provider", "model", "base_url", "key_env", "timeout_s", "retries",
-                   "price_per_m"}
+                   "price_per_m", "calibration"}
```

### 3.2 `litellm_jah/points.py`

```diff
+++ b/litellm_jah/points.py  (Ask, L523 附近)
     p: dict[str, float] = field(default_factory=dict)
     model: str = ""
+    calibrated: bool = False # the probabilities live in the card's calibrated space (else raw T=1)
     cost_usd: float = 0.0
```
```diff
+++ b/litellm_jah/points.py  (Asker.ask, L584-585)
             a.p = {o: float(d.p.get(o, 0.0)) for o in q.options}
             a.model, a.cost_usd = d.model, float(d.cost_usd or 0.0)
+            a.calibrated = bool(getattr(d, "calibrated", False)) # rules/none backends have no probabilities to temper
```
```diff
+++ b/litellm_jah/points.py  (Ask.record, L560)
-            "cost_usd": round(self.cost_usd, 8), "model": self.model or None, "rounds": self.rounds,
+            "cost_usd": round(self.cost_usd, 8), "model": self.model or None, "rounds": self.rounds,
+            "calibrated": self.calibrated if self.p else None,
```

### 3.3 `litellm_jah/docker/jah-config.yaml`（评审通过后才加）

```yaml
decision:
  type: jev
  provider: local
  model: AgentDecision-4B
  base_url: http://ssh-tunnel:8000
  key_env: DECISION_API_KEY
  price_per_m: 0
  # the model card's calibration.json (softmax(logits / T)). Our server answers with raw T=1 probabilities
  # (proved by a 21-synonym-option probe: no one-hot), so the plugin applies the card's temperatures here.
  # tau and `jah fit` then live in the calibrated space, comparable with the card's thresholds.
  # NOTE YAML: the keys contain ':' and must stay quoted; values like off/on would parse as booleans.
  calibration:
    global: 0.7720006824268275
    buckets:
      "choice:3-5": 0.9676007232026532
      "choice:17-52": 0.05000000000004328
      "noul:2": 0.4230827372940139
```

## 4. 数值影响（先给预期，别指望校准救 4B）

routing/stop 是二选项问题 → T=global 0.772，等价指数 1/T=1.2953，sharpen 很轻：

| 原始 p_hard | 校准后 |
|---|---|
| 0.60 | 0.628 |
| 0.66（4B 现网观测到的最大值） | 0.703 |
| 0.70 | 0.750 |
| 0.75 | 0.806 |

- 校准空间的 tau 0.75 ≈ 原始空间的 0.70：在 preview 模型分布上 fires 会从 25.4% 升到 ~28.5%；
- **4B 现网 p_hard max 0.66 → 校准后 0.70，tau 0.75 仍然 0 fires**——校准不解决 4B 的问题，P-1 重校照做；
- 真正的重头戏本应是 17-52 桶（T=0.05，×20 sharpen），但我们的 tool_select 已 off，暂无消费者；stop 若将来出现多选项形态则自动受益。

## 5. 测试清单（实现时）

1. `_bucket` 边界：2/3/5/6/16/17/52；
2. 桶缺失回退 global（`choice:2`、`choice:6-16`）；
3. 二选项变换与上表一致（误差 <1e-6）；变换保序、和为 1；
4. 往返：raw→calibrated→用 T 反向幂 → 还原 raw；
5. `_check_calibration` 拒绝：非 mapping、global 缺/0/负/NaN/inf、未知桶名、桶值非数（全部启动期 `ValueError`）；
6. `make_backend` 拒绝未知键的行为不变（calibration 进入白名单）；
7. 无 `calibration` 键 → `_parse` 输出与今天逐位一致（回归基线）；
8. `Asker.ask` 对 RulesBackend 行 `calibrated=False`；decisions 行含 `"calibrated"` 字段；
9. 现有 218 测试全绿 + ruff clean。

## 6. 上线与回滚

1. 评审通过 → 实现 + 测试 → `--build` 重建（python 变更）；
2. **先在配置里只开 calibration、tau 不动**：观察 1–2 天，routing 行 `calibrated: true`、fires 比例应 ≈ 表 4 的位移；
3. 之后 `jah fit` / P-1 重校**一律在 calibrated 空间**做；分析新旧 era 时按行的 `calibrated` + `model` 两个字段切段；
4. 回滚：删 `calibration` 块 + `docker compose restart`（纯挂载配置变更），回到原始空间——tau 语义随之回退，重校结论作废，需按表 4 换算或重跑。

## 7. 未决问题（不阻塞本提案，但必须有人拍板）

- **许可证：CC BY-NC 4.0**（adapter + calibration.json + thresholds.json 全非商用）。本部署是生产流量——**这是合规问题，先于一切技术细节**。要么确认公司使用场景获准，要么换可商用模型/自训。
- 服务端 `serve_jev` 是否其实传了 `--calibration` 但实现不同（探针结论强，但值得服务方确认一次）。
- 服务端把 prompt 截到**最后 4096 tokens**：长步骤的 routing 输入实际被截断，P-1 重校的样本要覆盖长上下文层。
- 卡片数据（0.445/0.996、温度值）也是 CC BY-NC；写进 repo 注释时保持引用 URL 即可，避免整段复制。
