# Drex vs JAH: option readout heads, the RLAF objective, and the inference stack

**Status (2026-11-14):** research note. Compares the *option readout head*, the *training
objective*, and the *inference/serving stack* of the open **Drex** decision models
(Nace.AI, "Towards Foundation Decision Models: Drex", arXiv:2611.07854v2) against this
repository's decision deciders (`training/decision_model.py`, `training/serve_jev.py`).

**Scope.** Three questions, answered side by side:

1. **Readout head.** How does each model turn a single forward pass into a probability
   distribution over the user's options? (JAH `option_logits` vs Drex Max `W_r`, with the
   Drex *pointer* head kept as the cost-referenced third case.)
2. **Objective.** What does each model train against — in particular Drex's
   **RLAF** (reinforcement learning from adjusted feedback) stage and its KL-regularized
   Max variant — versus JAH's soft-CE/Brier, DPO, and policy-gradient losses?
3. **Inference stack.** How does each model serve a request cheaply: prefix caching,
   batching, calibration, and the measured latency/cost.

Both are "single-pass typed deciders": given a state + a user-supplied option set, one
forward pass yields a distribution over options, **no autoregressive decoding**. They are
different points in the same family: JAH sits on the *constrained-LM-head* side
(decider-4b / JevK5 lineage), Drex Max on the *pplx-decider* side (non-causal backbone +
dedicated readout head).

**Code map:**

| Concern | JAH (this repo) | Drex (paper) |
|---|---|---|
| Option codes | `decision/types.OPTION_LABELS` (52) | §5.2, codes γ₁..γ₂₅₅ |
| Render / prompt | `training/data.render`, `decision_model.chat_text` | §5.2 eq (4) |
| Readout head | `training/decision_model.option_logits` | §5.2 eq (5), `W_r` |
| Prefix / incremental scoring | `decision_model.option_logits_cached`, `prefill`, `SessionCache` | §9 shared prefill |
| Losses | `decision_model.decision_loss`, `train_decision.py` | §6 eq (6)–(14) |
| Calibration / temperature | `calibrate.py`, `serve_jev.Engine.temperature` | §6.6 eq (14), `T_dep` |
| > max-options | `serve_jev` knockout | §5.2 (255-code head) |
| Serving | `training/serve_jev.py` | §9, Table 8 |

---

## 0. Bottom line

| Axis | JAH (option-code readout) | Drex Max (readout head) | Drex pointer (DLM / v1.5 / v2.0) |
|---|---|---|---|
| Where the logits come from | **LM head**, gathered to 52 code-token rows | **new** bias-free linear `W_r` on the `<ans>` hidden state | pointer head, dot product of two hidden states |
| New parameters | **0** (LoRA only on the backbone) | **~1.3M** (`W_r`, 255×5120) | ~1.0–2.1M (pointer head, `d_p=256`) |
| Read position | last token (next-token position) | `<ans>` special token | `<decide>` (query) × each `</opt>` (key) |
| Option codes | **52** (A..Z, a..z; one token each) | **255** (A..Z, AA..; one token each) | unbounded (one `</opt>` per option) |
| Attention | **fully causal** | 48 DeltaNet (causal) + 16 full-attention **non-causal** (bidirectional) | causal, **block-masked** (questions never see each other) |
| State encoding | **once**, shared across k questions via prefix cache | **per question** (bidirectionality couples state to question) | **once**, shared via block-causal mask |
| Options compare to each other | no (causal) | **yes** (bidirectional full attention) | yes, through the pointer dot product |
| Multimodal | text only | ViT image encoder | text only |
| Order robustness | **data-side**: option permutation at train time | **loss-side**: `λ_π` permutation-KL + `λ_s` state-shuffle-KL | inherited; measured in §8.5 |
| Lineage | decider-4b / JevK5 / Imajev | **pplx-decider** (attention + readout) | open **Kev** code base |

The one-sentence trade-off: **JAH = causal + shared prefix cache + zero new params + 52
codes + data-side debiasing → cheap, cacheable, easy to fine-tune**; **Drex Max =
bidirectional + per-question re-encode + 1.3M new params + 255 codes + loss-side
consistency + vision → more accurate (especially knowledge / multi-step / image) but pays
the state cost per question**.

---

## 1. Background: the single-pass typed decider

A decision model answers a *question* about a shared *state*. The state is long and
expensive; the question is short and cheap. The design constraint is to read the answer as
a distribution over the user's options in a **single forward pass**, without generating
tokens.

The JAH render (this repo, `training/data.render`) is byte-stable across training,
calibration, and serving:

```
STATE:
{state}

QUESTION: {question}
OPTIONS:
A. {opt_1}
B. {opt_2}
...

Answer with the option label only.
```

wrapped in the system prompt

```
You are a fast decision module inside an AI agent. Read the state and answer the question
with exactly one token: the label of your answer.
```

and run through the tokenizer's chat template (Qwen3-style, `enable_thinking=False`).
Option labels are `OPTION_LABELS = "A..Z" + "a..z"` — 52 single-token codes
(`decision_model.code_token_ids` raises if a label is not exactly one token).

Drex Max encodes each question as its own sequence (paper §5.2, eq 4):

$$
x_k = \big[\, \mathcal{I}\,;\, S\,;\, u_k\,;\, \gamma_1\!:\!o_{k,1}\,;\, \dots\,;\,
\gamma_{K_k}\!:\!o_{k,K_k}\,;\, \texttt{<ans>} \,\big],
$$

where $\mathcal{I}$ are the image tokens (ViT), $S$ the state, $u_k$ the instruction, the
options are written as `code + ":" + text` (e.g. `A: refund request`), and `<ans>` is a
fixed special token that ends the sequence. The option codes $\gamma_1,\dots,\gamma_{255} =
\texttt{A},\dots,\texttt{Z},\texttt{AA},\dots$ are each a single vocabulary token.

Both read the answer from **one position at the end of the sequence** and take K logits —
they diverge entirely in *where those K logits come from*.

---

## 2. The option readout heads

### 2.1 JAH: constrained LM-head gather (no new parameters)

`decision_model.option_logits` runs one forward pass and gathers the **pre-trained LM
head's** next-token logits onto the option-code tokens:

```python
out = model(input_ids=..., attention_mask=..., use_cache=False, logits_to_keep=1)
last = out.logits[:, -1, :].float()          # full-vocabulary logits at the last position
z    = torch.gather(last, 1, code_ids)       # keep only the 52 code-token rows
return z.masked_fill(code_ids < 0, float("-inf"))
```

`DecisionCollator` **left-pads** prompts so the decision position is always index `-1`.
The probability is $p = \mathrm{softmax}(z/T)$ over the K presented codes.

This is a **constrained decoder**: the model is literally asked "what is the next token, but
the only tokens it may pick are A..z." The readout is the intersection of the pre-trained
next-token distribution with the 52 code tokens. Consequences:

- **Zero new parameters.** Only LoRA (r=32) on the Qwen3-4B backbone is trained. There is no
  overfitting surface in the readout itself.
- **The readout inherits the LM's prior over letters.** Each code row is a real,
  pre-trained vocabulary entry (`A.` is a common token), so the code logits are shaped by
  the LM's token statistics, not by a purpose-built classifier.
- **Calibration is tied to the LM head.** `calibrate.py` fits a per-(kind, K) temperature
  over the gathered logits; `serve_jev.Engine.temperature(kind, k)` applies it at serve time.

>52 options are answered by a **knockout** in `serve_jev` (chunks of 52, top-2 advance, a
final read over the finalists), because one readout pass is bounded by the 52 code tokens.

### 2.2 Drex Max: dedicated `W_r` classifier head

Drex Max replaces the LM head with a **new, bias-free linear layer** applied to the hidden
state at `<ans>` (paper §5.2, eq 5). Let $h_k \in \mathbb{R}^d$ (d=5120) be the
final-layer hidden state at the last position of $x_k$:

$$
z_k = W_r h_k \in \mathbb{R}^{255}, \qquad
p_{k,j} = \frac{\exp(z_{k,j}/T)}{\sum_{i=1}^{K_k} \exp(z_{k,i}/T)}, \quad j=1,\dots,K_k,
$$

where $W_r \in \mathbb{R}^{255 \times d}$ has **no bias**, so it is a pure linear map from
the hidden state to one logit per option code. Only the first $K_k$ logits (the K options
of the question) are kept. $W_r$ has $255 \times 5120 \approx 1.3\text{M}$ parameters. The
shipped temperature is $T = 1.009$, fitted after the last training stage (eq 14); it is
close to 1, so it changes confidence but not the most-probable option.

Consequences:

- **~1.3M dedicated parameters**, independent of the LM head's token distribution. It can
  assign arbitrary scores to the codes and separate options more flexibly than a constrained
  gather.
- **More capacity to separate options**, at the cost of a new trainable surface and a
  slightly larger calibration target.
- **Symbol binding is the readout's job.** Option j is identified *only* by its code
  $\gamma_j$; the backbone must associate each code token with the text of its option — the
  multiple-choice symbol-binding problem. A permutation of the options changes which of the
  255 logits corresponds to which option, so **order robustness is entirely learned** (and
  measured, §8.5 of the paper).

### 2.3 Drex pointer head (the cost-referenced third case)

For completeness, the Drex *pointer* models (DLM / v1.5 / v2.0) score each option as a
scaled dot product of two hidden states inside one causal encoding:

$$
z_j = \mathbf{k}_j^{\top} \mathbf{q} / \sqrt{d_p}, \qquad p = \mathrm{softmax}(z/T),
$$

where $\mathbf{q} = W_q h_{\texttt{<decide>}}$ (the query at the question's `<decide>`
token) and $\mathbf{k}_j = W_k h_{\texttt{</opt>}}$ (the key at each option's `</opt>`
token). The state is encoded **once** and shared by every question through a **block-causal
mask** (a question block attends to the state but never to another question block), so
question independence is structural. $d_p = 256$; the head is ~1.0–2.1M params, <0.03% of
the backbone. JAH's causal + shared-prefix design is architecturally closer to this
pointer model than to Drex Max, even though JAH's readout is an LM-head gather rather than a
pointer dot product.

### 2.4 Side-by-side

| | JAH gather | Drex Max `W_r` | Drex pointer |
|---|---|---|---|
| Logit source | LM head ∩ code tokens | new `W_r` on `h_<ans>` | `k_j^T q / √d_p` |
| New params | 0 | ~1.3M | ~1.0–2.1M |
| Codes | 52 | 255 | unbounded |
| Read position | last token | `<ans>` | `<decide>` × `</opt>` |
| Option identity | code token (in vocab) | code token (in vocab) | `</opt>` hidden state |
| Symbol binding | required | required (explicit) | not required (text is the key) |

### 2.5 The axes that actually differ

1. **Where the logits come from.** JAH re-uses the pre-trained LM head (a *constrained
   decoder*); Drex Max adds a *classifier head*. This is the deepest architectural split.
   JAH: zero new params, no new overfitting surface, but the code logits carry the LM's
   letter prior. Drex Max: 1.3M new params, more flexible option separation, a new
   calibration target.

2. **Attention direction.** JAH is fully causal. Drex Max makes 16 of 64 layers
   **non-causal** (every 4th full-attention layer) so the context and the start of the
   option list can attend to every later token, including all options and the instruction —
   the options and context are compared **bidirectionally** before the answer. A causal
   model (JAH, pointer) cannot. The paper notes that the same weights with the original
   causal mask do **not** reproduce the intended scores — the non-causal mask is load-bearing.

3. **Cost model — opposite.**
   - **JAH: state is expensive, questions are cheap.** The shared `STATE:` prefix is
     prefilled **once**; the k questions of a request (and the next steps of an episode)
     reuse it. `option_logits_cached` appends right-padded suffixes onto the cached prefix —
     exact under causality, since a token's hidden state never depends on later tokens.
   - **Drex Max: every question re-encodes the state.** Because the full-attention layers
     are non-causal, the state's representation depends on the question, so it **cannot be
     cached across questions** (paper §5.3). The state is paid once *per question*. For
     short contexts the gap is small (launch median 183 ms vs 151 ms for v2.0); for many
     questions over a long document, the shared-cache designs (JAH, pointer) stay cheaper.

4. **Order robustness — two routes, same goal.** Both readouts identify options by a code
   token, so a permutation re-wires which logit maps to which option. JAH buys robustness
   **data-side** (options are shuffled at train time, `permuted()`). Drex buys it **loss-side**
   with an explicit symmetric-KL permutation-consistency term `λ_π` and a state-shuffle term
   `λ_s` (eq 8). Paper ablation: removing the consistency terms raises the random-permutation
   flip rate 4.2% → 6.3% (K≥6: 13.0% → 17.8%) with no accuracy change — i.e. Drex pays for
   robustness explicitly; JAH gets it implicitly.

5. **Multimodality.** JAH is text-only. Drex Max keeps the backbone's ViT image encoder;
   image tokens enter ahead of the state, so every question in a request can attend to every
   image. This is what lifts image-only Decision Bench accuracy 63.3% → 98.3%.

---

## 3. Training objective (the RLAF stage)

All Drex models share the data and the supervised objective, then are trained with
**RLAF**. JAH's objective family is listed alongside so the differences are explicit.

### 3.1 Stage 1 — supervised fine-tuning

For predicted logits $z \in \mathbb{R}^K$, target $t \in \Delta^{K-1}$ (one-hot, or a
teacher distribution), and the correct-level index $y$ (score questions only), the
per-question loss is (paper eq 6)

$$
\ell_{\mathrm{sup}}(z) = -\sum_j t_j \log p_j \;+\; \lambda_{\mathrm{ord}}\,
\mathbb{1}[\tau{=}\texttt{score}]\,\mathrm{D}_{\mathrm{C}}^{2}(p, e_y),
\qquad p = \mathrm{softmax}(z),
$$

where $\mathrm{D}_{\mathrm{C}}^{2}$ is the **squared Cramér distance** (eq 7) between $p$
and the one-hot $e_y$ whose cumulative distribution is $\mathbb{1}[i \ge y]$ — the ranked
probability score from forecast verification:

$$
\mathrm{D}_{\mathrm{C}}^{2}(p, q) = \frac{1}{K-1}\sum_{i=1}^{K-1}
\Bigl(\textstyle\sum_{j \le i} p_j - \sum_{j \le i} q_j\Bigr)^2.
$$

Two consistency regularizers (eq 8) add robustness for backbones that cannot isolate
options architecturally, with $\overline{\mathrm{KL}}(a,b) = \tfrac12[\mathrm{KL}(a\|b) +
\mathrm{KL}(b\|a)]$:

$$
\ell_{\pi} = \overline{\mathrm{KL}}\big(p(\cdot\mid S, b),\ \pi^{-1} p(\cdot\mid S, \pi b)\big),
\qquad
\ell_{s} = \mathrm{KL}\big(\mathrm{sg}[p(\cdot\mid S, b)] \,\|\, p(\cdot\mid \tilde S, b)\big),
$$

where $\pi$ is a random option permutation, $\tilde S$ is the state with its sentences/fields
shuffled, and `sg` stops gradients. The Stage-1 objective (eq 9) is

$$
\mathcal{L}_{\mathrm{SFT}} = \mathbb{E}\big[\ell_{\mathrm{sup}} + \lambda_{\pi}\ell_{\pi} +
\lambda_{s}\ell_{s}\big].
$$

The data is **1,150,460** verified text decision records over **11** domains (mostly
teacher-distilled, plus open-source and verifiable procedurally-generated data). Options of
every choice question with $K \ge 3$ are **shuffled each time** the record is seen; in a
random fraction of records a **none** or **distractor** option is inserted so the model
learns not to force a choice.

> **JAH counterpart.** `decision_model.decision_loss` implements the supervised half as
> `soft_ce` ($-\sum q_k \log p_k$) and `soft_ce_brier` (adds `brier_weight·Σ(p_k−q_k)²`, a
> proper scoring rule that helps calibration). JAH's position debiasing is data-side
> (`permuted()`), **not** an explicit `λ_π` consistency loss — the two routes from §2.5(4).

### 3.2 Stage 2 — RLAF (pointer models)

One-hot labels train over-confidence, so RLAF replaces them with an **adjusted target**
$\tilde q \in \Delta^{K-1}$: a teacher proposes a distribution $q^{\mathrm{teacher}}$, which
is then adjusted by the **verified outcome** $y^\star$ (gold label, execution result, or
game-engine output) so that $\tilde q$ is consistent with what is known to be true while
retaining the teacher's uncertainty over the remaining options. The reward is a **proper
scoring rule** applied to the prediction and $\tilde q$ (eq 10):

$$
R(p, \tilde q) = \sum_j \tilde q_j \log p_j
\;+\; w_{\mathrm{sph}}\, \frac{\tilde q^{\top} p}{\lVert p \rVert_2}
\;-\; w_{\mathrm{cr}}\, \mathbb{1}[\tau{=}\texttt{score}]\, \mathrm{D}_{\mathrm{C}}^{2}(p, \tilde q),
$$

with $w_{\mathrm{sph}} = 0.5$ and $w_{\mathrm{cr}} = 1$. Every term is a proper scoring
rule, so $\mathbb{E}[R]$ is maximized at $p = \tilde q$ (eq 11) — the objective pushes
**accuracy and calibration together** rather than trading one for the other:

$$
\max_{\theta}\ \mathbb{E}\big[R(p_{\theta}, \tilde q)\big].
$$

It is optimized by **direct differentiation**, which makes it a *deterministic* loss on soft
targets (cross-entropy w.r.t. $\tilde q$ plus spherical and squared-Cramér terms); the
"reinforcement" is in how $\tilde q$ is built. A REINFORCE estimator was implemented but did
not outperform direct differentiation.

### 3.3 Specialization stage (v2.0 only)

For a weak task family, an extra stage runs **after** SFT and **before** RLAF. It trains
**LoRA adapters** (not all weights) on (1) solver-verified decisions in the family, (2)
difficult mined examples, and (3) replayed earlier-stage examples, and adds a **KL penalty**
$\mathrm{KL}(p_{\mathrm{ref}} \| p_\theta)$ where $p_{\mathrm{ref}}$ is the distribution at
the stage's start, to prevent drift. The checkpoint is selected on a held-out split of the
stage's own data, not on a benchmark.

> **JAH counterpart.** JAH's DPO and policy-gradient losses (`dpo`, `pg` in
> `decision_loss`) and its GRPO/OPD stages (`grpo_online.py`, `grpo_trl.py`, `opd.py`)
> play the same "push beyond one-hot SFT" role. JAH's `pg` is an exact
> expected-reward policy gradient with a KL-to-reference term
> `$-\sum p_k Q_k + \text{kl\_weight}\,\mathrm{KL}(p\|p_{\mathrm{ref}})$` — structurally
> the same KL-anchored form as the Max RLAF below.

### 3.4 RLAF for Drex Max (KL-regularized)

For Max, the verified outcome and a **judge** enter as *rewards* rather than being merged
into a target. Let $\pi_\theta(a\mid x) = p_{\theta,a}$ be the policy (eq 5), $\pi_0$ the
frozen SFT model, $y^\star$ the verified outcome (when one exists), and $q^J \in
\Delta^{K-1}$ the distribution produced by a **judge** — here the **generative Qwen3.8-27B
at high reasoning effort**, i.e. the generative form of Max's *own* base (not a stronger
model). The judge contributes a second opinion by generating token-by-token with long
chain-of-thought, which a single-pass decider cannot. The reward for answer $a$ (eq 12) is

$$
r(a) = \rho\, \mathbb{1}[a = y^\star] \;+\; \eta \log q^{J}_{a},
$$

with $\rho = 0$ when no verified outcome exists. RLAF maximizes (eq 13)

$$
J(\theta) = \mathbb{E}_x\Big[\sum_{a \in O} \pi_\theta(a\mid x) r(a)
\;-\; \beta\, \mathrm{KL}\big(\pi_\theta(\cdot\mid x) \,\|\, \pi_0(\cdot\mid x)\big)\Big]
\;-\; \lambda_\pi \mathbb{E}[\ell_\pi] \;-\; \lambda_s \mathbb{E}[\ell_s],
$$

keeping two of the SFT consistency losses. $\beta$ controls how far the policy may move from
the SFT model: larger $\beta$ stays closer to $\pi_0$; smaller $\beta$ weights the judge and
the verified outcome more. For Max, $\rho = 0.2$, $\eta = 0.05$ (paper Table 11).

Two differences from the pointer RLAF: the KL term anchors the policy to $\pi_0$, and the
verified outcome + judge enter as *rewards* rather than as a merged target $\tilde q$. The
paper notes (untested) that a judge that over-weights a wrong option can pull the optimal
policy toward it, and that judge-alignment can sharpen the policy toward over-confidence —
a plausible contributor to Max's mild over-confidence (§8.4).

**Contamination filtering.** No Decision Index / JevBench evaluation item is used for
training, and records whose state exactly matches, or shares more than two 8-word sequences
with, any benchmark example are dropped. The filters miss paraphrased items and benchmark
knowledge distilled through teacher models trained on the benchmarks.

### 3.5 Calibration / temperature (both)

Drex applies a temperature to the logits before the final softmax (eqs 2, 5). The shipped
value $T_{\mathrm{dep}}$ is 1 for the released pointer heads (no scaling) and **1.009** for
Max, fitted after the last stage by minimizing negative log-likelihood on held-out training
records (eq 14):

$$
T = \arg\min_{T>0} \sum_{(z,y) \in \mathcal{D}} -\log\big(\mathrm{softmax}(z/T)\big)_{y}.
$$

JAH does the same with `calibrate.py`, fitting a per-(question-kind, option-count)
temperature and applying it in `serve_jev.Engine.temperature(kind, k)`; the logprob
decider (`decision/llm_logprob.py`) applies the same calibration table so training,
logprob, and serving stay byte-consistent.

---

## 4. Inference / serving stack

### 4.1 Drex (paper §9)

- **Two-step forward.** One request = a **state pass** (cost scales with context length —
  131k tokens ≈ 2.7 s, the 2-second target) plus a fast **question pass**. With a **cached
  114–117k state, each subsequent question costs ~21.6 ms**, faster than a fresh 2k
  context (35.6 ms) — this is the pointer-model shared-encoding advantage.
- **Shared prefill.** Bit-identical to the stateless path; 6–8× at 8k tokens, 11–12× at
  32k with 16 questions. Not used below ~1k tokens.
- **CUDA Graphs.** −2/3 latency at 1k tokens; negligible at 6–8k.
- **Kernel changes.** Each is gated by a **paired-bootstrap accuracy gate** before being
  adopted; FP8 vs bf16 shows **0.00** accuracy change.
- **Measured.** Median end-to-end latency: Max **183 ms**, v2.0 151 ms, v1.5 145 ms,
  Jev 196 ms, GPT-6 Luna 326 ms, GPT-6 Sol ~2.0 s, Claude Opus 5.5 ~3.5 s. Cost per 1k
  decisions: Max / v2.0 **$0.06**, v1.5 / Jev $0.04, Luna $0.10, Sol $3.33, Opus $10.17.

### 4.2 JAH (`training/serve_jev.py`, `decision_model.py`)

- **Shared prefix, exact under causality.** The k questions of a request share one prefill
  of their common token prefix (the whole `STATE:` block). `option_logits_cached` appends
  right-padded suffixes onto the cached prefix — exact because a causal token's hidden state
  never depends on later tokens; `prefill` chunks the prefix, `expand_cache`/`clone_cache`/
  `crop_cache` manage the per-row caches. `JEV_SERVE_NO_CACHE=1` forces the stateless path.
- **Session cache.** `SessionCache` is an LRU of `(token prefix, cache snapshot)` pairs,
  bounded by total cached tokens and entry count, so the **next step of an episode**
  prefills only its delta. `Engine._probe_cache` learns once whether the cache is croppable
  and its bytes-per-token, and validates croppability numerically against a fresh prefill.
- **Batched suffix pass / fuse_multi.** Several questions on one state run their suffixes in
  one batched pass; on a croppable cache, `fuse_multi` runs the prefix delta together with
  the **longest** question's suffix in a single forward, then crops the cache back.
- **Knockout for >52 options.** Chunks of 52, top-2 advance, a final read over the
  finalists; several forward passes composed into one distribution (`--knockout-probs`).
- **Calibration.** Per-(kind, K) temperature from `calibrate.py`.
- **Profiling.** `JEV_SERVE_PROFILE=1` accumulates wall time per phase (encode, lookup,
  prefill, store, suffix) with device syncs, for benchmarking only.

### 4.3 Side-by-side

| | Drex Max | Drex pointer | JAH |
|---|---|---|---|
| State cost | per question (no cross-q cache) | once per request | once per request (prefix) + session cache across steps |
| Multi-question reuse | ✗ | ✓ (block-causal) | ✓ (shared prefix + `SessionCache`) |
| Batching | shared prefill / question pass | shared encoding | batched suffix pass, `fuse_multi` |
| Graphs / kernels | CUDA Graphs, kernel changes w/ accuracy gates | same stack | standard HF forward, `logits_to_keep` gather |
| Calibration | `T_dep` (eq 14) | `T=1` shipped | per-(kind,K) `calibrate.py` |
| > max options | 255-code head (no knockout) | unbounded | 52-code + knockout |
| Measured latency | 183 ms (Max) | 145–151 ms (v1.5/v2.0) | not reported in this repo (bench planned) |

---

## 5. Implications and open questions

1. **Cost vs accuracy is the core trade.** JAH's causal + shared-cache design is the
   cheaper choice for *many questions over one long state* (the agent-routing use case).
   Drex Max's bidirectional + per-question design is the more accurate choice for
   *knowledge-intensive / multi-step / image* decisions, at the price of re-encoding the
   state per question.
2. **Readout head choice is a zero-param vs 1.3M-param decision.** JAH's constrained gather
   has no new overfitting surface and inherits the LM's letter prior; Drex Max's `W_r` buys
   more flexible option separation and a dedicated calibration target. A natural JAH A/B
   would add a `--head {lm_gather, w_r}` switch in `train_decision.py` / `option_logits`.
3. **Order robustness is bought by two different levers** (data-side permutation vs
   loss-side `λ_π`). JAH currently uses the former; adopting the symmetric-KL consistency
   term would make the robustness explicit and testable (flip-rate under random permutation,
   as in Drex §8.5).
4. **The RLAF reward is the closest formal bridge** between the two: a proper scoring rule
   (log-likelihood + spherical + squared-Cramér) on a soft target, or the KL-anchored
   reward form (Max) that JAH's `pg` loss already mirrors. Porting Drex's adjusted-target
   construction (teacher distribution + verified outcome → $\tilde q$) is the single
   highest-leverage idea to lift calibration without a new head.
5. **Unverified in this repo.** Drex's exact per-stage hyperparameters, the kernel-level
   inference changes, and the enterprise-only Drex Max v1.1 / Drex for Finance are outside
   the open weights; JAH's serving bench (`bench_serving.py` referenced in the training
   README) is not yet implemented, so no JAH latency/cost figure is quoted above.

---

## 6. References

- "Towards Foundation Decision Models: Drex", arXiv:2611.07854v2 (Nace.AI). §5 (Drex Max
  architecture), §6 (training / RLAF, eqs 6–14), §8.4–8.8 (calibration, order robustness,
  ablations), §9 (deployment), §10 (discussion).
- This repo: `research/src/jevharness/training/decision_model.py` (readout, prefix caching),
  `training/data.py` (`render`, `permuted`), `training/serve_jev.py` (serving, `SessionCache`,
  knockout), `training/calibrate.py`, `training/train_decision.py`,
  `research/src/jevharness/decision/types.py` (`OPTION_LABELS`),
  `research/docs/TRAINING_RECIPES.md`.
