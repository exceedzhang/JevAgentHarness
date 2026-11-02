# ADR-002: One shared engine per process (deepcopy fix)

Status: Accepted
Date: 2026-11-02

## Context

The decision engine is stateful by design: it owns the in-memory session store, the journal file
handles, and the in-flight decision futures for the current step. That state must be **single per
process**, or routing decisions and their log rows diverge.

LiteLLM's proxy resolves the configured plugin instances (`classifier_plugin:
litellm_jah.classifier.jah_classifier`) into the config dict at load time — then deepcopies the
whole config (`_ConfigWithBaseline`). A deepcopy skips `__init__`, so the complexity router ended up
holding a *copy* of the plugin with a deep-copied `JAHRouter` underneath: an independent session
store and journal. Symptom: every step produced two routing rows, and the classifier's session
lifecycle (tier, streaks) was invisible to the pre/post-call hooks.

## Decision

- `_JahPluginBase.__deepcopy__` returns a shallow clone (`object.__new__(type(self))`) that **shares
  its `cfg` and its `router`** with the original. The config can copy the plugin freely; the engine
  never gets copied.
- Both module-level instances are constructed with no arguments (`jah_classifier = JahClassifier()`, `jah_routing_plugin = JahRoutingPlugin()`)
  each defaults to the lazy process-wide shared engine (``_shared_engine()``), so the two wiring
  styles (classifier plugin vs routing plugin) — and the pre-call, post-call and step-logger callbacks —
  agree by construction, not by accident. An explicit ``cfg`` or ``router`` builds a dedicated engine
  (unit tests use this).
- A regression test (`test_deepcopying_a_plugin_shares_its_engine`) deepcopies both plugin classes
  and asserts engine identity.

## Alternatives Considered

- **Build the engine lazily inside `classify()`/`run()`**: a fresh engine per call would silently
  fork the session store; a process-global lazy singleton would add import-order fragility for no
  benefit. Rejected.
- **Install a global `copy.deepcopy` override for `JAHRouter`**: too invasive for a plugin; other
  users of the venv would inherit it. Rejected.
- **Reconstruct state in `__setstate__`**: reconstructing a *second* working engine is exactly the
  bug being fixed. Rejected.

## Consequences

- The journal, session store, and decision futures are process-level singletons; a multi-replica
  proxy still needs a shared `SessionStore` (see the README's Sessions section) — deepcopies within
  one process are no longer a hazard.
- If LiteLLM ever changes how it copies configs (e.g. `model_copy(deep=True)` on a pydantic field
  holding the plugin), the regression test is the tripwire.
