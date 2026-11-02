"""The LiteLLM-facing plugin classes against a duck-typed routing context."""

from __future__ import annotations

from litellm_jah.classifier import JahClassifier, JahRoutingPlugin, jah_classifier, jah_routing_plugin
from litellm_jah.config import JAHCfg
from litellm_jah.engine import JAHRouter

from conftest import Ctx, FakeBackend, decision, msgs


def cls(p_hard=0.3, **cfg):
    c = JAHCfg(decision={"type": "rules"}, **cfg)
    be = FakeBackend(lambda n, s, q: decision(p_hard))
    return JahClassifier(cfg=c, router=JAHRouter(c, backend=be)), be


def test_classifier_returns_the_tier_label(run):
    async def go():
        cl, be = cls(p_hard=0.9)
        assert await cl.classify(Ctx(msgs(0))) == "COMPLEX"     # first step
        assert await cl.classify(Ctx(msgs(1))) == "COMPLEX"     # p(hard) >= tau
        cl2, _ = cls(p_hard=0.1)
        assert await cl2.classify(Ctx(msgs(1))) == "SIMPLE"     # p(hard) < tau
        await cl.close()
        await cl2.close()
    run(go())


def test_classifier_custom_labels(run):
    async def go():
        c = JAHCfg(decision={"type": "rules"}, strong_label="STRONG", fast_label="FAST")
        cl = JahClassifier(cfg=c, router=JAHRouter(c, backend=FakeBackend(lambda n, s, q: decision(0.1))))
        assert await cl.classify(Ctx(msgs(1))) == "FAST"
        await cl.close()
    run(go())


def test_classifier_declines_on_an_engine_failure(run, monkeypatch):
    async def go():
        cl, _ = cls()
        async def boom(messages, metadata=None):
            raise ValueError("broken")
        monkeypatch.setattr(cl.router, "decide", boom)
        assert await cl.classify(Ctx(msgs(1))) is None
        await cl.close()
    run(go())


def test_classifier_uses_explicit_session_id_from_metadata(run):
    async def go():
        cl, be = cls(p_hard=0.9, stick_strong=1)
        await cl.classify(Ctx(msgs(1), metadata={"jah_session": "s1"}))
        assert len(cl.router.store._d) == 1 and "s1" in cl.router.store._d
        await cl.close()
    run(go())


def test_routing_plugin_narrows_to_the_tier(run):
    async def go():
        c = JAHCfg(decision={"type": "rules"}, strong_models=["gemini/argon"], fast_models=["gemini/flash-lite"])
        rp = JahRoutingPlugin(cfg=c, router=JAHRouter(c, backend=FakeBackend(lambda n, s, q: decision(0.9))))
        both = ["gemini/argon", "gemini/flash-lite"]
        ctx = Ctx(msgs(1), candidate_models=both)
        await rp.run(ctx)
        assert ctx.candidate_models == ["gemini/argon"]
        assert ctx.signals["jah-router"]["tier"] == "strong"
        rp2 = JahRoutingPlugin(cfg=c, router=JAHRouter(c, backend=FakeBackend(lambda n, s, q: decision(0.1))))
        ctx2 = Ctx(msgs(1), candidate_models=both)
        await rp2.run(ctx2)
        assert ctx2.candidate_models == ["gemini/flash-lite"]
        await rp.close()
        await rp2.close()
    run(go())


def test_routing_plugin_fail_safe_keeps_strong(run):
    async def go():
        c = JAHCfg(decision={"type": "rules"}, on_error="strong",
                   strong_models=["gemini/argon"], fast_models=["gemini/flash-lite"])

        def down(n, s, q):
            raise RuntimeError("down")

        rp = JahRoutingPlugin(cfg=c, router=JAHRouter(c, backend=FakeBackend(down)))
        ctx = Ctx(msgs(1), candidate_models=["gemini/argon", "gemini/flash-lite"])
        await rp.run(ctx)
        assert ctx.candidate_models == ["gemini/argon"]  # decision error -> strong tier
        await rp.close()
    run(go())


def test_routing_plugin_leaves_pool_untouched_without_model_lists(run):
    async def go():
        rp = JahRoutingPlugin(cfg=JAHCfg(decision={"type": "rules"}),
                              router=JAHRouter(JAHCfg(decision={"type": "rules"}),
                                               backend=FakeBackend(lambda n, s, q: decision(0.9))))
        ctx = Ctx(msgs(1), candidate_models=["a/x", "b/y"])
        await rp.run(ctx)
        assert ctx.candidate_models == ["a/x", "b/y"]
        await rp.close()
    run(go())


def test_run_survives_a_context_without_router_attributes(run):
    async def go():
        rp = JahRoutingPlugin(cfg=JAHCfg(decision={"type": "rules"}, strong_models=("openai/strong",),
                                         fast_models=("openai/fast",)))

        class Bare:  # a context that only carries the messages
            raw_messages = msgs(1)
            structured_messages = msgs(1)
            metadata = {"session_id": "s1"}

        ctx = Bare()
        assert await rp.run(ctx) is ctx  # no exception, context returned untouched
        await rp.close()
    run(go())


def test_default_construction_shares_the_process_engine():
    # a config that names the CLASS instead of the singleton must not build a second engine
    a = JahClassifier()
    b = JahRoutingPlugin()
    assert a.router is b.router is jah_classifier.router is jah_routing_plugin.router
    assert a.cfg is jah_classifier.cfg is b.cfg


def test_an_explicit_engine_is_dedicated(run):
    async def go():
        c = JAHCfg(decision={"type": "rules"})
        d = JahClassifier(cfg=c)
        assert d.router is not jah_classifier.router and d.cfg is c
        e = JahClassifier(router=jah_classifier.router)
        assert e.router is jah_classifier.router # an explicit router is used as-is
        await d.close()
    run(go())
