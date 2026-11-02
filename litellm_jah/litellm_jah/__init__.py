"""JevAgentHarness for the LiteLLM proxy: the standalone proxy's decision points as LiteLLM plugins. See README.md.

Scope (mirrors the standalone proxy): routing (cascade / shadow / guard / off), the finish check, the stop
guard / shadow / trigger, tool select (cascade / assist / guard / shadow), reflection and the action guard —
wired as a complexity-router classifier / routing plugin (pre-routing) plus pre-call, post-call and step-logger
callbacks. Every decision point and mode of the standalone proxy is ported; ``tool_params`` and the memory
points have no proxy equivalent (see points.py).
"""

from .backend import (
    Decision,
    DecisionBackend,
    JevBackend,
    NoneBackend,
    Question,
    RulesBackend,
    StepView,
    make_backend,
)
from .classifier import JahClassifier, JahRoutingPlugin, jah_classifier, jah_routing_plugin
from .config import JAHCfg
from .engine import JAHRouter, RouteResult
from .hook import JahFinishCheck, jah_finish_check
from .journal import Journal
from .logger import JahStepLogger, jah_step_logger
from .postcall import JahPostCall, jah_post_call
from .precall import JahPreCall, jah_pre_call
from .sessions import (
 InMemorySessionStore,
 RedisSessionStore,
 Session,
 SessionStore,
 make_store,
)
from .state import HARD, QUESTION, render_state, session_key, step_index

__all__ = [
    "Decision", "DecisionBackend", "JevBackend", "NoneBackend", "Question", "RulesBackend", "StepView",
    "make_backend", "JahClassifier", "JahRoutingPlugin", "jah_classifier", "jah_routing_plugin", "JAHCfg",
    "JAHRouter", "RouteResult", "Journal", "JahStepLogger", "jah_step_logger",
    "JahFinishCheck", "jah_finish_check", "JahPostCall", "jah_post_call", "JahPreCall", "jah_pre_call",
    "InMemorySessionStore", "RedisSessionStore", "Session", "SessionStore", "make_store",
    "HARD", "QUESTION", "render_state", "session_key", "step_index",
]
