"""Decision points and their modes: where the decision model acts in the proxy, and how.

Ported from ``jevagentharness.points`` (MIT). The standalone proxy offers five points (routing, tool select,
stop, reflection, action); this plugin offers all of them, in every mode, including the fourth routing mode
(``guard``: the fast tier writes the step, and the decision model checks the reply after the call).
``tool_params`` and the memory points are not offered by a proxy at all.

``shadow`` asks the question and logs the answer next to what the agent did, and changes nothing — except at
the stop point, where a shadowed finish check re-runs the checked reply on the strong tier and only logs the
verdict (the agent's reply is not held). A decision-model failure or timeout at a point falls back to that
point's LLM-only behaviour, so it never stops the agent.

Responses and requests are LiteLLM shapes (``choices[0].message.tool_calls``, a list of tool dicts, message
dicts); the helpers here duck-type them so the checks unit-test without a LiteLLM install.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from .backend import DecisionBackend, Question, RulesBackend, StepView

POINTS = ("routing", "tool_select", "stop", "reflection", "action")
MODES: dict[str, tuple[str, ...]] = {
"routing": ("off", "shadow", "cascade", "guard"),
"tool_select": ("off", "shadow", "assist", "cascade", "guard"),
"stop": ("off", "shadow", "strong", "guard", "trigger"),
"reflection": ("off", "shadow", "trigger", "always"),
"action": ("off", "shadow", "guard"),
}
# names accepted for a mode that equals another one in the proxy (docs/modes.md)
ALIASES = {("routing", "trigger"): "cascade"}
KEYS: dict[str, set[str]] = {
"routing": {"mode", "tau"},
"tool_select": {"mode", "tau", "k", "min_tools", "max_options", "tier", "force"},
"stop": {"mode", "tau", "scope", "submit"},
"reflection": {"mode", "tau", "every"},
"action": {"mode", "tau", "prefilter", "tools", "patterns", "on_error"},
}
# default thresholds. Tool choice is delegated only when the decision model is (almost) certain until `jah fit`
# has fitted a threshold on your own runs: the paper's fitted tool threshold was 1.00 (Jev, Gemini 3.5 Flash),
# and the same delegation at a fixed 0.5 lost 21.5 points of task success.
TAU = {"tool_select": 0.99, "stop": 0.9, "reflection": 0.8, "action": 0.5}

PRESETS: dict[str, dict[str, dict[str, Any]]] = {
"route": {}, # the default: routing cascade + finish check
"paper": {"tool_select": {"mode": "cascade"}}, # + the paper's tool-choice cascade (fit tau!)
"safe": {"action": {"mode": "guard"}}, # + a second look at risky actions
"shadow": {"tool_select": {"mode": "shadow"}, "stop": {"mode": "shadow"},
"reflection": {"mode": "shadow"}, "action": {"mode": "shadow"}}, # log everything, for `jah fit`
}


@dataclass(frozen=True)
class PointCfg:
    point: str
    mode: str = "off"
    tau: float = 0.5
    k: int = 5 # tool_select assist: keep the k likeliest tools
    min_tools: int = 12 # tool_select assist: only shortlist requests with at least this many tools
    max_options: int = 100 # tool_select: options per question (more tools: a two-round knockout)
    tier: str | None = None # tool_select cascade: the tier that writes the arguments (default: the fast tier)
    force: bool = True # tool_select cascade: force the tool (tool_choice); false = only pick the tier
    scope: str = "fast" # stop guard / shadow: check final answers of the fast tier, or of every tier
    every: int = 5 # reflection always: every N steps
    prefilter: str = "risky" # action: risky (pattern match) | command (command-like tools) | all
    tools: tuple[str, ...] = () # action: extra regexes for command-like tool names
    patterns: tuple[str, ...] = () # action: extra regexes for risky call text
    submit: tuple[str, ...] = () # stop: regexes on a call's text that mark it as the agent ending its turn
    on_error: str = "accept" # action guard: what a decision-model failure does (accept | recheck)

    @property
    def on(self) -> bool:
        return self.mode != "off"


@dataclass(frozen=True)
class Decisions:
    routing: PointCfg
    tool_select: PointCfg
    stop: PointCfg
    reflection: PointCfg
    action: PointCfg

    def modes(self) -> dict[str, str]:
        return {p: getattr(self, p).mode for p in POINTS}

    def only_routing(self) -> bool:
        """True for the default behaviour (routing + finish check): the step record stays as it always was."""
        return all(getattr(self, p).mode == "off" for p in ("tool_select", "reflection", "action")) \
            and self.stop.mode in ("off", "strong")


def _mode_name(v: Any) -> Any:
    return "off" if v is False else v # YAML 1.1 reads a bare `off` as false


def _norm_point(point: str, v: Any, where: str) -> dict[str, Any]:
    v = _mode_name(v)
    if isinstance(v, str):
        v = {"mode": v}
    if not isinstance(v, Mapping):
        raise ValueError(f"{where}.{point}: expected a mapping like {{mode: shadow}}, got {v!r}")
    d = dict(v)
    bad = set(d) - KEYS[point]
    if bad:
        raise ValueError(f"{where}.{point}: unknown keys {sorted(bad)} (allowed: {sorted(KEYS[point])})")
    if "mode" not in d:
        raise ValueError(f"{where}.{point}.mode is required (one of {', '.join(MODES[point])})")
    d["mode"] = _mode_name(d["mode"])
    m = ALIASES.get((point, str(d["mode"])), str(d["mode"]))
    if m not in MODES[point]:
        raise ValueError(f"{where}.{point}.mode must be one of {', '.join(MODES[point])}, got {d['mode']!r}")
    d["mode"] = m
    if "tau" in d and d["tau"] is not None:
        t = float(d["tau"])
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"{where}.{point}.tau must be in [0, 1], got {d['tau']!r}")
        d["tau"] = t
    for k, lo, hi in (("k", 1, None), ("min_tools", 0, None), ("every", 1, None), ("max_options", 2, 255)):
        if k in d:
            try:
                d[k] = int(d[k])
            except (TypeError, ValueError):
                raise ValueError(f"{where}.{point}.{k} must be an integer, got {d[k]!r}") from None
            if d[k] < lo or (hi is not None and d[k] > hi):
                raise ValueError(f"{where}.{point}.{k} must be in [{lo}, {hi or 'inf'}]"
                                 + (" (Jev's limit is 255 options)" if k == "max_options" else ""))
    if "force" in d:
        d["force"] = bool(d["force"])
    if d.get("tier") is not None:
        d["tier"] = str(d["tier"])
    for k, allowed in (("scope", ("fast", "all")), ("prefilter", ("risky", "command", "all")),
                       ("on_error", ("accept", "recheck"))):
        if k in d and d[k] not in allowed:
            raise ValueError(f"{where}.{point}.{k} must be one of {', '.join(allowed)}, got {d[k]!r}")
    for k in ("tools", "patterns", "submit"):
        if k in d:
            vals = d[k] if isinstance(d[k], (list, tuple)) else [d[k]]
            for r in vals:
                try:
                    re.compile(str(r))
                except re.error as e:
                    raise ValueError(f"{where}.{point}.{k}: bad regex {r!r}: {e}") from e
            d[k] = tuple(str(r) for r in vals)
    return d


def validate(decisions: Mapping[str, Any] | None, where: str = "decisions") -> dict[str, dict[str, Any]]:
    """Check a ``decisions:`` block; returns it normalized (string shorthands expanded, aliases resolved)."""
    out: dict[str, dict[str, Any]] = {}
    for point, v in (decisions or {}).items():
        if point not in KEYS:
            raise ValueError(f"{where}: unknown decision point {point!r} (one of {', '.join(POINTS)})")
        out[point] = _norm_point(point, v, where)
    return out


def check_preset(name: Any, where: str = "preset") -> str | None:
    if name in (None, ""):
        return None
    if name not in PRESETS:
        raise ValueError(f"{where}: unknown preset {name!r} (one of {', '.join(PRESETS)})")
    return str(name)


def _merge(base: dict[str, dict[str, Any]], over: Mapping[str, Any]) -> None:
    for point, v in validate(over).items():
        base[point] = {**base[point], **v}


def resolve(router_tau: float, verify_finish: bool = True, preset: str | None = None,
            decisions: Mapping[str, Any] | None = None, routing_mode: str = "cascade") -> Decisions:
    """The decision points of the plugin: the defaults (routing at ``router_tau`` in ``routing_mode``; the
    finish check when ``verify_finish``), then the ``preset``, then the ``decisions`` block. Later layers win."""
    base: dict[str, dict[str, Any]] = {p: {} for p in POINTS}
    base["routing"]["mode"] = routing_mode
    base["stop"]["mode"] = "strong" if verify_finish else "off"
    if preset:
        _merge(base, PRESETS[preset])
    _merge(base, decisions or {})
    pts = {}
    for p in POINTS:
        d = dict(base[p])
        if d.get("tau") is None:
            d["tau"] = router_tau if p == "routing" else TAU[p]
        pts[p] = PointCfg(point=p, **d)
    return Decisions(**pts)


# ---------------------------------------------------------------------------------------------- questions and notes
_WATCH = "You are watching an AI agent work on the task below, one tool call at a time. "
NO_TOOL = "reply_without_tool"
NO_TOOL_DESC = "Call no tool: reply in text (the final answer, a report, or a question to the user)"
TOOL_TEXT = _WATCH + "Which tool should the agent call next?"

ROUTE_GUARD = Question(
    id="fast_step_ok",
    text=(_WATCH + "A fast, cheap model wrote the agent's next step, shown at the end as PROPOSED NEXT STEP. Decide "
          "whether to keep it. 'keep': a sensible next move that follows from what the agent knows. 'redo': the step "
          "looks wrong, careless or premature (a wrong command or edit, a repeated failed attempt, or a claim that "
          "the task is done without checking), so a stronger model should write this step instead."),
    options=["keep", "redo"],
    descriptions=["The proposed step is a sensible next move", "A stronger model should write this step instead"])
STOP = Question(
    id="task_done",
    text=(_WATCH + "The agent now wants to end its turn with the reply shown at the end as PROPOSED FINAL REPLY, "
          "with no further tool calls. Is ending here right: is the task actually complete and the reply "
          "correct (or does the agent rightly need input from the user)?"),
    options=["done", "not_done"],
    descriptions=["Ending the turn here is right: the task is complete and the reply is correct, or the agent "
                  "rightly asks the user for input",
                  "Work remains, a result was not checked, or the reply is wrong or incomplete"])
STOP_TRIGGER = Question(
    id="already_done",
    text=_WATCH + "Has the task already been fully completed, so the agent should stop now and give its final "
                 "answer?",
    options=["yes", "no"],
    descriptions=["The task is complete: the agent should stop calling tools and answer",
                  "Work remains"])
REFLECT = Question(
    id="needs_reflection",
    text=(_WATCH + "Should the agent pause and re-check before its next action: is it stuck or off track, "
          "repeating itself, or likely to make a wrong action or give a wrong final answer next?"),
    options=["yes", "no"],
    descriptions=["The agent should pause and re-check its progress and plan before acting",
                  "The agent is making progress; no pause needed"])
ACTION = Question(
    id="action_risk",
    text=(_WATCH + "The agent proposes the action shown at the end as PROPOSED ACTION. Could it be irreversible "
          "or risky: delete or overwrite data that is not easy to restore, rewrite version-control history or "
          "force-push, change systems outside the task's workspace (deploy, publish, send messages, spend "
          "money), or weaken security? Answer 'risky' only if running it without a second look could cause "
          "harm that is hard to undo."),
    options=["safe", "risky"],
    descriptions=["Running it is fine or easy to undo", "It could cause harm that is hard to undo"])

NOTE = "[Note from the harness] "
REFLECT_NOTE = (NOTE + "Before your next action, pause and check your progress: re-read the task, look for a "
               "mistake or a wrong assumption so far, and decide the best next step. Then continue.")
STOP_TRIGGER_NOTE = (NOTE + "The task may already be complete. Check the result against the task. If it is "
                    "complete, stop calling tools and give your final answer now; otherwise continue.")
STOP_REJECT_NOTE = (NOTE + "The task may not be complete yet, or the final reply may be wrong. Re-check the task "
                     "and the results before you end your turn.")
CHARS_PROPOSAL = 1500

MAX_ACTION_ASKS = 4 # proposed calls checked per reply (more: first four)
REDO_ORDER = ["action_guard", "tool_guard", "route_guard", "stop_guard", "finish_check"]
MOVABLE = {"routine", "guard", "routing_off", "shadow", "pinned"}

# request keys that are not valid acompletion parameters for a strong re-run (same exclusions as LiteLLM's
# skills hook)
_EXCLUDED = frozenset({"messages", "model", "tools", "metadata", "litellm_metadata", "container",
                       "user_api_key_dict",
                       # the tier deployment's credentials: a re-run is the strong model, with the strong
                       # credentials (cfg.strong_api_key/strong_api_base) or the environment, not the tier's
                       "api_key", "api_base", "api_version",
                       # LiteLLM/proxy plumbing: forwarding it makes the re-run reuse the original request's
                       # logging object (and claim its success log), its call/trace ids, client, session, ...
                       "client", "caching", "shared_session", "proxy_server_request", "model_info",
                       "secret_fields", "use_in_pass_through", "use_litellm_proxy", "use_xai_oauth",
                       # client-controlled provider redirect: the re-run must keep the strong tier's own endpoint and
                       # provider, never one the client requested (otherwise the strong credentials go to the client's
                       # endpoint: credential exfiltration)
                       "base_url", "custom_llm_provider", "extra_headers", "extra_body", "api_type",
                       "azure_ad_token", "organization",
                       "allow_client_keepalive_override", "complexity_router_config"})


def pass_through(request_data: dict) -> dict[str, Any]:
    """The original request's completion parameters, minus the keys a re-run sets itself and LiteLLM's
    internal plumbing (``litellm_*`` / ``_*`` keys, the logging object, call/trace ids, credentials, ...).
    A re-run is a fresh top-level call: it must not inherit the original request's logging context, or the
    original request's success log never fires."""
    return {k: v for k, v in request_data.items()
            if k not in _EXCLUDED and not k.startswith(("litellm_", "_"))}


# ----------------------------------------------------------------------------------------------- response helpers
def tc_name(c: Any) -> str:
    f = c.get("function") if isinstance(c, dict) else getattr(c, "function", None)
    if f is None:
        return ""
    return str(f.get("name") if isinstance(f, dict) else getattr(f, "name", "") or "")


def tc_args(c: Any) -> str:
    f = c.get("function") if isinstance(c, dict) else getattr(c, "function", None)
    if f is None:
        return ""
    a = f.get("arguments") if isinstance(f, dict) else getattr(f, "arguments", None)
    return str(a) if a is not None else ""


def resp_tool_calls(resp: Any) -> list[Any]:
    try:
        c = resp.choices[0]
        m = c.message if hasattr(c, "message") else c.get("message")
        tcs = m.tool_calls if hasattr(m, "tool_calls") else m.get("tool_calls")
        return list(tcs) if tcs else []
    except Exception: # noqa: BLE001
        return []


def resp_text(resp: Any) -> str:
    try:
        c = resp.choices[0]
        m = c.message if hasattr(c, "message") else c.get("message")
        t = m.content if hasattr(m, "content") else m.get("content")
        return str(t) if t is not None else ""
    except Exception: # noqa: BLE001
        return ""


def call_text(c: Any, n: int = 300) -> str:
    s = f"{tc_name(c)}({tc_args(c)})"
    return s if len(s) <= n else s[: n - 4] + " ..."


def tool_guard_note(resp: Any) -> str:
    calls = resp_tool_calls(resp)
    what = call_text(calls[0]) if calls else "a final reply without a tool call"
    return (NOTE + f"A check doubts the proposed next step `{what}`. Re-read the task and the latest results, "
                   "then make the right next call (the same one if it is right).")


def action_note(calls: list[Any]) -> str:
    what = "; ".join(f"`{call_text(c)}`" for c in calls)
    return (NOTE + f"The proposed action {what} may be irreversible or risky (it could delete data, rewrite "
                   f"history, or change systems outside the task). Check that it is needed and safe. If it is, "
                   f"make the same call again; if not, take a safer next step.")


def with_notes(messages: list[dict[str, Any]], notes: list[str]) -> list[dict[str, Any]]:
    """The messages with harness notes appended to the last message (a tool result or a user message), so the
    conversation keeps its shape on every API and a cached prefix stays a prefix."""
    if not notes:
        return messages
    text = "\n\n".join(notes)
    msgs = list(messages)
    if msgs and msgs[-1].get("role") in ("tool", "user"):
        m = dict(msgs[-1])
        prev = str(m.get("content") or "")
        m["content"] = (prev + "\n\n" if prev else "") + text
        msgs[-1] = m
    else:
        msgs.append({"role": "user", "content": text})
    return msgs


def proposal_text(resp: Any) -> str:
    calls = resp_tool_calls(resp)
    if calls:
        s = "; ".join(f"{tc_name(c)}({tc_args(c)})" for c in calls)
    else:
        s = f"message: {resp_text(resp)}"
    s = " ".join(s.split())
    if len(s) > CHARS_PROPOSAL: # keep the head and the tail: agents end long replies with the answer
        s = s[:300] + " ... " + s[-(CHARS_PROPOSAL - 305):]
    return s


def _canon(v: Any) -> Any:
    if isinstance(v, str):
        return " ".join(v.split())
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [_canon(x) for x in v]
    return v


def _args_obj(a: Any) -> Any:
    s = a if isinstance(a, str) else json.dumps(a)
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return s


def same_calls(a: list[Any], b: list[Any]) -> bool:
    """Two replies make the same calls: equal names and arguments, up to JSON formatting and runs of whitespace."""
    return len(a) == len(b) and all(tc_name(x) == tc_name(y) and _canon(_args_obj(tc_args(x))) ==
                                    _canon(_args_obj(tc_args(y))) for x, y in zip(a, b))


def ends_turn(resp: Any, submit: re.Pattern[str] | None) -> bool:
    """The reply ends the agent's turn: no tool call, or a call matching a ``stop.submit`` pattern."""
    calls = resp_tool_calls(resp)
    if not calls:
        return True
    return submit is not None and any(submit.search(f"{tc_name(c)}({tc_args(c)})") for c in calls)


def submit_re(cfg: PointCfg) -> re.Pattern[str] | None:
    return re.compile("|".join(f"(?:{p})" for p in cfg.submit)) if cfg.submit else None


def recheck_kind(before: Any, after: Any) -> str:
    """What a strong redo did with a checked reply: ``same`` calls (up to formatting), ``changed`` calls, or
    ``no_call`` (the redo replied without a tool call)."""
    after_calls = resp_tool_calls(after)
    if not after_calls:
        return "no_call"
    return "same" if before is not None and same_calls(resp_tool_calls(before), after_calls) else "changed"


def llm_choice(resp: Any) -> str:
    """What the LLM chose at the tool-choice point: its first tool call, or no tool."""
    calls = resp_tool_calls(resp)
    return tc_name(calls[0]) if calls else NO_TOOL


def tool_options(tools: list[Any], tool_choice: Any) -> tuple[list[str], list[str]] | None:
    """The tool-choice question's options for this request, or None when the choice is not open (no tools, the
    client forced a tool or forbade tools, or fewer than two options)."""
    if not tools or tool_choice == "none" or isinstance(tool_choice, dict):
        return None
    opts: list[str] = []
    descs: list[str] = []
    for t in tools:
        f = t.get("function") if isinstance(t, dict) else None
        name = str(f.get("name") or "") if f else ""
        if name and name not in opts:
            opts.append(name)
            first = next((ln.strip() for ln in str(f.get("description") or "").splitlines() if ln.strip()), "")
            descs.append(first[:160])
    if tool_choice != "required" and NO_TOOL not in opts:
        opts.append(NO_TOOL)
        descs.append(NO_TOOL_DESC)
    return (opts, descs) if len(opts) >= 2 else None


def used_tools(messages: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for m in messages:
        for c in m.get("tool_calls") or []:
            if isinstance(c, dict):
                f = c.get("function") or {}
                if f.get("name"):
                    out.add(str(f["name"]))
    return out


# ---------------------------------------------------------------------------------------------- action pre-filter
COMMAND_TOOLS = [
    r"bash", r"shell", r"exec", r"command", r"\bcmd\b", r"terminal", r"\brun", r"local_shell", r"apply_patch",
    r"write", r"edit", r"create", r"delete", r"remove", r"move", r"rename", r"patch", r"\bgit", r"deploy",
    r"publish", r"send", r"post", r"push", r"upload", r"sql", r"query", r"kubectl", r"docker", r"http", r"request",
    r"transfer", r"pay", r"cancel", r"refund", r"update", r"drop",
]
# a command position: the start of the text or of a JSON string, after ; & | ( ` or an escaped newline, or after
# sudo / xargs / exec (so `grep -rn rm .` is not a deletion)
_CMD = r"""(?:^|["';&|(`]|\\n|\bsudo\s+|\bxargs\s+(?:-\S+\s+)*|\bexec\s+)\s*"""
RISKY_PATTERNS = [
    # files
    _CMD + r"rm(?![\w.-])", _CMD + r"(?:rmdir|unlink|shred)\b", r"\bfind\b[^\n]*-delete\b",
    r"\bfind\b[^\n]*-exec\s+rm\b", r"shutil\.rmtree", r"os\.(remove|unlink|rmdir)\b", r"\*\*\* Delete File:",
    r"\btruncate\s+-s\s*0\b", r">\s*/dev/sd",
    # version control
    r"\bgit\s+push\b", r"\bgit\s+reset\s+--hard\b", r"\bgit\s+clean\s+-\w*f", r"\bgit\s+checkout\s+(--\s+)?\.(\s|$)",
    r"\bgit\s+restore\s+(--\S+\s+)*\.(\s|$)", r"\bgit\s+branch\s+-D\b", r"\bgit\s+stash\s+(drop|clear)\b",
    r"\bgit\s+rebase\b", r"\bgit\s+filter-(branch|repo)\b", r"\bgit\s+commit\s+[^\n]*--amend\b",
    r"\bgit\s+tag\s+-d\b", r"\bgit\s+update-ref\s+-d\b",
    # databases
    r"(?i:\b(drop|truncate)\s+(table|database|schema|index)\b)", r"(?i:\bdelete\s+from\b)",
    r"(?i:\balter\s+table\b[^\n]*\bdrop\b)", r"(?i:\bflushall\b)", r"(?i:\bdropdatabase\b)",
    # system
    r"\bmkfs(\.\w+)?\b", r"\bdd\s+[^\n]*\bof=", r"\bchmod\s+(-R|777)\b", r"\bchown\s+-R\b", r"\bsudo\b",
    r"\b(shutdown|reboot|halt|poweroff)\b", r"\bkill\s+-9\b", r"\b(pkill|killall)\b", r":\(\)\s*\{",
    r"\b(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(ba|z)?sh\b",
    # publishing, deploying, cloud and remote systems
    r"\bnpm\s+(publish|unpublish)\b", r"\btwine\s+upload\b", r"\bcargo\s+publish\b", r"\bgem\s+push\b",
    r"\bdocker\s+(push|system\s+prune|volume\s+rm|rm)\b", r"\bkubectl\s+(delete|apply|replace|scale|drain)\b",
    r"\bhelm\s+(uninstall|delete|upgrade)\b", r"\bterraform\s+(apply|destroy)\b",
    r"\baws\s+\S+\s+(delete|rm|terminate|remove)\S*",     r"\bgcloud\b[^\n]*\b(delete|destroy)\b",
    r"\baz\b[^\n]*\bdelete\b", r"\bgh\s+(pr\s+merge|release\s+(create|delete)|repo\s+delete)\b",
    r"(?i:\bcurl\b[^\n]*-X\s*(DELETE|PUT|POST|PATCH)\b)",
]
DESTRUCTIVE_TOOL_NAMES = r"(?i)(^|[_\W])(delete|remove|drop|destroy|purge|wipe|cancel|refund|transfer|pay|send|" \
                          r"publish|deploy|merge|force)([_\W]|$)"


class ActionFilter:
    """Cheap deterministic pre-filter: which proposed calls go to the decision model at all."""

    def __init__(self, cfg: PointCfg) -> None:
        self.mode = cfg.prefilter
        self.tools = re.compile("|".join(COMMAND_TOOLS + list(cfg.tools)), re.I)
        self.risky = re.compile("|".join(f"(?:{p})" for p in RISKY_PATTERNS + list(cfg.patterns)))
        self.destructive = re.compile(DESTRUCTIVE_TOOL_NAMES)

    def is_risky(self, c: Any) -> bool:
        return bool(self.destructive.search(tc_name(c)) or self.risky.search(tc_args(c) or ""))

    def candidates(self, calls: list[Any]) -> list[Any]:
        if self.mode == "all":
            return list(calls)
        if self.mode == "command":
            return [c for c in calls if self.tools.search(tc_name(c)) or self.is_risky(c)]
        return [c for c in calls if self.is_risky(c)]


# ---------------------------------------------------------------------------------------------- asking
@dataclass
class Ask:
    """One decision call (or a two-round knockout over many tools) and what came of it; one line of
    decisions.jsonl. ``label`` is what the LLM itself chose at this point when that is known, and ``fit_ok``
    says whether that choice was made without the decision model's influence (so `jah fit` may use it)."""

    point: str
    mode: str
    question: str
    options: list[str]
    tau: float | None = None
    act_option: str | None = None # the option whose probability is compared with tau
    p: dict[str, float] = field(default_factory=dict)
    model: str = ""
    calibrated: bool = False  # p is in the model card's calibrated space (backend.py applies its temperatures)
    truncated: bool = False  # the state was sent as a tail cut to max_state_chars (backend.py)
    state: str | None = None  # the exact state string sent (backend-provided, post-truncation); logged to the
    # decision row only when the config enables it - the GRPO training data is built from these rows
    cost_usd: float = 0.0
    latency_s: float = 0.0
    rounds: int = 1
    error: str | None = None
    acted: bool = False
    effect: str | None = None
    label: str | None = None
    label_tier: str | None = None
    fit_ok: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.p)

    @property
    def choice(self) -> str | None:
        return max(self.p, key=lambda k: self.p[k]) if self.p else None

    @property
    def confidence(self) -> float | None:
        return max(self.p.values()) if self.p else None

    def p_of(self, option: str | None = None) -> float:
        return float(self.p.get(option or self.act_option or "", 0.0))

    def record(self) -> dict[str, Any]:
        top = sorted(self.p.items(), key=lambda kv: -kv[1])[:10]
        agree = (self.choice == self.label) if self.p and self.label in self.options else None
        rec: dict[str, Any] = {
            "point": self.point, "mode": self.mode, "question": self.question, "n_options": len(self.options),
            "options": self.options[:300], "p": {k: round(v, 4) for k, v in top}, "choice": self.choice,
            "confidence": None if self.confidence is None else round(self.confidence, 4), "tau": self.tau,
            "act_option": self.act_option, "p_act": round(self.p_of(), 4) if self.act_option and self.p else None,
            "acted": self.acted, "effect": self.effect, "label": self.label, "label_tier": self.label_tier,
            "agree": agree, "fit_ok": self.fit_ok, "latency_s": round(self.latency_s, 4),
            "cost_usd": round(self.cost_usd, 8), "model": self.model or None, "rounds": self.rounds,
            "calibrated": self.calibrated if self.p else None,
            "truncated": self.truncated if self.p else None,
            "error": self.error}
        if self.state is not None:
            rec["state"] = self.state
        rec.update(self.extra)
        return rec


class Asker:
    """Asks the decision model one question with a time limit and turns any failure into an ``Ask`` with
    ``error``."""

    def __init__(self, backend: DecisionBackend, log_state: bool = False) -> None:
        self.backend = backend
        # when on, each successful Ask keeps the exact state string sent to the server so the journal row
        # (decision point + state + question) is a complete prompt for training-data harvests
        self.log_state = log_state

    async def _decide(self, state: str, q: Question, timeout: float, view: StepView | None) -> Any:
        if isinstance(self.backend, RulesBackend):
            return await self.backend.decide(state, q, view)
        return await asyncio.wait_for(self.backend.decide(state, q), timeout=timeout)

    async def ask(self, point: str, mode: str, state: str, q: Question, timeout: float, tau: float | None = None,
                  act_option: str | None = None, view: StepView | None = None) -> Ask:
        a = Ask(point=point, mode=mode, question=q.id, options=list(q.options), tau=tau, act_option=act_option)
        t0 = time.monotonic()
        try:
            d = await self._decide(state, q, timeout, view)
            a.p = {o: float(d.p.get(o, 0.0)) for o in q.options}
            a.model, a.cost_usd = d.model, float(d.cost_usd or 0.0)
            a.calibrated = bool(getattr(d, "calibrated", False))  # rules/none backends have no temperatures to apply
            a.truncated = bool(getattr(d, "truncated", False))
            if not any(a.p.values()):
                raise RuntimeError("the decision model returned no probability for any option")
        except Exception as e: # noqa: BLE001 - a decision-model failure falls back to the LLM-only behaviour
            a.error = f"{type(e).__name__}: {str(e)[:200]}"
            a.p = {}
        if self.log_state and a.p:
            a.state = getattr(d, "state", None)
        a.latency_s = time.monotonic() - t0
        return a

    async def ask_choice(self, point: str, mode: str, state: str, text: str, options: list[str], descs: list[str],
                         timeout: float, tau: float, max_options: int, keep: int = 5) -> Ask:
        """A choice among options; more than ``max_options`` options -> a knockout: the chunks are asked
        concurrently, the ``keep`` likeliest of each go to one final question. The final question's
        probabilities are logged (renormalized among the finalists)."""
        if len(options) <= max_options:
            return await self.ask(point, mode, state, Question(id="next_tool", text=text, options=options,
                                                               descriptions=descs), timeout, tau)
        keep = max(2, min(keep, max_options - 1))
        n_chunks = -(-len(options) // max_options)
        size = -(-len(options) // n_chunks)
        chunks = [list(range(i, min(i + size, len(options)))) for i in range(0, len(options), size)]
        t0 = time.monotonic()
        firsts = await asyncio.gather(*[self.ask(point, mode, state, Question(
            id="next_tool", text=text, options=[options[i] for i in ch], descriptions=[descs[i] for i in ch]),
            timeout, tau) for ch in chunks])
        bad = next((a for a in firsts if not a.ok), None)
        if bad is not None:
            bad.latency_s, bad.rounds, bad.options = time.monotonic() - t0, 2, list(options)
            bad.cost_usd = sum(a.cost_usd for a in firsts)
            return bad
        finalists: list[str] = []
        for a in firsts:
            finalists += [o for o, _ in sorted(a.p.items(), key=lambda kv: -kv[1])[:keep]]
        if len(finalists) > max_options: # very many tools: keep the likeliest finalists overall
            pr = {o: p for a in firsts for o, p in a.p.items()}
            finalists = sorted(finalists, key=lambda o: -pr[o])[:max_options]
        if NO_TOOL in options and NO_TOOL not in finalists:
            finalists[-1] = NO_TOOL # the no-tool option always reaches the final round
        idx = {o: i for i, o in enumerate(options)}
        final = await self.ask(point, mode, state, Question(
            id="next_tool", text=text, options=finalists, descriptions=[descs[idx[o]] for o in finalists]),
            timeout, tau)
        final.options = list(options)
        final.rounds = 2
        final.cost_usd += sum(a.cost_usd for a in firsts)
        final.latency_s = time.monotonic() - t0
        final.extra["finalists"] = len(finalists)
        return final
