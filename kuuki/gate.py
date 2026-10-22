"""Decide whether an AI participant should speak up in a conversation, and why.

Two reusable pieces:

* `InterjectionGate` -- give it the topic/reference text and the conversation so far, get
  back a typed `GateResult` (speak or stay quiet, intent, probabilities, latency, reason).
  Put it in front of an LLM call so the LLM only runs when there is something to say.
* `InterjectionIntervention` -- the same judgement as a Strands `InterventionHandler`.
  Attach it to an agent that has a "say something" tool (post to Slack, reply in chat...);
  `before_tool_call` returns `Proceed` or a `Guide` telling the agent to end its turn quietly.

The model only classifies: five narrow yes/no questions, one per reason to speak. The
policy (per-question thresholds, priority order, cooldown) is plain Python in
`decide_policy`, shared with the LLM baseline in `kuuki.baseline`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from strands.interventions import Deny, Guide, InterventionHandler, Proceed

from .client import Decider

# --------------------------------------------------------------------------- the questions
#
# Five narrow yes/no questions in one decider request (the state is encoded once and
# shared). Each maps to one intent. `{ai_name}` is filled in by the gate.

INTENTS = ("fact_check", "answer_question", "recap", "nudge_back", "stay_silent")
SILENT = "stay_silent"

QUESTIONS = {
    "contradicts": {
        "type": "noul",
        "instructions": "Does the LATEST LINE state a number, date, status or fact that contradicts the REFERENCE FACTS?",
        "criteria": {
            "true": "the latest line says something the reference facts show is wrong",
            "false": "the latest line agrees with the reference facts, or states nothing checkable",
        },
    },
    "unanswered": {
        "type": "noul",
        "instructions": (
            "Is a question from the recent lines still unanswered or answered only with a guess, "
            "and do the REFERENCE FACTS contain the answer?"
        ),
        "criteria": {
            "true": "the humans failed to answer a question that the reference facts answer",
            "false": (
                "no open question, a human just asked and others have not had a chance yet, "
                "or the reference facts do not contain the answer"
            ),
        },
    },
    "drifted": {
        "type": "noul",
        "instructions": "Have the last four or more lines all been small talk unrelated to the agenda?",
        "criteria": {
            "true": "several consecutive recent lines are off-agenda chatter",
            "false": "the discussion is on the agenda, or only one or two lines drifted",
        },
    },
    "closing": {
        "type": "noul",
        "instructions": (
            "Does the LATEST LINE end the meeting or move to the next agenda item while an earlier "
            "point is still wrong or undecided?"
        ),
        "criteria": {
            "true": "the meeting is wrapping up or switching topic and a summary of decisions would help",
            "false": "the discussion continues on the same topic",
        },
    },
    "addressed": {
        "type": "noul",
        "instructions": "Does the LATEST LINE speak directly to {ai_name}, the AI assistant, for example by name?",
        "criteria": {
            "true": "the latest line addresses {ai_name}",
            "false": "the latest line is said to the human participants",
        },
    },
}

# Priority order: the first question at or above its threshold decides the intent.
RULES = (
    ("addressed", "answer_question"),
    ("contradicts", "fact_check"),
    ("unanswered", "answer_question"),
    ("closing", "recap"),
    ("drifted", "nudge_back"),
)


def build_questions(ai_name: str) -> dict[str, dict[str, Any]]:
    def fill(q: dict[str, Any]) -> dict[str, Any]:
        out = dict(q)
        out["instructions"] = q["instructions"].format(ai_name=ai_name)
        out["criteria"] = {k: v.format(ai_name=ai_name) for k, v in q["criteria"].items()}
        return out

    return {name: fill(q) for name, q in QUESTIONS.items()}


# --------------------------------------------------------------------------- types


@dataclass(frozen=True)
class Utterance:
    speaker: str
    text: str


@dataclass(frozen=True)
class Thresholds:
    """Per-question thresholds (P(yes) at or above counts as yes) and the cooldown.

    Defaults were picked on the development script only.
    """

    addressed: float = 0.30
    contradicts: float = 0.35
    unanswered: float = 0.55
    closing: float = 0.50
    drifted: float = 0.62
    cooldown_lines: int = 3  # after speaking, the next N lines cannot trigger (except `addressed`)

    def for_question(self, name: str) -> float:
        return float(getattr(self, name))

    def as_dict(self) -> dict[str, float]:
        return {name: self.for_question(name) for name in QUESTIONS}


@dataclass
class GateResult:
    interject: bool
    intent: str  # stay_silent when not interjecting
    probs: dict[str, float]  # P(yes) for each question in QUESTIONS
    thresholds: dict[str, float]  # the threshold each probability was compared with
    trigger: str | None  # question that fired (None when silent)
    latency_ms: float  # wall-clock time of the classification round trip
    reason: str
    backend: str
    llm_calls: int = 0  # LLM invocations spent on this decision (0 for the local decider)
    input_tokens: int = 0  # tokens sent to the classifier (decider or Bedrock)
    output_tokens: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def coerce_history(history: Iterable[Utterance | tuple[str, str] | str]) -> list[Utterance]:
    out: list[Utterance] = []
    for item in history:
        if isinstance(item, Utterance):
            out.append(item)
        elif isinstance(item, tuple):
            out.append(Utterance(item[0], item[1]))
        else:
            speaker, _, text = str(item).partition(": ")
            out.append(Utterance(speaker, text) if text else Utterance("unknown", speaker))
    return out


def lines_since_ai(history: Sequence[Utterance], ai_name: str) -> tuple[int | None, str | None]:
    """How many lines ago the AI last spoke (None if never), and what it said."""
    for back, u in enumerate(reversed(history)):
        if u.speaker == ai_name:
            return back, u.text
    return None, None


def build_state(
    context: str,
    history: Sequence[Utterance],
    ai_name: str,
    recent_lines: int,
    since_ai: int | None = None,
) -> str:
    since, last = lines_since_ai(history, ai_name)
    if since_ai is not None:
        since = since_ai
    if since is None:
        ai_note = f"{ai_name} has not spoken yet."
    elif last is None:
        ai_note = f"{ai_name} last spoke {since} lines ago."
    else:
        ai_note = f'{ai_name} last spoke {since} lines ago: "{last}"'
    window = [f"{u.speaker}: {u.text}" for u in history[-recent_lines:]]
    earlier = "\n".join(window[:-1]) or "(none)"
    latest = window[-1] if window else "(none)"
    # The latest line stays last in the state, as in the tuning experiment (bench/exp_narrow.py).
    return (
        f"{context.strip()}\n\n--- {ai_name.upper()} ---\n{ai_note}\n\n"
        f"--- EARLIER LINES ---\n{earlier}\n\n--- LATEST LINE ---\n{latest}"
    )


def decide_policy(
    probs: dict[str, float], *, since_ai: int | None, thresholds: Thresholds
) -> tuple[bool, str, str | None, str]:
    """Turn probabilities into (interject, intent, trigger, reason). First rule over its threshold wins."""
    # `since_ai` counts lines after the AI's own line: 1 means the AI spoke right before the latest line.
    cooling = since_ai is not None and since_ai <= thresholds.cooldown_lines
    blocked: list[str] = []
    for name, intent in RULES:
        t = thresholds.for_question(name)
        if probs[name] < t:
            continue
        if cooling and name != "addressed":
            blocked.append(f"{name} {probs[name]:.3f} >= {t}")
            continue
        return True, intent, name, f"{name} {probs[name]:.3f} >= {t}"
    if blocked:
        return False, SILENT, None, f"cooldown ({since_ai} of {thresholds.cooldown_lines} lines): " + ", ".join(blocked)
    top = max(QUESTIONS, key=lambda n: probs[n] - thresholds.for_question(n))
    return (
        False,
        SILENT,
        None,
        f"nothing over threshold (closest: {top} {probs[top]:.3f} < {thresholds.for_question(top)})",
    )


# --------------------------------------------------------------------------- the gate


class InterjectionGate:
    """Should the AI speak now? One local decider call per new line.

    The gate is stateless between calls: cooldown is derived from `history`, where lines
    whose speaker equals `ai_name` are the AI's own (append the AI's line after it speaks),
    or passed explicitly as `since_ai` when the AI's lines are not in the history.
    """

    backend = "decider"

    def __init__(
        self,
        context: str,
        decider: Decider | None = None,
        *,
        ai_name: str = "Kuuki",
        thresholds: Thresholds | None = None,
        recent_lines: int = 8,
    ) -> None:
        self.context = context
        self.decider = decider or Decider()
        self.ai_name = ai_name
        self.thresholds = thresholds or Thresholds()
        self.recent_lines = recent_lines
        self.questions = build_questions(ai_name)

    def classify(self, state: str) -> tuple[dict[str, float], dict[str, Any]]:
        """Return (P(yes) per question, usage) for a state. Override point for other backends."""
        answers = self.decider.ask(state, self.questions)
        probs = {name: float(answers[name]["noul"]) for name in QUESTIONS}
        usage = {
            "llm_calls": 0,
            "input_tokens": self.decider.last_usage.get("input_tokens", 0),
            "output_tokens": self.decider.last_usage.get("output_tokens", 0),
            "extra": {"server_latency_ms": self.decider.last_latency_ms},
        }
        return probs, usage

    def state_for(self, history: Iterable[Utterance | tuple[str, str] | str], since_ai: int | None = None) -> str:
        return build_state(self.context, coerce_history(history), self.ai_name, self.recent_lines, since_ai)

    def decide(self, history: Iterable[Utterance | tuple[str, str] | str], since_ai: int | None = None) -> GateResult:
        lines = coerce_history(history)
        if since_ai is None:
            since_ai, _ = lines_since_ai(lines, self.ai_name)
        state = build_state(self.context, lines, self.ai_name, self.recent_lines, since_ai)
        t0 = time.perf_counter()
        probs, usage = self.classify(state)
        latency_ms = (time.perf_counter() - t0) * 1000
        interject, intent, trigger, reason = decide_policy(probs, since_ai=since_ai, thresholds=self.thresholds)
        return GateResult(
            interject=interject,
            intent=intent,
            probs=probs,
            thresholds=self.thresholds.as_dict(),
            trigger=trigger,
            latency_ms=latency_ms,
            reason=reason,
            backend=self.backend,
            **usage,
        )


# --------------------------------------------------------------------------- the intervention

QUIET_FEEDBACK = (
    "Do not post anything this time. The conversation does not need your input right now "
    "({reason}). End your turn now without calling any tool and without a reply."
)
CAP_REASON = "Posting was already declined in this turn. Do not call this tool again; end your turn without a reply."


class InterjectionIntervention(InterventionHandler):
    """Gate a "speak" tool with `InterjectionGate`.

    `before_tool_call` on any tool in `speak_tools` returns `Proceed` when the gate says
    speak, otherwise `Guide` with "end your turn without speaking". After `max_guides`
    guides in one invocation the handler stops asking the gate and returns `Deny`, so an
    agent that keeps retrying cannot loop.

    `history` maps the agent to the conversation the gate should judge. By default it reads
    `agent.messages`: user text is split into "Speaker: text" lines, and only speak-tool
    calls this handler let through count as the AI's lines (attributed to `ai_name`).
    """

    name = "kuuki-interjection"

    def __init__(
        self,
        gate: InterjectionGate,
        speak_tools: Iterable[str] = ("post_message",),
        *,
        max_guides: int = 1,
        history: Callable[[Any], list[Utterance]] | None = None,
    ) -> None:
        self.gate = gate
        self.speak_tools = set(speak_tools)
        self.max_guides = max_guides
        self._history = history or self._history_from_messages
        self._guides_this_turn = 0
        self._proceeded: set[str] = set()
        self.log: list[dict[str, Any]] = []

    def _history_from_messages(self, agent: Any) -> list[Utterance]:
        out: list[Utterance] = []
        for message in agent.messages:
            role = message.get("role")
            for block in message.get("content", []):
                if not isinstance(block, dict):
                    continue
                if role == "assistant":
                    # Assistant prose never reached the channel; only speak calls that ran did.
                    use = block.get("toolUse")
                    if use and use.get("toolUseId") in self._proceeded:
                        said = next((v for v in (use.get("input") or {}).values() if isinstance(v, str)), "")
                        out.append(Utterance(self.gate.ai_name, said))
                    continue
                if "text" not in block:
                    continue
                for line in block["text"].splitlines():
                    if line.strip():
                        out.extend(coerce_history([line.strip()]))
        return out

    def before_invocation(self, event, **kwargs):
        self._guides_this_turn = 0
        return Proceed()

    def before_tool_call(self, event, **kwargs):
        tool = event.tool_use["name"]
        if tool not in self.speak_tools:
            return Proceed()
        if self._guides_this_turn >= self.max_guides:
            self.log.append({"tool": tool, "verdict": "denied", "reason": "retry cap reached"})
            return Deny(reason=CAP_REASON)

        result = self.gate.decide(self._history(event.agent))
        self.log.append({"tool": tool, "verdict": "proceed" if result.interject else "guide", **result.to_dict()})
        if result.interject:
            self._proceeded.add(event.tool_use.get("toolUseId", ""))
            return Proceed(reason=result.reason)
        self._guides_this_turn += 1
        return Guide(feedback=QUIET_FEEDBACK.format(reason=result.reason), reason=result.reason)
