"""Load a meeting script (JSONL) and play it through a gate, yielding events.

Script format, one JSON object per line:

    {"meta": {"title": "...", "agenda": "...", "reference": "..."}}       # first line
    {"speaker": "Maya", "text": "...", "expected": "stay_silent"}         # one utterance
    {"speaker": "Dev", "text": "...", "expected": "answer_question", "addressed": true}

`expected` (optional) is what the AI should do right after the line: stay_silent,
fact_check, answer_question, recap or nudge_back. `addressed` (optional) marks lines that
speak to the AI directly. `t` (optional) is seconds from the start, used by realtime replay.

`meeting_events` is a generator of plain dicts so a CLI and a web UI can share it.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gate import INTENTS, SILENT, InterjectionGate, Utterance


@dataclass(frozen=True)
class Line:
    speaker: str
    text: str
    expected: str | None = None
    addressed: bool = False
    t: float | None = None


@dataclass
class Script:
    title: str
    agenda: str
    reference: str
    lines: list[Line]

    @property
    def meeting(self) -> str:
        return f"Meeting: {self.title}. Agenda: {self.agenda}"

    @property
    def context(self) -> str:
        """Topic and reference text for the gate's state."""
        return f"{self.meeting}\nReference facts: {self.reference.strip()}"


def load_script(path: str | Path) -> Script:
    meta: dict[str, str] = {}
    lines: list[Line] = []
    for n, raw in enumerate(Path(path).read_text().splitlines(), 1):
        if not raw.strip():
            continue
        obj = json.loads(raw)
        if "meta" in obj:
            meta = obj["meta"]
            continue
        if "speaker" not in obj or "text" not in obj:
            raise ValueError(f"{path}:{n}: each utterance needs 'speaker' and 'text'")
        expected = obj.get("expected")
        if expected is not None and expected not in INTENTS:
            raise ValueError(f"{path}:{n}: expected must be one of {INTENTS}, got {expected!r}")
        lines.append(Line(obj["speaker"], obj["text"], expected, bool(obj.get("addressed")), obj.get("t")))
    return Script(meta.get("title", ""), meta.get("agenda", ""), meta.get("reference", ""), lines)


def expected_speaks(expected: str) -> bool:
    return expected != SILENT


def _pause_after(line: Line) -> float:
    # Rough speaking time at ~150 words per minute, so lines do not arrive back to back.
    return max(1.0, len(line.text.split()) / 2.5)


def percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def _gap_before(line: Line, prev: Line | None) -> float:
    if prev is None:
        return 0.0
    if line.t is not None and prev.t is not None:
        return max(0.0, line.t - prev.t)
    return _pause_after(prev)


def meeting_events(
    script: Script,
    gate: InterjectionGate,
    speaker: Any | None = None,
    *,
    realtime: bool = False,
    warmup: bool = True,
    before_line: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Feed lines to the gate one by one; when it says speak, generate (and say) a line.

    `speaker` is a `kuuki.speaker.Speaker` or None (decide only). The AI's generated lines
    are appended to the history under `gate.ai_name`. Cooldown is counted from the last
    line the gate chose to speak on, with or without a speaker.

    With `realtime`, each line waits for the previous line's speaking time, measured from
    when the previous line was processed, so a pause in `before_line` (called before each
    line; block in it to pause) does not make the following lines arrive in a burst.
    """
    yield {"type": "start", "backend": gate.backend, "lines": len(script.lines), "title": script.title}

    if warmup and script.lines:
        # The first decider request after start-up is slow; pay it before the meeting starts.
        first = script.lines[0]
        w = gate.decide([Utterance(first.speaker, first.text)])
        yield {"type": "warmup", "latency_ms": w.latency_ms}

    history: list[Utterance] = []
    decisions: list[dict[str, Any]] = []
    speeches: list[dict[str, Any]] = []
    prev: Line | None = None
    prev_done = time.monotonic()
    last_spoke: int | None = None

    for i, line in enumerate(script.lines):
        if before_line is not None:
            before_line()
        if realtime:
            wait = _gap_before(line, prev) - (time.monotonic() - prev_done)
            if wait > 0:
                time.sleep(wait)
        prev = line

        history.append(Utterance(line.speaker, line.text))
        yield {
            "type": "line",
            "index": i,
            "speaker": line.speaker,
            "text": line.text,
            "expected": line.expected,
            "addressed": line.addressed,
        }

        since_ai = None if last_spoke is None else i - last_spoke
        result = gate.decide(history, since_ai=since_ai)
        match = None if line.expected is None else (result.interject == expected_speaks(line.expected))
        decision = {"type": "decision", "index": i, "expected": line.expected, "match": match, **result.to_dict()}
        decisions.append(decision)
        yield decision

        if result.interject:
            last_spoke = i
            if speaker is not None:
                speech = speaker.generate(result.intent, history)
                history.append(Utterance(gate.ai_name, speech.text))
                event = {"type": "speech", "index": i, "speaker": gate.ai_name, **speech.__dict__}
                speeches.append(event)
                yield event
                speaker.say(speech.text)
        prev_done = time.monotonic()

    yield summarize(gate, decisions, speeches)


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def summarize(gate: InterjectionGate, decisions: list[dict], speeches: list[dict]) -> dict[str, Any]:
    lat = [d["latency_ms"] for d in decisions]
    labelled = [d for d in decisions if d["match"] is not None]
    should_speak = [d for d in labelled if expected_speaks(d["expected"])]
    should_quiet = [d for d in labelled if not expected_speaks(d["expected"])]
    intent_hits = sum(d["interject"] and d["intent"] == d["expected"] for d in should_speak)

    gate_llm_calls = sum(d["llm_calls"] for d in decisions)
    gate_in = sum(d["input_tokens"] for d in decisions)
    gate_out = sum(d["output_tokens"] for d in decisions)
    speaker_model_calls = sum(s["model_calls"] for s in speeches)
    bedrock_gate = gate.backend != "decider"
    return {
        "type": "summary",
        "backend": gate.backend,
        "decisions": len(decisions),
        "interjections": sum(d["interject"] for d in decisions),
        "latency_median_ms": statistics.median(lat) if lat else None,
        "latency_p95_ms": percentile(lat, 95) if lat else None,
        "gate_llm_calls": gate_llm_calls,
        "speaker_invocations": len(speeches),
        "speaker_model_calls": speaker_model_calls,
        "llm_calls_total": gate_llm_calls + speaker_model_calls,
        # what the same meeting would cost if an LLM judged every line: one call per line + generation
        "llm_calls_if_llm_gate": len(decisions) + speaker_model_calls,
        "bedrock_input_tokens": (gate_in if bedrock_gate else 0) + sum(s["input_tokens"] for s in speeches),
        "bedrock_output_tokens": (gate_out if bedrock_gate else 0) + sum(s["output_tokens"] for s in speeches),
        "decider_input_tokens": 0 if bedrock_gate else gate_in,
        "labelled": len(labelled),
        # speak / stay-quiet agreement over every labelled line
        "speak_silent_match": _ratio(sum(d["match"] for d in labelled), len(labelled)),
        "expected_speak": len(should_speak),
        "spoke_when_expected": sum(d["interject"] for d in should_speak),
        "expected_silent": len(should_quiet),
        "quiet_when_expected": sum(not d["interject"] for d in should_quiet),
        # over lines that should get an interjection: spoke *and* with the expected intent
        "intent_match": _ratio(intent_hits, len(should_speak)),
        "intent_match_count": intent_hits,
        "false_interjections": sum(d["interject"] for d in should_quiet),
    }
