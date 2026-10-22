"""Generate the AI's line once the gate has decided to speak (Strands Agent on Bedrock)."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass

from strands import Agent, tool
from strands.models import BedrockModel

from .gate import Utterance

DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

BASE_PROMPT = (
    "You are {ai_name}, an AI participant in a live team meeting. You speak rarely; you were "
    "asked to speak now. Reply with exactly what you will say out loud: one or two short, "
    "plain English sentences, no preamble, no markdown, no speaker label.\n\n"
    "Meeting context:\n{agenda}\n\nYour task now: {task}"
)

INTENT_TASKS = {
    "fact_check": (
        "Someone just stated a number or fact that may contradict the reference material. Call "
        "lookup_reference first, then politely correct it with the right figure."
    ),
    "answer_question": (
        "There is a question nobody has answered, or someone asked you directly. Call "
        "lookup_reference if the answer could be in the reference material, then answer it. "
        "If the material does not say, say you do not know."
    ),
    "recap": "The group is moving on while a decision is unclear. Briefly restate what was decided and what is still open.",
    "nudge_back": "The discussion has drifted off the agenda. Gently steer it back to the current agenda item.",
}


def _chunks(reference: str) -> list[str]:
    return [c.strip() for c in re.split(r"(?<=[.!?])\s+|\n+", reference) if c.strip()]


@dataclass
class Speech:
    text: str
    intent: str
    latency_ms: float
    model_calls: int
    input_tokens: int
    output_tokens: int
    tools_used: list[str]


class Speaker:
    """One short spoken line per call. A fresh Agent each time keeps calls independent."""

    def __init__(
        self,
        agenda: str,
        reference: str,
        *,
        ai_name: str = "Kuuki",
        model_id: str = DEFAULT_MODEL_ID,
        region: str = "us-east-1",
        speak_aloud: bool = True,
        voice: str | None = None,
        recent_lines: int = 8,
    ) -> None:
        self.agenda = agenda
        self.reference = reference
        self.ai_name = ai_name
        self.model = BedrockModel(model_id=model_id, region_name=region, temperature=0.3, max_tokens=200)
        self.speak_aloud = speak_aloud and shutil.which("say") is not None
        self.voice = voice
        self.recent_lines = recent_lines
        self.calls = 0
        self.model_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def _lookup_tool(self):
        chunks = _chunks(self.reference)

        @tool
        def lookup_reference(query: str) -> str:
            """Search the meeting's reference material and return the matching passages.

            Args:
                query: Keywords to look for, e.g. "release date" or "velocity".
            """
            words = {w for w in re.findall(r"[a-z0-9-]+", query.lower()) if len(w) > 2}
            scored = [(sum(w in c.lower() for w in words), c) for c in chunks]
            hits = [c for s, c in sorted(scored, key=lambda x: -x[0]) if s > 0][:3]
            return "\n".join(hits) if hits else "No matching passage. Full reference:\n" + self.reference

        return lookup_reference

    def generate(self, intent: str, history: Sequence[Utterance]) -> Speech:
        task = INTENT_TASKS.get(intent, INTENT_TASKS["answer_question"])
        agent = Agent(
            model=self.model,
            tools=[self._lookup_tool()],
            system_prompt=BASE_PROMPT.format(ai_name=self.ai_name, agenda=self.agenda, task=task),
            callback_handler=None,
        )
        recent = "\n".join(f"{u.speaker}: {u.text}" for u in history[-self.recent_lines :])
        t0 = time.perf_counter()
        result = agent(f"Recent transcript:\n{recent}\n\nWhat do you say now?")
        latency_ms = (time.perf_counter() - t0) * 1000

        summary = result.metrics.get_summary()
        usage = summary["accumulated_usage"]
        speech = Speech(
            text=str(result).strip(),
            intent=intent,
            latency_ms=latency_ms,
            model_calls=summary["total_cycles"],
            input_tokens=usage["inputTokens"],
            output_tokens=usage["outputTokens"],
            tools_used=list(summary.get("tool_usage", {}).keys()),
        )
        self.calls += 1
        self.model_calls += speech.model_calls
        self.input_tokens += speech.input_tokens
        self.output_tokens += speech.output_tokens
        return speech

    def say(self, text: str) -> None:
        """Read the line aloud with macOS `say` (blocking). No-op when disabled."""
        if not self.speak_aloud:
            return
        cmd = ["say"] + (["-v", self.voice] if self.voice else []) + [text]
        subprocess.run(cmd, check=False)
