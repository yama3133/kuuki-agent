"""Baseline B: the same five yes/no judgements, answered by Claude Haiku 4.5 on Bedrock as JSON.

Uses the same questions, state, thresholds and policy as `InterjectionGate`, so the only
difference in a comparison is who classifies: a local 2B decider or a Bedrock LLM call per line.
"""

from __future__ import annotations

import json
import re
from typing import Any

import boto3

from .gate import QUESTIONS, InterjectionGate

DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

SYSTEM_PROMPT = """You classify a moment in a meeting. You never speak in the meeting.
Answer each yes/no question below about the state you are given, as the probability (0 to 1)
that the answer is yes. Reply with JSON only, no prose.

Questions:
{questions}

Reply with exactly this JSON shape:
{{{shape}}}"""


def _describe(questions: dict[str, dict[str, Any]]) -> str:
    return "\n".join(
        f"- {name}: {q['instructions']}\n    yes: {q['criteria']['true']}\n    no: {q['criteria']['false']}"
        for name, q in questions.items()
    )


def _parse_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in model output: {text!r}")
    return json.loads(match.group(0))


class LLMInterjectionGate(InterjectionGate):
    """`InterjectionGate` with the classifier swapped for one Bedrock call per decision."""

    backend = "baseline"

    def __init__(
        self,
        context: str,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        region: str = "us-east-1",
        **kwargs: Any,
    ) -> None:
        # The decider client is never called here; pass a dummy so the base does not build one.
        super().__init__(context, decider=object(), **kwargs)  # type: ignore[arg-type]
        self.model_id = model_id
        self.bedrock = boto3.client("bedrock-runtime", region_name=region)
        self.system = SYSTEM_PROMPT.format(
            questions=_describe(self.questions),
            shape=", ".join(f'"{name}": <p>' for name in QUESTIONS),
        )
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def classify(self, state: str) -> tuple[dict[str, float], dict[str, Any]]:
        response = self.bedrock.converse(
            modelId=self.model_id,
            system=[{"text": self.system}],
            messages=[{"role": "user", "content": [{"text": f"STATE:\n{state}"}]}],
            inferenceConfig={"maxTokens": 200, "temperature": 0.0},
        )
        usage = response.get("usage", {})
        self.calls += 1
        self.input_tokens += usage.get("inputTokens", 0)
        self.output_tokens += usage.get("outputTokens", 0)

        text = "".join(b.get("text", "") for b in response["output"]["message"]["content"])
        data = _parse_json(text)
        probs = {name: float(data[name]) for name in QUESTIONS}
        return probs, {
            "llm_calls": 1,
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "extra": {"server_latency_ms": response.get("metrics", {}).get("latencyMs")},
        }
