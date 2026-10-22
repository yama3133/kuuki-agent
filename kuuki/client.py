"""Minimal synchronous client for a running strands-decider server.

Adapted from strands-decider's examples/strands/_client.py (Apache-2.0): plain `urllib`,
blocking, one request at a time, which is how the server is best driven.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

NAME = "strands-decider"
ENV_URL = "STRANDS_DECIDER_URL"
SERVE_HINT = "strands-decider serve <checkpoint> --port 8099"
MODEL_ID = "hobson-latest"  # ignored by the server; sent only to show the full wire shape
DEFAULT_URL = os.environ.get(ENV_URL, "http://127.0.0.1:8099")


class ServerUnavailable(RuntimeError):
    """The server is not answering. Raised with the command that starts one."""


class Decider:
    """One state, many typed questions, one round trip."""

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 120.0) -> None:
        self.url = url.rstrip("/")
        # Generous: the first request after start-up (or at a new input length) can take seconds.
        self.timeout = timeout
        self.calls = 0
        self.input_tokens = 0
        self.last_latency_ms: float | None = None
        self.last_usage: dict[str, int] = {}

    @staticmethod
    def noul(instructions: str, criteria: Mapping[str, str] | None = None) -> dict[str, Any]:
        question: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria:
            question["criteria"] = dict(criteria)
        return question

    @staticmethod
    def choice(instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
        return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}

    def _unavailable(self, exc: Exception) -> ServerUnavailable:
        return ServerUnavailable(f"no {NAME} at {self.url} ({exc}). Start one with:\n    {SERVE_HINT}")

    def ask(self, state: Any, questions: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
        """Evaluate `state` against `questions`; answers come back under the same keys."""
        body = json.dumps({"state": state, "model": MODEL_ID, "questions": dict(questions)}).encode()
        request = urllib.request.Request(
            f"{self.url}/v1/systemone", data=body, headers={"content-type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.URLError as exc:
            raise self._unavailable(exc) from exc
        self.calls += 1
        self.last_usage = payload.get("usage", {})
        self.input_tokens += self.last_usage.get("input_tokens", 0)
        self.last_latency_ms = payload.get("latency_ms")
        return payload["answers"]

    def health(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=10) as response:
                return json.loads(response.read())
        except urllib.error.URLError as exc:
            raise self._unavailable(exc) from exc

    def banner(self) -> str:
        h = self.health()
        return f"{NAME}: {h['checkpoint']} [{h['device']}, window {h['max_length']}] at {self.url}"
