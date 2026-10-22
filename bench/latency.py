"""Measure Decider latency on this machine for meeting-sized states.

Sends the three Kuuki questions in one request, at several transcript lengths.
For each length the first request is recorded separately (cold), then N warm repeats.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8099/v1/systemone"

AGENDA = (
    "Meeting: Sprint 42 planning. Agenda: 1) carry-over tickets 2) release date 3) on-call rota.\n"
    "Reference facts: release is scheduled for Nov 14. Team velocity last sprint was 31 points. "
    "The payments migration ticket (PAY-210) is blocked on the vendor."
)

LINES = [
    "Maya: Okay, let's start with carry-over. We have four tickets from last sprint.",
    "Dev: PAY-210 is still open, I think it's almost done actually.",
    "Maya: Great, so we can close it this week?",
    "Sam: Our velocity was like 45 points last sprint, so we have room.",
    "Dev: Yeah, we should pull in the search revamp too.",
    "Maya: By the way did anyone watch the game last night?",
    "Sam: Oh man, that last quarter was wild.",
    "Dev: I couldn't believe the referee call at the end.",
    "Maya: Ha. Anyway, what's the release date again?",
    "Sam: Not sure, end of November maybe?",
    "Dev: Kuuki, do you know?",
    "Maya: Let's move on to on-call.",
]

QUESTIONS = {
    "should_interject": {
        "type": "noul",
        "instructions": "Would a thoughtful meeting facilitator interject right now, after the latest line?",
        "criteria": {
            "true": "speaking now adds clear value the participants are missing",
            "false": "the conversation is fine without an interruption",
        },
    },
    "intent": {
        "type": "choice",
        "instructions": "If the facilitator spoke now, what would it do?",
        "criteria": {
            "fact_check": "a number or fact just stated contradicts the reference facts",
            "answer_question": "a question is hanging that nobody has answered",
            "recap": "the group is moving on while the decision is still unclear",
            "nudge_back": "the discussion has drifted far from the agenda",
            "stay_silent": "nothing needs to be said",
        },
    },
    "addressed_to_ai": {
        "type": "noul",
        "instructions": "Is the latest line speaking directly to the AI assistant named Kuuki?",
        "criteria": {
            "true": "the latest line asks or addresses Kuuki",
            "false": "the latest line is between the human participants",
        },
    },
}


def state_for(n_lines: int) -> str:
    recent = "\n".join(LINES[-n_lines:])
    return f"{AGENDA}\n\n--- RECENT TRANSCRIPT ---\n{recent}"


def ask(state: str) -> tuple[float, dict]:
    body = json.dumps({"state": state, "questions": QUESTIONS}).encode()
    req = urllib.request.Request(URL, data=body, headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        payload = json.loads(r.read())
    wall = (time.perf_counter() - t0) * 1000
    return wall, payload


def main() -> int:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    for n in (4, 8, 12):
        state = state_for(n)
        cold_wall, payload = ask(state)
        tokens = payload["usage"]["input_tokens"]
        warm = [ask(state)[0] for _ in range(repeats)]
        print(
            f"lines={n:2d} input_tokens={tokens:4d} cold={cold_wall:7.0f}ms "
            f"warm_median={statistics.median(warm):6.0f}ms warm_max={max(warm):6.0f}ms "
            f"server_latency_ms={payload['latency_ms']}"
        )
    _, payload = ask(state_for(12))
    print(json.dumps(payload["answers"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
