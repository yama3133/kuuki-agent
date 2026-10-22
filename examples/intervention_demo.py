"""Minimal example: InterjectionIntervention on a chat bot with a `post_message` tool.

The bot is told to always reply via `post_message` (an eager bot, the failure mode). Each
turn it receives the newest channel messages. Before `post_message` runs, the intervention
asks the local decider whether a facilitator would speak now; if not, it returns `Guide`
("end your turn without posting"), and after one guide per turn it returns `Deny`.

    python examples/intervention_demo.py

Needs the decider server (default http://127.0.0.1:8099) and AWS credentials for Bedrock.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strands import Agent, tool
from strands.models import BedrockModel

from kuuki.client import Decider, ServerUnavailable
from kuuki.gate import InterjectionGate, InterjectionIntervention

CONTEXT = (
    "Channel: #team-v32. Topic: shipping release v3.2.\n"
    "Reference facts: Release v3.2 is scheduled for Friday Nov 14. PAY-210 is blocked on the vendor until Nov 3."
)

POSTED: list[str] = []


@tool
def post_message(text: str) -> str:
    """Post a message to the team's chat channel.

    Args:
        text: The message to post.
    """
    POSTED.append(text)
    return "posted"


SYSTEM_PROMPT = (
    "You are Kuuki, a bot in a team chat channel. You receive the newest channel messages. "
    "On every turn, including small talk, you must call post_message with a short reply. "
    f"Facts you know:\n{CONTEXT}"
)

TURNS = [
    "Maya: I'll write the v3.2 release notes today.\nSam: Thanks, I'll review the notes this afternoon.",
    "Dev: Hey Kuuki, can you tell us the v3.2 release date? None of us remember.",
]


def main() -> int:
    decider = Decider()
    try:
        print(decider.banner(), "\n")
    except ServerUnavailable as exc:
        print(exc)
        return 1

    gate = InterjectionGate(CONTEXT, decider)
    intervention = InterjectionIntervention(gate, speak_tools=["post_message"], max_guides=1)
    agent = Agent(
        model=BedrockModel(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", region_name="us-east-1"),
        tools=[post_message],
        interventions=[intervention],
        system_prompt=SYSTEM_PROMPT,
        callback_handler=None,
    )

    for turn in TURNS:
        print(f"--- new channel messages ---\n{turn}")
        before_log, before_posted = len(intervention.log), len(POSTED)
        agent(turn)
        for row in intervention.log[before_log:]:
            if row["verdict"] == "denied":
                print(f"  [gate] denied: {row['reason']}")
            else:
                print(
                    f"  [gate] {row['verdict']}: "
                    + " ".join(f"{k}={v:.2f}" for k, v in row["probs"].items())
                    + f" {row['latency_ms']:.0f}ms | {row['reason']}"
                )
        new = POSTED[before_posted:]
        print(f"  posted: {new if new else '(nothing)'}\n")

    guided = sum(r["verdict"] == "guide" for r in intervention.log)
    denied = sum(r["verdict"] == "denied" for r in intervention.log)
    print(f"{decider.calls} decider calls; {len(intervention.log)} post attempts, {guided} guided, {denied} denied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
