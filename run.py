"""Play a meeting script through the interjection gate and print what happens.

python run.py scripts/smoke.jsonl --gate decider --no-speak
python run.py scripts/dev.jsonl --gate baseline --realtime
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys

from kuuki.baseline import LLMInterjectionGate
from kuuki.client import Decider, ServerUnavailable
from kuuki.gate import InterjectionGate
from kuuki.speaker import Speaker
from kuuki.transcript import load_script, meeting_events


def build(args: argparse.Namespace):
    script = load_script(args.script)
    if args.gate == "decider":
        decider = Decider()
        print(decider.banner())
        gate = InterjectionGate(script.context, decider)
    else:
        gate = LLMInterjectionGate(script.context)
    speaker = None if args.no_generate else Speaker(script.meeting, script.reference, speak_aloud=not args.no_speak)
    return script, gate, speaker


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def print_event(e: dict) -> None:
    kind = e["type"]
    if kind == "start":
        print(f"\n== {e['title']} ({e['lines']} lines, gate={e['backend']}) ==\n")
    elif kind == "warmup":
        print(f"(warm-up request: {e['latency_ms']:.0f} ms, excluded from stats)\n")
    elif kind == "line":
        print(f"[{e['index']:2d}] {e['speaker']}: {e['text']}")
    elif kind == "decision":
        probs = " ".join(f"{k[:5]}={v:.2f}" for k, v in e["probs"].items())
        verdict = f"SPEAK:{e['intent']}" if e["interject"] else "quiet"
        if e["match"] is None:
            mark = ""
        elif e["match"] and (not e["interject"] or e["intent"] == e["expected"]):
            mark = "  ok"
        elif e["match"]:
            mark = f"  ok (intent expected {e['expected']})"
        else:
            mark = f"  MISS (expected {e['expected']})"
        print(f"     -> {verdict:<22} {probs} {e['latency_ms']:.0f}ms | {e['reason']}{mark}")
    elif kind == "speech":
        print(
            f"     >> {e['speaker']}: {e['text']}  "
            f"({e['latency_ms']:.0f} ms, {e['model_calls']} model calls, tools={e['tools_used']})"
        )
    elif kind == "summary":
        print("\n== summary ==")
        print(f"gate: {e['backend']}   decisions: {e['decisions']}   interjections: {e['interjections']}")
        print(f"decision latency: median {e['latency_median_ms']:.0f} ms, p95 {e['latency_p95_ms']:.0f} ms")
        print(
            f"LLM calls: {e['llm_calls_total']} total "
            f"(gate {e['gate_llm_calls']}, speaker {e['speaker_model_calls']} model calls "
            f"in {e['speaker_invocations']} lines); an LLM judging every line would need "
            f"{e['llm_calls_if_llm_gate']}"
        )
        print(f"Bedrock tokens: in {e['bedrock_input_tokens']}, out {e['bedrock_output_tokens']}")
        if e["decider_input_tokens"]:
            print(f"Decider tokens (local): in {e['decider_input_tokens']}")
        if e["labelled"]:
            print(
                f"speak/quiet match: {fmt_pct(e['speak_silent_match'])} of {e['labelled']} "
                f"(spoke {e['spoke_when_expected']}/{e['expected_speak']} when expected, "
                f"quiet {e['quiet_when_expected']}/{e['expected_silent']} when expected)"
            )
            print(
                f"intent match on should-speak lines: {fmt_pct(e['intent_match'])} "
                f"({e['intent_match_count']}/{e['expected_speak']}); "
                f"spoke when it should stay quiet: {e['false_interjections']}"
            )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("script", help="meeting script (JSONL)")
    p.add_argument("--gate", choices=["decider", "baseline"], default="decider")
    p.add_argument("--no-speak", action="store_true", help="do not read lines aloud with macOS say")
    p.add_argument("--no-generate", action="store_true", help="decide only; never call the speaker LLM")
    p.add_argument("--realtime", action="store_true", help="replay lines with speaking-time gaps")
    p.add_argument("--no-warmup", action="store_true", help="skip the warm-up request before the meeting")
    p.add_argument("--jsonl", help="also write every event as JSON lines to this file")
    args = p.parse_args()

    try:
        script, gate, speaker = build(args)
    except ServerUnavailable as exc:
        print(exc, file=sys.stderr)
        return 1
    # Only the local decider has a cold start worth hiding; a warm-up would cost the baseline a Bedrock call.
    warmup = args.gate == "decider" and not args.no_warmup
    with contextlib.ExitStack() as stack:
        out = stack.enter_context(open(args.jsonl, "w")) if args.jsonl else None
        for event in meeting_events(script, gate, speaker, realtime=args.realtime, warmup=warmup):
            print_event(event)
            if out:
                out.write(json.dumps(event) + "\n")
                out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
