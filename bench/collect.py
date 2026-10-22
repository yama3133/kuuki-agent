"""Collect raw decider numbers for every line of a transcript (no AI lines inserted)."""
import json, sys
from kuuki.gate import InterjectionGate, Utterance
from kuuki.transcript import load_script

script = load_script(sys.argv[1])
gate = InterjectionGate(script.context)
hist, rows = [], []
gate.classify("warmup")
for i, line in enumerate(script.lines, 1):
    hist.append(Utterance(line.speaker, line.text))
    from kuuki.gate import build_state
    nums, _ = gate.classify(build_state(gate.context, hist, gate.ai_name, gate.recent_lines))
    rows.append({"i": i, "expected": line.expected, "addressed": line.addressed, **nums})
    print(i, line.expected, f"pi={nums['p_interject']:.2f} pa={nums['p_addressed']:.2f}", nums["raw_intent"], f"c={nums['confidence']:.2f}", {k[:4]: round(v, 2) for k, v in nums["intent_probs"].items()}, flush=True)
json.dump(rows, open(sys.argv[2], "w"), indent=1)
