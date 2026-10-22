"""Experiment: narrow per-intent yes/no questions focused on the latest line (dev only)."""
import json, sys, time
from kuuki.client import Decider
from kuuki.transcript import load_script

Q = {
 "contradicts": {"type": "noul", "instructions": "Does the LATEST LINE state a number, date, status or fact that contradicts the REFERENCE FACTS?",
   "criteria": {"true": "the latest line says something the reference facts show is wrong", "false": "the latest line agrees with the reference facts, or states nothing checkable"}},
 "unanswered": {"type": "noul", "instructions": "Is a question from the recent lines still unanswered or answered only with a guess, and do the REFERENCE FACTS contain the answer?",
   "criteria": {"true": "the humans failed to answer a question that the reference facts answer", "false": "no open question, a human just asked and others have not had a chance yet, or the reference facts do not contain the answer"}},
 "drifted": {"type": "noul", "instructions": "Have the last four or more lines all been small talk unrelated to the agenda?",
   "criteria": {"true": "several consecutive recent lines are off-agenda chatter", "false": "the discussion is on the agenda, or only one or two lines drifted"}},
 "closing": {"type": "noul", "instructions": "Does the LATEST LINE end the meeting or move to the next agenda item while an earlier point is still wrong or undecided?",
   "criteria": {"true": "the meeting is wrapping up or switching topic and a summary of decisions would help", "false": "the discussion continues on the same topic"}},
 "addressed": {"type": "noul", "instructions": "Does the LATEST LINE speak directly to Kuuki, the AI assistant, for example by name?",
   "criteria": {"true": "the latest line addresses Kuuki", "false": "the latest line is said to the human participants"}},
}

def state(ctx, hist):
    earlier = "\n".join(hist[-8:-1]) or "(none)"
    return f"{ctx}\n\n--- EARLIER LINES ---\n{earlier}\n\n--- LATEST LINE ---\n{hist[-1]}"

s = load_script(sys.argv[1]); d = Decider(); d.ask("warmup", {"x": Q["addressed"]})
hist, rows = [], []
for i, l in enumerate(s.lines, 1):
    hist.append(f"{l.speaker}: {l.text}")
    t = time.perf_counter(); a = d.ask(state(s.context, hist), Q); ms = (time.perf_counter() - t) * 1000
    v = {k: a[k]["noul"] for k in Q}
    rows.append({"i": i, "expected": l.expected, **v, "ms": ms})
    print(f"{i:2d} {l.expected:15s} " + " ".join(f"{k[:5]}={v[k]:.2f}" for k in Q) + f" {ms:.0f}ms", flush=True)
json.dump(rows, open(sys.argv[2], "w"), indent=1)
