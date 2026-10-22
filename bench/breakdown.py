"""Latency breakdown: number of questions, question length, state length (server-side latency_ms)."""
import json, statistics, sys, urllib.request
sys.path.insert(0, ".")
from kuuki.gate import build_questions, build_state, Utterance
from kuuki.transcript import load_script

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
s = load_script("scripts/dev.jsonl")
hist = [Utterance(l.speaker, l.text) for l in s.lines[:14]]
Q = build_questions("Kuuki")
names = list(Q)

def ask(state, qs, n=6):
    out = []
    for _ in range(n):
        body = json.dumps({"state": state, "questions": qs}).encode()
        r = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/v1/systemone", data=body, headers={"content-type": "application/json"})).read())
        out.append(r["latency_ms"])
    return statistics.median(out[1:]), r["usage"]["input_tokens"]

st8 = build_state(s.context, hist, "Kuuki", 8)
st4 = build_state(s.context, hist, "Kuuki", 4)
short = {k: {"type": "noul", "instructions": v["instructions"]} for k, v in Q.items()}
for k in (1, 2, 3, 5):
    print(f"state8 questions={k} full   ", ask(st8, {n: Q[n] for n in names[:k]}))
print("state8 questions=5 no-criteria", ask(st8, short))
print("state4 questions=5 full       ", ask(st4, Q))
print("tiny state questions=5 full   ", ask("Maya: hello", Q))
