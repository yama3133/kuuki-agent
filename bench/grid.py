"""Offline grid search of per-question thresholds on collected dev numbers."""
import itertools, json, sys
rows = json.load(open(sys.argv[1]))
MAP = [("addressed", "answer_question"), ("contradicts", "fact_check"), ("unanswered", "answer_question"), ("closing", "recap"), ("drifted", "nudge_back")]

def run(th, cooldown=3, verbose=False):
    last = -99; ok_sw = ok_int = n_pos = fp = 0; out = []
    for r in rows:
        dec = "stay_silent"
        for q, intent in MAP:
            if r[q] >= th[q] and (q == "addressed" or r["i"] - last > cooldown):
                dec = intent; break
        if dec != "stay_silent": last = r["i"]
        exp = r["expected"]; speak_e = exp != "stay_silent"; speak_d = dec != "stay_silent"
        ok_sw += speak_e == speak_d; fp += speak_d and not speak_e
        if speak_e: n_pos += 1; ok_int += dec == exp
        out.append((r["i"], exp, dec))
    if verbose:
        for o in out:
            if o[1] != "stay_silent" or o[2] != "stay_silent": print(*o)
    return ok_sw / len(rows), ok_int, n_pos, fp

grid = {"addressed": [0.3, 0.35], "contradicts": [0.3, 0.35, 0.4], "unanswered": [0.5, 0.52, 0.55], "closing": [0.5, 0.55, 0.6, 0.65], "drifted": [0.6, 0.62, 0.65]}
best = sorted((run(dict(zip(grid, v))) + (dict(zip(grid, v)),) for v in itertools.product(*grid.values())), key=lambda x: (x[1], x[0]), reverse=True)
for b in best[:8]: print(f"acc={b[0]:.2f} intent={b[1]}/{b[2]} fp={b[3]}", b[4])
print("--- best detail"); run(best[0][4], verbose=True)
