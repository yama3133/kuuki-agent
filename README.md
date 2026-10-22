English | [日本語](README.ja.md)

# Kuuki

An AI meeting agent that knows when to stay quiet.

*Kuuki* (空気) is Japanese for "the air", as in *kuuki wo yomu*, "reading the air": sensing when to speak and when not to. For every line in a meeting, a local [Strands Decider 2B](https://github.com/strands-labs/strands-decider) model decides whether to interject and why. Only when it decides to speak does a Strands Agent on Amazon Bedrock (Claude Haiku 4.5) write a one- or two-sentence line, which is read aloud with macOS `say`.

Decision pattern: Policy. Each line is classified into stay quiet, fact check, answer a question, recap or nudge back to the agenda before the agent acts.

## How it works

```
meeting line ──► InterjectionGate (Strands Decider 2B, local, five yes/no questions in one request)
                   │
                   ▼
                 policy in plain Python (thresholds, priority order, cooldown)
                   ├─ stay quiet ──► nothing is sent to an LLM
                   └─ speak(intent) ─► Strands Agent on Amazon Bedrock writes the line ──► say
```

| Question (noul = yes/no) | Intent when it fires | Threshold |
|---|---|---|
| `addressed`: the latest line speaks to the AI by name | `answer_question` | 0.30 |
| `contradicts`: the latest line contradicts the reference facts | `fact_check` | 0.35 |
| `unanswered`: a question is left unanswered and the reference has the answer | `answer_question` | 0.55 |
| `closing`: moving on or wrapping up while a point is open | `recap` | 0.50 |
| `drifted`: the last four or more lines are off-agenda small talk | `nudge_back` | 0.62 |

In the order above, the first question at or above its threshold decides the intent. For three lines after the AI speaks, only `addressed` can fire. The model only classifies; the code decides.

## Results

Two hand-labelled meeting scripts in `scripts/`: a sprint planning meeting for development (32 lines) and an incident postmortem for evaluation (35 lines, 6 places where the agent should speak). Thresholds were chosen on the development script only and left unchanged for the evaluation script. The baseline asks the same five questions to Claude Haiku 4.5 on Bedrock as JSON, with the same thresholds and policy.

Evaluation script, one run each, Apple M1 (16 GB), decider served with `--device mlx`:

| | Decider gate | Haiku 4.5 gate |
|---|---|---|
| Speak/stay-quiet agreement | 86% (30/35) | 86% (30/35) |
| Intent correct where it should speak | 3/6 | 4/6 |
| Spoke when it should stay quiet | 2 | 3 |
| Decision latency, median | 1,691 ms | 945 ms |
| Bedrock calls | 8 | 47 |
| Bedrock input / output tokens | 7,299 / 414 | 38,274 / 2,552 |

- Both gates agree with the labels equally often, while the decider gate sends about 81% fewer input tokens to Bedrock
- On this M1 the decider is slower than the Haiku 4.5 gate. Trimming the questions and the state saved at most about 20%, and MPS was about twice as slow as MLX. The 115 ms in the upstream README is per question on an RTX 3090
- With 13 labelled interjection points in total, these are tendencies, not significant differences

Details and raw events: [`results/SUMMARY.md`](results/SUMMARY.md) (Japanese), `results/eval_*.jsonl`.

What we learned: one broad five-way `choice` question ("what would a facilitator do now?") was pulled along by the whole recent transcript. Once a question had been asked, "answer a question" stayed high for many lines. Five narrow yes/no questions that point at the `LATEST LINE` separated the cases much better. On the development script, speak/stay-quiet agreement went from 81% (the first design, provisional thresholds) to 88% (narrow questions, tuned thresholds), and correct intents went from 1/7 to 4/7.

## Setup

Requirements: an Apple-silicon Mac, Python 3.12, [uv](https://docs.astral.sh/uv/), and AWS credentials with Amazon Bedrock access to Claude Haiku 4.5 in `us-east-1`.

```bash
git clone https://github.com/yama3133/kuuki-agent && cd kuuki-agent
git clone --depth 1 https://github.com/strands-labs/strands-decider vendor/strands-decider
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e "vendor/strands-decider[mlx]" -r requirements.txt
```

The `mlx` extra is not on PyPI yet, so strands-decider is installed from a clone. `botocore[crt]` is needed only when your credentials come from `aws login`.

Start the decider (the first run downloads the model, about 4 GB):

```bash
.venv/bin/strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --device mlx --port 8099
```

## Run

Live page for the demo:

```bash
.venv/bin/python -m web          # http://127.0.0.1:8000
```

Pick a script and press Start. The left column is the meeting (Kuuki's lines in orange). The right column shows the five probabilities for the latest line against their thresholds, the chosen intent and the decision time. The footer counts Bedrock calls against what an LLM judging every line would need.

Command line:

```bash
.venv/bin/python run.py scripts/eval.jsonl --gate decider --no-speak
.venv/bin/python run.py scripts/eval.jsonl --gate baseline --no-speak
```

| Option | Effect |
|---|---|
| `--gate decider` / `baseline` | Local decider gate (default) or the Haiku 4.5 gate |
| `--no-speak` | Do not read lines aloud |
| `--no-generate` | Decide only; never call the speaker LLM |
| `--realtime` | Replay lines with speaking-time gaps |
| `--jsonl FILE` | Also write every event as JSON lines |

Script format, one JSON object per line:

```json
{"meta": {"title": "Sprint 42 planning", "agenda": "1) carry-over 2) release date", "reference": "Release is Nov 14."}}
{"speaker": "Maya", "text": "Let's start with carry-over.", "expected": "stay_silent"}
{"speaker": "Dev", "text": "Kuuki, do you know?", "expected": "answer_question", "addressed": true}
```

`expected` is what the agent should do right after the line: `stay_silent`, `fact_check`, `answer_question`, `recap` or `nudge_back`.

## Reusable decision pattern (`kuuki/gate.py`)

### `InterjectionGate`: put it in front of an LLM call

```python
from kuuki.gate import InterjectionGate

gate = InterjectionGate("Meeting: ... Reference facts: ...")
result = gate.decide([("Maya", "Let's start."), ("Dev", "Kuuki, what's the release date?")])
if result.interject:
    ...  # call your LLM with result.intent
```

`decide` returns a `GateResult` with `interject`, `intent`, `probs` (one per question), `thresholds`, `trigger`, `latency_ms` and `reason`. The questions are the `QUESTIONS` constant at the top of `gate.py`, and the thresholds are `kuuki.gate.Thresholds`. Rewrite both for your own setting and tune the thresholds on your own labelled data.

### `InterjectionIntervention`: the same judgement as a Strands `InterventionHandler`

```python
from kuuki.gate import InterjectionGate, InterjectionIntervention

intervention = InterjectionIntervention(InterjectionGate(context), speak_tools=["post_message"], max_guides=1)
agent = Agent(model=..., tools=[post_message], interventions=[intervention])
```

Before a speak tool such as `post_message` runs, it returns `Proceed` when the gate says speak, otherwise `Guide` telling the agent to end its turn without posting. After `max_guides` guides in one invocation it returns `Deny`, so an agent that keeps retrying cannot loop.

```bash
.venv/bin/python examples/intervention_demo.py
```

Other places the pattern fits:

- a bot in a group chat
- an agent that reacts to a live stream's comments
- a pair-programming assistant that should only interrupt when it matters

Known limitation: on `Proceed` the intervention does not pass the intent to the agent, so the agent decides what to post. If the intent matters, use `InterjectionGate` directly as `run.py` does.

## Repository layout

| Path | Contents |
|---|---|
| `kuuki/` | gate, Bedrock speaker, LLM baseline, transcript playback |
| `web/` | live demo page (FastAPI and Server-Sent Events) |
| `scripts/` | meeting scripts with labels (`LABELS-REVIEW.md` explains them, in Japanese) |
| `bench/` | latency and threshold experiments |
| `results/` | measurements |
| `docs/design.md` | design notes (Japanese) |

## License

Apache License 2.0
