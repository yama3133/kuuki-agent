[English](README.md) | 日本語

# Kuuki

空気を読んで、黙るべきときは黙る会議エージェントです。

会議の発言が1つ入るたびに、手元の Mac で動く [Strands Decider 2B](https://github.com/strands-labs/strands-decider) が「口を挟むべきか、挟むなら何をするか」を判定します。挟むと決めたときだけ、Amazon Bedrock 上の Strands Agent(Claude Haiku 4.5)が1〜2文の発言を作り、macOS の `say` で読み上げます。

決定パターン: Policy。発言ごとに「黙る/事実確認/質問に答える/まとめる/議題に戻す」に分類してから動きます。

## 仕組み

```
会議の発言 ──► InterjectionGate(Strands Decider 2B、ローカル、yes/no 質問5つを1回で送る)
                 │
                 ▼
               方針(ふつうの Python: しきい値、優先順、クールダウン)
                 ├─ 黙る ──────► LLM には何も送らない
                 └─ 挟む(意図)─► Amazon Bedrock 上の Strands Agent が発言を作る ──► 読み上げ
```

| 質問(noul = yes/no) | 当てはまったときの意図 | しきい値 |
|---|---|---|
| `addressed`: 最新の発言が AI に名指しで話しかけている | `answer_question` | 0.30 |
| `contradicts`: 最新の発言が参考資料と食い違う | `fact_check` | 0.35 |
| `unanswered`: 質問が宙に浮いていて、答えが参考資料にある | `answer_question` | 0.55 |
| `closing`: 論点が残ったまま次の話題に移る、または会議を締める | `recap` | 0.50 |
| `drifted`: 直近4発言以上が議題と関係ない雑談 | `nudge_back` | 0.62 |

上から順に見て、最初にしきい値を超えた質問で意図を決めます。AI が発言した後の3発言は、`addressed` 以外では口を挟みません。モデルは分類だけを行い、どうするかはコードが決めます。

## 結果

`scripts/` に、正解ラベルを付けた会議の台本が2本あります。開発用はスプリント計画会議(32発言)、評価用は障害の振り返り会議(35発言、口を挟むべき箇所は6)です。しきい値は開発用の台本だけで決め、評価用では変えていません。比較対象は、同じ5つの質問を Bedrock の Claude Haiku 4.5 に JSON で答えさせる構成で、しきい値と方針も共通です。

評価用の台本、各1回の実行、Apple M1(16GB)、Decider は `--device mlx` で起動:

| | Decider で判定 | Haiku 4.5 で判定 |
|---|---|---|
| 挟む/黙るが正解と一致 | 86%(30/35) | 86%(30/35) |
| 挟むべき箇所で意図まで一致 | 3/6 | 4/6 |
| 黙るべき箇所で挟んだ | 2 | 3 |
| 判定にかかった時間(中央値) | 1,691ms | 945ms |
| Bedrock の呼び出し | 8回 | 47回 |
| Bedrock の入力/出力トークン | 7,299 / 414 | 38,274 / 2,552 |

- 正解との一致率は同じで、Bedrock に送る入力トークンは Decider のほうが約81%少ない
- この M1 では、判定は Haiku 4.5 のほうが速い。質問と入力を削っても短縮は最大約2割で、MPS は MLX の約2倍遅かった。公式 README の 115ms は RTX 3090 での1問あたりの値
- 口を挟むべき箇所は2本合わせて13しかないため、差ではなく傾向として見てほしい

詳しい数字と生データ: [`results/SUMMARY.md`](results/SUMMARY.md)、`results/eval_*.jsonl`

分かったこと: 最初は「進行役なら今どうするか」を5択で1問だけ聞いていました。すると判定が直近の会話全体に引きずられ、一度質問が出ると「質問に答える」の値が何発言も高いまま残りました。「最新の発言」を明示した狭い yes/no 質問5つに分けると、見分けがはっきりしました。開発用の台本では、挟む/黙るの一致が 81%(最初の設計、仮のしきい値)から 88%(狭い質問、調整後のしきい値)に、意図の一致が 1/7 から 4/7 に上がりました。

## 準備

必要なもの: Apple シリコンの Mac、Python 3.12、[uv](https://docs.astral.sh/uv/)、us-east-1 で Claude Haiku 4.5 を呼べる Amazon Bedrock の認証情報

```bash
git clone https://github.com/yama3133/kuuki-agent && cd kuuki-agent
git clone --depth 1 https://github.com/strands-labs/strands-decider vendor/strands-decider
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e "vendor/strands-decider[mlx]" -r requirements.txt
```

`mlx` 付きの strands-decider はまだ PyPI にないため、clone したものを入れます。`botocore[crt]` は、認証情報を `aws login` で取得している場合にだけ必要です。

Decider を起動します(初回はモデル約4GBをダウンロードします)。

```bash
.venv/bin/strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --device mlx --port 8099
```

## 動かし方

デモ用の画面:

```bash
.venv/bin/python -m web          # http://127.0.0.1:8000
```

台本を選んで Start を押します。左が会議の発言(Kuuki の発言はオレンジ)、右が最新の発言に対する5つの確率としきい値、選ばれた意図、判定時間です。下の欄では、Bedrock の呼び出し回数を「毎発言 LLM で判定した場合」と比べて表示します。

コマンドライン:

```bash
.venv/bin/python run.py scripts/eval.jsonl --gate decider --no-speak
.venv/bin/python run.py scripts/eval.jsonl --gate baseline --no-speak
```

| オプション | 内容 |
|---|---|
| `--gate decider` / `baseline` | ローカルの Decider で判定(既定)、または Haiku 4.5 で判定 |
| `--no-speak` | 読み上げない |
| `--no-generate` | 判定だけ行い、発言を作る LLM を呼ばない |
| `--realtime` | 話す時間の間隔をあけて発言を流す |
| `--jsonl FILE` | すべてのイベントを JSON Lines でも書き出す |

台本の形式(1行に JSON 1つ):

```json
{"meta": {"title": "Sprint 42 planning", "agenda": "1) carry-over 2) release date", "reference": "Release is Nov 14."}}
{"speaker": "Maya", "text": "Let's start with carry-over.", "expected": "stay_silent"}
{"speaker": "Dev", "text": "Kuuki, do you know?", "expected": "answer_question", "addressed": true}
```

`expected` は、その発言の直後にエージェントがすべきことです。`stay_silent`、`fact_check`、`answer_question`、`recap`、`nudge_back` のいずれかを書きます。

## 再利用できる決定パターン(`kuuki/gate.py`)

### `InterjectionGate`: LLM を呼ぶ前に置く

```python
from kuuki.gate import InterjectionGate

gate = InterjectionGate("Meeting: ... Reference facts: ...")
result = gate.decide([("Maya", "Let's start."), ("Dev", "Kuuki, what's the release date?")])
if result.interject:
    ...  # result.intent を渡して LLM を呼ぶ
```

`decide` は `GateResult` を返します。中身は `interject`、`intent`、`probs`(質問ごとの確率)、`thresholds`、`trigger`、`latency_ms`、`reason` です。質問は `gate.py` 先頭の定数 `QUESTIONS`、しきい値は `kuuki.gate.Thresholds` にあります。使う場面に合わせて両方を書き換え、しきい値はご自身の正解付きデータで調整してください。

### `InterjectionIntervention`: 同じ判定を Strands の `InterventionHandler` として使う

```python
from kuuki.gate import InterjectionGate, InterjectionIntervention

intervention = InterjectionIntervention(InterjectionGate(context), speak_tools=["post_message"], max_guides=1)
agent = Agent(model=..., tools=[post_message], interventions=[intervention])
```

`post_message` のような発言用ツールが実行される前に判定します。ゲートが「挟む」と判定すれば `Proceed` を、そうでなければ「投稿せずにターンを終える」よう促す `Guide` を返します。1回の呼び出しの中で `Guide` が `max_guides` 回に達すると `Deny` を返すので、エージェントが再試行を繰り返して止まらなくなることはありません。

```bash
.venv/bin/python examples/intervention_demo.py
```

ほかに使えそうな場面:

- グループチャットに常駐するボット
- 配信のコメント欄に反応するエージェント
- 必要なときだけ口を挟むペアプログラミング支援

既知の制約: `Proceed` のとき、意図はエージェントに渡しません。何を投稿するかはエージェント任せです。意図を使いたい場合は、`run.py` のように `InterjectionGate` を直接使ってください。

## ファイル構成

| パス | 内容 |
|---|---|
| `kuuki/` | ゲート、Bedrock の発言役、LLM で判定する比較構成、台本の再生 |
| `web/` | デモ画面(FastAPI と Server-Sent Events) |
| `scripts/` | 正解ラベル付きの会議の台本(`LABELS-REVIEW.md` に解説) |
| `bench/` | 遅延としきい値の実験 |
| `results/` | 計測結果 |
| `docs/design.md` | 設計メモ |

## ライセンス

Apache License 2.0
