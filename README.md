# Bracket

**Passive age-band inference for chat safety — with the confidence score actually measured.**

Bracket reads how someone writes and what they talk about, maintains a live estimate of their age band (child / teen / adult / unknown), and emits a `safety_posture` the host product can act on. It never touches the reply path.

---

## What I built

Two pieces of work: a measurement layer, and code-mixed language support.

### 1. The measurement layer — `src/bracket_eval/`

The pipeline reported **100% accuracy**. That number was circular: one script generated the test conversations, and the same keyword lexicon that informed them then graded the results. A rule system recovering its own trigger words from text built around those words.

Underneath that sat a bigger problem. The architecture's central claim is that confidence is computed deterministically in Python rather than by the model, so the model cannot be confidently wrong. The first half is true and enforced in code. The second does not follow — determinism guarantees reproducibility, not meaning. A function returning `0.9` for every input is perfectly deterministic and perfectly useless. Nobody had checked whether `0.7` meant *right 70% of the time*, and that unchecked score gates a `blocked` posture at cut points (`0.4`, `0.7`) with no derivation anywhere.

So I built the evaluation that didn't exist:

| Module | Purpose |
|---|---|
| `pipeline.py` | Synchronous deterministic driver — walks the real modules in real order, no LLM, no network, so a threshold sweep replays thousands of transcripts in seconds |
| `metrics.py` | Confusion matrix with abstention accounted separately, ECE, MCE, Brier score, reliability bins. Pure stdlib — no numpy, no sklearn |
| `calibration.py` | Asymmetric safety cost function + threshold grid search |
| `datasets.py` | Held-out loader + Blog Authorship Corpus loader |

Plus 35 held-out conversations written as ordinary chat rather than keyword bait, a 5-case evasion suite, and `scripts/run_eval.py`.

**Result on held-out data:**

```
true \ pred        child      teen     adult   unknown
------------------------------------------------------
child                  7         2         0         0
teen                   0         6         0         6
adult                  1         1         5         7

accuracy (abstention counts as wrong)   0.514
accuracy (committed predictions only)   0.818
abstention rate                         0.371

UNSAFE ERRORS — children classed adult   0
UNSAFE ERRORS — any minor classed adult  0
```

51.4%, not 100%. The number matters less than its shape:

- **Zero unsafe errors.** No minor was ever classified as an adult. The fail-closed design works exactly as claimed — that is the part that survived testing.
- **37% abstention.** The real failure mode is silence, not error. When it commits, it is right 82% of the time.
- **Adult recall 0.36.** Adults are hardest, because "adult" is the *absence* of child signals rather than the presence of adult ones.

### The finding worth defending

Measuring calibration showed those two results are the same bug.

```bash
python scripts/show_underconfident_turns.py
```

```
correct: 20/21  =  95.2%
the system's own stated confidence on these turns: ~33.6%
Underconfidence gap: +61.7%

Of these, 21 scored below the 0.4 cut point in policy_decision/table.py,
and 20 of those 21 were CORRECT — right answers parked as 'low confidence'.
```

Twenty-one turns where the system rated itself ~34% sure. It was right on twenty. Every one falls below the `0.4` threshold and gets filed as low-confidence, so the product does almost nothing with them.

The system isn't reckless. It's underconfident, and the abstention rate is a symptom of that. Accuracy alone shows a mediocre 51% and tells you nothing about what to fix.

### Fitted thresholds

Those cut points were hand-picked with no derivation. `calibration.py` grid-searches them against an **asymmetric** cost function, because the errors are not symmetric and accuracy pretends they are:

```python
COST_CHILD_TREATED_AS_ADULT  = 10.0   # a minor gets an unrestricted product
COST_TEEN_TREATED_AS_ADULT   =  6.0
COST_MINOR_UNDER_PROTECTED   =  3.0   # right band, posture too loose
COST_ABSTENTION_ON_MINOR     =  2.0   # safe failure, still a failure
COST_ADULT_OVER_RESTRICTED   =  1.0   # one tap to resolve via step-up
```

This deliberately fits *thresholds*, not the confidence function. Temperature scaling or isotonic regression would reshape the score itself and destroy the auditability that makes the architecture defensible. A fitted threshold is still a constant in a table — deterministic, inspectable, explainable to a regulator.

```bash
python scripts/run_eval.py --fit-thresholds
```

---

### 2. Hinglish — `src/signal_extraction/hinglish.py`

The pipeline detected non-English text and abstained. Correct for languages nobody has written cues for. Wrong for Hinglish:

```
English : "mom said I have to log off at 9"   → guardian_reference, weight 0.7
Hinglish: "mummy ne bola 9 baje tak"          → ZERO cues, band unknown
```

Same child, same signal, same need to protect them, completely invisible. Romanized Hinglish is the default chat register for a very large teenage population.

Seven cue subtypes across both scripts, preserving the existing weight ordering — disclosure `0.9` > topic `0.7` > style `0.25`. Style stays weak deliberately: a non-native adult and a native child look identical on lexical features, so letting style establish a band would systematically misjudge non-native speakers.

**Measured as a true ablation** — the same 35 conversations scored twice with the lexicon enabled and disabled, not by dropping the Hinglish conversations (which would compare two different datasets):

| | Lexicon OFF | Lexicon ON |
|---|---|---|
| Accuracy | 0.229 | **0.514** |
| Abstention rate | 0.600 | **0.371** |

On six **Hindi-dominant** cases, where the signal is carried only by Hindi words with no English loanword to fall back on:

| | Lexicon OFF | Lexicon ON |
|---|---|---|
| Accuracy | 0.000 | **1.000** |
| Abstention rate | 1.000 | **0.000** |

It abstains on every single one without the lexicon. That is the population this exists for.

```bash
python scripts/run_eval.py --no-hinglish   # run the ablation yourself
```

**Three design decisions:**

Adult Hinglish cues are included on purpose — without them the module could only push toward "child," drifting every code-mixed speaker young. Flesch-Kincaid is skipped on this path, because it is calibrated on English syllable structure and would tag every Hinglish adult as a low-reading-level writer. There is no Hinglish adult-self-claim detector: claim phrasing in code-mixed chat is too varied to enumerate safely, and a missed claim fails safe while a false one fails open.

**A real bug this caught:** detection splits markers into unambiguous (`nahi`, `kyun`, `hai`) and ambiguous (`the`, `main`, `hi`, `ka` — also English words). One unambiguous marker triggers; ambiguous ones need four. Without that split, **"the main point"** classifies as Hinglish. Now a regression test.

---

## Two mistakes I made while measuring this

Both are easy to make and both distort the headline number.

**The first ablation compared different datasets.** `--no-hinglish` originally dropped the Hinglish conversations, scoring 21 against 29. The resulting "+8.1pp" measured which subset was harder, not what the module did. Fixed: the flag now disables the lexicon and keeps every conversation. The true effect turned out to be larger — +28.5pp.

**The Hinglish path replaced the English lexicon instead of merging with it.** Code-mixed text is code-mixed: `"abhi homework bhi baaki hai"` carries an English school cue the Hinglish lexicon does not enumerate, and swapping lexicons silently dropped it. Measured, that made the Hinglish path score *lower* confidence than the fallback it replaced. Fixed: both lexicons run, with Hinglish taking precedence on overlap so evidence is never double-counted.

---

## Run it

Python 3.12+. No GPU, no model, no network.

```bash
pip install -r requirements.txt

python scripts/run_eval.py                       # accuracy + calibration + evasion
python scripts/run_eval.py --fit-thresholds      # sweep the policy cut points
python scripts/run_eval.py --no-hinglish         # true ablation
python scripts/run_eval.py --show-failures       # every miss, with its failure mode
python scripts/show_underconfident_turns.py      # the calibration finding, turn by turn

pytest -q                                        # 545 tests
```

Full run and deployment guide: [`RUN_AND_DEPLOY.md`](RUN_AND_DEPLOY.md)

### Full stack, still no GPU

```bash
# terminal 1 — agent
BRACKET_INFERENCE_MODE=deterministic SKIP_AMD_CHECK=true \
  uvicorn src.orchestration.api:app --host 0.0.0.0 --port 8080 --reload

# terminal 2 — UI at http://localhost:5173
cd src/ui && npm install && npm run dev
```

Hinglish through the live service:

```bash
curl -s -X POST http://localhost:8080/v1/turn -H "Content-Type: application/json" \
  -d '{"session_id":"hi","turn_text":"mummy ne bola 9 baje tak hi phone milega, homework bhi baaki hai","turn_number":1}'
```

→ `band=child`, `posture=caution`, cues `guardian_hinglish` + `curfew_hinglish`.

---

## Architecture

```
TurnEvent
  → Gateway/Session (M1)       — ingest, session context
  → Gate (M1.5)                — analyze or reuse?
  → Signal Extraction (M2)     — extract cues → SignalSet
  → Evidence Fabric (M3)       — accumulate, decay, corroborate
  → Bracket Inference (M4)     — propose band; Python computes confidence
  → Policy Decision (M5)       — deterministic table: band × confidence
  → Enforcement (M6)           — emit safety_posture
  → Step-Up Verification (M7)  — confirmed-only persist
  → Audit/Fairness (M8)        — ephemeral trace
  Orchestration (M9/M10)       — planner-supervisor loop + guardrails
```

All deterministic modules are pure Python with no LLM calls. Only `signal_extraction`, `band_estimator` and `stepup_composer` delegate to a model, and the model never assigns a weight or a confidence.

My additions:

```
src/bracket_eval/                    pipeline · metrics · calibration · datasets
src/signal_extraction/hinglish.py    code-mixed cue lexicon
data/                                holdout_en · holdout_hinglish · adversarial
scripts/                             run_eval.py · show_underconfident_turns.py
```

## What is still wrong

- 35 held-out conversations is a small sample; the threshold fit is flat as a result.
- The held-out set is author-written, not collected. Real chat with verified ages would be better.
- Hinglish coverage is keyword-based and misses unenumerated spellings. It abstains when it misses — the safe direction.
- The evasion suite is 5 cases, with a 2/5 attack success rate. Real, but small.

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
