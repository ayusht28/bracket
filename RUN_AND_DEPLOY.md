# Run and Deploy

Everything here runs with **no GPU and no model**. The deterministic path is a
first-class mode, not a stub: same gate, same lexicon weights, same confidence
formula, same policy table, same guardrails. Only the two LLM delegates (cue
extraction, band proposal) are swapped for their rule equivalents.

Verified on Python 3.11 and 3.12, Node 22.

---

## 0. Prerequisites

| | Needed | Why |
|---|---|---|
| Python 3.12+ | yes | upstream `requires-python = ">=3.12"` |
| Node 20+ | only for the UI | React/Vite SPA |
| GPU | **no** | deterministic mode covers the whole pipeline |
| Model endpoint | **no** | optional; see section 5 |

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> On Python 3.11 the 20 tests in `tests/unit/contracts/test_protocols.py` fail.
> They assert on `__protocol_attrs__`, a `typing` internal that only exists in
> 3.12+. Nothing to do with this code — use 3.12 and they pass.

---

## 1. See the evaluation (fastest — 5 seconds, no servers)

This is the core of the project, and it needs nothing running.

```bash
python scripts/run_eval.py                      # accuracy + calibration + evasion
python scripts/run_eval.py --fit-thresholds     # sweep the policy cut points
python scripts/run_eval.py --show-failures      # every miss, with its failure mode
python scripts/run_eval.py --no-hinglish        # ablation: what Hinglish bought
python scripts/run_eval.py --json report.json   # machine-readable
```

Expected headline:

```
accuracy (abstention counts as wrong)   0.414
accuracy (committed predictions only)   0.750
abstention rate                         0.448
UNSAFE ERRORS — children classed adult   0
expected calibration error 0.1964  (severely miscalibrated, underconfident)
attack success rate (ever reached adult)  2/5
```

## 2. Run the tests

```bash
pytest -q                 # 542 tests
python -m pytest -q       # equivalent; both work
```

Bare `pytest` works because `pythonpath = ["."]` is set in
`pyproject.toml`. Upstream required `PYTHONPATH=. pytest`, which still works.

---

## 3. See it running — the agent service

Terminal 1:

```bash
BRACKET_INFERENCE_MODE=deterministic SKIP_AMD_CHECK=true \
  uvicorn src.orchestration.api:app --host 0.0.0.0 --port 8080 --reload
```

Check it:

```bash
curl -s http://localhost:8080/health
```

Send a turn:

```bash
curl -s -X POST http://localhost:8080/v1/turn \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","turn_text":"my mom says i have to log off at 9, still got homework for 8th grade","turn_number":1}' \
  | python3 -m json.tool
```

Returns `band=teen`, `confidence=0.54`, `posture.level=restricted`.

**The Hinglish path, live** — the code-mixed lexicon going through the real
service, not a unit test:

```bash
curl -s -X POST http://localhost:8080/v1/turn \
  -H "Content-Type: application/json" \
  -d '{"session_id":"hi","turn_text":"mummy ne bola 9 baje tak hi phone milega, homework bhi baaki hai","turn_number":1}' \
  | python3 -m json.tool
```

Returns `band=child`, `posture=caution`, cues `guardian_hinglish` +
`curfew_hinglish`. On upstream the same sentence returns `band=unknown` with
zero cues.

| Route | What it does |
|---|---|
| `GET /health` | liveness + AMD telemetry (degrades cleanly with no GPU) |
| `POST /v1/turn` | one turn → band, confidence, posture, evidence, planner trace |
| `POST /v1/chat/completions` | same pipeline, OpenAI-compatible shape (UI uses this) |
| `POST /v1/confirm` | persist a **confirmed** band — the only thing that ever persists |
| `POST /v1/roster` | replay a Discord channel export, one session per author |

## 4. See it running — the UI

Terminal 2:

```bash
cd src/ui
npm install
npm run dev
```

Open **http://localhost:5173**. Vite proxies `/v1/` to `localhost:8080`, so the
agent must be running from step 3.

Tabs: **Session** (live band/confidence/posture per turn, plus the AMD telemetry
badge when a GPU is present) and **Roster** (risk-ranked per-user table).

```bash
npm run build   # production bundle → src/ui/dist
npm test        # 12 vitest tests
```

## 5. Optional — the LLM path

Not needed for anything above. On a 16GB Mac, use the 4B model; **27B will not
fit** (upstream benchmarked it on a 192GB MI300X).

```bash
ollama pull gemma3:4b
export BRACKET_INFERENCE_MODE=llm
export LOCAL_API_BASE=http://localhost:11434/v1
export LOCAL_MODEL=gemma3:4b
export LOCAL_API_KEY=EMPTY
```

Dual-model serving (small extractor, larger estimator):

```bash
export EXTRACTOR_MODEL=gemma3:4b
export ESTIMATOR_MODEL=gemma3:12b
```

---

## 6. Deploy — Docker

```bash
docker build -t bracket-agent:1.0.0 .
docker build -f src/ui/Dockerfile.ui -t bracket-ui:1.0.0 src/ui/
```

Run the agent with no GPU:

```bash
docker run --rm -p 8080:8080 \
  -e BRACKET_INFERENCE_MODE=deterministic \
  -e SKIP_AMD_CHECK=true \
  bracket-agent:1.0.0
```

Run the UI (nginx, port 8081):

```bash
docker run --rm -p 8081:80 bracket-ui:1.0.0
```

Makefile equivalents: `make docker-build-all`, `make docker-run`,
`make docker-run-ui`, `make docker-push-all IMAGEREPO=<registry>`.

## 7. Deploy — Kubernetes / Helm

```bash
helm lint helm/bracket

helm install bracket ./helm/bracket \
  --set agent.env.BRACKET_INFERENCE_MODE=deterministic \
  --set agent.env.SKIP_AMD_CHECK=true
```

With a real vLLM endpoint:

```bash
helm install bracket ./helm/bracket \
  --set agent.env.LOCAL_API_BASE=http://vllm-service:8000/v1 \
  --set agent.env.EXTRACTOR_MODEL=google/gemma-3-4b-it \
  --set agent.env.ESTIMATOR_MODEL=google/gemma-3-27b-it
```

One command from source to a running cluster:

```bash
make helm-release IMAGEREPO=<your-registry> VERSION=1.0.0
```

Builds both images, pushes them, and installs the chart with
`agent.image.*` / `ui.image.*` pointed at what it just pushed.

Key values: `agent.replicaCount` (2), `agent.autoscaling.enabled` (false),
`ui.enabled` (true), `ingress.enabled` (false).

## 8. Deploy — AMD ROCm / vLLM

```bash
vllm serve google/gemma-3-27b-it --host 0.0.0.0 --port 8000 --device rocm
```

Then point `LOCAL_API_BASE` at it and drop `SKIP_AMD_CHECK`. On startup the
agent verifies the endpoint is reachable and the model is loaded; the UI's
Session tab shows live GPU utilisation and vLLM throughput.

---

## 9. Where to put this for a demo

| Audience | Path |
|---|---|
| Recruiter skimming GitHub | README + `python scripts/run_eval.py` output pasted in |
| Live 5-minute demo | agent + UI (sections 3–4), then the Hinglish curl |
| Judge with a laptop | `make install-notebook && make notebook` → Run All |
| Production pilot | Helm (section 7) with vLLM (section 8) |

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`** — run from the repo root, not
from inside `scripts/`.

**20 failures in `test_protocols.py`** — you are on Python 3.11. Use 3.12.

**UI shows no data** — the agent is not running, or not on 8080. Check
`curl localhost:8080/health` first.

**`langdetect` wheel fails to build** — `pip install wheel` first, or
`pip install --no-build-isolation langdetect`.

**Regenerate the held-out data** — edit `scripts/_build_holdout.py`, then
`python scripts/_build_holdout.py`.
