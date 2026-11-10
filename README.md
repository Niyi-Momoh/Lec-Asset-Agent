# LEC Asset Sync Agent

[![CI](https://github.com/Niyi-Momoh/Lec-Asset-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Niyi-Momoh/Lec-Asset-Agent/actions/workflows/ci.yml)

An agent that maintains a live, best-known inventory of physical assets (a parcel, a
vehicle, and a piece of equipment) across three independent, concurrently-updating
operational systems. It actively decides what to check next, detects write-write
conflicts between sources, resolves them deterministically (not with last-write-wins),
and stays correct under real concurrent writes.

Built for the LEC AI build assessment. See [`WALKTHROUGH.md`](WALKTHROUGH.md) for a
full explanation of every design decision, written so I can defend each one on camera.

## What it does, in one run

```
python demo/run_demo.py
```

This seeds two genuine conflicts into three stub systems, lets the agent poll them on
its own schedule for 6 steps (with a live reasoning log), and prints:

1. The raw, contradictory ground truth sitting in each stub source.
2. The agent's own decision log — which (asset, source) pair it chose to poll at each
   step, and why, in its own words.
3. Both conflicts it resolved, with a **side-by-side comparison against what naive
   last-write-wins would have produced** — and confirmation naive LWW would have been
   wrong both times.
4. A live concurrency check: the same two conflicting updates applied in both possible
   arrival orders, proving the final state is identical either way.

## Requirements checklist (how each is met)

| Requirement | Where |
|---|---|
| Agent actively decides what to query next (staleness/priority-driven, not passive) | [`agent.py`](src/lec_asset_agent/agent.py) `AssetSyncAgent._candidates` / `.step` |
| Decision log with reasoning across ≥3 sequential steps | `AssetSyncAgent.decision_log`, exposed at `GET /decision-log`; demo runs 6 steps |
| Detects & resolves a genuine write-write conflict, better than naive LWW | [`resolver.py`](src/lec_asset_agent/resolver.py); demo scenario 1 (recency) + scenario 2 (reliability tiebreak) |
| Concurrent writes don't corrupt state; deterministic regardless of arrival order | [`store.py`](src/lec_asset_agent/store.py) per-asset `asyncio.Lock` + deterministic resolver; proven in `tests/test_concurrency.py` |
| Query interface returning current best-known state | REST API, [`api.py`](src/lec_asset_agent/api.py) |
| Local model | Ollama (`llama3.2:1b` by default), pluggable with a deterministic fallback — see [`llm.py`](src/lec_asset_agent/llm.py) |

## How to run

### 1. Set up the environment

```bash
python -m venv .venv
.venv/Scripts/activate        # on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. (Optional but recommended) Install Ollama and pull a small model

The agent runs perfectly well without this — it falls back to a deterministic
template-based reasoner automatically. With Ollama, the decision log and conflict
explanations are generated live by a real local model.

```bash
# https://ollama.com/download
ollama pull llama3.2:1b
```

### 3. Run the tests

```bash
pytest -v
```

11 tests covering: deterministic conflict resolution, order-independence (folding
updates forwards vs. backwards produces the same state), a 300-update concurrent
stress test checked against an independent recomputation, and scheduler/decision-log
behavior.

### 4. Run the demo

```bash
python demo/run_demo.py
```

This is what the walkthrough video shows running.

### 5. Run the REST API

```bash
uvicorn lec_asset_agent.api:app --app-dir src --port 8000
```

On startup it seeds the same scenario as the demo and runs 9 sync steps (covering
every asset/source pair once) before serving requests.

```bash
curl http://localhost:8000/assets                 # every asset's current best-known state
curl http://localhost:8000/assets/PKG-1042         # one asset, with provenance
curl http://localhost:8000/decision-log            # the agent's scheduling reasoning
curl http://localhost:8000/conflicts               # every conflict detected + how it was resolved
curl -X POST http://localhost:8000/step            # manually trigger one more sync step
```

## Architecture, briefly

```
sources.py   -- 3 stubbed external systems (warehouse scanner, GPS tracker,
                dispatcher log), each with its own reliability tier and latency
resolver.py  -- pure, deterministic conflict resolution (no LLM, no I/O):
                compares event_time -> source reliability -> seq -> source_id
store.py     -- async-safe store: one lock per asset, applies resolver's decision
agent.py     -- scheduler: scores every (asset, source) pair by staleness x
                priority, polls the top one, logs its reasoning
llm.py       -- pluggable reasoning backend (Ollama, with a heuristic fallback)
                used ONLY to narrate decisions already made deterministically
api.py       -- FastAPI query interface
```

The one decision worth flagging up front: **the LLM never decides which value wins a
conflict.** That's a deterministic comparison in `resolver.py`. The LLM is used to
decide *what to poll next* and to *explain* decisions in plain language for the
decision log. This is why the "deterministic regardless of arrival order" requirement
can be satisfied by a test, rather than hoped for from a model. Full reasoning for this
split, and every other decision, is in [`WALKTHROUGH.md`](WALKTHROUGH.md).

## What I'd do next with more time

- **Persistence.** Everything is in-memory; a restart loses state. I'd add SQLite (or
  Redis if this needed to run distributed) with the same resolver logic applied on
  write, so recovery replays deterministically to the same state.
- **Clock-skew handling.** The resolver trusts each source's `event_time` as given. In
  reality, a cheap GPS unit's clock might drift minutes off true time, which would
  silently corrupt recency-based resolution. I'd add a per-source clock-offset estimate
  (e.g. compare source-reported time to receipt time over many samples) and correct for
  it before comparing.
- **Structured field merges.** Right now each field is a single opaque value with one
  winner. A real location update might have lat/lon/heading/speed — I'd want field-level
  merge (accept lat/lon from the more reliable source, speed from the more recent one)
  rather than one wins-take-all per field.
- **Smarter scheduling.** The current scorer is staleness × priority. I'd move to
  something closer to a multi-armed bandit: sources that have recently gone unreachable
  should be backed off, not repeatedly hammered every cycle.
- **Bigger local model.** `llama3.2:1b` occasionally gets a comparison backwards in its
  prose explanation (e.g. describes the more reliable source as less reliable) even
  though the underlying resolution is correct — the resolver doesn't trust the LLM's
  wording, but a 3B+ model would make the decision log more trustworthy to read.
- **Push instead of poll for the query side.** A WebSocket/SSE stream of state changes
  would suit a real ops dashboard better than polling `GET /assets`.
- **Auth + multi-tenant asset scoping**, since a real fleet system would need this
  before the API could be exposed to anyone.
