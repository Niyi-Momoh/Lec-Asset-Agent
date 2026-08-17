"""REST query interface.

GET  /assets              -- current best-known state of every asset
GET  /assets/{asset_id}   -- current best-known state of one asset, with provenance
GET  /decision-log        -- the agent's step-by-step scheduling reasoning
GET  /conflicts           -- every write-write conflict detected and how it was resolved
POST /step                -- manually trigger one more scheduling step (handy for the demo/video)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .world import build_world

STARTUP_STEPS = 9  # 3 assets x 3 sources -- covers every seeded pair at least once


@asynccontextmanager
async def lifespan(app: FastAPI):
    store, sources, agent = await build_world()
    await agent.run(STARTUP_STEPS)
    app.state.store = store
    app.state.agent = agent
    yield


app = FastAPI(title="LEC Asset Sync Agent", lifespan=lifespan)


@app.get("/assets")
async def list_assets():
    return [a.snapshot() for a in app.state.store.all_assets()]


@app.get("/assets/{asset_id}")
async def get_asset(asset_id: str):
    asset = app.state.store.get(asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail=f"unknown asset '{asset_id}'")
    return asset.snapshot()


@app.get("/decision-log")
async def decision_log():
    return [e.to_dict() for e in app.state.agent.decision_log]


@app.get("/conflicts")
async def conflicts():
    return [
        {
            "asset_id": c.asset_id,
            "field": c.field,
            "winner_value": c.winner.value,
            "winner_source": c.winner.source_id,
            "naive_lww_would_pick": c.naive_lww_would_pick.value,
            "naive_lww_was_wrong": c.naive_lww_was_wrong,
        }
        for c in app.state.store.conflict_log
    ]


@app.post("/step")
async def step():
    entry = await app.state.agent.step()
    return entry.to_dict()
