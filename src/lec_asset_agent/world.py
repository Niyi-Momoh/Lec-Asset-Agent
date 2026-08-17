"""Wires together the store, sources, and agent, and seeds the demo scenario.

Shared by both the REST API and the standalone demo script so they show
exactly the same scenario.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .agent import AssetSyncAgent
from .llm import build_reasoner
from .models import AssetKind
from .sources import default_sources
from .store import AssetStore


async def build_world(model: str = "llama3.2:1b") -> tuple[AssetStore, dict, AssetSyncAgent]:
    sources = default_sources()
    store = AssetStore(sources={sid: s.config for sid, s in sources.items()})

    store.register_asset("PKG-1042", AssetKind.PARCEL, priority=3)
    store.register_asset("TRUCK-07", AssetKind.VEHICLE, priority=2)
    store.register_asset("EQUIP-88", AssetKind.EQUIPMENT, priority=1)

    now = datetime.now(timezone.utc)

    # -- Scenario 1: recency conflict --------------------------------------
    # The GPS tracker's last ping before delivery is genuinely OLDER than the
    # warehouse scan that confirms delivery, but the GPS update is queued
    # behind extra network latency (see poll_latency_ms in sources.py) and so
    # arrives at the agent AFTER the scanner's update. A naive last-write-wins
    # (whichever the agent processes last) would let the stale GPS ping
    # clobber the correct "delivered" status. Our resolver compares
    # event_time, not arrival order, so it doesn't matter which one the
    # agent happens to poll first.
    sources["warehouse_scanner"].seed(
        "PKG-1042", "status", "delivered", event_time=now - timedelta(seconds=5)
    )
    sources["gps_tracker"].seed(
        "PKG-1042", "status", "in_transit", event_time=now - timedelta(seconds=40)
    )

    # -- Scenario 2: reliability conflict (event_time tie) ------------------
    # Two sources report a genuinely simultaneous, contradictory observation
    # for the truck's custodian field (coarse-grained manual log entry vs.
    # scanner, both timestamped to the same minute). With event_time tied,
    # the resolver falls back to source reliability.
    tie_time = now - timedelta(minutes=2)
    sources["dispatcher_log"].seed("TRUCK-07", "custodian", "J. Alabi (manual override)", event_time=tie_time)
    sources["warehouse_scanner"].seed("TRUCK-07", "custodian", "R. Chen (badge scan)", event_time=tie_time)

    # -- Uncontested baseline data, just so every asset has some state ------
    sources["gps_tracker"].seed("TRUCK-07", "location", "A2 Northbound, mile 14", event_time=now - timedelta(seconds=10))
    sources["warehouse_scanner"].seed("EQUIP-88", "status", "dispatched", event_time=now - timedelta(minutes=30))

    reasoner = await build_reasoner(model=model)
    agent = AssetSyncAgent(store=store, sources=sources, reasoner=reasoner)
    return store, sources, agent
