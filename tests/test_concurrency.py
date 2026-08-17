"""Proves the store handles genuinely concurrent writes safely:
no corruption, and a result that's independent of scheduling order.

Two separate properties are checked, deliberately:
  - determinism: firing the same updates in either order produces the same
    final state (order-independence, from the resolver).
  - no corruption: firing MANY concurrent updates across MANY assets at
    once never produces a state that disagrees with recomputing from the
    full update log directly (proves the locking never drops or interleaves
    a write, which order-independence alone would not catch).
"""

import asyncio
import random
from datetime import datetime, timedelta, timezone

import pytest

from lec_asset_agent.models import AssetKind, FieldUpdate, SourceConfig
from lec_asset_agent.resolver import fold_updates
from lec_asset_agent.store import AssetStore

SOURCES = {
    "scanner": SourceConfig("scanner", "Warehouse Scanner", reliability=3, poll_latency_ms=0),
    "gps": SourceConfig("gps", "GPS Tracker", reliability=2, poll_latency_ms=0),
    "manual": SourceConfig("manual", "Dispatcher Log", reliability=1, poll_latency_ms=0),
}
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_concurrent_conflicting_writes_are_order_independent():
    a = FieldUpdate("PKG-1", "status", "delivered", "scanner", NOW - timedelta(seconds=5))
    b = FieldUpdate("PKG-1", "status", "in_transit", "gps", NOW - timedelta(seconds=40))

    store_ab = AssetStore(SOURCES)
    store_ab.register_asset("PKG-1", AssetKind.PARCEL, priority=1)
    store_ba = AssetStore(SOURCES)
    store_ba.register_asset("PKG-1", AssetKind.PARCEL, priority=1)

    # asyncio.gather schedules both coroutines and lets the event loop
    # interleave them -- this is genuine concurrent access to the store,
    # not a sequential call in disguise.
    await asyncio.gather(store_ab.apply_update(a), store_ab.apply_update(b))
    await asyncio.gather(store_ba.apply_update(b), store_ba.apply_update(a))

    assert store_ab.get("PKG-1").fields["status"].value == "delivered"
    assert store_ba.get("PKG-1").fields["status"].value == "delivered"


@pytest.mark.asyncio
async def test_no_corruption_under_many_concurrent_writers():
    """Fire a large, randomized batch of concurrent updates across several
    assets/fields/sources and confirm the store's final state exactly
    matches an independent fold of the same update log. If the per-asset
    locking ever let two writers interleave a read-modify-write, this would
    show up as a mismatch (a "lost update").
    """
    random.seed(42)
    store = AssetStore(SOURCES)
    assets = ["A1", "A2", "A3"]
    for a in assets:
        store.register_asset(a, AssetKind.EQUIPMENT, priority=1)

    all_updates: list[FieldUpdate] = []
    for i in range(300):
        asset = random.choice(assets)
        field = random.choice(["status", "location"])
        source = random.choice(list(SOURCES))
        value = f"val-{random.randint(0, 5)}"
        event_time = NOW - timedelta(seconds=random.randint(0, 1000))
        all_updates.append(FieldUpdate(asset, field, value, source, event_time, seq=i))

    # Fire all 300 updates concurrently (event loop decides interleaving).
    await asyncio.gather(*(store.apply_update(u) for u in all_updates))

    for asset in assets:
        for field in ["status", "location"]:
            relevant = [u for u in all_updates if u.asset_id == asset and u.field == field]
            if not relevant:
                continue
            expected = fold_updates(relevant, SOURCES)
            actual = store.get(asset).fields[field]
            assert actual.value == expected.value, (
                f"corruption on {asset}.{field}: store has {actual.value!r}, "
                f"independent fold says {expected.value!r}"
            )
