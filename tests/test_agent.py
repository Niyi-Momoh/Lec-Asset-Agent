"""Scheduler behaviour: active staleness/priority-driven polling and a
decision log with reasoning across multiple sequential steps.

Uses the heuristic reasoning backend explicitly so this test is
deterministic and doesn't depend on Ollama being installed/running.
"""

from datetime import datetime, timedelta, timezone

import pytest

from lec_asset_agent.agent import AssetSyncAgent
from lec_asset_agent.llm import HeuristicBackend
from lec_asset_agent.models import AssetKind
from lec_asset_agent.sources import default_sources
from lec_asset_agent.store import AssetStore

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_scheduler_prefers_higher_priority_never_polled_asset():
    sources = default_sources()
    store = AssetStore({sid: s.config for sid, s in sources.items()})
    store.register_asset("LOW", AssetKind.EQUIPMENT, priority=1)
    store.register_asset("HIGH", AssetKind.PARCEL, priority=3)

    agent = AssetSyncAgent(store=store, sources=sources, reasoner=HeuristicBackend())
    entry = await agent.step()

    # Both assets are equally "never polled" (infinite/capped staleness) at
    # step 1, so the tie is broken by priority -- the scheduler should not
    # pick arbitrarily/alphabetically, it should pick the higher-priority asset.
    assert entry.asset_id == "HIGH"


@pytest.mark.asyncio
async def test_decision_log_has_reasoning_across_at_least_three_steps():
    sources = default_sources()
    store = AssetStore({sid: s.config for sid, s in sources.items()})
    store.register_asset("PKG-1042", AssetKind.PARCEL, priority=3)
    store.register_asset("TRUCK-07", AssetKind.VEHICLE, priority=2)

    sources["warehouse_scanner"].seed("PKG-1042", "status", "delivered", NOW - timedelta(seconds=5))
    sources["gps_tracker"].seed("PKG-1042", "status", "in_transit", NOW - timedelta(seconds=40))

    agent = AssetSyncAgent(store=store, sources=sources, reasoner=HeuristicBackend())
    entries = await agent.run(4)

    assert len(entries) == 4
    for e in entries:
        assert e.reasoning  # non-empty natural-language justification
        assert e.asset_id and e.source_id

    # the resolved conflict should show up somewhere in the log
    all_conflicts = [c for e in entries for c in e.conflicts]
    assert any(c["field"] == "status" for c in all_conflicts)


@pytest.mark.asyncio
async def test_scheduler_does_not_repeatedly_poll_the_same_pair_when_others_are_staler():
    sources = default_sources()
    store = AssetStore({sid: s.config for sid, s in sources.items()})
    store.register_asset("A", AssetKind.EQUIPMENT, priority=1)
    store.register_asset("B", AssetKind.EQUIPMENT, priority=1)

    agent = AssetSyncAgent(store=store, sources=sources, reasoner=HeuristicBackend())
    seen_pairs = set()
    for _ in range(len(sources) * 2):  # enough steps to cover both assets x all sources once
        entry = await agent.step()
        seen_pairs.add((entry.asset_id, entry.source_id))

    # every (asset, source) pair should have been visited at least once
    # before any pair is revisited a third+ time -- i.e. it's spreading
    # coverage by staleness, not fixating on one pair.
    expected_pairs = {("A", sid) for sid in sources} | {("B", sid) for sid in sources}
    assert seen_pairs == expected_pairs
