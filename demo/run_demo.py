"""End-to-end demo, meant to be run and screen-recorded as-is.

Run with:  python demo/run_demo.py

Walks through, in order:
  1. The raw (conflicting) ground truth seeded into each stub source.
  2. The agent actively scheduling and running several sync steps, with its
     reasoning logged at each step.
  3. The two conflicts it hit, resolved value vs. what naive last-write-wins
     (whichever update the agent happened to process last) would have picked.
  4. A concurrency-safety check: firing the same two conflicting updates in
     both possible orders and proving the final state is identical either way.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lec_asset_agent.models import FieldUpdate
from lec_asset_agent.world import build_world


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


async def main() -> None:
    hr("1. GROUND TRUTH SEEDED INTO EACH STUB SOURCE (independent systems)")
    store, sources, agent = await build_world()
    for sid, source in sources.items():
        for asset_id, fields in source._events.items():
            for f, u in fields.items():
                print(f"  [{sid:18s}] {asset_id:10s} {f:10s} = {u.value!r:35s} @ {u.event_time.isoformat()}")

    print(f"\n  Reasoning backend in use: {agent.reasoner.name}"
          + ("  (install Ollama + `ollama pull llama3.2:1b` for live LLM reasoning)"
             if agent.reasoner.name == "heuristic" else ""))

    hr("2. AGENT ACTIVELY SCHEDULING SYNC STEPS")
    entries = await agent.run(6)
    for e in entries:
        print(f"\n  step {e.step} [{e.reasoner_used}] -> polled {e.source_id} for {e.asset_id} "
              f"(staleness {e.staleness_s:.1f}s, score {e.score:.2f})")
        print(f"    reasoning: {e.reasoning}")
        for u in e.updates_applied:
            print(f"    applied: {u['field']} = {u['value']!r} (changed={u['changed']})")
        for c in e.conflicts:
            print(f"    CONFLICT on {c['field']}: kept {c['winner_value']!r} from {c['winner_source']}")
            print(f"      naive last-write-wins would have picked: {c['naive_lww_would_pick']!r}"
                  f"  (wrong={c['naive_lww_was_wrong']})")
            print(f"      explanation: {c['explanation']}")

    hr("3. FINAL BEST-KNOWN STATE OF EVERY ASSET")
    for asset in store.all_assets():
        print(f"\n  {asset.asset_id} ({asset.kind.value}, priority={asset.priority})")
        for field, fs in asset.fields.items():
            flag = "  <-- was contested" if len(fs.history) > 1 else ""
            print(f"    {field:10s} = {fs.value!r:35s} (source: {fs.winning_update.source_id}){flag}")

    hr("4. CONFLICT SUMMARY: agent vs. naive last-write-wins")
    if not store.conflict_log:
        print("  (no conflicts recorded)")
    for c in store.conflict_log:
        verdict = "NAIVE LWW WAS WRONG" if c.naive_lww_was_wrong else "naive LWW happened to agree here"
        print(f"\n  {c.asset_id}.{c.field}:")
        print(f"    incumbent : {c.incumbent.value!r:20s} from {c.incumbent.source_id:18s} @ {c.incumbent.event_time.isoformat()}")
        print(f"    incoming  : {c.incoming.value!r:20s} from {c.incoming.source_id:18s} @ {c.incoming.event_time.isoformat()}")
        print(f"    agent kept: {c.winner.value!r:20s} from {c.winner.source_id}")
        print(f"    -> {verdict}")

    hr("5. CONCURRENCY-SAFETY CHECK: same two conflicting updates, both orders")
    a = FieldUpdate(asset_id="ORDER-TEST", field="status", value="A-wins",
                     source_id="warehouse_scanner", event_time=store.get("PKG-1042").fields["status"].winning_update.event_time)
    b = FieldUpdate(asset_id="ORDER-TEST", field="status", value="B-wins",
                     source_id="gps_tracker", event_time=a.event_time)  # exact tie -> reliability breaks it
    store.register_asset("ORDER-TEST", store.get("PKG-1042").kind, priority=1)

    from lec_asset_agent.store import AssetStore
    store_ab = AssetStore(sources=store._sources)
    store_ab.register_asset("ORDER-TEST", store.get("PKG-1042").kind, priority=1)
    store_ba = AssetStore(sources=store._sources)
    store_ba.register_asset("ORDER-TEST", store.get("PKG-1042").kind, priority=1)

    await asyncio.gather(store_ab.apply_update(a), store_ab.apply_update(b))
    await asyncio.gather(store_ba.apply_update(b), store_ba.apply_update(a))

    final_ab = store_ab.get("ORDER-TEST").fields["status"].value
    final_ba = store_ba.get("ORDER-TEST").fields["status"].value
    print(f"  applied [A then B] concurrently -> final value = {final_ab!r}")
    print(f"  applied [B then A] concurrently -> final value = {final_ba!r}")
    assert final_ab == final_ba, "ORDER DEPENDENCE BUG"
    print(f"  MATCH -- final state is deterministic regardless of arrival order "
          f"(reliability tiebreak picked {final_ab!r}, the higher-reliability source)")


if __name__ == "__main__":
    asyncio.run(main())
