"""The active scheduling agent.

This is the piece that satisfies the "must actively decide which system to
query next... rather than passively responding to requests" requirement.
Each step it:

  1. Scores every (asset, source) pair it knows about by staleness (how long
     since that pair was last polled) weighted by the asset's priority.
  2. Picks the highest-scoring pair and polls that source for that asset.
  3. Applies whatever updates come back through the store (which does the
     deterministic conflict resolution).
  4. Records a decision-log entry explaining the choice in natural language
     (via the pluggable LLM/heuristic reasoner) and what happened as a
     result -- including a plain explanation of any conflict it resolved.

Nothing here decides *which value wins* a conflict -- see resolver.py. This
module only decides *what to look at next* and *how to narrate it*.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone

from .llm import ReasoningBackend
from .models import FieldUpdate
from .sources import SimulatedSource
from .store import AssetStore


@dataclass
class DecisionLogEntry:
    step: int
    timestamp: datetime
    asset_id: str
    source_id: str
    staleness_s: float
    score: float
    reasoning: str
    updates_applied: list[dict]
    conflicts: list[dict]
    reasoner_used: str

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "timestamp": self.timestamp.isoformat(),
            "asset_id": self.asset_id,
            "source_id": self.source_id,
            "staleness_s": round(self.staleness_s, 2),
            "score": round(self.score, 2),
            "reasoning": self.reasoning,
            "updates_applied": self.updates_applied,
            "conflicts": self.conflicts,
            "reasoner_used": self.reasoner_used,
        }


class AssetSyncAgent:
    def __init__(
        self,
        store: AssetStore,
        sources: dict[str, SimulatedSource],
        reasoner: ReasoningBackend,
    ):
        self.store = store
        self.sources = sources
        self.reasoner = reasoner
        self.decision_log: list[DecisionLogEntry] = []
        # (asset_id, source_id) -> monotonic time of last poll; unpolled pairs
        # get float("inf") staleness so every pair is tried at least once.
        self._last_polled: dict[tuple[str, str], float] = {}
        self._step = 0

    def _candidates(self) -> list[dict]:
        now = time.monotonic()
        candidates = []
        for asset in self.store.all_assets():
            for source_id, source in self.sources.items():
                last = self._last_polled.get((asset.asset_id, source_id))
                staleness = float("inf") if last is None else now - last
                # Cap "infinite" staleness to something large-but-finite so the
                # score stays a normal float for logging/display purposes.
                staleness_capped = min(staleness, 3600.0)
                score = staleness_capped * asset.priority
                candidates.append({
                    "asset_id": asset.asset_id,
                    "source_id": source_id,
                    "source_name": source.config.name,
                    "priority": asset.priority,
                    "staleness_s": staleness_capped,
                    "score": score,
                })
        return candidates

    async def step(self) -> DecisionLogEntry:
        self._step += 1
        candidates = self._candidates()
        chosen = max(candidates, key=lambda c: c["score"])

        reasoning = await self.reasoner.explain_scheduling_choice(
            chosen=f"{chosen['asset_id']} via {chosen['source_name']}",
            candidates=candidates,
        )

        updates: list[FieldUpdate] = await self.sources[chosen["source_id"]].poll(chosen["asset_id"])
        self._last_polled[(chosen["asset_id"], chosen["source_id"])] = time.monotonic()

        applied = []
        conflicts = []
        for update in updates:
            result = await self.store.apply_update(update)
            applied.append({"field": update.field, "value": update.value, "changed": result.changed})
            if result.conflict is not None:
                c = result.conflict
                loser = c.incumbent if c.winner is not c.incumbent else c.incoming
                reason = (
                    "more recent event_time" if c.winner.event_time != loser.event_time
                    else "higher-reliability source, event_time tied"
                )
                explanation = await self.reasoner.explain_conflict(
                    asset_id=c.asset_id,
                    field=c.field,
                    winner={"value": c.winner.value, "source": c.winner.source_id,
                            "event_time": c.winner.event_time.isoformat()},
                    loser={"value": loser.value, "source": loser.source_id,
                           "event_time": loser.event_time.isoformat()},
                    reason=reason,
                )
                conflicts.append({
                    "field": c.field,
                    "winner_value": c.winner.value,
                    "winner_source": c.winner.source_id,
                    "naive_lww_would_pick": c.naive_lww_would_pick.value,
                    "naive_lww_was_wrong": c.naive_lww_was_wrong,
                    "explanation": explanation,
                })

        entry = DecisionLogEntry(
            step=self._step,
            timestamp=datetime.now(timezone.utc),
            asset_id=chosen["asset_id"],
            source_id=chosen["source_id"],
            staleness_s=chosen["staleness_s"],
            score=chosen["score"],
            reasoning=reasoning,
            updates_applied=applied,
            conflicts=conflicts,
            reasoner_used=self.reasoner.name,
        )
        self.decision_log.append(entry)
        return entry

    async def run(self, steps: int) -> list[DecisionLogEntry]:
        return [await self.step() for _ in range(steps)]
