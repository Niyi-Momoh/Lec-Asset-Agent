"""Core data types shared across the agent.

An Asset has several independently-updatable fields (status, location, custodian).
Each field update carries the metadata the resolver needs to make a correct
decision without ever looking at when the agent happened to receive it:

    source_id     -- which external system produced this observation
    event_time    -- when the SOURCE says the observation occurred (not arrival time)
    seq           -- monotonic counter *within that source*, used to break ties
                     between two updates from the same source with identical
                     event_time (e.g. a coarse clock on a cheap GPS unit)
    value         -- the observed value

Deliberately NOT included: an "arrival timestamp" at the agent. Arrival order
is exactly the thing a naive last-write-wins implementation gets wrong, so the
resolver must never depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AssetKind(str, Enum):
    VEHICLE = "vehicle"
    PARCEL = "parcel"
    EQUIPMENT = "equipment"


class AssetStatus(str, Enum):
    DISPATCHED = "dispatched"
    IN_TRANSIT = "in_transit"
    DELAYED = "delayed"
    DELIVERED = "delivered"
    RETURNED = "returned"


@dataclass(frozen=True)
class SourceConfig:
    """Static, known-in-advance configuration for one external system.

    reliability is a small integer tier (higher = more trusted) used only as
    a tie-breaker when two updates claim the exact same event_time. It is
    NOT used to override a genuinely more recent observation from a less
    reliable source -- recency of the observed event wins first.
    """

    source_id: str
    name: str
    reliability: int  # 1 (low) .. 3 (high)
    poll_latency_ms: int  # simulated network/processing latency for this source


@dataclass(frozen=True)
class FieldUpdate:
    """One observation of a single field on a single asset from a single source."""

    asset_id: str
    field: str  # "status" | "location" | "custodian"
    value: object
    source_id: str
    event_time: datetime
    seq: int = 0

    def resolution_key(self, reliability: int) -> tuple:
        """Sort key used by the resolver. Higher sorts as "wins".

        Order of precedence, each only consulted if the previous one ties:
          1. event_time      -- what actually happened most recently, per the source
          2. reliability     -- which system do we trust more, given a true tie
          3. seq              -- monotonic per-source counter, last-ditch tiebreak
          4. source_id        -- deterministic total order, so two truly identical
                                  updates never depend on dict/set iteration order
        """
        return (self.event_time, reliability, self.seq, self.source_id)


@dataclass
class FieldState:
    """Current best-known value for one field, plus full provenance."""

    value: object
    winning_update: FieldUpdate
    # every update ever considered for this field, in the order the resolver
    # evaluated them -- kept so the API/decision log can show *why* a value won
    history: list[FieldUpdate] = field(default_factory=list)


@dataclass
class AssetState:
    asset_id: str
    kind: AssetKind
    priority: int  # 1 (low) .. 3 (high) -- used by the scheduler, not the resolver
    fields: dict[str, FieldState] = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "kind": self.kind.value,
            "priority": self.priority,
            "fields": {
                name: {
                    "value": fs.value,
                    "as_of": fs.winning_update.event_time.isoformat(),
                    "source": fs.winning_update.source_id,
                    "contested": len(fs.history) > 1,
                }
                for name, fs in self.fields.items()
            },
        }
