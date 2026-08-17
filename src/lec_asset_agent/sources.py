"""Stubbed external operational systems.

Per the assessment brief, these are in-memory stand-ins for the kind of
systems a real fleet/warehouse operation would have. The realism of the
transport doesn't matter -- what matters is that each one behaves like a
genuinely independent system:

  * its own clock (event_time it reports is its own, not the agent's)
  * its own reliability characteristics
  * its own latency, so "who the agent asks last" and "who is telling the
    truth" are decoupled -- this is what makes out-of-order arrival possible
    to simulate deterministically in a demo/test rather than relying on
    real network jitter.

Each source exposes an async `poll(asset_id)` that returns zero or more
FieldUpdates "currently available" for that asset -- i.e. this call is the
thing the agent's scheduler decides to invoke, and when.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from .models import FieldUpdate, SourceConfig


class SimulatedSource:
    def __init__(self, config: SourceConfig):
        self.config = config
        # asset_id -> field -> FieldUpdate, the events "seeded" into this system
        self._events: dict[str, dict[str, FieldUpdate]] = {}
        self._seq = 0

    def seed(self, asset_id: str, field: str, value: object, event_time: datetime) -> FieldUpdate:
        """Load an observation into this source, as if it had just recorded it."""
        self._seq += 1
        update = FieldUpdate(
            asset_id=asset_id,
            field=field,
            value=value,
            source_id=self.config.source_id,
            event_time=event_time,
            seq=self._seq,
        )
        self._events.setdefault(asset_id, {})[field] = update
        return update

    async def poll(self, asset_id: str) -> list[FieldUpdate]:
        """Simulate the latency of actually reaching this system over the network."""
        await asyncio.sleep(self.config.poll_latency_ms / 1000)
        return list(self._events.get(asset_id, {}).values())


def default_sources() -> dict[str, SimulatedSource]:
    configs = [
        SourceConfig(source_id="warehouse_scanner", name="Warehouse Barcode Scanner",
                     reliability=3, poll_latency_ms=20),
        SourceConfig(source_id="gps_tracker", name="Vehicle GPS Telematics",
                     reliability=2, poll_latency_ms=180),
        SourceConfig(source_id="dispatcher_log", name="Manual Dispatcher Log",
                     reliability=1, poll_latency_ms=60),
    ]
    return {c.source_id: SimulatedSource(c) for c in configs}
