"""Concurrency-safe asset store.

Two independent hazards are handled here, and it's worth being explicit
about which is which:

1. Data-race safety: two coroutines calling `apply_update` for the SAME
   asset at the "same time" must not interleave their read-modify-write and
   corrupt state (e.g. one update's history list getting silently dropped).
   Fixed with one asyncio.Lock per asset_id, acquired for the full
   read-modify-write. Different assets use different locks, so unrelated
   assets never block each other.

2. Order-independence: even when the two updates are properly serialized by
   the lock (no corruption either way), the RESULT must be the same
   regardless of which one the lock let through first. This is not a locking
   problem -- it's what resolver.resolve_field's deterministic comparison
   guarantees. The lock stops corruption; the resolver stops
   order-dependence. Both are required; neither alone is sufficient.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field as dc_field

from .models import AssetKind, AssetState, FieldUpdate, SourceConfig
from .resolver import ConflictReport, resolve_field


@dataclass
class ApplyResult:
    asset: AssetState
    conflict: ConflictReport | None
    changed: bool


class AssetStore:
    def __init__(self, sources: dict[str, SourceConfig]):
        self._sources = sources
        self._assets: dict[str, AssetState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()  # protects creation of per-asset locks
        self.conflict_log: list[ConflictReport] = []

    def register_asset(self, asset_id: str, kind: AssetKind, priority: int) -> None:
        if asset_id not in self._assets:
            self._assets[asset_id] = AssetState(asset_id=asset_id, kind=kind, priority=priority)

    async def _lock_for(self, asset_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            if asset_id not in self._locks:
                self._locks[asset_id] = asyncio.Lock()
            return self._locks[asset_id]

    async def apply_update(self, update: FieldUpdate) -> ApplyResult:
        lock = await self._lock_for(update.asset_id)
        async with lock:
            asset = self._assets[update.asset_id]
            current = asset.fields.get(update.field)
            new_state, conflict = resolve_field(current, update, self._sources)

            changed = current is None or current.value != new_state.value
            asset.fields[update.field] = new_state

            if conflict is not None:
                self.conflict_log.append(conflict)

            return ApplyResult(asset=asset, conflict=conflict, changed=changed)

    def get(self, asset_id: str) -> AssetState | None:
        return self._assets.get(asset_id)

    def all_assets(self) -> list[AssetState]:
        return list(self._assets.values())
