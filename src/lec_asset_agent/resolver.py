"""Deterministic, order-independent conflict resolution.

This module is the one piece of the system that MUST be pure and
deterministic -- the assessment requires the final state to be "correct
regardless of arrival order", and an LLM call is not a fit for that
guarantee. Everything here is plain comparison logic; the LLM is used
elsewhere (agent.py) for the fuzzier "what should I look at next" decision,
never for "which write wins".

The core idea is a per-field, per-asset Last-Writer-Wins register, but keyed
by the SOURCE's claimed event_time rather than the AGENT's receipt time, with
source reliability as an explicit tiebreak. This is what makes it correct
where naive LWW (keyed by receipt/processing order) is not: two updates
resolve to the same winner no matter which one the agent happens to process
first, because the winner is computed from data attached to the updates
themselves, not from processing order.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import AssetState, FieldState, FieldUpdate, SourceConfig


@dataclass
class ConflictReport:
    asset_id: str
    field: str
    incoming: FieldUpdate
    incumbent: FieldUpdate
    winner: FieldUpdate
    naive_lww_would_pick: FieldUpdate  # whichever arrived later at the agent
    naive_lww_was_wrong: bool


def resolve_field(
    current: FieldState | None,
    incoming: FieldUpdate,
    sources: dict[str, SourceConfig],
) -> tuple[FieldState, ConflictReport | None]:
    """Apply `incoming` to `current`, returning the new FieldState and,
    if this update genuinely conflicted with the incumbent value, a report.

    A "conflict" here means: a different update was already on file for this
    field, AND the two updates disagree on value. Agreeing updates (or a
    first-ever update) are not conflicts, just normal writes.
    """
    incoming_reliability = sources[incoming.source_id].reliability

    if current is None:
        return FieldState(value=incoming.value, winning_update=incoming, history=[incoming]), None

    incumbent = current.winning_update
    history = current.history + [incoming]

    if incumbent.value == incoming.value:
        # Same value reported again (e.g. a redundant poll) -- not a conflict.
        # Still worth keeping whichever update is more authoritative on file.
        pass

    incumbent_reliability = sources[incumbent.source_id].reliability
    incoming_key = incoming.resolution_key(incoming_reliability)
    incumbent_key = incumbent.resolution_key(incumbent_reliability)

    winner = incoming if incoming_key > incumbent_key else incumbent

    new_state = FieldState(value=winner.value, winning_update=winner, history=history)

    if incumbent.value == incoming.value:
        return new_state, None

    # Genuine disagreement -- figure out what naive last-write-wins (i.e.
    # "whichever update the agent happened to apply last") would have done,
    # so we can show, concretely, when it would have been wrong. In this
    # call, `incoming` is-by-construction the one being applied last, so
    # naive LWW always picks `incoming`.
    naive_pick = incoming
    naive_wrong = naive_pick.value != winner.value

    report = ConflictReport(
        asset_id=incoming.asset_id,
        field=incoming.field,
        incoming=incoming,
        incumbent=incumbent,
        winner=winner,
        naive_lww_would_pick=naive_pick,
        naive_lww_was_wrong=naive_wrong,
    )
    return new_state, report


def fold_updates(
    updates: list[FieldUpdate],
    sources: dict[str, SourceConfig],
) -> FieldState:
    """Recompute a field's state from scratch given ALL updates, in any
    order. Used by tests as an independent invariant check: applying updates
    one-by-one through the live store must always agree with folding the
    full update log, no matter what order either happens in.
    """
    state: FieldState | None = None
    for u in updates:
        state, _ = resolve_field(state, u, sources)
    assert state is not None
    return state
