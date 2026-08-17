"""Correctness of the deterministic conflict resolver, independent of the
agent/scheduler entirely -- these tests construct FieldUpdates directly."""

from datetime import datetime, timedelta, timezone

import pytest

from lec_asset_agent.models import FieldUpdate, SourceConfig
from lec_asset_agent.resolver import fold_updates, resolve_field

SOURCES = {
    "scanner": SourceConfig("scanner", "Warehouse Scanner", reliability=3, poll_latency_ms=0),
    "gps": SourceConfig("gps", "GPS Tracker", reliability=2, poll_latency_ms=0),
    "manual": SourceConfig("manual", "Dispatcher Log", reliability=1, poll_latency_ms=0),
}

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def upd(source, value, seconds_ago, seq=0, field="status", asset="X"):
    return FieldUpdate(
        asset_id=asset, field=field, value=value, source_id=source,
        event_time=NOW - timedelta(seconds=seconds_ago), seq=seq,
    )


def test_more_recent_event_time_wins_even_if_it_arrives_first():
    older = upd("gps", "in_transit", seconds_ago=40)
    newer = upd("scanner", "delivered", seconds_ago=5)

    state, conflict = resolve_field(None, older, SOURCES)
    state, conflict = resolve_field(state, newer, SOURCES)

    assert state.value == "delivered"
    assert conflict is not None
    assert conflict.naive_lww_was_wrong is False  # newer applied last == naive agrees here


def test_naive_lww_would_be_wrong_when_stale_update_arrives_last():
    """This is the case the assessment asks us to demonstrate: an
    out-of-order arrival where a naive 'whatever I processed most recently
    wins' implementation gets the wrong answer, but our resolver does not.
    """
    newer_arrives_first = upd("scanner", "delivered", seconds_ago=5)
    older_arrives_second = upd("gps", "in_transit", seconds_ago=40)

    state, _ = resolve_field(None, newer_arrives_first, SOURCES)
    state, conflict = resolve_field(state, older_arrives_second, SOURCES)

    # correct: recency-aware resolution keeps "delivered"
    assert state.value == "delivered"
    # and we can show naive LWW (= whatever was applied last) would have said "in_transit"
    assert conflict.naive_lww_would_pick.value == "in_transit"
    assert conflict.naive_lww_was_wrong is True


def test_tied_event_time_breaks_on_reliability():
    low = upd("manual", "returned", seconds_ago=120)
    high = upd("scanner", "delivered", seconds_ago=120)

    state, _ = resolve_field(None, low, SOURCES)
    state, conflict = resolve_field(state, high, SOURCES)
    assert state.value == "delivered"

    # and the reverse application order gives the same answer
    state2, _ = resolve_field(None, high, SOURCES)
    state2, _ = resolve_field(state2, low, SOURCES)
    assert state2.value == "delivered"


def test_tied_event_time_and_reliability_breaks_on_seq_then_source_id():
    a = upd("scanner", "A", seconds_ago=10, seq=1)
    b = upd("scanner", "B", seconds_ago=10, seq=2)  # same source, later internal sequence number
    state, _ = resolve_field(None, a, SOURCES)
    state, _ = resolve_field(state, b, SOURCES)
    assert state.value == "B"


def test_resolution_is_independent_of_application_order():
    updates = [
        upd("gps", "in_transit", seconds_ago=40),
        upd("scanner", "delivered", seconds_ago=5),
        upd("manual", "delayed", seconds_ago=100),
    ]
    forward = fold_updates(updates, SOURCES)
    backward = fold_updates(list(reversed(updates)), SOURCES)
    assert forward.value == backward.value == "delivered"


def test_agreeing_updates_are_not_reported_as_conflicts():
    a = upd("scanner", "delivered", seconds_ago=10)
    b = upd("gps", "delivered", seconds_ago=5)  # different source, same value
    state, _ = resolve_field(None, a, SOURCES)
    state, conflict = resolve_field(state, b, SOURCES)
    assert conflict is None
    assert state.value == "delivered"
