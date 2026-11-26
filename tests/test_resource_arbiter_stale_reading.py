"""A snapshot answered from the last reading says so.

While a tenant hangs, the read that refreshes the cards and the tenants is
outstanding, so the card figures freeze along with the tenant's. The bounded
wait is pinned in test_resource_arbiter.py; this pins the marker.
"""

import threading

from pixlstash.resources.arbiter import SNAPSHOT_PROBE_WAIT_S, Tenant
from tests.test_resource_arbiter import HungComfy, _wait_for, make


def test_a_snapshot_from_the_last_reading_is_marked_stale():
    arb, _fake, _ = make(used_mb=12000, thread=False)
    comfy = HungComfy(usage=7000)
    arb.register_probe(Tenant.COMFYUI, comfy)

    assert arb.snapshot()["stale"] is False

    comfy.hang.set()
    try:
        timed_out = arb.snapshot()
        outstanding = arb.snapshot()
        assert timed_out["stale"] is True, "waited out, yet reported fresh"
        assert outstanding["stale"] is True, "a read outstanding, yet fresh"
        assert timed_out["devices"][0]["used_mb"] == 12000
    finally:
        comfy.release.set()
    comfy.hang.clear()
    assert _wait_for(lambda: arb.snapshot()["stale"] is False), (
        "still marked stale once the tenants answer again"
    )


# --- Two pollers of one read -------------------------------------------------
# The second caller to arrive while a read runs used to answer stale at once,
# even when that read came in a moment later, so two pollers (two tabs, or the
# progress poll beside the training room) saw the flag flap. It also answered
# fresh from a read that had failed.


def _first_snapshot_in_a_thread(arb):
    answer = {}

    def first():
        try:
            answer["snapshot"] = arb.snapshot()
        except BaseException as exc:
            answer["error"] = exc

    thread = threading.Thread(target=first, name="test-first-poller")
    thread.start()
    return thread, answer


def _release_later(event, seconds):
    timer = threading.Timer(seconds, event.set)
    timer.start()
    return timer


def test_a_second_poller_waits_for_the_read_in_flight_and_reads_fresh():
    arb, _fake, _ = make(used_mb=12000, thread=False)
    comfy = HungComfy(usage=7000)
    arb.register_probe(Tenant.COMFYUI, comfy)
    assert arb.snapshot()["stale"] is False

    comfy.hang.set()
    first, answer = _first_snapshot_in_a_thread(arb)
    try:
        assert _wait_for(lambda: comfy.reads == 2), "the first read never started"
        # The read answers well inside its deadline, after the second poller
        # has joined it.
        timer = _release_later(comfy.release, SNAPSHOT_PROBE_WAIT_S * 0.4)
        second = arb.snapshot()
        first.join(5)
        timer.join(5)
    finally:
        comfy.release.set()
    assert "error" not in answer, answer
    assert answer["snapshot"]["stale"] is False
    assert second["stale"] is False, "the second poller answered stale at once"
    assert comfy.reads == 2, "the second poller started a read of its own"


def test_a_poller_that_joined_a_read_that_failed_reads_stale():
    arb, fake, _ = make(used_mb=12000, thread=False)
    assert arb.snapshot()["stale"] is False

    entered, release = threading.Event(), threading.Event()

    def broken(max_age_s=1.0, strict=False):
        entered.set()
        release.wait(10)
        raise RuntimeError("test-ledger-failure")

    fake.devices = broken
    first, answer = _first_snapshot_in_a_thread(arb)
    try:
        assert entered.wait(5), "the first read never started"
        timer = _release_later(release, SNAPSHOT_PROBE_WAIT_S * 0.4)
        second = arb.snapshot()
        first.join(5)
        timer.join(5)
    finally:
        release.set()
    # The caller that started the read gets its error, as an inline read did;
    # the one that joined it answers from the last reading and says so.
    assert isinstance(answer.get("error"), RuntimeError), answer
    assert second["stale"] is True, "fresh figures from a read that failed"
    assert second["devices"][0]["used_mb"] == 12000
