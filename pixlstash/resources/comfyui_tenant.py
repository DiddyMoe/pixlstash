"""ComfyUI as a tenant of the VRAM arbiter.

What the arbiter needs from ComfyUI, read over its HTTP API so it works on
every platform and inside Docker:

* **which card it is on**: a ComfyUI is on this machine when a card NVML sees
  has its name and size, *and* ComfyUI's free memory tracks that card's within
  ``SAME_HOST_FREE_TOLERANCE_MB`` on two reads a moment apart. The address
  alone cannot say: PixlStash in Docker reaches a host ComfyUI through
  ``host.docker.internal``, and a remote box can have the same card. A remote
  ComfyUI has no card here, and its runs are granted at once;
* **what it holds**: ``/system_stats`` ``devices[].torch_vram_total``,
  ComfyUI's reserved allocator bytes, cache included, or, for a ComfyUI on
  this machine, its process's memory as NVML reports it when that is more.
  ComfyUI's dynamic VRAM (``comfy-aimdo``) keeps model weights outside
  torch's allocator, so ``torch_vram_total`` alone reads a fraction of a GB
  while ComfyUI holds most of the card. The process is the one NVML lists on
  ComfyUI's card that listens on ComfyUI's port;
* **whether it is ours to fill**: ``/queue`` holding nothing PixlStash did not
  submit (``busy``), or nothing at all (``idle``);
* **freeing**: ``POST /free``, only on hand-off to a tenant that is not
  ComfyUI (the arbiter's reclaim ladder) or when the owner asks;
* **quitting** (#2558): what ``/free`` cannot release, its CUDA context, goes
  only when the process exits. ``kept_mb`` is that floor, the least ComfyUI has
  been seen to hold, and ``quit`` ends the process on its card that listens on
  its port, only when idle and only once the owner has said yes.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TypeVar
from urllib.parse import urlparse

import psutil
import requests

from pixlstash.pixl_logging import get_logger
from pixlstash.resources.arbiter import Tenant, arbiter
from pixlstash.resources.ledger import UNMEASURED_DEVICE, ledger
from pixlstash.utils.bounded_http import bounded_get, bounded_request

logger = get_logger(__name__)

#: How old a ``/system_stats`` or ``/queue`` read may be.
STATS_TTL_S = 1.0
#: How long a failed read stands: a hung ComfyUI costs one timeout per this,
#: not one per progress poll. An explicit fresh read (``max_age_s=0``) still asks.
FAILED_READ_TTL_S = 10.0
#: How long one ``/system_stats`` or ``/queue`` read may take in all, body
#: included, so a ComfyUI that trickles its answer is given up on.
READ_DEADLINE_S = 10.0
#: The most a ``/system_stats``, ``/queue`` or ``/free`` answer may be. A
#: queue carries every pending prompt's whole graph.
READ_MAX_BYTES = 32 * 1024 * 1024
#: How long a same-host decision stands before it is checked again.
MAPPING_TTL_S = 60.0
#: How closely ComfyUI's free memory must track the NVML card's.
SAME_HOST_FREE_TOLERANCE_MB = 512
#: How closely ComfyUI's card size must match the NVML card's. ComfyUI reports
#: CUDA's total, which leaves out what the driver keeps back: 439 MB less than
#: NVML's on a 32 GB card. Cards of one name differ by whole GBs.
SAME_CARD_TOTAL_TOLERANCE_MB = 1024
#: How long ``free`` waits for ComfyUI's memory to settle: ``unload_all_models``
#: moves weights to RAM first, so the figure falls over seconds, not at once.
FREE_SETTLE_TIMEOUT_S = 30.0
#: Two reads this close together count as settled.
FREE_SETTLE_MB = 256
#: How often a run's peak is sampled.
PEAK_SAMPLE_S = 0.5
#: How long ``quit`` waits for ComfyUI to exit after asking it to.
QUIT_WAIT_S = 30.0

_MB = 1024 * 1024
# What ``_map_device`` answers when a read failed partway: unknown, not remote.
_UNREAD = object()
_NO_PROXY = {"http": None, "https": None, "all": None}
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")
_T = TypeVar("_T")


class _Flight:
    """One read in progress, which later callers wait for instead of repeating.

    Args:
        generation: The probe's generation when the read began; its answer is
            kept only while that is still the probe's.
    """

    def __init__(self, generation: int) -> None:
        self.generation = generation
        self.done = threading.Event()
        self.value: object = None
        self.error: BaseException | None = None


class ComfyUIProbe:
    """One ComfyUI address, as the arbiter's ``TenantProbe``.

    Args:
        client_id: The client id PixlStash submits under; a queue entry with
            any other is somebody else's.
    """

    def __init__(self, client_id: str) -> None:
        self._client_id = client_id
        self._lock = threading.Lock()
        self._base_url: str | None = None
        self._stats: tuple[float, dict | None] = (0.0, None)
        self._queue: tuple[float, dict | None] = (0.0, None)
        self._mapping: tuple[float, str | None] = (0.0, None)
        self._pid: int | None = None
        # The card's pids last searched for ComfyUI's, so a search that found
        # nothing is not repeated on every poll until they change.
        self._searched: frozenset[int] = frozenset()
        # The least this ComfyUI has been seen to hold: what /free leaves.
        self._floor_mb: int | None = None
        # Bumped whenever what was read stops being this ComfyUI's (another
        # address, or its process quit): a read begun before is not kept.
        self._generation = 0
        # The reads in progress, by kind: one at a time, the rest wait for it.
        self._inflight: dict[str, _Flight] = {}

    @property
    def base_url(self) -> str | None:
        return self._base_url

    def watch(self, base_url: str) -> None:
        """Point the probe at the ComfyUI PixlStash is about to submit to."""
        with self._lock:
            if base_url != self._base_url:
                self._base_url = base_url
                self._mapping = (0.0, None)
                self._forget_process_locked()

    # ── reads ────────────────────────────────────────────────────────────

    def _get(self, path: str):
        return _read_json(f"{self._base_url}{path}")

    def _read(self, path: str, what: str) -> dict | None:
        """*path*'s answer, or None when ComfyUI cannot be read."""
        try:
            return self._get(path)
        except (requests.RequestException, ValueError) as exc:
            logger.debug(
                "ComfyUI's %s at %s cannot be read: %s", what, self._base_url, exc
            )
            return None

    def _shared(
        self, key: str, slot: str, max_age_s: float, fetch: Callable[[], _T]
    ) -> _T | None:
        """*slot*'s cached answer while it stands, else one *fetch* shared by
        every caller that arrives while it is in flight.

        The fetch runs outside the lock. Its answer is kept in *slot* only if
        the generation it began in is still the probe's (``watch`` to another
        address and ``quit`` start a new one) and it is not ``_UNREAD``.
        Whoever waited gets that answer, or the exception the fetch raised. A
        fresh read (``max_age_s=0``) must be sent after the call, so it never
        joins one begun before; it leads its own, which later callers may join.
        """
        with self._lock:
            if self._base_url is None:
                return None
            read_at, value = getattr(self, slot)
            # A failed read is cached too: a hung ComfyUI must cost one
            # timeout per TTL, not one per caller.
            if read_at and time.monotonic() - read_at < _ttl(value, max_age_s):
                return value
            flight = self._inflight.get(key) if max_age_s else None
            leading = flight is None
            if leading:
                flight = _Flight(self._generation)
                # A fresh read beside one in flight leads its own, unlisted.
                self._inflight.setdefault(key, flight)
        if not leading:
            flight.done.wait()
            if flight.error is not None:
                raise flight.error
            return flight.value
        try:
            flight.value = fetch()
        except BaseException as exc:
            flight.error = exc
            raise
        finally:
            with self._lock:
                if self._inflight.get(key) is flight:
                    del self._inflight[key]
                if flight.error is None and flight.value is not _UNREAD:
                    if flight.generation == self._generation:
                        setattr(self, slot, (time.monotonic(), flight.value))
                    else:
                        logger.debug(
                            "Not keeping ComfyUI's %s read: it began before "
                            "the address or the process changed (now %s)",
                            key,
                            self._base_url,
                        )
            flight.done.set()
        return flight.value

    def stats(self, max_age_s: float = STATS_TTL_S) -> dict | None:
        """ComfyUI's ``/system_stats``, or None when it cannot be read."""
        return self._shared(
            "/system_stats",
            "_stats",
            max_age_s,
            lambda: self._read("/system_stats", "stats"),
        )

    def _queue_now(self, max_age_s: float = STATS_TTL_S) -> dict | None:
        return self._shared(
            "/queue", "_queue", max_age_s, lambda: self._read("/queue", "queue")
        )

    def _entries(self, max_age_s: float = STATS_TTL_S) -> list | None:
        queue = self._queue_now(max_age_s)
        if not isinstance(queue, dict):
            return None
        return [
            item
            for key in ("queue_running", "queue_pending")
            for item in queue.get(key) or []
            if isinstance(item, (list, tuple))
        ]

    def _device_row(self, max_age_s: float = STATS_TTL_S) -> dict | None:
        stats = self.stats(max_age_s)
        devices = (stats or {}).get("devices") or []
        return devices[0] if devices and isinstance(devices[0], dict) else None

    # ── TenantProbe ──────────────────────────────────────────────────────

    def device(self) -> str | None:
        """The UUID of the card this ComfyUI runs on, or None for a remote one."""
        # One match at a time: its two fresh reads 0.3 s apart are shared.
        found = self._shared("mapping", "_mapping", MAPPING_TTL_S, self._find_device)
        if found is _UNREAD:
            # Unreachable is not remote: keep the last answer and ask again,
            # rather than admit this ComfyUI's runs unchecked for a minute.
            checked_at, mapped = self._mapping
            return self._unread_answer(checked_at, mapped)
        return found

    def _find_device(self) -> str | None | object:
        if self.stats() is None:
            return _UNREAD
        return self._map_device()

    @staticmethod
    def _unread_answer(checked_at: float, mapped: str | None) -> str | None:
        """The card to assume when ComfyUI cannot be read: the last one found,
        or, never found yet, PixlStash's own. None would read as remote, whose
        runs are granted unchecked; sharing PixlStash's card keeps them
        fit-checked until a read says otherwise."""
        return mapped if checked_at else ledger().pixlstash_device()

    def _map_device(self) -> str | None | object:
        # Fresh, as the NVML read below is: a cached row compared with a live
        # card misses by whatever a model load moved in between.
        first = self._device_row(max_age_s=0.0)
        if first is None:
            return _UNREAD
        cards = ledger().devices(max_age_s=0.0)
        loopback = (urlparse(self._base_url or "").hostname or "") in _LOOPBACK_HOSTS
        if cards and not cards[0].measurable:
            # No NVML to compare with (Metal, ROCm). A ComfyUI on a loopback
            # address shares this machine's one unmeasurable device.
            return UNMEASURED_DEVICE if loopback else None
        name = str(first.get("name") or "")
        total = int(first.get("vram_total") or 0) // _MB
        candidates = [
            card
            for card in cards
            if card.name
            and card.name in name
            and abs(card.total_mb - total) <= SAME_CARD_TOTAL_TOLERANCE_MB
        ]
        if not candidates:
            return None
        if loopback:
            # Proof rather than likeness: the process on the card that listens
            # on ComfyUI's port. Free memory cannot tell while a model loads,
            # and a miss there reads as remote, granted unchecked. Loopback
            # only: a remote ComfyUI may share its port with a local one.
            for card in candidates:
                if self._listening_pid(ledger().processes(card.uuid)) is not None:
                    logger.info(
                        "ComfyUI at %s runs on this machine's %s (%s): its "
                        "process is on that card",
                        self._base_url,
                        card.name,
                        card.uuid,
                    )
                    return card.uuid
        # Two reads a moment apart: an identical card on another machine has
        # the same name and size, but not the same free memory twice over.
        for attempt in range(2):
            row = first if attempt == 0 else self._device_row(max_age_s=0.0)
            if row is None:
                return _UNREAD
            free = int(row.get("vram_free") or 0) // _MB
            readings = {card.uuid: card for card in ledger().devices(max_age_s=0.0)}
            candidates = [
                card
                for card in candidates
                if card.uuid in readings
                and abs(readings[card.uuid].free_mb - free)
                <= SAME_HOST_FREE_TOLERANCE_MB
            ]
            if not candidates:
                return None
            if attempt == 0:
                time.sleep(0.3)
        logger.info(
            "ComfyUI at %s runs on this machine's %s (%s)",
            self._base_url,
            candidates[0].name,
            candidates[0].uuid,
        )
        return candidates[0].uuid

    def usage_mb(self) -> int | None:
        """What ComfyUI holds, MB; None when it cannot be read."""
        generation = self._generation
        row = self._device_row()
        if row is None:
            return None
        held = self._process_mb()
        if row.get("torch_vram_total") is None:
            usage = held
        else:
            usage = max(int(row.get("torch_vram_total") or 0) // _MB, held or 0)
        if usage is not None:
            with self._lock:
                # Read before ComfyUI changed: another one's floor.
                if generation == self._generation:
                    self._floor_mb = (
                        usage if self._floor_mb is None else min(usage, self._floor_mb)
                    )
        return usage

    def kept_mb(self) -> int | None:
        """What ComfyUI keeps after ``/free``, MB: the least it has held.

        Its CUDA context (about 780 MB on an RTX 5090, more with dynamic VRAM)
        is released only when the process exits. Before any ``/free`` the
        least seen may still include models, which overstates it; the lease
        re-reads the card after the ladder, so that only ever errs towards
        asking. None until ComfyUI has been read.
        """
        return self._floor_mb

    def can_quit(self) -> bool:
        """Whether PixlStash knows ComfyUI's process here and could end it.

        False for a remote ComfyUI, one in Docker or one NVML cannot attribute
        (no pid listening on its port on its card).
        """
        return self._pid is not None

    def quit(self, wait_s: float = QUIT_WAIT_S) -> bool:
        """End an idle ComfyUI's process: terminate, then wait for it.

        PixlStash did not start ComfyUI and cannot start it again; the caller
        has the owner's yes. Returns True once the process has exited.
        """
        with self._lock:
            # Read as one pair: a ``watch`` between two unlocked reads could
            # pair the old address's pid with the new address's generation.
            pid = self._pid
            generation = self._generation
        # Asked again now: a pid read a while ago may name another process,
        # and a queue read a second ago may have missed a job just submitted.
        if pid is not None and self._listening_pid([pid]) is None:
            pid = None
        if pid is None or not self.idle(max_age_s=0.0):
            logger.info(
                "Not quitting ComfyUI at %s: %s",
                self._base_url,
                "no process of its is known" if pid is None else "it is not idle",
            )
            return False
        with self._lock:
            if generation != self._generation:
                # Another address (or a quit) since the pid was read: the
                # port check and the queue read may have been the new
                # ComfyUI's, not this process's.
                logger.info(
                    "Not quitting ComfyUI process %s: ComfyUI changed while its "
                    "queue was read (another address or a quit; now %s)",
                    pid,
                    self._base_url,
                )
                return False
        try:
            process = psutil.Process(pid)
            process.terminate()
            process.wait(timeout=wait_s)
        except psutil.NoSuchProcess:
            logger.info("ComfyUI's process %s had already exited", pid)
        except (psutil.Error, OSError) as exc:
            # TimeoutExpired is a psutil.Error too: not escalated to a kill.
            # (On Windows psutil's terminate is already TerminateProcess.)
            logger.warning(
                "Could not quit ComfyUI at %s (process %s): %s",
                self._base_url,
                pid,
                exc,
            )
            return False
        logger.warning(
            "Quit ComfyUI at %s (process %s) to make room for training, as "
            "the owner asked; it has to be started again by hand",
            self._base_url,
            pid,
        )
        with self._lock:
            # The next ComfyUI started here is a new process with its own
            # floor, and a read begun before the quit was the old one's.
            self._forget_process_locked()
        return True

    def _forget_process_locked(self) -> None:
        """Drop what was read of ComfyUI's process; the caller holds the lock.

        A new generation, so a read still in flight from before is neither
        joined nor kept.
        """
        self._generation += 1
        self._inflight.clear()
        self._stats = (0.0, None)
        self._queue = (0.0, None)
        self._pid = None
        self._searched = frozenset()
        self._floor_mb = None

    def _process_mb(self) -> int | None:
        """ComfyUI's process memory on its card per NVML, or None unknown.

        Only for a ComfyUI already mapped to a card here. None where NVML
        cannot attribute memory (Windows WDDM, Docker) or the process is not
        found, and the allocator's figure stands alone.
        """
        with self._lock:
            generation = self._generation
            card = self._mapping[1]
        if not card or card == UNMEASURED_DEVICE:
            return None
        processes = ledger().processes(card)
        # The search under the lock, so concurrent callers do not each scan
        # every process's sockets; it is local, no HTTP.
        with self._lock:
            if generation != self._generation:
                # ComfyUI changed since its card was read: not this one's.
                return None
            if self._pid not in processes and frozenset(processes) != self._searched:
                self._searched = frozenset(processes)
                found = self._listening_pid(processes)
                if found is not None and self._pid is not None and found != self._pid:
                    # ComfyUI was restarted: its floor was another process's.
                    self._floor_mb = None
                self._pid = found
                if self._pid is None and processes:
                    logger.warning(
                        "No process on ComfyUI's card listens on %s, so its "
                        "memory is read from its allocator alone and may show "
                        "as other apps'",
                        self._base_url,
                    )
            pid = self._pid
        return processes.get(pid)

    def _listening_pid(self, pids) -> int | None:
        """Which of *pids* listens on ComfyUI's port, or None."""
        url = urlparse(self._base_url or "")
        try:
            port = url.port or (443 if url.scheme == "https" else 80)
        except ValueError as exc:
            logger.debug("ComfyUI's address %s has no port: %s", self._base_url, exc)
            return None
        for pid in pids:
            try:
                connections = psutil.Process(pid).net_connections(kind="inet")
            except psutil.Error as exc:
                logger.debug("Could not read process %s's sockets: %s", pid, exc)
                continue
            if any(
                # An empty tuple for a socket with no local address yet.
                conn.status == psutil.CONN_LISTEN
                and conn.laddr
                and conn.laddr.port == port
                for conn in connections
            ):
                return pid
        return None

    def busy(self) -> bool:
        """Work PixlStash did not submit is queued or running on ComfyUI.

        An unreadable queue is not busy. This only picks a waiting lease's
        reason, "busy with work started outside PixlStash", which would be
        false for an unreachable ComfyUI. Memory is guarded elsewhere: every
        lease is admitted against the card's free memory, which counts what
        ComfyUI holds, and reclaim asks ``idle``, which is False here.
        """
        entries = self._entries()
        if entries is None:
            return False
        return any(
            not (
                len(item) > 3
                and isinstance(item[3], dict)
                and item[3].get("client_id") == self._client_id
            )
            for item in entries
        )

    def idle(self, max_age_s: float = STATS_TTL_S) -> bool:
        """Nothing at all is queued or running on ComfyUI, per a queue read at
        most *max_age_s* old (0: one sent after the call)."""
        entries = self._entries(max_age_s)
        return entries is not None and not entries

    def jobs(self) -> int:
        """How many prompts are queued or running on ComfyUI, anyone's."""
        return len(self._entries() or [])

    def reclaim(self) -> bool:
        return self.free() is not None

    def free(self) -> int | None:
        """Ask an idle ComfyUI to unload its models; MB freed, or None if refused.

        Waits until two reads of its memory agree, at most
        ``FREE_SETTLE_TIMEOUT_S``: ``/free`` is applied by ComfyUI's idle prompt
        worker, not by the request, and moves weights to RAM first.
        """
        if self._base_url is None or not self.idle():
            return None
        before = self.usage_mb() or 0
        try:
            response = bounded_request(
                "POST",
                f"{self._base_url}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=10,
                deadline_s=10,
                max_bytes=READ_MAX_BYTES,
                proxies=_NO_PROXY,
            )
            if response.status_code != 200:
                raise requests.HTTPError(f"/free answered {response.status_code}")
        except requests.RequestException as exc:
            logger.warning(
                "ComfyUI at %s could not be asked to free memory: %s",
                self._base_url,
                exc,
            )
            return None
        deadline = time.monotonic() + FREE_SETTLE_TIMEOUT_S
        last = before
        while time.monotonic() < deadline:
            time.sleep(1.0)
            now = self.stats(max_age_s=0.0)
            current = self.usage_mb()
            if now is None or current is None:
                break
            if abs(current - last) <= FREE_SETTLE_MB and current < before:
                last = current
                break
            last = current
        logger.info(
            "Asked ComfyUI at %s to free memory: %s MB -> %s MB",
            self._base_url,
            before,
            last,
        )
        return max(0, before - last)


def _read_json(url: str):
    """*url*'s JSON answer, read within ``READ_DEADLINE_S`` and
    ``READ_MAX_BYTES``; any answer but a 200 is ``requests.HTTPError``, a
    redirect included, since it is not followed."""
    response = bounded_get(
        url,
        deadline_s=READ_DEADLINE_S,
        max_bytes=READ_MAX_BYTES,
        timeout=5,
        proxies=_NO_PROXY,
    )
    if response.status_code != 200:
        raise requests.HTTPError(f"{url} answered {response.status_code}")
    return response.json()


def _ttl(value, max_age_s: float) -> float:
    """How long a read stands: longer for a failed one, unless asked fresh."""
    return (
        max(max_age_s, FAILED_READ_TTL_S) if value is None and max_age_s else max_age_s
    )


class PeakSampler:
    """Samples ComfyUI's allocation while one run executes; ``peak_mb`` is the max.

    Runs from the moment ComfyUI starts the run until ``stop``. A run's peak
    is ComfyUI's whole allocation, cache included, which is what its next run
    will grow to again. ``room_mb`` is the room the run had at that peak: what
    ComfyUI held plus what was still free on *device*, None when unread. A
    peak that left less free than the arbiter's safety margin filled it.
    """

    def __init__(
        self, probe: ComfyUIProbe, started: callable, device: str | None = None
    ) -> None:
        self._probe = probe
        self._started = started
        self._device = device
        self._stop = threading.Event()
        self.peak_mb: int | None = None
        self.room_mb: int | None = None
        self._thread = threading.Thread(
            target=self._run, name="comfyui-peak", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(PEAK_SAMPLE_S):
            try:
                if self._started():
                    self._sample()
            except Exception:
                # One probe failure must not end the sampler: the peak it reports
                # feeds the arbiter's room figure for the rest of the run.
                logger.error(
                    "ComfyUI peak sampler failed; trying again in %ss",
                    PEAK_SAMPLE_S,
                    exc_info=True,
                )

    def _sample(self) -> None:
        self._probe.stats(max_age_s=PEAK_SAMPLE_S / 2)
        usage = self._probe.usage_mb()
        if usage is None or usage <= (self.peak_mb or 0):
            return
        self.peak_mb = usage
        # Fresh, read right after the usage: a cached card reading from
        # before it would misjudge the room while a model loads.
        card = ledger().device(self._device, max_age_s=0.0)
        self.room_mb = (
            usage + card.free_mb if card is not None and card.measurable else None
        )

    def stop(self) -> int | None:
        self._stop.set()
        self._thread.join(timeout=2)
        return self.peak_mb


_probe: ComfyUIProbe | None = None
_probe_lock = threading.Lock()


def probe(client_id: str, base_url: str | None = None) -> ComfyUIProbe:
    """The one ComfyUI probe, registered with the arbiter; pointed at *base_url*."""
    global _probe
    with _probe_lock:
        if _probe is None:
            _probe = ComfyUIProbe(client_id)
            arbiter().register_probe(Tenant.COMFYUI, _probe)
    if base_url:
        _probe.watch(base_url)
    return _probe
