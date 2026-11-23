"""What happened to each picture a background task was handed.

A task that reports only the pictures it changed leaves its finder guessing
about the rest. A picture that failed for a reason that will pass (a missing
model, a full disk, an out-of-memory decode) looks exactly like one that was
never attempted, so the finder selects it again on the next sweep, a few
seconds later, and the pipeline spins on the same failing pictures. This module
is the vocabulary a task uses to say, for every picture it was given, which of
four things happened:

- ``SUCCEEDED``: the stage's output was written.
- ``FAILED_RETRYABLE``: the environment failed (backend, I/O, memory, writer,
  a decode that did not return in time); the file is presumed healthy and is
  retried later, never marked broken.
- ``FAILED_PERMANENT``: the file's bytes were read and cannot be decoded.
- ``DEFERRED``: not attempted (cancelled, or not started before the task
  stopped waiting for its decodes).

and :class:`RetryHold`, the in-memory capped, jittered exponential backoff a
finder applies to retryable failures.

Stdlib only, on purpose: tasks, finders and the work planner all import it, and
none of them may pull another in through it.
"""

import random
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable


class Outcome(str, Enum):
    """The fate of one picture in one task."""

    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_PERMANENT = "failed_permanent"
    DEFERRED = "deferred"


class Reason(str, Enum):
    """Why a picture did not succeed."""

    UNDECODABLE = "undecodable"
    NO_FRAMES = "no_frames"
    UNREACHABLE = "unreachable"
    IO_TRANSIENT = "io_transient"
    RESOURCE = "resource"
    LOAD_ERROR = "load_error"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    INFERENCE_ERROR = "inference_error"
    SCORER_ERROR = "scorer_error"
    PERSIST_ERROR = "persist_error"
    CANCELLED = "cancelled"
    ADMISSION_REFUSED = "admission_refused"
    DECODE_TIMEOUT = "decode_timeout"


#: How many picture ids a log line names; the rest are counted.
LOG_ID_LIMIT = 20


def id_sample(ids) -> list:
    """The first :data:`LOG_ID_LIMIT` of ``ids``, sorted, for a log line."""
    return sorted(ids)[:LOG_ID_LIMIT]


def outcomes_payload(
    entries: Iterable[tuple[int, Outcome, "Reason | None"]],
) -> dict:
    """The ``outcomes`` part of a task result, one ``(pid, outcome, reason)``
    per picture, in the order given.
    """
    return {
        "outcomes": [
            (int(pid), Outcome(outcome), Reason(reason) if reason else None)
            for pid, outcome, reason in entries
        ]
    }


@dataclass(frozen=True)
class OutcomeSummary:
    """A task's outcomes, grouped by picture id.

    Attributes:
        succeeded: Pictures whose stage output was written.
        failed_retryable: Pictures that failed for a reason that may pass.
        failed_permanent: Pictures whose file cannot be decoded.
        deferred: Pictures that were not attempted.
        reasons: The reason recorded for each picture that did not succeed.
    """

    succeeded: frozenset
    failed_retryable: frozenset
    failed_permanent: frozenset
    deferred: frozenset
    reasons: dict

    @property
    def failed(self) -> frozenset:
        """Every picture that was attempted and did not succeed."""
        return self.failed_retryable | self.failed_permanent


def read_outcomes(result) -> "OutcomeSummary | None":
    """Group a task result's ``outcomes`` by kind.

    Returns ``None`` for a result that carries no ``outcomes`` (a task that has
    not adopted this contract), so a caller can count its pictures as unknown
    rather than as done. An outcome value that is not an :class:`Outcome`
    raises ``ValueError``: it is a bug in the task, and guessing what it meant
    would hide it.
    """
    if not isinstance(result, dict) or "outcomes" not in result:
        return None
    groups: dict[Outcome, set] = {outcome: set() for outcome in Outcome}
    reasons: dict = {}
    for pid, outcome, reason in result["outcomes"]:
        groups[Outcome(outcome)].add(pid)
        if reason is not None:
            reasons[pid] = Reason(reason)
    return OutcomeSummary(
        succeeded=frozenset(groups[Outcome.SUCCEEDED]),
        failed_retryable=frozenset(groups[Outcome.FAILED_RETRYABLE]),
        failed_permanent=frozenset(groups[Outcome.FAILED_PERMANENT]),
        deferred=frozenset(groups[Outcome.DEFERRED]),
        reasons=reasons,
    )


class RetryHold:
    """Keys that may not be retried yet, with capped, jittered exponential backoff.

    The n-th consecutive :meth:`hold` of a key keeps it out for a random wait in
    ``[d/2, d]``, where ``d = min(cap_s, base_s * 2**n)``. The half-delay floor
    is there because a background sweeper must never retry at once; the jitter
    keeps a batch of pictures that failed together from coming back together.
    Past the cap, attempts go on at the cap's rate. A hold is never escalated to
    a permanent mark: retryable means the file is presumed healthy.

    In memory, per instance: a restart forgets every hold.
    """

    #: ``2.0 ** n`` overflows a float a little past n = 1023; any exponent this
    #: large is already far past the cap.
    _MAX_EXPONENT = 64

    def __init__(
        self,
        base_s: float = 2.0,
        cap_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._base_s = float(base_s)
        self._cap_s = float(cap_s)
        self._clock = clock
        self._rng = rng
        self._lock = threading.Lock()
        # key -> (consecutive holds, time it may be retried)
        self._entries: dict = {}

    def hold(self, key) -> float:
        """Hold ``key`` once more and return how long it is held for, in seconds."""
        with self._lock:
            attempts = self._entries.get(key, (0, 0.0))[0] + 1
            delay = min(
                self._cap_s,
                self._base_s * 2.0 ** min(attempts, self._MAX_EXPONENT),
            )
            wait = delay / 2.0 + self._rng() * delay / 2.0
            self._entries[key] = (attempts, self._clock() + wait)
            return wait

    def clear(self, key) -> None:
        """Forget ``key``: its next failure starts the backoff from the beginning."""
        with self._lock:
            self._entries.pop(key, None)

    def active(self) -> set:
        """The keys whose hold has not expired yet."""
        with self._lock:
            now = self._clock()
            return {
                key for key, (_, ready_at) in self._entries.items() if ready_at > now
            }
