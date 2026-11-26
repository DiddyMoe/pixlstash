"""Removing a penalised tag: the owner's decision and the cleanup it starts.

The penalised-tag list is the owner's (the hub's user row). Removing a tag
from it with X does three things, in this order:

1. drops its weight and records it, with the tags merged into it, under
   ``excluded_tags`` in the owner's tagger settings, bumping the exclusion
   generation so every library's tagger and import path stops producing it;
2. records the smart-score invalidation the weight change owes, before the hub
   write, as ``PATCH /users/me/config`` does;
3. starts a :class:`TagExclusionCleanupTask` that takes the tag off the active
   library's pictures.

Adding a tag back with a weight lifts the exclusion; nothing removed earlier
is restored. The job list is in memory, so after a restart ``status`` reports
the work still to do as ``interrupted`` from a live count, and a retry
(``start_cleanup``) finishes it.
"""

import json
import threading

from fastapi import HTTPException
from sqlmodel import Session

from pixlstash.database import DBPriority
from pixlstash.db_models import User
from pixlstash.db_models.tag import (
    DEFAULT_SMART_SCORE_PENALIZED_TAG_WEIGHT,
    DEFAULT_SMART_SCORE_PENALIZED_TAGS,
)
from pixlstash.pixl_logging import get_logger
from pixlstash.tasks.tag_exclusion_cleanup_task import (
    ACTIVE_STATES,
    TagExclusionCleanupTask,
    pictures_carrying,
)
from pixlstash.utils.quality.smart_score_utils import smart_score_penalised_tags
from pixlstash.utils.quality.tag_exclusion import (
    exclusion_generation,
    exclusions_from_tagger_settings,
    expand_aliases,
    normalise_tag,
    with_exclusions,
)
from pixlstash.utils.service.smart_score_invalidation import (
    apply_pending_invalidations,
    changed_penalised_tags,
    record_pending_invalidation,
)
from pixlstash.utils.service.user_settings_utils import apply_user_config_patch

logger = get_logger(__name__)

#: Finished jobs kept per library for the status route.
JOBS_KEPT = 20

_LOCK = threading.Lock()
_JOBS: dict[str, list[TagExclusionCleanupTask]] = {}


def status(vault) -> dict:
    """What is removed from tagging, what is left to clean up, and the jobs.

    ``remaining`` counts the unlocked pictures, the Scrapheap included, that
    still carry a removed tag and ``locked`` the locked ones; both are read
    live, so they are true after a restart too. ``state`` is ``running``
    while a job is queued or running, ``interrupted`` when none is and
    pictures are left, otherwise ``idle``.
    """
    excluded = vault.db.tag_exclusions
    remaining, locked = vault.db.run_immediate_read_task(_carrying, excluded)
    jobs = [task.snapshot() for task in reversed(_jobs(vault))]
    if any(job["state"] in ACTIVE_STATES for job in jobs):
        state = "running"
    elif remaining:
        state = "interrupted"
    else:
        state = "idle"
    return {
        "library_uuid": vault.library_uuid,
        "excluded": sorted(excluded),
        "remaining": remaining,
        "locked": locked,
        "state": state,
        "jobs": jobs,
    }


def start_cleanup(vault, names, generation: int) -> dict:
    """Start a job removing *names* from this library's pictures.

    Returns the job as the status route reports it. With no task runner the
    job is recorded as ``not_started`` and the status reads ``interrupted``.
    The job is listed before it is submitted, so a submission that raises
    (a dead worker whose replacement cannot start) marks it ``not_started``
    too before the error reaches the caller: left ``queued``, it would read
    as running and ``retry_cleanup`` would refuse with 409 until a restart.
    """
    task = TagExclusionCleanupTask(vault, names, generation)
    with _LOCK:
        jobs = _JOBS.setdefault(vault.library_uuid, [])
        jobs.append(task)
        del jobs[:-JOBS_KEPT]
    try:
        task_id = vault.submit_task(task)
    except BaseException as exc:
        task.mark_not_started()
        logger.error(
            "Submitting the cleanup of %s in library %s raised %s: %s; the job "
            "is marked not started and reads as interrupted until retried.",
            sorted(task.params["names"]),
            vault.library_uuid,
            type(exc).__name__,
            exc,
        )
        raise
    if task_id is None:
        task.mark_not_started()
        logger.warning(
            "No task runner took the cleanup of %s in library %s; it reads as "
            "interrupted until it is retried.",
            sorted(task.params["names"]),
            vault.library_uuid,
        )
    return task.snapshot()


def retry_cleanup(vault, names=None) -> dict:
    """Start a job for *names* (default: every removed tag) in this library.

    Raises:
        HTTPException: ``422`` when a name is not removed from tagging,
            ``409`` while a job is still running here.
        TaskAdmissionRefused: A library restore has paused task admission
            (409), asked before the job is listed.
    """
    vault.refuse_while_admission_paused()
    excluded = vault.db.tag_exclusions
    wanted = excluded if names is None else frozenset(normalise_tag(n) for n in names)
    if not wanted <= excluded:
        raise HTTPException(
            status_code=422,
            detail=(
                "Only a tag removed from tagging can be cleaned up: "
                + ", ".join(sorted(wanted - excluded))
            ),
        )
    if any(task.snapshot()["state"] in ACTIVE_STATES for task in _jobs(vault)):
        raise HTTPException(
            status_code=409,
            detail="Removed tags are already being taken off pictures here.",
        )
    return start_cleanup(vault, wanted, vault.db.tag_exclusion_generation)


def remove_penalised_tag(server, user_id: int, name: str) -> dict:
    """Remove *name* from the penalised tags and from tagging, then clean up.

    Raises:
        HTTPException: ``404`` when *name* is neither penalised nor removed.
        TaskAdmissionRefused: A library restore has paused task admission
            (409). Asked before the owner's settings are written, since the
            removal exists to start the cleanup job.
    """
    server.vault.refuse_while_admission_paused()
    result, newly = _change_owner_settings(server, user_id, name, None)
    return {**result, "cleanup": _carry_on(server.vault, newly, result["excluded"])}


def set_penalised_tag(server, user_id: int, name: str, weight: int) -> dict:
    """Penalise *name* with *weight*, and let the tagger produce it again.

    Lifting an exclusion moves the generation, so a job still running for
    other removed tags stops at its next batch; a new job finishes those.
    """
    result, _newly = _change_owner_settings(server, user_id, name, weight)
    return {**result, "cleanup": _carry_on(server.vault, set(), result["excluded"])}


def _jobs(vault) -> list[TagExclusionCleanupTask]:
    with _LOCK:
        return list(_JOBS.get(vault.library_uuid, []))


def _carry_on(vault, names, excluded) -> dict | None:
    """Start a job for *names* and for the names of jobs that are stopping.

    A queued or running job planned on an older generation stops at its next
    batch; its names still removed, and not already carried by a job on the
    current generation, go into the new job. Returns ``None`` when there is
    nothing to start.
    """
    generation = vault.db.tag_exclusion_generation
    stopping: set[str] = set()
    carried: set[str] = set()
    for task in _jobs(vault):
        if task.snapshot()["state"] not in ACTIVE_STATES:
            continue
        if task.params["generation"] == generation:
            carried.update(task.params["names"])
        else:
            stopping.update(task.params["names"])
    wanted = (set(names) | (stopping - carried)) & set(excluded)
    if not wanted:
        return None
    return start_cleanup(vault, wanted, generation)


def _carrying(session: Session, names) -> tuple[int, int]:
    return (
        len(pictures_carrying(session, names)),
        len(pictures_carrying(session, names, locked=True)),
    )


def _resolved_weights(raw) -> dict:
    return smart_score_penalised_tags(
        raw,
        DEFAULT_SMART_SCORE_PENALIZED_TAGS,
        default_weight=DEFAULT_SMART_SCORE_PENALIZED_TAG_WEIGHT,
    )


def _planned(user: User, name: str, weight: int | None):
    """The weights and tagger settings *user* has after the change.

    ``weight=None`` removes *name* and the tags merged into it; otherwise
    *name* gets *weight* and leaves the exclusions with those tags.
    """
    weights = _resolved_weights(user.smart_score_penalised_tags)
    settings = json.loads(user.tagger_settings or "{}")
    excluded = exclusions_from_tagger_settings(settings)
    family = expand_aliases([name])
    if weight is None:
        if name not in {normalise_tag(tag) for tag in weights} | excluded:
            raise HTTPException(
                status_code=404, detail=f"'{name}' is not a penalised tag."
            )
        new_weights = {
            tag: value
            for tag, value in weights.items()
            if normalise_tag(tag) not in family
        }
        new_excluded = excluded | family
    else:
        new_weights = {
            tag: value for tag, value in weights.items() if normalise_tag(tag) != name
        }
        new_weights[name] = weight
        new_excluded = excluded - family
    if new_excluded != excluded:
        settings = with_exclusions(settings, new_excluded)
    return weights, new_weights, settings, new_excluded - excluded


def _change_owner_settings(server, user_id: int, name: str, weight: int | None):
    """Write the owner's new weights and exclusions, as the config PATCH does.

    Returns the response body and the names this write newly removed.

    The weight diff is recorded as a smart-score invalidation in the vault
    before the hub write (see ``PATCH /users/me/config``), applied after it,
    and the new exclusions reach this process's tagging at once.
    """

    def preview(session: Session) -> set[str]:
        user = session.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        old, new, _settings, _newly = _planned(user, name, weight)
        return changed_penalised_tags(old, new)

    def write(session: Session):
        user = session.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        # Planned again from the row this turn writes, so the diff below is
        # the one this write makes even if another write landed in between.
        old, new, settings, newly = _planned(user, name, weight)
        apply_user_config_patch(user, {"smart_score_penalised_tags": new})
        user.tagger_settings = json.dumps(settings)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user, changed_penalised_tags(old, new), newly

    recorded = server.hub_engine.run_immediate_read_task(preview)
    if recorded:
        _record_invalidation(server, recorded)
    user, changed, newly = server.hub_engine.run_task(
        write, priority=DBPriority.IMMEDIATE
    )
    server.auth.user = user
    server._user = user
    late = changed - recorded
    if late:
        logger.warning(
            "Penalised tags changed between preview and write; recording the "
            "invalidation for %s after the write.",
            ", ".join(sorted(late)),
        )
        _record_invalidation(server, late)
    if changed:
        try:
            server.vault.db.run_task(
                apply_pending_invalidations, priority=DBPriority.IMMEDIATE
            )
        except Exception:
            logger.exception(
                "Penalised tags changed (%s) and the invalidation is recorded, "
                "but applying it now failed. Scores for pictures carrying those "
                "tags stay stale until the pending-invalidation finder drains "
                "the record.",
                ", ".join(sorted(changed)),
            )
        server.vault.wake()
    settings = json.loads(user.tagger_settings or "{}")
    server.vault.set_tagger_settings(settings)
    logger.info(
        "Penalised tag '%s' %s; removed from tagging: %s (generation %s).",
        name,
        "removed" if weight is None else f"set to {weight}",
        sorted(exclusions_from_tagger_settings(settings)),
        exclusion_generation(settings),
    )
    response = {
        "smart_score_penalised_tags": _resolved_weights(
            user.smart_score_penalised_tags
        ),
        "excluded": sorted(exclusions_from_tagger_settings(settings)),
    }
    return response, newly


def _record_invalidation(server, tags: set[str]) -> None:
    server.vault.db.run_task(
        lambda session: (
            record_pending_invalidation(session, tags),
            session.commit(),
        ),
        priority=DBPriority.IMMEDIATE,
    )
