"""A finished run's import waits for the library and never fails the run for it.

``Vault._import_training_run_under_lease`` admits the run only while its vault
is the active library and no restore or switch holds admission, and refuses it
before anything lands otherwise. The copy to the shelf (the hub's, which a
switch does not replace) runs with no lease held, so a large LoRA never makes a
switch wait out its readers; the attachment to the run's set or character
writes the vault's database, so it alone runs under a read lease. A refusal
there leaves the LoRA on the shelf, unattached, and the retry answers from the
shelf and attaches it. The refusal's reason is read under the same lock as the
refusal (``acquire_read_or_raise``). ``TrainingService._finalize`` answers the
refusal:

- SWITCHING: it waits and asks again; the LoRA is the owner's work.
- UNAVAILABLE, or another vault active: it leaves the row as it is, and the
  reconcile the next time the library opens imports the run.

Never "LoRA import failed". A run whose row is gone imports nothing and leaves
its files where they are.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

import pytest

from pixlstash.event_types import EventType
from pixlstash.hub import training_runs
from pixlstash.hub.db import now_iso
from pixlstash.services import training_service as training_service_module
from pixlstash.services.library_generation_coordinator import (
    LibraryGenerationCoordinator,
    LibraryNotReadyError,
    LibraryReadLease,
    SwitchState,
)
from pixlstash.services.model_shelf_service import UnknownAttachmentEntityError
from pixlstash.training.backend import TrainingProgress
from pixlstash.vault import NotTheActiveLibraryError, Vault, _move_trained_attachment
from tests.training import test_training_service as service_tests
from tests.training.test_training_service import (
    _RESOLUTION,
    _SET_ID,
    _CapturingNotify,
    _make_dataset_builder,
    _make_service,
    _StubLease,
    _write_template,
)

# The fake ai-toolkit checkout the service tests run against.
repo_dir = service_tests.repo_dir


class _Coordinator:
    """The coordinator's lease surface, counting the leases still held."""

    def __init__(self, vault=None, state=SwitchState.READY):
        self.vault = vault
        self.state = state
        self.held = 0
        self.released = 0

    def acquire_read(self):
        if self.state is not SwitchState.READY:
            return None
        self.held += 1
        return LibraryReadLease(1, "library-under-test", self.vault, object())

    def acquire_read_or_raise(self):
        lease = self.acquire_read()
        if lease is None:
            raise LibraryNotReadyError(self.state)
        return lease

    def release_read(self, lease):
        self.held -= 1
        self.released += 1


def _stand_in(coordinator, body):
    """A vault with a coordinator, whose import body is *body*."""
    vault = SimpleNamespace(
        library_coordinator=coordinator,
        image_root="/home/me/library",
        _import_training_run=body,
    )
    if coordinator is not None and coordinator.vault is None:
        coordinator.vault = vault
    return vault


_INFO = {"id": 3, "name": "Clementine", "set_id": None, "character_id": None}
_COVER = "c" * 64


@pytest.fixture
def attachments(monkeypatch):
    """Each attachment the import makes, with the leases held while it ran."""
    made = []

    def attach(vault, info, previous, sha):
        made.append(
            SimpleNamespace(
                vault=vault,
                info=info,
                previous=previous,
                sha=sha,
                held=vault.library_coordinator.held
                if vault.library_coordinator is not None
                else None,
            )
        )

    monkeypatch.setattr("pixlstash.vault._move_trained_attachment", attach)
    return made


def _landing(coordinator, cover=_COVER, then=None):
    """An import body: records the leases held, runs *then*, lands *cover*."""
    seen = []

    def body(run_dir, info, *, attach=True):
        seen.append({"held": coordinator.held, "attach": attach})
        if then is not None:
            then()
        return {"stack_id": 7, "imported": 1, "cover_sha256": cover}

    return body, seen


def test_the_copy_runs_with_no_lease_held(attachments):
    coordinator = _Coordinator()
    body, seen = _landing(coordinator, cover=None)
    vault = _stand_in(coordinator, body)

    Vault._import_training_run_under_lease(vault, "/home/me/run", _INFO)

    assert seen == [{"held": 0, "attach": False}], "the copy held a library lease"
    assert attachments == []
    assert coordinator.held == 0 and coordinator.released == 1


def test_the_attachment_runs_under_a_lease_and_gives_it_back(attachments):
    coordinator = _Coordinator()
    body, seen = _landing(coordinator)
    vault = _stand_in(coordinator, body)

    result = Vault._import_training_run_under_lease(vault, "/home/me/run", _INFO)

    assert result["cover_sha256"] == _COVER
    assert seen == [{"held": 0, "attach": False}]
    assert [(a.vault, a.info, a.previous, a.sha, a.held) for a in attachments] == [
        (vault, _INFO, None, _COVER, 1)
    ], "the attachment wrote the vault without a lease"
    assert coordinator.held == 0 and coordinator.released == 2


def test_a_switch_that_begins_during_the_copy_refuses_the_attachment(attachments):
    coordinator = _Coordinator()

    def begin_switch():
        coordinator.state = SwitchState.SWITCHING

    body, seen = _landing(coordinator, then=begin_switch)
    vault = _stand_in(coordinator, body)

    with pytest.raises(LibraryNotReadyError) as raised:
        Vault._import_training_run_under_lease(vault, "/home/me/run", _INFO)

    assert raised.value.state is SwitchState.SWITCHING
    assert len(seen) == 1 and attachments == []
    assert coordinator.held == 0


def test_another_vault_active_after_the_copy_attaches_nothing(attachments):
    coordinator = _Coordinator()

    def switch_away():
        coordinator.vault = object()

    body, seen = _landing(coordinator, then=switch_away)
    vault = _stand_in(coordinator, body)

    with pytest.raises(NotTheActiveLibraryError):
        Vault._import_training_run_under_lease(vault, "/home/me/run", _INFO)

    assert len(seen) == 1
    assert attachments == [], "attached in a library that did not train the run"
    assert coordinator.held == 0


def test_a_failed_attachment_still_gives_its_lease_back(monkeypatch):
    coordinator = _Coordinator()
    body, _ = _landing(coordinator)

    def attach(vault, info, previous, sha):
        raise RuntimeError("example-injected attachment failure")

    monkeypatch.setattr("pixlstash.vault._move_trained_attachment", attach)
    vault = _stand_in(coordinator, body)

    with pytest.raises(RuntimeError):
        Vault._import_training_run_under_lease(vault, "/home/me/run", _INFO)
    assert coordinator.held == 0 and coordinator.released == 2


def test_a_failed_import_still_gives_its_lease_back():
    coordinator = _Coordinator()

    def body(run_dir, info, **kw):
        raise RuntimeError("example-injected importer failure")

    vault = _stand_in(coordinator, body)
    with pytest.raises(RuntimeError):
        Vault._import_training_run_under_lease(vault, "/home/me/run", _INFO)
    assert coordinator.held == 0


@pytest.mark.parametrize("state", [SwitchState.SWITCHING, SwitchState.UNAVAILABLE])
def test_no_lease_means_nothing_is_imported(state):
    calls = []
    vault = _stand_in(_Coordinator(state=state), lambda *a: calls.append(a))

    with pytest.raises(LibraryNotReadyError) as raised:
        Vault._import_training_run_under_lease(vault, "/home/me/run", _INFO)

    assert raised.value.state is state
    assert calls == []


def test_a_vault_that_is_no_longer_the_active_library_imports_nothing():
    other = object()
    coordinator = _Coordinator(vault=other)
    calls = []
    vault = _stand_in(coordinator, lambda *a: calls.append(a))

    with pytest.raises(NotTheActiveLibraryError) as raised:
        Vault._import_training_run_under_lease(vault, "/home/me/run", _INFO)

    assert raised.value.state is SwitchState.READY
    assert calls == [], "imported into a library that did not train the run"
    assert coordinator.held == 0


def test_a_vault_nobody_published_imports_without_a_lease(caplog):
    calls = []
    vault = _stand_in(None, lambda *a: calls.append(a) or {})

    with caplog.at_level(logging.INFO, logger="pixlstash.vault"):
        Vault._import_training_run_under_lease(vault, "/home/me/run", _INFO)

    assert len(calls) == 1
    assert "imports without a library lease" in caplog.text


class _Registry:
    def __init__(self, library):
        self.library = library

    def active_library(self):
        return self.library


def _coordinating(vault):
    """A real coordinator over a coherent runtime whose vault is *vault*."""
    db = SimpleNamespace(is_open=True)
    vault.is_open = True
    vault.db = db
    server = SimpleNamespace(
        vault=vault,
        auth=SimpleNamespace(vault_db=db),
        library_registry=_Registry(
            SimpleNamespace(uuid="library-under-test", path=vault.image_root)
        ),
    )
    coordinator = LibraryGenerationCoordinator(server)
    vault.library_coordinator = coordinator
    return coordinator


class _SwitchEndsOnUnlock:
    """The coordinator's lock, where the switch ends the moment it is let go:
    a reason read after the refusal, outside the lock, reads READY."""

    def __init__(self, coordinator):
        self._inner = coordinator._condition
        self._coordinator = coordinator

    def __enter__(self):
        return self._inner.__enter__()

    def __exit__(self, *exc):
        try:
            return self._inner.__exit__(*exc)
        finally:
            self._coordinator.state = SwitchState.READY

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_a_coherent_library_admits_the_run_and_has_its_leases_back(monkeypatch):
    body_saw, attach_saw = [], []
    vault = _stand_in(None, None)
    coordinator = _coordinating(vault)

    def body(run_dir, info, *, attach=True):
        body_saw.append(dict(coordinator._readers))
        return {"stack_id": 7, "imported": 1, "cover_sha256": _COVER}

    def attach(vault_, info, previous, sha):
        attach_saw.append(dict(coordinator._readers))

    vault._import_training_run = body
    monkeypatch.setattr("pixlstash.vault._move_trained_attachment", attach)

    Vault._import_training_run_under_lease(vault, "/home/me/run", _INFO)

    assert body_saw == [{}] and attach_saw == [{0: 1}]
    assert coordinator._readers == {}


def test_a_refusal_during_a_switch_says_switching_though_the_switch_ends_at_once():
    calls = []
    vault = _stand_in(None, lambda *a, **kw: calls.append(a))
    coordinator = _coordinating(vault)
    coordinator.state = SwitchState.SWITCHING
    coordinator._condition = _SwitchEndsOnUnlock(coordinator)

    with pytest.raises(LibraryNotReadyError) as raised:
        Vault._import_training_run_under_lease(vault, "/home/me/run", _INFO)

    # READY would read as another library being active, and the run would
    # wait for the next start instead of for the switch.
    assert raised.value.state is SwitchState.SWITCHING
    assert not isinstance(raised.value, NotTheActiveLibraryError)
    assert calls == []


class _SwitchBeginsOnUnlock(_SwitchEndsOnUnlock):
    """The coordinator's lock, where a switch begins the moment it is let go:
    a reason read after the lease was granted, outside the lock, reads
    SWITCHING."""

    def __exit__(self, *exc):
        try:
            return self._inner.__exit__(*exc)
        finally:
            self._coordinator.state = SwitchState.SWITCHING


def _make_active(coordinator, vault):
    """Publish *vault* as the coherent active library of *coordinator*."""
    vault.is_open = True
    vault.db = coordinator.server.auth.vault_db
    coordinator.server.vault = vault


@pytest.mark.parametrize("refused_at", ["admission", "attachment"])
def test_another_vault_active_says_ready_though_a_switch_begins_at_once(
    refused_at, attachments
):
    coordinator = _coordinating(_stand_in(None, None))
    vault = _stand_in(None, None)
    vault.library_coordinator = coordinator
    other = SimpleNamespace(image_root=vault.image_root)
    copies = []

    def body(run_dir, info, *, attach=True):
        copies.append(attach)
        _make_active(coordinator, other)
        coordinator._condition = _SwitchBeginsOnUnlock(coordinator)
        return {"stack_id": 7, "imported": 1, "cover_sha256": _COVER}

    vault._import_training_run = body
    if refused_at == "admission":
        _make_active(coordinator, other)
        coordinator._condition = _SwitchBeginsOnUnlock(coordinator)
    else:
        _make_active(coordinator, vault)

    with pytest.raises(NotTheActiveLibraryError) as raised:
        Vault._import_training_run_under_lease(vault, "/home/me/run", _INFO)

    # The lease was granted while admission was READY, and another vault being
    # active is why it was refused. SWITCHING, read after the lock let go,
    # would send the run to wait out a switch that is not the reason.
    assert raised.value.state is SwitchState.READY
    assert copies == ([] if refused_at == "admission" else [False])
    assert attachments == []
    assert coordinator._readers == {}


def test_admission_with_no_coherent_library_open_is_unavailable():
    calls = []
    vault = _stand_in(None, lambda *a, **kw: calls.append(a))
    coordinator = _coordinating(vault)
    # Auth bound to another database than the vault's: never admitted.
    coordinator.server.auth.vault_db = SimpleNamespace(is_open=True)

    with pytest.raises(LibraryNotReadyError) as raised:
        Vault._import_training_run_under_lease(vault, "/home/me/run", _INFO)

    assert raised.value.state is SwitchState.UNAVAILABLE
    assert calls == []
    assert coordinator.state is SwitchState.READY and coordinator._readers == {}


def _on_the_shelf(hub, run_id, *models):
    """Put *models* ``(sha256, training_step, stack_position)`` of a run on the
    shelf, in one stack; answer the stack's id."""
    with hub.transaction() as conn:
        stack_id = conn.execute(
            "INSERT INTO adapter_stack (name, created_at) VALUES (?, ?)",
            ("Clementine", now_iso()),
        ).lastrowid
        for sha, step, position in models:
            conn.execute(
                "INSERT INTO model (file_kind, kind, sha256, provenance, training_run_id, "
                "training_step, stack_id, stack_position) "
                "VALUES ('adapter', 'lora', ?, 'trained', ?, ?, ?, ?)",
                (sha, run_id, step, stack_id, position),
            )
    return stack_id


@pytest.mark.parametrize("attach", [True, False], ids=["attaches", "leased-apart"])
def test_a_retry_answers_from_the_shelf_and_attaches_its_cover(
    hub, attachments, attach
):
    """Refused at its attachment, a run is on the shelf, unattached. The retry
    imports nothing again; it answers from the shelf, the cover being the head
    of the run's stack (the owner's chosen step, here step 2)."""
    stack_id = _on_the_shelf(hub, _INFO["id"], ("e" * 64, None, 1), ("d" * 64, 2, 0))
    vault = SimpleNamespace(_hub=hub, library_coordinator=None)

    result = Vault._import_training_run(vault, "/home/me/run", _INFO, attach=attach)

    assert result == {"stack_id": stack_id, "imported": 2, "cover_sha256": "d" * 64}
    expected = [(vault, _INFO, None, "d" * 64)] if attach else []
    assert [(a.vault, a.info, a.previous, a.sha) for a in attachments] == expected


def test_a_run_whose_set_is_gone_keeps_its_lora_on_the_shelf(monkeypatch, caplog):
    monkeypatch.setattr("pixlstash.vault.fetch_attachments", lambda vault, sha256: {})

    def refuse(vault, sha, entities):
        raise UnknownAttachmentEntityError("set", 9)

    monkeypatch.setattr("pixlstash.vault.replace_attachments", refuse)
    run = {"id": 3, "name": "Clementine", "set_id": 9, "character_id": None}

    with caplog.at_level(logging.WARNING, logger="pixlstash.vault"):
        _move_trained_attachment(object(), run, None, "b" * 64)

    assert "Training run 3 (Clementine): its set 9 no longer exists" in caplog.text


@pytest.fixture
def finishing(hub, repo_dir, tmp_path, monkeypatch):
    """A service, its run's row, and a final checkpoint on disk."""
    monkeypatch.setattr(training_service_module, "_LIBRARY_NOT_READY_RETRY_S", 0.01)
    notify = _CapturingNotify()
    imports = []
    answers = []

    def importer(run_dir, info):
        imports.append(run_dir)
        answer = answers.pop(0) if answers else None
        if isinstance(answer, BaseException):
            raise answer
        if callable(answer):
            return answer()
        return {"stack_id": 7, "imported": 1, "cover_sha256": "a" * 64}

    service = _make_service(
        hub,
        repo_dir,
        lease=_StubLease(grant=True),
        notify=notify,
        dataset_builder=_make_dataset_builder(),
        importer=importer,
    )
    job = service.enqueue(
        set_id=_SET_ID,
        name="admission",
        base_model="m",
        trigger_word="t",
        template_path=_write_template(repo_dir, steps=4, save_every=2),
        resolution=_RESOLUTION,
    )
    final = tmp_path / "run" / "admission.safetensors"
    final.parent.mkdir()
    final.write_bytes(b"example-not-a-lora")
    progress = TrainingProgress(
        status="completed",
        info="",
        step=4,
        total_steps=4,
        speed="",
        final_checkpoint=str(final),
    )
    info = training_runs.get_run_by_uuid(hub, job["uuid"])
    return SimpleNamespace(
        hub=hub,
        service=service,
        notify=notify,
        imports=imports,
        answers=answers,
        info=info,
        progress=progress,
        final=final,
    )


def _finalize(env):
    env.service._finalize(None, None, env.info, env.progress)
    return training_runs.get_run_by_uuid(env.hub, env.info["uuid"])


def _failed_messages(env):
    return [
        data.get("message")
        for event, data in env.notify.events
        if event == EventType.PLUGIN_PROGRESS
        and isinstance(data, dict)
        and data.get("status") == "error"
    ]


def test_an_import_refused_by_a_restore_waits_and_lands(finishing):
    finishing.answers.extend(
        [
            LibraryNotReadyError(SwitchState.SWITCHING),
            LibraryNotReadyError(SwitchState.SWITCHING),
        ]
    )

    row = _finalize(finishing)

    assert len(finishing.imports) == 3, "the import did not ask again"
    assert row["status"] == "completed"
    assert row["error"] is None
    assert _failed_messages(finishing) == []


@pytest.mark.parametrize(
    "refusal",
    [
        LibraryNotReadyError(SwitchState.UNAVAILABLE),
        NotTheActiveLibraryError(SwitchState.READY),
    ],
    ids=["unavailable", "another-vault-active"],
)
def test_an_import_refused_for_good_leaves_the_row(finishing, refusal):
    before = dict(finishing.info)
    finishing.answers.append(refusal)

    row = _finalize(finishing)

    assert len(finishing.imports) == 1, "asked again where no answer can change"
    assert row["status"] == before["status"]
    assert row["error"] is None and row["completed_at"] is None
    assert _failed_messages(finishing) == []


def test_a_stop_while_the_import_waits_leaves_the_row(finishing):
    def stop_then_refuse():
        finishing.service._stop.set()
        raise LibraryNotReadyError(SwitchState.SWITCHING)

    finishing.answers.append(stop_then_refuse)
    before = dict(finishing.info)

    finisher = threading.Thread(target=_finalize, args=(finishing,), daemon=True)
    finisher.start()
    finisher.join(5)

    assert not finisher.is_alive(), "the wait did not end at the stop"
    row = training_runs.get_run_by_uuid(finishing.hub, finishing.info["uuid"])
    assert len(finishing.imports) == 1
    assert row["status"] == before["status"] and row["error"] is None
    assert _failed_messages(finishing) == []


def test_a_run_whose_row_is_gone_imports_nothing_and_keeps_its_files(finishing, caplog):
    with finishing.hub.transaction() as conn:
        conn.execute(
            "DELETE FROM training_run WHERE uuid = ?", (finishing.info["uuid"],)
        )

    with caplog.at_level(logging.WARNING, logger=training_service_module.__name__):
        finishing.service._finalize(None, None, finishing.info, finishing.progress)

    assert finishing.imports == []
    assert finishing.final.exists()
    assert (
        f"Training run {finishing.info['uuid']} (id {finishing.info['id']}) "
        f"finished, but its row is gone; {finishing.final} is left in place"
    ) in caplog.text
