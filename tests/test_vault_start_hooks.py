"""What `Vault.start` and the face-write settle callback guarantee.

- A pending person on a picture that already has face rows is resolved at
  every start with background work enabled, without waiting for a new face
  write. The pass itself is tested in
  tests/test_pending_people_reconciliation.py; this file tests that start runs it.
- `_after_face_writes_land`'s settle callback never raises, and a cancelled or
  failed face write logs the pictures it affected.
"""

import logging
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from sqlmodel import select

from pixlstash.db_models import Character, Face, Picture
from pixlstash.vault import Vault


@pytest.fixture
def vault(tmp_path):
    v = Vault(
        image_root=str(tmp_path),
        disable_background_workers=False,
        daily_snapshots_enabled=False,
    )
    try:
        yield v
    finally:
        v.close()


def _seed_stranded(session):
    alice = Character(name="Alice")
    session.add(alice)
    session.flush()
    stranded = Picture(
        file_path="stranded.png",
        format="PNG",
        width=100,
        height=100,
        pending_character_id=alice.id,
    )
    session.add(stranded)
    session.flush()
    face = Face(picture_id=stranded.id, face_index=0, bbox=[0, 0, 40, 40])
    session.add(face)
    session.commit()
    return {"alice": alice.id, "stranded": stranded.id, "face": face.id}


def test_start_resolves_a_pending_person_whose_picture_already_has_faces(vault):
    ids = vault.db.run_task(_seed_stranded)

    vault.start()

    def read(session):
        pic = session.get(Picture, ids["stranded"])
        face = session.exec(select(Face).where(Face.id == ids["face"])).one()
        return pic.pending_character_id, face.character_id

    pending, face_character = vault.db.run_task(read)
    assert pending is None, "start must close a pending person stranded on faces"
    assert face_character == ids["alice"]


@pytest.mark.parametrize("outcome", ["cancelled", "failed"])
def test_a_face_write_that_did_not_land_is_logged_and_never_raises(
    tmp_path, caplog, outcome
):
    v = Vault(
        image_root=str(tmp_path),
        disable_background_workers=True,
        daily_snapshots_enabled=False,
    )
    try:
        write: Future = Future()
        if outcome == "cancelled":
            assert write.cancel()
        else:
            write.set_exception(RuntimeError("example-write-failure"))
        task = SimpleNamespace(pending_writes=[write])
        with caplog.at_level(logging.WARNING):
            # The write is already done, so add_done_callback runs the
            # settle callback inline; concurrent.futures swallows and logs
            # anything it raises as "exception calling callback".
            v._after_face_writes_land(task, [], [7, 11])

        messages = [r.getMessage() for r in caplog.records]
        assert not any("exception calling callback" in m for m in messages), messages
        landed = [m for m in messages if "did not land" in m]
        assert landed and "[7, 11]" in landed[0], messages
        assert not v._face_followups, "the follow-up is released"
    finally:
        v.close()
