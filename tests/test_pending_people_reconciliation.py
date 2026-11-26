"""A pending person on a picture that already has faces is resolved at start-up.

Face extraction writes a picture's face rows and matches its pending people in
two transactions. A crash between them, a stop that gave up waiting for the
match, or a failed face write on a picture that already had faces leaves the
person pending on a picture the face finder never offers again (it has
faces). ``reconcile_pending_people_with_faces`` is the start-up pass that
closes it.
"""

import pytest
from sqlmodel import select

from pixlstash.db_models import Character, Face, Picture, PicturePendingPerson
from pixlstash.services.folder_structure_commit_service import (
    reconcile_pending_people_with_faces,
)
from pixlstash.tasks.face_extraction_task import FaceExtractionTask
from pixlstash.tasks.missing_face_extraction_finder import FACE_EXTRACTION_BATCH_LIMIT
from pixlstash.vault import Vault


@pytest.fixture
def vault(tmp_path):
    v = Vault(
        image_root=str(tmp_path),
        disable_background_workers=True,
        daily_snapshots_enabled=False,
    )
    try:
        yield v
    finally:
        v.close()


def _seed(session):
    alice = Character(name="Alice")
    bob = Character(name="Bob")
    session.add_all([alice, bob])
    session.flush()

    def picture(name, **kw):
        pic = Picture(
            file_path=f"{name}.png", format="PNG", width=100, height=100, **kw
        )
        session.add(pic)
        session.flush()
        return pic

    # Stranded scalar: pending person, two faces, nobody joined them.
    stranded = picture("stranded", pending_character_id=alice.id)
    small = Face(picture_id=stranded.id, face_index=0, bbox=[0, 0, 10, 10])
    large = Face(picture_id=stranded.id, face_index=1, bbox=[0, 0, 50, 50])
    # Stranded people row: one face. Bob has no reference faces to score, so
    # the matcher gives him the largest free face.
    group = picture("group")
    group_face = Face(picture_id=group.id, face_index=0, bbox=[0, 0, 30, 30])
    session.add(PicturePendingPerson(picture_id=group.id, character_id=bob.id))
    # Extraction ran and found nothing: only the no-face marker row.
    empty = picture("empty", pending_character_id=bob.id)
    marker = Face(picture_id=empty.id, face_index=-1)
    # Not extracted yet: no face rows. Must stay pending for extraction.
    waiting = picture("waiting", pending_character_id=alice.id)
    session.add(PicturePendingPerson(picture_id=waiting.id, character_id=bob.id))
    session.add_all([small, large, group_face, marker])
    session.commit()
    return {
        "alice": alice.id,
        "bob": bob.id,
        "stranded": stranded.id,
        "large": large.id,
        "small": small.id,
        "group": group.id,
        "group_face": group_face.id,
        "empty": empty.id,
        "waiting": waiting.id,
    }


def test_start_up_pass_resolves_pending_people_on_pictures_with_faces(vault):
    ids = vault.db.run_task(_seed)

    handed = reconcile_pending_people_with_faces(vault)

    assert handed == 3, "stranded, group and empty have face rows; waiting has none"

    def read(session):
        pending = {
            p.id: p.pending_character_id for p in session.exec(select(Picture)).all()
        }
        people = {
            (r.picture_id, r.character_id)
            for r in session.exec(select(PicturePendingPerson)).all()
        }
        faces = {f.id: f.character_id for f in session.exec(select(Face)).all()}
        return pending, people, faces

    pending, people, faces = vault.db.run_task(read)
    assert pending[ids["stranded"]] is None
    assert faces[ids["large"]] == ids["alice"], "the largest face takes the person"
    assert faces[ids["small"]] is None
    assert faces[ids["group_face"]] == ids["bob"]
    assert (ids["group"], ids["bob"]) not in people, "the people row is consumed"
    assert pending[ids["empty"]] is None, "no real face: discarded, as extraction does"
    assert pending[ids["waiting"]] == ids["alice"], "not extracted yet: still pending"
    assert (ids["waiting"], ids["bob"]) in people

    assert reconcile_pending_people_with_faces(vault) == 0, "nothing left to resolve"


def test_vault_start_resolves_pending_people_before_the_workers_start(tmp_path):
    # The pass above only helps if start-up runs it. A vault with workers but
    # no finders and no hub: ``start`` runs its real sequence, and nothing
    # else touches the pictures.
    vault = Vault(
        image_root=str(tmp_path),
        planner_finders=(),
        daily_snapshots_enabled=False,
    )
    try:
        ids = vault.db.run_task(_seed)
        seen = {}
        start_runner = vault._task_runner.start

        def _start_runner_after_reading_what_it_would_see():
            seen["stranded"] = vault.db.run_task(
                lambda session: (
                    session.get(Picture, ids["stranded"]).pending_character_id
                )
            )
            return start_runner()

        vault._task_runner.start = _start_runner_after_reading_what_it_would_see
        vault.start()

        assert seen == {"stranded": None}, "resolved before the task runner started"
        large = vault.db.run_task(lambda session: session.get(Face, ids["large"]))
        assert large.character_id == ids["alice"]
        waiting = vault.db.run_task(
            lambda session: session.get(Picture, ids["waiting"])
        )
        assert waiting.pending_character_id == ids["alice"], "not extracted yet"
    finally:
        vault.close()


def test_a_face_task_writes_its_rows_in_one_chunk():
    """`Vault._after_face_writes_land` settles a task's writes all or nothing.

    One failed write withholds every picture's announcement and pending match,
    including pictures whose own chunk landed.
    That is only right while a task carries one write: no finder may hand a
    task more pictures than `_run_task` flushes at once.
    """
    assert FACE_EXTRACTION_BATCH_LIMIT <= FaceExtractionTask._FLUSH_CHUNK_SIZE, (
        "a face task would write in several chunks; make "
        "_after_face_writes_land settle each chunk's pictures on their own"
    )
