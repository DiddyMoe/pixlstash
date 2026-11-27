"""A pending person on a picture that already has faces is resolved at start-up.

Face extraction writes a picture's face rows and matches its pending people in
two transactions. A stop or crash between them leaves the person pending on a
picture the face finder never offers again, because it has faces.
"""

from sqlmodel import select

from pixlstash.db_models import Character, Face, Picture, PicturePendingPerson
from pixlstash.vault import Vault


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

    # Faces landed, the match did not: the largest face takes the person.
    stranded = picture("stranded", pending_character_id=alice.id)
    small = Face(picture_id=stranded.id, face_index=0, bbox=[0, 0, 10, 10])
    large = Face(picture_id=stranded.id, face_index=1, bbox=[0, 0, 50, 50])
    # The same for a people row. Bob has no reference faces to score, so the
    # matcher gives him the largest free face.
    group = picture("group")
    group_face = Face(picture_id=group.id, face_index=0, bbox=[0, 0, 30, 30])
    session.add(PicturePendingPerson(picture_id=group.id, character_id=bob.id))
    # Extraction ran and found nothing: only the no-face marker row.
    empty = picture("empty", pending_character_id=bob.id)
    marker = Face(picture_id=empty.id, face_index=-1)
    # Not extracted yet: no face rows, so it waits for extraction.
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


def _read(session):
    pending = {
        p.id: p.pending_character_id for p in session.exec(select(Picture)).all()
    }
    people = {
        (r.picture_id, r.character_id)
        for r in session.exec(select(PicturePendingPerson)).all()
    }
    faces = {f.id: f.character_id for f in session.exec(select(Face)).all()}
    return pending, people, faces


def test_start_resolves_pending_people_on_pictures_whose_faces_exist(tmp_path):
    # Workers on, no finders: `start` runs its real sequence and no task
    # touches the pictures.
    vault = Vault(
        image_root=str(tmp_path),
        planner_finders=(),
        daily_snapshots_enabled=False,
    )
    try:
        ids = vault.db.run_task(_seed)

        vault.start()

        pending, people, faces = vault.db.run_task(_read)
        assert faces[ids["large"]] == ids["alice"], "the largest face takes the person"
        assert faces[ids["small"]] is None
        assert pending[ids["stranded"]] is None
        assert faces[ids["group_face"]] == ids["bob"], "a people row is matched too"
        assert (ids["group"], ids["bob"]) not in people
        assert pending[ids["empty"]] is None, (
            "no real face: discarded, as extraction does"
        )
        assert pending[ids["waiting"]] == ids["alice"], (
            "not extracted yet: still pending"
        )
        assert (ids["waiting"], ids["bob"]) in people
    finally:
        vault.close()
