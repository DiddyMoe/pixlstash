"""The face-crop pass is skipped when every crop tag is excluded.

`test_tag_task.py` proves the skip against a stand-in database. Here a real
TagTask runs on a real vault: the exclusions come from the database's own
setter, the preload runs on its own thread, the faces come from Face rows and
the tags are written to the Tag table. Only the tagging model is a stand-in.
"""

import os

import pytest
from PIL import Image
from sqlmodel import Session, select

from pixlstash.db_models import Face, Picture, Tag
from pixlstash.tasks.tag_task import CROP_TAGS_BY_KIND, TagTask
from pixlstash.vault import Vault

ALL_CROP_TAGS = frozenset().union(*CROP_TAGS_BY_KIND.values())


class _Tagger:
    """The tagging workflow with its models replaced: fixed tags, a record of
    every crop it was handed."""

    is_pixlstash_tagger_enabled = True

    def __init__(self):
        self.crop_kinds = []

    def ensure_active_plugin_ready(self, engine_override=None):
        return None

    def active_plugin_name(self, engine_override=None):
        return engine_override or "pixlstash_tagger"

    def active_model_version(self, engine_override=None):
        return "pixlstash_tagger:test"

    def full_pass_input(self, image, engine_override=None):
        return None

    def whole_face_crop(self):
        return True

    def pixlstash_tagger_image_size_quality_crop(self):
        return 64

    def tag_images(self, image_paths, out_raw_scores=None, **_kwargs):
        for path in image_paths:
            if out_raw_scores is not None:
                out_raw_scores[path] = {"portrait": 0.9}
        return {path: ["portrait"] for path in image_paths}

    def tag_quality_crops(self, items, out_raw_scores=None, **_kwargs):
        self.crop_kinds.extend(key for key, _crop in items)
        return {key: [] for key, _crop in items}


@pytest.fixture(scope="module")
def vault(tmp_path_factory):
    root = tmp_path_factory.mktemp("crop-skip")
    # Never started: no planner tags the pictures behind the test's back.
    with Vault(image_root=str(root)) as vault:
        yield vault


@pytest.fixture
def face_picture(vault):
    """A picture with one face big enough for a face crop."""
    name = f"face-{len(os.listdir(vault.db.image_root))}.png"
    path = os.path.join(vault.db.image_root, name)
    Image.new("RGB", (256, 256), (120, 90, 60)).save(path)

    def add(session: Session):
        picture = Picture(file_path=path, format="png", width=256, height=256)
        session.add(picture)
        session.commit()
        session.add(Face(picture_id=picture.id, bbox=[40, 40, 200, 200]))
        session.commit()
        return picture.id

    pid = vault.db.run_task(add)
    previous = (vault.db.tag_exclusions, vault.db.tag_exclusion_generation)
    yield pid
    vault.db.set_tag_exclusions(previous[0], previous[1])


def _tag(vault, pid, monkeypatch):
    """Run one TagTask on *pid* the way the runner does; return the stand-in
    tagger, how many times faces were read, and the tags written."""
    face_reads = []
    real_fetch = TagTask._fetch_faces_for_pictures

    def counted(session, picture_ids):
        face_reads.append(list(picture_ids))
        return real_fetch(session, picture_ids)

    monkeypatch.setattr(TagTask, "_fetch_faces_for_pictures", staticmethod(counted))
    tagger = _Tagger()
    picture = vault.db.run_task(lambda session: session.get(Picture, pid))
    task = TagTask(database=vault.db, tagging_workflow=tagger, pictures=[picture])
    task.on_queued()
    task.run()
    tags = vault.db.run_task(
        lambda session: set(
            session.exec(select(Tag.tag).where(Tag.picture_id == pid)).all()
        )
    )
    return tagger, face_reads, tags


def _exclude(vault, names):
    vault.db.set_tag_exclusions(frozenset(names), vault.db.tag_exclusion_generation + 1)


def test_no_face_crop_is_read_or_tagged_when_every_crop_tag_is_excluded(
    vault, face_picture, monkeypatch
):
    _exclude(vault, ALL_CROP_TAGS)

    tagger, face_reads, tags = _tag(vault, face_picture, monkeypatch)

    assert tagger.crop_kinds == [], "the crop model ran with every crop tag excluded"
    assert face_reads == [], "faces were read for crops nobody keeps"
    assert "portrait" in tags, "the full pass must still write its tags"


def test_one_crop_tag_still_allowed_keeps_the_face_crop(
    vault, face_picture, monkeypatch
):
    # The positive control: the same picture and face with one crop tag left
    # allowed takes the face crop, so the test above cannot pass on a picture
    # that never qualified for one.
    kept = sorted(CROP_TAGS_BY_KIND["face"])[0]
    _exclude(vault, ALL_CROP_TAGS - {kept})

    tagger, face_reads, tags = _tag(vault, face_picture, monkeypatch)

    assert len(tagger.crop_kinds) == 1, tagger.crop_kinds
    assert "#centre" not in tagger.crop_kinds[0], "expected the face crop"
    assert face_reads and all(ids == [face_picture] for ids in face_reads)
    assert "portrait" in tags
