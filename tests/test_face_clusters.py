"""Face clusters, on synthetic embeddings (no server)."""

import tracemalloc

import numpy as np
import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from pixlstash.db_models import Character, Face, Picture
from pixlstash.services import face_clusters
from pixlstash.services.face_clusters import (
    NotThePersonsFace,
    centrality,
    cluster_faces,
    reference_cluster_faces,
    reference_cluster_ids,
    set_reference_cluster,
    vector,
)

RNG = np.random.default_rng(3)
DIM = 64


def _person():
    return RNG.standard_normal(DIM)


def _faces_of(person, ids, spread=0.5):
    return [(pid, person + RNG.standard_normal(DIM) * spread) for pid in ids]


def test_two_people_and_a_stranger_give_two_clusters_and_one_unclustered():
    a, b, stranger = _person(), _person(), _person()
    faces = _faces_of(a, range(1, 9)) + _faces_of(b, range(11, 16))
    faces.append((99, stranger))
    clusters, unclustered = cluster_faces(faces, 0.45)
    assert [sorted(pid for pid, _ in c.members) for c in clusters] == [
        list(range(1, 9)),
        list(range(11, 16)),
    ]
    assert unclustered == [99]
    assert all(c.cohesion > 0.45 for c in clusters)


def test_the_planted_centre_is_the_most_central_member():
    person = _person()
    faces = [(1, person)] + _faces_of(person, range(2, 12), spread=0.8)
    (cluster,), _ = cluster_faces(faces, 0.3)
    assert cluster.members[0][0] == 1
    scores = [c for _, c in cluster.members]
    assert scores == sorted(scores, reverse=True)
    assert np.allclose(centrality([person, person * 2]), [1.0, 1.0])


def _pairwise_cohesion(units):
    """The mean likeness of every two rows, read off the full likeness matrix."""
    n = len(units)
    return float(((units @ units.T).sum() - n) / (n * (n - 1)))


def test_cohesion_is_the_mean_likeness_of_every_two_faces():
    rng = np.random.default_rng(17)
    person = rng.standard_normal(DIM)
    for units in (
        face_clusters._unit(rng.standard_normal((2, DIM))),
        face_clusters._unit(rng.standard_normal((300, DIM))),
        face_clusters._unit(person + rng.standard_normal((500, DIM)) * 0.2),
    ):
        assert face_clusters._cohesion(units) == pytest.approx(
            _pairwise_cohesion(units), abs=1e-12
        )
    twins = face_clusters._unit(np.tile(person, (5, 1)))
    assert face_clusters._cohesion(twins) == pytest.approx(1.0, abs=1e-12)
    opposed = face_clusters._unit([person, -person])
    assert face_clusters._cohesion(opposed) == pytest.approx(-1.0, abs=1e-12)
    assert face_clusters._cohesion(face_clusters._unit([person])) == 1.0


def test_each_clusters_cohesion_is_over_its_own_members():
    rng = np.random.default_rng(29)
    a, b = rng.standard_normal(DIM), rng.standard_normal(DIM)
    faces = [(pid, a + rng.standard_normal(DIM) * 0.3) for pid in range(1, 21)]
    faces += [(pid, b + rng.standard_normal(DIM) * 0.6) for pid in range(31, 46)]
    clusters, unclustered = cluster_faces(faces, 0.3)
    by_id = dict(faces)
    assert unclustered == []
    assert [sorted(pid for pid, _ in c.members) for c in clusters] == [
        list(range(1, 21)),
        list(range(31, 46)),
    ]
    for cluster in clusters:
        units = face_clusters._unit([by_id[pid] for pid, _ in cluster.members])
        assert cluster.cohesion == pytest.approx(_pairwise_cohesion(units), abs=1e-12)


def test_cohesion_builds_no_pair_matrix():
    """A likeness matrix of 3,000 faces is 72 MB of float64; the sum needs one row."""
    units = face_clusters._unit(np.random.default_rng(23).standard_normal((3000, DIM)))
    tracemalloc.start()
    try:
        face_clusters._cohesion(units)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 3000 * 3000 * 8 // 100


def test_a_single_face_is_unclustered():
    assert cluster_faces([(7, _person())]) == ([], [7])
    assert cluster_faces([]) == ([], [])


def test_a_stricter_threshold_splits_a_loose_cluster():
    a = _person()
    # A second person halfway to the first: one person loosely, two strictly.
    b = a + _person() * 0.9
    faces = _faces_of(a, range(1, 7), 0.3) + _faces_of(b, range(11, 17), 0.3)
    loose, _ = cluster_faces(faces, 0.35)
    strict, _ = cluster_faces(faces, 0.75)
    assert [len(c.members) for c in loose] == [12]
    assert len(strict) == 2


def test_an_embedding_without_a_direction_is_no_vector():
    assert vector(np.zeros(8, dtype=np.float32).tobytes()) is None
    assert vector(np.full(8, np.nan, dtype=np.float32).tobytes()) is None
    assert vector(b"") is None
    assert vector(np.ones(8, dtype=np.float32).tobytes()) is not None


def test_a_face_without_an_embedding_does_not_carry_the_person():
    """The reference cluster holds only faces that can steer (an in-memory vault)."""
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        person = Character(name="Ida")
        session.add(person)
        pictures = [Picture() for _ in range(3)]
        session.add_all(pictures)
        session.commit()
        first, second = _person(), _person()
        both, bare, plain = (p.id for p in pictures)
        # Picture `both` carries the person twice: its first face counts.
        session.add_all(
            [
                Face(
                    picture_id=both,
                    face_index=0,
                    character_id=person.id,
                    features=first.astype(np.float32).tobytes(),
                ),
                Face(
                    picture_id=both,
                    face_index=1,
                    character_id=person.id,
                    features=second.astype(np.float32).tobytes(),
                ),
                Face(picture_id=bare, face_index=0, character_id=person.id),
                Face(
                    picture_id=plain,
                    face_index=0,
                    character_id=person.id,
                    features=first.astype(np.float32).tobytes(),
                ),
            ]
        )
        session.commit()
        flat = Picture()
        session.add(flat)
        session.commit()
        session.add(
            Face(
                picture_id=flat.id,
                face_index=0,
                character_id=person.id,
                features=np.zeros(64, dtype=np.float32).tobytes(),
            )
        )
        session.commit()
        # No embedding, or one with no direction: neither carries the person.
        with pytest.raises(NotThePersonsFace) as refused:
            set_reference_cluster(session, person.id, [both, bare, flat.id])
        assert refused.value.picture_ids == [bare, flat.id]
        assert reference_cluster_ids(session, person.id) == []
        set_reference_cluster(session, person.id, [both, plain])
        assert reference_cluster_ids(session, person.id) == [both, plain]
        chosen = {
            f.picture_id: f.face_index
            for f in reference_cluster_faces(session, person.id, 10)
        }
        assert chosen == {both: 0, plain: 0}
        # A first face whose embedding has no direction gives way to the next.
        first_face = session.exec(
            select(Face).where(Face.picture_id == both, Face.face_index == 0)
        ).one()
        first_face.features = np.zeros(64, dtype=np.float32).tobytes()
        session.commit()
        chosen = {
            f.picture_id: f.face_index
            for f in reference_cluster_faces(session, person.id, 10)
        }
        assert chosen == {both: 1, plain: 0}


def test_person_clusters_are_recomputed_when_a_face_changes(monkeypatch):
    """The grid filter's cache never answers for faces it was not built from."""
    a, b = _person(), _person()
    faces = [
        face_clusters.PersonFace(pid, vec, None, None)
        for pid, vec in _faces_of(a, [1, 2, 3], 0.1) + _faces_of(b, [4, 5], 0.1)
    ]
    monkeypatch.setattr(face_clusters, "read_person", lambda *_: (faces, []))
    clusters = face_clusters.person_cluster_ids(None, 99)
    assert [sorted(c) for c in clusters] == [[1, 2, 3], [4, 5]]
    # Picture 3's face is re-read as the other person's.
    faces[2] = face_clusters.PersonFace(
        3, b + RNG.standard_normal(DIM) * 0.1, None, None
    )
    clusters = face_clusters.person_cluster_ids(None, 99)
    assert [sorted(c) for c in clusters] == [[3, 4, 5], [1, 2]]
