"""Unit tests for the tiered duplicate detection service.

Covers, per tier and per rule:

* **tier 1 (exact)** - groups on the indexed ``pixel_sha``, refuses to group two
  files that share a digest but differ in ``size_bytes`` (the sampled-digest
  guard), and stays blind to the scrapheap;
* **tier 2 (bucketed near)** - candidate buckets come from the precomputed
  columns, comparison happens only inside a bucket, and the group's confidence is
  its weakest link;
* **tier 3 (embedding)** - reuses the shipped likeness edge table;
* **the tier policy** - exact is always on, each looser tier requires the tier
  above it, and the 0.65 floor is a hard error rather than a silent clamp;
* **cover selection** - the 2026-07-30 lexicographic ranking: smart score in
  quarter-star buckets (unknown ranks neutral, never zero), then pixel count,
  then sharpness, then stars/tags/RAW/bytes, ties to the oldest capture;
* **evidence** - matching pills and evidence-against pills, both directions;
* **the queue** - paged by confidence descending, verdict-resolved groups never
  re-offered, and scope-narrowed counts.
"""

import gc
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event, text
from sqlmodel import select

from pixlstash.database import DBPriority
from pixlstash.db_models import (
    Character,
    Face,
    Picture,
    PictureProjectMember,
    PictureSet,
    PictureSetMember,
    PictureStack,
    Project,
)
from pixlstash.db_models.dedup import (
    SCAN_PENDING,
    SCAN_RUNNING,
    VERDICT_KEEP_SEPARATE,
    DedupGroup,
    DedupGroupMember,
    DedupScan,
    DedupVerdict,
)
from pixlstash.db_models.picture_likeness import PictureLikeness
from pixlstash.db_models.quality import Quality
from pixlstash.db_models.tag import Tag
from pixlstash.server import Server
from pixlstash.services import dedup_tier_service as tiers
from pixlstash.services import dedup_verdict_service as verdicts
from pixlstash.services.dedup_tier_service import (
    CandidateMember,
    DedupScope,
    DedupTier,
    ScopeType,
    TierPolicy,
)
from pixlstash.tasks import dedup_scan_task as dedup_scan_task_module
from pixlstash.tasks.dedup_scan_task import DedupScanTask
from pixlstash.tasks.dedup_scan_finder import DedupScanFinder
from pixlstash.tasks.task_type import TaskType
from pixlstash.task_runner import TaskRunner
from pixlstash.utils.query.folder_prefix import PATHS_FOLD_CASE

_BASE_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# A 64-bit dHash is 16 hex chars. These differ from ZERO by a controlled number
# of set bits, so the Hamming distance (and therefore the similarity) is exact.
PHASH_ZERO = "0000000000000000"
PHASH_ONE_BIT = "0000000000000001"  # 1 bit  -> similarity 63/64 = 0.984375
PHASH_FOUR_BITS = "000000000000000f"  # 4 bits -> similarity 60/64 = 0.9375
PHASH_FAR = "ffffffffffffffff"  # 64 bits -> similarity 0.0


@pytest.fixture
def server():
    temp_dir = tempfile.TemporaryDirectory()
    config_path = os.path.join(temp_dir.name, "server-config.json")
    with open(config_path, "w") as fh:
        fh.write(json.dumps({"port": 8000, "disable_background_workers": True}))
    Server.DEFAULT_FORCE_CPU = True
    srv = Server(config_path)
    try:
        yield srv
    finally:
        srv.close()
        temp_dir.cleanup()
        gc.collect()


def _run(server, fn, *args):
    """Run *fn(session, \\*args)* on the DB worker and return its result."""
    return server.vault.db.run_task(fn, *args, priority=DBPriority.IMMEDIATE)


def _seed(server, specs):
    """Insert one picture per spec; return the ids in order.

    Recognised keys: ``pixel_sha``, ``perceptual_hash``, ``size_bytes``,
    ``width``, ``height``, ``score``, ``smart_score``, ``sharpness`` (writes a
    ``Quality`` row), ``format``, ``file_path``, ``created_at`` (an offset in
    seconds from ``_BASE_TIME``), ``deleted``, ``tags``,
    ``import_source_folder``, ``reference_folder_id``.
    """

    def insert(session):
        picture_ids = []
        for index, spec in enumerate(specs):
            width = spec.get("width", 4000)
            height = spec.get("height", 3000)
            created_offset = spec.get("created_at")
            pic = Picture(
                file_path=spec.get("file_path", f"/vault/pic_{index}.png"),
                format=spec.get("format", "png"),
                width=width,
                height=height,
                size_bin_index=(width << 32) + height,
                size_bytes=spec.get("size_bytes", 1000),
                score=spec.get("score"),
                smart_score=spec.get("smart_score"),
                pixel_sha=spec.get("pixel_sha"),
                perceptual_hash=spec.get("perceptual_hash"),
                import_source_folder=spec.get("import_source_folder"),
                reference_folder_id=spec.get("reference_folder_id"),
                deleted=bool(spec.get("deleted", False)),
                created_at=(
                    _BASE_TIME + timedelta(seconds=created_offset)
                    if created_offset is not None
                    else None
                ),
            )
            session.add(pic)
            session.flush()
            for tag in spec.get("tags", []):
                session.add(Tag(picture_id=int(pic.id), tag=tag))
            if "sharpness" in spec:
                session.add(
                    Quality(picture_id=int(pic.id), sharpness=spec["sharpness"])
                )
            picture_ids.append(int(pic.id))
        session.commit()
        return picture_ids

    return _run(server, insert)


def _member(**kwargs) -> CandidateMember:
    """A bare :class:`CandidateMember` for the pure-function tests."""
    defaults = {
        "id": 1,
        "width": 4000,
        "height": 3000,
        "format": "jpeg",
        "score": 0,
        "tag_count": 0,
    }
    defaults.update(kwargs)
    return CandidateMember(**defaults)


# ── the tier policy ───────────────────────────────────────────────────────────


def test_exact_tier_is_always_on_and_cannot_be_switched_off():
    assert TierPolicy().tiers == (DedupTier.EXACT,)
    assert TierPolicy().includes(DedupTier.EXACT)
    assert not TierPolicy().includes(DedupTier.NEAR)


def test_each_looser_tier_requires_the_tier_above_it():
    assert TierPolicy(near_enabled=True).tiers == (DedupTier.EXACT, DedupTier.NEAR)
    assert TierPolicy(near_enabled=True, embedding_enabled=True).tiers == (
        DedupTier.EXACT,
        DedupTier.NEAR,
        DedupTier.EMBEDDING,
    )
    with pytest.raises(ValueError, match="requires near_enabled"):
        TierPolicy(embedding_enabled=True)


def test_threshold_default_is_090_and_the_floor_is_a_hard_error():
    assert TierPolicy().threshold == pytest.approx(0.90)
    # Below the floor is a 400-worthy error, never a silent clamp: a low
    # threshold produces confident-looking garbage and destroys the count.
    with pytest.raises(ValueError, match="0.65"):
        TierPolicy(threshold=0.5)
    assert TierPolicy(threshold=tiers.MIN_THRESHOLD).threshold == pytest.approx(0.65)


def test_group_size_bounds_are_validated():
    with pytest.raises(ValueError, match="min_group_size"):
        TierPolicy(min_group_size=1)
    with pytest.raises(ValueError, match="max_group_size"):
        TierPolicy(min_group_size=4, max_group_size=3)


# ── cover selection (the 2026-07-30 lexicographic ranking) ────────────────────


def test_smart_score_dominates_every_size_advantage():
    """Tier 1 beats tier 2: a 40 MP blurry scan must not outrank a sharp
    original the scorer rated higher - the exact failure of the old weighted
    sum, where pixels alone could buy the cover."""
    sharp_original = _member(id=1, width=4000, height=3000, smart_score=4.5)
    blurry_scan = _member(
        id=2, width=8000, height=5000, smart_score=2.0, tag_count=9, score=5
    )
    assert tiers.select_cover([blurry_scan, sharp_original]) == 1


def test_smart_scores_inside_one_bucket_fall_through_to_size():
    """A lead smaller than the 0.25 bucket is scoring noise, not a decision:
    both land in the same bucket and the bigger picture wins."""
    slightly_better = _member(id=1, width=1000, height=1000, smart_score=4.30)
    bigger = _member(id=2, width=4000, height=3000, smart_score=4.26)
    assert tiers.select_cover([slightly_better, bigger]) == 2
    # A genuine bucket lead decides, regardless of size.
    clearly_better = _member(id=3, width=1000, height=1000, smart_score=4.55)
    assert tiers.select_cover([bigger, clearly_better]) == 3


def test_an_unknown_smart_score_ranks_neutral_never_zero():
    """NULL (not yet computed) and -1.0 (the failed-metric sentinel) both read
    as unknown and rank at the neutral midpoint: an unscored copy still loses
    to a known-good one, still beats a known-bad one, and two unknowns fall
    through to size - never buried below a scored-terrible sibling."""
    unknown = _member(id=1, width=1000, height=1000)
    known_bad = _member(id=2, width=8000, height=8000, smart_score=2.0)
    known_good = _member(id=3, width=1000, height=1000, smart_score=4.0)
    assert tiers.select_cover([known_bad, unknown]) == 1
    assert tiers.select_cover([unknown, known_good]) == 3
    failed = _member(id=4, width=4000, height=3000, smart_score=-1.0)
    # Failed (-1.0) is neutral too: it ties the NULL member and wins on size.
    assert tiers.select_cover([unknown, failed]) == 4


def test_size_beats_sharpness_at_equal_smart_bucket():
    bigger_softer = _member(
        id=1, width=4000, height=3000, smart_score=4.0, sharpness=0.10
    )
    smaller_sharper = _member(
        id=2, width=2000, height=1500, smart_score=4.0, sharpness=0.45
    )
    assert tiers.select_cover([smaller_sharper, bigger_softer]) == 1


def test_sharpness_decides_at_equal_smart_score_and_pixels():
    soft = _member(id=1, sharpness=0.15, score=5, tag_count=9)
    sharp = _member(id=2, sharpness=0.40)
    assert tiers.select_cover([soft, sharp]) == 2
    # Unknown sharpness (missing row or the -1.0 sentinel) is neutral (0.25):
    # it loses to a known-sharper copy and beats a known-softer one.
    unknown = _member(id=3)
    failed = _member(id=4, sharpness=-1.0)
    assert tiers.select_cover([unknown, sharp]) == 2
    assert tiers.select_cover([soft, failed]) == 4


def test_lower_order_signals_break_full_quality_ties():
    """Stars, then tags, then RAW, then bytes - in that order, only after the
    quality and size tiers tie."""
    starred = _member(id=1, score=4)
    tagged = _member(id=2, tag_count=7)
    assert tiers.select_cover([tagged, starred]) == 1
    raw = _member(id=3, format="arw")
    assert tiers.select_cover([tagged, raw]) == 2  # tags outrank RAW
    heavy = _member(id=4, size_bytes=9_000_000)
    light = _member(id=5, size_bytes=1_000)
    assert tiers.select_cover([light, heavy]) == 4


def test_ties_break_to_the_oldest_capture_then_the_lowest_id():
    old = _member(id=10, created_at=_BASE_TIME)
    new = _member(id=11, created_at=_BASE_TIME + timedelta(hours=5))
    assert tiers.select_cover([new, old]) == 10
    # No timestamps at all: deterministic on the lowest id.
    a = _member(id=21)
    b = _member(id=22)
    assert tiers.select_cover([b, a]) == 21


def test_raw_is_detected_by_format_or_extension():
    assert _member(format="ARW").is_raw
    assert _member(format="jpeg", file_path="/shoots/A7R0912.arw").is_raw
    assert not _member(format="jpeg", file_path="/shoots/x.jpg").is_raw


def test_the_legacy_cover_score_field_is_unchanged():
    """`cover_score` is deprecated wire-compat, not the selection rule - but
    while it ships it must keep its documented value."""
    member = _member(width=4000, height=3000, tag_count=2, score=3)
    assert member.megapixels == pytest.approx(12.0)
    assert member.cover_score == pytest.approx(48.0 + 6.0 + 6.0)
    raw = _member(id=1, format="arw", width=1000, height=1000)
    jpeg = _member(id=2, format="jpeg", width=1000, height=1000)
    assert raw.cover_score - jpeg.cover_score == pytest.approx(tiers.COVER_RAW_BONUS)


def test_serialization_carries_the_ranking_signals_null_safe():
    scored = _member(smart_score=4.256, sharpness=0.312)
    assert scored.as_dict()["smart_score"] == pytest.approx(4.256)
    assert scored.as_dict()["sharpness"] == pytest.approx(0.312)
    # NULL and the -1.0 failed sentinel both serialize as null, never a fake
    # number the Compare view would display as real.
    blank = _member()
    assert blank.as_dict()["smart_score"] is None
    assert blank.as_dict()["sharpness"] is None
    failed = _member(smart_score=-1.0, sharpness=-1.0)
    assert failed.as_dict()["smart_score"] is None
    assert failed.as_dict()["sharpness"] is None


# ── signature ─────────────────────────────────────────────────────────────────


def test_signature_is_order_independent_and_content_derived():
    assert tiers.group_signature(["b", "a"]) == tiers.group_signature(["a", "b"])
    assert tiers.group_signature(["a", "b"]) != tiers.group_signature(["a", "c"])


def test_signature_falls_back_to_the_picture_id_when_no_hash_exists():
    hashed = _member(id=7, pixel_sha="deadbeef", size_bytes=100)
    unhashed = _member(id=7)
    # The hash is never the identity on its own: it is sampled above 128 KiB, so
    # the size travels with it (see test_content_key_carries_the_size_co_key).
    assert hashed.content_key == "deadbeef:100"
    assert unhashed.content_key == "id:7"


# ── evidence ──────────────────────────────────────────────────────────────────


def test_group_evidence_reports_both_directions():
    members = [
        _member(id=1, width=6000, height=4000, created_at=_BASE_TIME, format="jpeg"),
        _member(id=2, width=1920, height=1440, created_at=_BASE_TIME, format="webp"),
    ]
    pills = tiers.build_group_evidence(DedupTier.NEAR, 0.96, members)
    texts = {pill["text"]: pill["against"] for pill in pills}
    assert "96% visual match" not in texts
    assert texts["Different resolution"] is True
    assert texts["Different aspect ratio"] is True
    assert texts["Different file format"] is True
    assert texts["Same capture second"] is False


def test_exact_evidence_leads_with_the_hash_and_same_dimensions():
    members = [_member(id=1), _member(id=2)]
    pills = tiers.build_group_evidence(DedupTier.EXACT, 1.0, members)
    texts = [pill["text"] for pill in pills]
    assert texts[0] == "Identical file hash"
    assert "Same dimensions" in texts
    assert not any(pill["against"] for pill in pills)


def test_candidate_evidence_explains_the_preselection_both_ways():
    best = _member(id=1, width=6000, height=4000, tag_count=6, score=5)
    worst = _member(id=2, width=1080, height=1080)
    members = [best, worst]
    best_pills = tiers.build_candidate_evidence(best, members, cover_id=1)
    worst_pills = tiers.build_candidate_evidence(worst, members, cover_id=1)
    assert any(p["text"] == "Highest resolution" for p in best_pills)
    assert any(p["text"] == "Preselected as cover" for p in best_pills)
    assert any(p["against"] and "fewer pixels" in p["text"] for p in worst_pills)
    assert any(p["against"] and "Fewer tags" in p["text"] for p in worst_pills)
    # Nobody here has a smart score or sharpness: no pill invents one.
    for pills in (best_pills, worst_pills):
        assert not any("smart score" in p["text"].lower() for p in pills)
        assert not any("Sharpest" in p["text"] for p in pills)


def test_candidate_evidence_explains_the_smart_score_and_sharpness_tiers():
    """The pills must explain the NEW ranking, in its priority order."""
    best = _member(id=1, smart_score=4.3, sharpness=0.4)
    mid = _member(id=2, smart_score=4.26, sharpness=0.1)
    worse = _member(id=3, smart_score=3.1)
    unknown = _member(id=4)
    members = [best, mid, worse, unknown]

    best_pills = tiers.build_candidate_evidence(best, members, cover_id=1)
    assert best_pills[0]["text"] == "Best smart score (4.3)"
    assert best_pills[0]["against"] is False
    assert any(p["text"] == "Sharpest copy" for p in best_pills)

    # 4.26 sits in the same quarter-star bucket as 4.3: an effective tie both
    # read as best - the pill mirrors the decision unit, not float noise.
    mid_pills = tiers.build_candidate_evidence(mid, members, cover_id=1)
    assert mid_pills[0]["text"] == "Best smart score (4.3)"
    assert not any(p["text"] == "Sharpest copy" for p in mid_pills)

    worse_pills = tiers.build_candidate_evidence(worse, members, cover_id=1)
    assert worse_pills[0]["text"] == "Lower smart score (3.1 vs 4.3)"
    assert worse_pills[0]["against"] is True

    # An unknown score gets no pill either way: the null field is the honest
    # display, and a red pill would blame the picture for a pending task.
    unknown_pills = tiers.build_candidate_evidence(unknown, members, cover_id=1)
    assert not any("smart score" in p["text"].lower() for p in unknown_pills)


def test_reference_folder_pictures_expose_their_path_and_others_do_not():
    managed = _member(id=1, file_path="/vault/a.png")
    referenced = _member(id=2, file_path="/photos/a.png", reference_folder_id=3)
    assert managed.as_dict()["file_path"] is None
    assert referenced.as_dict()["file_path"] == "/photos/a.png"


# ── tier 1: exact ─────────────────────────────────────────────────────────────


def test_exact_tier_groups_on_the_indexed_hash(server):
    ids = _seed(
        server,
        [
            {"pixel_sha": "aaa", "size_bytes": 100, "score": 4},
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "bbb", "size_bytes": 100},
        ],
    )
    groups = _run(server, tiers.find_exact_groups_in_session, None)
    assert len(groups) == 1
    group = groups[0]
    assert sorted(group.picture_ids) == sorted(ids[:2])
    assert group.confidence == pytest.approx(1.0)
    assert group.tier is DedupTier.EXACT
    # Equal quality/size tiers: the star tier picks the higher human score.
    assert group.cover_picture_id == ids[0]


def test_exact_tier_refuses_to_group_on_a_digest_alone(server):
    """The sampled-digest guard: same hash, different size is not a match.

    ``pixel_sha`` samples large files rather than hashing every byte, so equal
    file size is a required co-key. Dropping it would let the queue claim an
    identity the digest does not actually prove.
    """
    _seed(
        server,
        [
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "aaa", "size_bytes": 999},
        ],
    )
    assert _run(server, tiers.find_exact_groups_in_session, None) == []


def test_exact_tier_ignores_the_scrapheap_and_unhashed_rows(server):
    _seed(
        server,
        [
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "aaa", "size_bytes": 100, "deleted": True},
            {"pixel_sha": None, "size_bytes": 100},
            {"pixel_sha": None, "size_bytes": 100},
        ],
    )
    assert _run(server, tiers.find_exact_groups_in_session, None) == []


def test_exact_tier_narrows_to_a_scope(server):
    ids = _seed(
        server,
        [
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "ccc", "size_bytes": 100},
            {"pixel_sha": "ccc", "size_bytes": 100},
        ],
    )

    def add_set(session):
        picture_set = PictureSet(name="Scope")
        session.add(picture_set)
        session.commit()
        session.refresh(picture_set)
        for picture_id in ids[:2]:
            session.add(
                PictureSetMember(set_id=int(picture_set.id), picture_id=picture_id)
            )
        session.commit()
        return int(picture_set.id)

    set_id = _run(server, add_set)
    scope = DedupScope(scope_type=ScopeType.SET, scope_id=str(set_id))
    scoped = _run(server, tiers.find_exact_groups_in_session, scope)
    assert len(scoped) == 1
    assert sorted(scoped[0].picture_ids) == sorted(ids[:2])
    assert len(_run(server, tiers.find_exact_groups_in_session, None)) == 2


def _group_shapes(groups):
    return [
        (group.signature, sorted(group.picture_ids), group.cover_picture_id)
        for group in groups
    ]


def test_paged_exact_groups_match_the_unpaged_result(server, monkeypatch):
    """Pages bound one writer slice; they must not change which groups exist.

    The keys come back oldest picture first (the queue serves equal-confidence
    groups in creation order), a key is never split across pages however large
    it is, and building page by page gives exactly the groups one build of every
    key gives.
    """
    ids = _seed(
        server,
        [
            {"pixel_sha": "pair", "size_bytes": 100},
            {"pixel_sha": "pair", "size_bytes": 100},
            {"pixel_sha": "pair", "size_bytes": 200},
            {"pixel_sha": "pair", "size_bytes": 200},
            {"pixel_sha": "nosize", "size_bytes": None},
            {"pixel_sha": "nosize", "size_bytes": None},
            *({"pixel_sha": "big", "size_bytes": 100} for _ in range(5)),
            {"pixel_sha": "big", "size_bytes": 100, "deleted": True},
            {"pixel_sha": "lone", "size_bytes": 100},
            {"pixel_sha": "lone", "size_bytes": 100, "deleted": True},
        ],
    )

    keys = _run(server, tiers.find_exact_keys_in_session, None)
    # Ordered by each key's lowest picture id, not by hash: alphabetically
    # "big" would come first and "pair" last.
    assert keys == [
        tiers.ExactKey("pair", 100, 2, ids[0]),
        tiers.ExactKey("pair", 200, 2, ids[2]),
        tiers.ExactKey("nosize", None, 2, ids[4]),
        tiers.ExactKey("big", 100, 5, ids[6]),
    ]
    unpaged = _run(server, tiers.build_exact_groups_for_keys_in_session, keys, None)

    monkeypatch.setattr(tiers, "EXACT_PAGE_MAX_KEYS", 2)
    monkeypatch.setattr(tiers, "EXACT_PAGE_MAX_MEMBERS", 4)
    pages = tiers.page_exact_keys(keys)
    # The five-member key is over the member bound and gets a page to itself.
    assert pages == [keys[0:2], keys[2:3], keys[3:4]]
    paged = [
        group
        for page in pages
        for group in _run(
            server, tiers.build_exact_groups_for_keys_in_session, page, None
        )
    ]

    assert _group_shapes(paged) == _group_shapes(unpaged)
    assert _group_shapes(
        _run(server, tiers.find_exact_groups_in_session, None)
    ) == _group_shapes(unpaged)
    assert [sorted(group.picture_ids) for group in paged] == [
        ids[0:2],
        ids[2:4],
        ids[4:6],
        ids[6:11],
    ]


def test_the_exact_member_lookup_is_planned_on_the_pixel_sha_index(server):
    """The member lookup must stay an index probe, not a read of the live table.

    PixlStash never runs ``ANALYZE``, and without statistics SQLite prefers a
    ``deleted`` equality to the ``pixel_sha IN (...)`` list, which made every
    lookup one full pass over the library. A scoped lookup checks each row's
    membership as a probe: as ``Picture.id IN (members)`` SQLite built the
    whole scope's list for every page and drove the read from it. The
    statement is captured as it is executed and explained against the same
    database.
    """
    ids = _seed(
        server,
        [
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "aaa", "size_bytes": 100},
        ],
    )
    set_scope = _collection_scopes(server, ids)[ScopeType.SET]

    def explain_member_lookup(session, scope):
        executed = []

        def capture(_conn, _cursor, statement, parameters, _context, _many):
            if "pixel_sha IN" in statement:
                executed.append((statement, parameters))

        engine = session.get_bind()
        event.listen(engine, "before_cursor_execute", capture)
        try:
            groups = tiers.build_exact_groups_for_keys_in_session(
                session, [("aaa", 100)], scope
            )
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        plans = [
            " | ".join(
                str(row[-1])
                for row in session.connection()
                .exec_driver_sql(f"EXPLAIN QUERY PLAN {statement}", parameters)
                .all()
            )
            for statement, parameters in executed
        ]
        return len(groups), plans

    group_count, plans = _run(server, explain_member_lookup, None)
    assert group_count == 1
    assert len(plans) == 1, plans
    assert "ix_picture_pixel_sha" in plans[0], plans
    assert "ix_picture_deleted" not in plans[0], plans

    group_count, plans = _run(server, explain_member_lookup, set_scope)
    assert group_count == 1
    assert len(plans) == 1, plans
    assert (
        "SEARCH picture USING INDEX ix_picture_pixel_sha (pixel_sha=?)" in (plans[0])
    ), plans
    assert "(set_id=? AND picture_id=?)" in plans[0], plans
    assert_scope_probed_per_row(plans[0], "picturesetmember")
    assert "rowid" not in plans[0], plans
    assert "ix_picture_deleted" not in plans[0], plans


SCOPE_TABLES = {
    ScopeType.SET: "picturesetmember",
    ScopeType.PROJECT: "pictureprojectmember",
    ScopeType.CHARACTER: "face",
}
"""The membership table each collection scope probes."""

_INDEX_READ = re.compile(
    r"^SEARCH (?P<table>\w+)\b[^(]*\bINDEX (?P<index>\w+) \((?P<key>[^)]*)\)$"
)


def assert_scope_probed_per_row(plan: str, table: str) -> None:
    """Assert *plan* reads *table* only as an index probe keyed on ``picture_id``.

    That is the property the scoped reads rely on: membership is checked for
    each row the hash index selects, never built once and walked. It is
    asserted on the shape of the line that reads the table, not on a label:
    SQLite up to 3.50 prints the probe under a ``CORRELATED SCALAR SUBQUERY``
    line and 3.53 prints it as ``SEARCH <table> EXISTS USING INDEX ...``. The
    ``IN (members)`` form this guards against reads the table keyed on the
    scope id alone (``(set_id=?)``), so it fails here on every version.
    *plan* is the ``" | "``-joined detail column of ``EXPLAIN QUERY PLAN``.
    """
    lines = [
        line
        for line in plan.split(" | ")
        if re.match(rf"^(SEARCH|SCAN) {table}\b", line)
    ]
    assert lines, (table, plan)
    for line in lines:
        read = _INDEX_READ.match(line)
        assert read is not None, (table, line, plan)
        assert "picture_id=?" in read["key"].split(" AND "), (table, line, plan)


def _explain_exact_key_reads(server, scope, shards):
    """Run the key discovery for each of *shards* and explain what it sent.

    Returns ``(keys per shard, plan per statement)``, the plans explained on
    the connection that ran the statement.
    """

    def explain(session):
        executed = []

        def capture(_conn, _cursor, statement, parameters, _context, _many):
            if "GROUP BY picture.pixel_sha" in statement:
                executed.append((statement, parameters))

        engine = session.get_bind()
        event.listen(engine, "before_cursor_execute", capture)
        try:
            keys = [
                tiers.find_exact_keys_in_session(session, scope, shard)
                for shard in shards
            ]
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        plans = [
            " | ".join(
                str(row[-1])
                for row in session.connection()
                .exec_driver_sql(f"EXPLAIN QUERY PLAN {statement}", parameters)
                .all()
            )
            for statement, parameters in executed
        ]
        return keys, plans

    return _run(server, explain)


def test_an_exact_key_shard_reads_only_its_slice_of_the_pixel_sha_index(server):
    """Each shard of the key discovery is a covering range of ``ix_picture_pixel_sha``.

    The shards exist so a stop can land between them; that is only worth it if
    a shard reads its sixteenth of the index. PixlStash never runs ``ANALYZE``,
    and without statistics SQLite rates the ``deleted`` equality above a range,
    walking ``ix_picture_deleted`` and the whole live table once per shard: 16
    full reads where there was one. The slice is grouped on the index alone
    (covering, no table row) and only the shared hashes' rows are then read,
    each by an equality on the same index. The first and last shards (one bound
    each) and a middle one (two bounds) are each pinned; the scoped shards are
    pinned by :func:`test_a_scoped_exact_key_shard_is_not_driven_by_the_scope`.
    """
    _seed(
        server,
        [
            {"pixel_sha": "0aa", "size_bytes": 100},
            {"pixel_sha": "0aa", "size_bytes": 100},
            {"pixel_sha": "5bb", "size_bytes": 100},
            {"pixel_sha": "5bb", "size_bytes": 100},
            {"pixel_sha": "fcc", "size_bytes": 100},
            {"pixel_sha": "fcc", "size_bytes": 100},
        ],
    )
    last = tiers.EXACT_KEY_SHARD_COUNT - 1
    keys, plans = _explain_exact_key_reads(server, None, [0, 5, last])

    assert [[key.pixel_sha for key in shard] for shard in keys] == [
        ["0aa"],
        ["5bb"],
        ["fcc"],
    ]
    assert len(plans) == 3, plans
    first, middle, final = plans
    # ``pixel_sha IS NOT NULL`` is the lower bound of the first shard.
    slice_read = (
        "USING COVERING INDEX ix_picture_pixel_sha (pixel_sha>? AND pixel_sha<?)"
    )
    assert slice_read in first, plans
    assert slice_read in middle, plans
    assert "USING COVERING INDEX ix_picture_pixel_sha (pixel_sha>?)" in final, plans
    for plan in plans:
        assert "SEARCH picture USING INDEX ix_picture_pixel_sha (pixel_sha=?)" in plan
        assert "ix_picture_deleted" not in plan, plans


def test_every_exact_key_is_in_exactly_one_shard(server):
    """The shards partition the keys: merged, they are the whole read, in order.

    Hashes sit on each side of a bound (``0f`` / ``1``), on the first and
    last shards' open ends (empty, ``ff``, ``zz``), and outside lowercase hex
    (``A1``, ``9z``), so a bound that drops or doubles a key shows. A key whose
    second copy is scrapheaped is no duplicate, and its lowest live id is the
    one a key is ordered by.
    """
    shas = ["1", "0f", "", "ff", "A1", "9z", "zz", "a"]
    specs = [
        {"pixel_sha": "gone", "size_bytes": 100, "deleted": True},
        {"pixel_sha": "gone", "size_bytes": 100},
        {"pixel_sha": "gone", "size_bytes": 100},
        {"pixel_sha": "half", "size_bytes": 100},
        {"pixel_sha": "half", "size_bytes": 100, "deleted": True},
    ]
    for sha in shas:
        specs += [{"pixel_sha": sha, "size_bytes": 7}] * 2
    ids = _seed(server, specs)

    whole = _run(server, tiers.find_exact_keys_in_session, None)
    shards = [
        _run(server, tiers.find_exact_keys_in_session, None, shard)
        for shard in range(tiers.EXACT_KEY_SHARD_COUNT)
    ]
    merged = sorted(
        (key for shard in shards for key in shard), key=lambda key: key.first_id
    )

    assert [key.pixel_sha for key in whole] == ["gone", *shas]
    assert whole[0] == tiers.ExactKey("gone", 100, 2, ids[1])
    assert merged == whole
    assert sum(len(shard) for shard in shards) == len(whole)
    with pytest.raises(ValueError, match="outside"):
        _run(
            server, tiers.find_exact_keys_in_session, None, tiers.EXACT_KEY_SHARD_COUNT
        )


def test_a_stop_during_the_key_discovery_lands_between_shards(server, monkeypatch):
    """A stop asked for while one shard is read stops the scan before the next.

    The discovery used to be one read of the whole scope, so a stop waited for
    all of it. Nothing is persisted and the scan is left restartable.
    """
    _seed(
        server,
        [
            {"pixel_sha": "0aa", "size_bytes": 100},
            {"pixel_sha": "0aa", "size_bytes": 100},
            {"pixel_sha": "9bb", "size_bytes": 100},
            {"pixel_sha": "9bb", "size_bytes": 100},
        ],
    )
    progress = _run(server, tiers.request_scan_in_session, TierPolicy(), None)
    task = DedupScanTask(server.vault.db, progress["scan_id"])
    read_shards: list[int] = []
    original_find = tiers.find_exact_keys_in_session

    def find_then_stop(session, scope=None, shard=None):
        read_shards.append(shard)
        if shard == 2:
            task.on_cancel()
        return original_find(session, scope, shard)

    monkeypatch.setattr(tiers, "find_exact_keys_in_session", find_then_stop)
    summary = task._run_task()

    assert summary == {
        "scan_id": progress["scan_id"],
        "status": SCAN_PENDING,
        "cancelled": True,
    }
    assert read_shards == [0, 1, 2]
    assert _run(server, lambda s: s.exec(select(DedupGroup)).all()) == []
    assert _run(server, tiers.scan_progress_in_session, None)["status"] == SCAN_PENDING


def test_a_failed_scan_stores_a_fixed_sentence_and_logs_the_error(
    server, monkeypatch, caplog
):
    """The scan row keeps a fixed sentence; the exception text goes to the log.

    The row's error is served to the UI, and an exception's text can carry
    paths or SQL (OWASP Error Handling Cheat Sheet).
    """
    _seed(server, [{"pixel_sha": "0aa", "size_bytes": 100}])
    progress = _run(server, tiers.request_scan_in_session, TierPolicy(), None)
    scan_id = progress["scan_id"]

    def fail_exact(_session, _scope=None, _shard=None):
        raise RuntimeError("example-secret-path/vault.db is locked")

    monkeypatch.setattr(tiers, "find_exact_keys_in_session", fail_exact)
    with caplog.at_level("ERROR", logger=dedup_scan_task_module.logger.name):
        with pytest.raises(RuntimeError, match="example-secret-path"):
            DedupScanTask(server.vault.db, scan_id)._run_task()

    served = _run(server, tiers.scan_progress_in_session, None)
    assert served["status"] == "failed"
    assert served["error"] == dedup_scan_task_module.SCAN_FAILED_MESSAGE
    assert "example-secret-path" not in repr(served)
    assert _run(server, lambda s: s.get(DedupScan, scan_id)).error == (
        dedup_scan_task_module.SCAN_FAILED_MESSAGE
    )
    logged = [r for r in caplog.records if f"scan_id={scan_id}" in r.getMessage()]
    assert logged, caplog.text
    assert "example-secret-path" in caplog.text


def test_the_sharded_discovery_pages_keys_in_the_unsharded_order(server, monkeypatch):
    """Shards split on the hash, so the merge restores lowest-picture-first.

    A hash late in the alphabet whose copies were imported first must still
    lead the pages, exactly as one unsharded read orders them.
    """
    _seed(
        server,
        [
            {"pixel_sha": sha, "size_bytes": 100}
            for sha in ("f1", "f1", "01", "01", "a7", "a7")
        ],
    )
    paged = []
    original_page_exact_keys = tiers.page_exact_keys

    def record(keys, *args, **kwargs):
        paged.append(list(keys))
        return original_page_exact_keys(keys, *args, **kwargs)

    monkeypatch.setattr(tiers, "page_exact_keys", record)
    assert _run_requested_scan(server)["status"] == "complete"

    assert [key.pixel_sha for key in paged[0]] == ["f1", "01", "a7"]
    assert paged == [_run(server, tiers.find_exact_keys_in_session)]


SCOPED_FOLDER = "/scoped"


def _collection_scopes(server, picture_ids):
    """A set, a project and a character holding exactly *picture_ids*.

    Returned with the folder scope of the pictures seeded with
    ``import_source_folder=SCOPED_FOLDER``, keyed by scope type.
    """

    def add(session):
        picture_set = PictureSet(name="Scoped")
        project = Project(name="Scoped")
        character = Character(name="Scoped")
        session.add_all([picture_set, project, character])
        session.flush()
        for picture_id in picture_ids:
            session.add(
                PictureSetMember(set_id=int(picture_set.id), picture_id=picture_id)
            )
            session.add(
                PictureProjectMember(project_id=int(project.id), picture_id=picture_id)
            )
            session.add(Face(picture_id=picture_id, character_id=int(character.id)))
        session.commit()
        return int(picture_set.id), int(project.id), int(character.id)

    set_id, project_id, character_id = _run(server, add)
    return {
        ScopeType.SET: DedupScope(ScopeType.SET, str(set_id)),
        ScopeType.PROJECT: DedupScope(ScopeType.PROJECT, str(project_id)),
        ScopeType.CHARACTER: DedupScope(ScopeType.CHARACTER, str(character_id)),
        ScopeType.FOLDER: DedupScope(ScopeType.FOLDER, SCOPED_FOLDER),
    }


def _create_index_last(server, name):
    """Drop index *name* and create it again, so it is the newest index.

    Asserts the newest index on its table really is *name* afterwards.
    """

    def recreate(session):
        found = session.execute(
            text("SELECT tbl_name, sql FROM sqlite_master WHERE name = :name"),
            {"name": name},
        ).one()
        session.execute(text(f"DROP INDEX {name}"))
        session.execute(text(found.sql))
        session.commit()
        return session.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
                " AND tbl_name = :table ORDER BY rowid DESC LIMIT 1"
            ),
            {"table": found.tbl_name},
        ).scalar()

    assert _run(server, recreate) == name


def test_a_scoped_exact_key_shard_is_not_driven_by_the_scope(server):
    """A scoped shard reads its slice of the hash index, not the whole scope.

    The scope used to be ``Picture.id IN (members)``, which SQLite builds once
    per statement and drives the read from by ``rowid``, the hash range only
    filtering: each of the sixteen shards then read the whole scope, 3-9x the
    one unsharded read. Every scope type is pinned to the hash-driven plan, the
    membership checked per candidate row (a correlated probe), and its keys
    must be the whole scoped read's: copies outside the scope, a scrapheaped
    copy and a copy at another size are no duplicate.
    """
    inside = {"import_source_folder": SCOPED_FOLDER}
    ids = _seed(
        server,
        [
            {"pixel_sha": "0zz", "size_bytes": 100},
            {"pixel_sha": "0zz", "size_bytes": 100},
            {"pixel_sha": "5bb", "size_bytes": 100},
            {"pixel_sha": "5bb", "size_bytes": 100, **inside},
            {"pixel_sha": "0aa", "size_bytes": 100, **inside},
            {"pixel_sha": "0aa", "size_bytes": 100, **inside},
            {"pixel_sha": "5bb", "size_bytes": 100, **inside},
            {"pixel_sha": "fcc", "size_bytes": 100, **inside},
            {"pixel_sha": "fcc", "size_bytes": 100, **inside},
            {"pixel_sha": "5dd", "size_bytes": 100, **inside},
            {"pixel_sha": "5dd", "size_bytes": 100},
            {"pixel_sha": "5ee", "size_bytes": 100, **inside},
            {"pixel_sha": "5ee", "size_bytes": 100, "deleted": True, **inside},
            {"pixel_sha": "5ff", "size_bytes": 100, **inside},
            {"pixel_sha": "5ff", "size_bytes": 200, **inside},
        ],
    )
    member_ids = [ids[3], *ids[4:10], ids[11], ids[12], ids[13], ids[14]]
    scopes = _collection_scopes(server, member_ids)
    last = tiers.EXACT_KEY_SHARD_COUNT - 1
    expected = [
        tiers.ExactKey("5bb", 100, 2, ids[3]),
        tiers.ExactKey("0aa", 100, 2, ids[4]),
        tiers.ExactKey("fcc", 100, 2, ids[7]),
    ]

    for scope_type, scope in scopes.items():
        whole = _run(server, tiers.find_exact_keys_in_session, scope)
        merged = sorted(
            (
                key
                for shard in range(tiers.EXACT_KEY_SHARD_COUNT)
                for key in _run(server, tiers.find_exact_keys_in_session, scope, shard)
            ),
            key=lambda key: key.first_id,
        )
        assert whole == expected, scope_type
        assert merged == expected, scope_type

        keys, plans = _explain_exact_key_reads(server, scope, [0, 5, last])
        assert [[key.pixel_sha for key in shard] for shard in keys] == [
            ["0aa"],
            ["5bb"],
            ["fcc"],
        ], scope_type
        assert len(plans) == 3, (scope_type, plans)
        slice_read = (
            "USING COVERING INDEX ix_picture_pixel_sha (pixel_sha>? AND pixel_sha<?)"
        )
        assert slice_read in plans[0], (scope_type, plans)
        assert slice_read in plans[1], (scope_type, plans)
        assert "USING COVERING INDEX ix_picture_pixel_sha (pixel_sha>?)" in plans[2]
        for plan in plans:
            assert "SEARCH picture USING INDEX ix_picture_pixel_sha (pixel_sha=?)" in (
                plan
            ), (scope_type, plans)
            assert "rowid" not in plan, (scope_type, plans)
            assert "INTEGER PRIMARY KEY" not in plan, (scope_type, plans)
            assert "ix_picture_deleted" not in plan, (scope_type, plans)
            if scope_type is not ScopeType.FOLDER:
                assert_scope_probed_per_row(plan, SCOPE_TABLES[scope_type])
        if scope_type is ScopeType.CHARACTER:
            # face has an index on picture_id and one on character_id. Without
            # ANALYZE the two tie, the one created last wins, and create_all
            # creates them in set order. Recreate the character's index so it
            # is the newer one: the probe must still take the picture's faces,
            # not every face of the character.
            _create_index_last(server, "ix_face_character_id")
            _, plans = _explain_exact_key_reads(server, scope, [5])
            assert "ix_face_picture_id (picture_id=?)" in plans[0], plans
            assert "ix_face_character_id" not in plans[0], plans


def test_a_small_collection_scope_reads_its_keys_in_one_step():
    """The shard plan costs a walk of the library's hash index, however small
    the scope, so a collection scope up to the limit is read whole, driven by
    its membership; the library and a folder always shard."""
    limit = tiers.EXACT_KEY_WHOLE_READ_MAX_PICTURES
    every_shard = tuple(range(tiers.EXACT_KEY_SHARD_COUNT))
    for scope_type in (ScopeType.SET, ScopeType.PROJECT, ScopeType.CHARACTER):
        scope = DedupScope(scope_type, "1")
        assert tiers.exact_key_shards(scope, 0) == (None,)
        assert tiers.exact_key_shards(scope, limit) == (None,)
        assert tiers.exact_key_shards(scope, limit + 1) == every_shard
    assert tiers.exact_key_shards(DedupScope(), 2) == every_shard
    folder = DedupScope(ScopeType.FOLDER, SCOPED_FOLDER)
    assert tiers.exact_key_shards(folder, 2) == every_shard


def test_a_small_scoped_scan_reads_keys_once_and_progress_in_one_pass(
    server, monkeypatch
):
    """A set scan: one whole key read, then progress in one pass of the set.

    The progress read used to count each page's id range with the scope's
    ``IN`` list, which SQLite drives from the list: every page re-read the
    whole scope in one reader step (1.8 s for 49 pages of a 100,000-picture
    set). It is now one statement whatever the number of pages, driven by the
    set's members, and pictures outside the set are not counted. The library
    keeps its count per page: each is a covering range of
    ``ix_picture_deleted``, together one pass over it.
    """
    outside = {"pixel_sha": "outside", "perceptual_hash": PHASH_FAR}
    specs = [outside, *_SETTLED_SPECS[:4], outside, *_SETTLED_SPECS[4:]]
    ids = _seed(server, specs)
    member_ids = [ids[1], *ids[2:5], *ids[6:]]
    scope = _collection_scopes(server, member_ids)[ScopeType.SET]

    keys = _run(server, tiers.find_exact_keys_in_session, scope)
    pages = tiers.page_exact_keys(keys, max_keys=1)
    assert [page[0].pixel_sha for page in pages] == ["a", "b", "c"]

    def explain_progress(session, scope, pages, total):
        executed = []

        def capture(_conn, _cursor, statement, parameters, _context, _many):
            if "FROM picture" in statement:
                executed.append((statement, parameters))

        engine = session.get_bind()
        event.listen(engine, "before_cursor_execute", capture)
        try:
            published = DedupScanTask._exact_progress_read(session, scope, pages, total)
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        plans = [
            " | ".join(
                str(row[-1])
                for row in session.connection()
                .exec_driver_sql(f"EXPLAIN QUERY PLAN {statement}", parameters)
                .all()
            )
            for statement, parameters in executed
        ]
        return published, plans

    published, plans = _run(server, explain_progress, scope, pages, 9)
    assert published == _SETTLED_AFTER_PAGES
    assert len(plans) == 1, plans
    assert "ix_picture_deleted (deleted=? AND rowid=?)" in plans[0], plans
    assert "picturesetmember" in plans[0], plans

    # The library: the two outside pictures are a key and count, 11 live.
    library_keys = _run(server, tiers.find_exact_keys_in_session)
    library_pages = tiers.page_exact_keys(library_keys, max_keys=1)
    assert [page[0].pixel_sha for page in library_pages] == [
        "outside",
        "a",
        "b",
        "c",
    ]
    published, plans = _run(server, explain_progress, DedupScope(), library_pages, 11)
    assert published == [1, 5, 8, 11]
    assert len(plans) == 3, plans
    for plan in plans:
        assert "USING COVERING INDEX ix_picture_deleted (deleted=? AND rowid" in (
            plan
        ), plans

    read_shards: list[int | None] = []
    original_find = tiers.find_exact_keys_in_session

    def record(session, scope=None, shard=None):
        read_shards.append(shard)
        return original_find(session, scope, shard)

    monkeypatch.setattr(tiers, "find_exact_keys_in_session", record)
    assert _run_requested_scan(server, scope=scope)["status"] == "complete"
    assert read_shards == [None]


# ── tier 2: bucketed near ─────────────────────────────────────────────────────


def test_buckets_come_from_the_precomputed_columns(server):
    _seed(
        server,
        [
            {"perceptual_hash": PHASH_ZERO, "width": 100, "height": 100},
            {"perceptual_hash": PHASH_ONE_BIT, "width": 100, "height": 100},
            {"perceptual_hash": PHASH_FAR, "width": 200, "height": 200},
        ],
    )
    buckets = _run(server, tiers.build_near_buckets, None)
    kinds = {bucket.kind for bucket in buckets}
    assert "size_bin" in kinds
    # The 200x200 picture is alone in its size bin, so that bucket is dropped:
    # a singleton bucket is not work and must not inflate the progress total.
    size_bins = [b for b in buckets if b.kind == "size_bin"]
    assert len(size_bins) == 1
    assert len(size_bins[0].picture_ids) == 2


def test_near_pairs_are_only_compared_inside_a_bucket(server):
    """A picture that shares no bucket is never compared, however similar it is.

    The third picture has a byte-identical perceptual hash but different
    dimensions *and* a different folder, so it shares no candidate bucket with
    the pair. Library-wide comparison would have pulled it in; bucketed
    comparison does not.
    """
    ids = _seed(
        server,
        [
            # Same dimensions and folder -> same bucket, 1 bit apart.
            {
                "perceptual_hash": PHASH_ZERO,
                "width": 100,
                "height": 100,
                "file_path": "/vault/a/one.png",
            },
            {
                "perceptual_hash": PHASH_ONE_BIT,
                "width": 100,
                "height": 100,
                "file_path": "/vault/a/two.png",
            },
            {
                "perceptual_hash": PHASH_ZERO,
                "width": 300,
                "height": 300,
                "file_path": "/vault/b/three.png",
            },
        ],
    )
    groups = _run(
        server, tiers.find_near_groups_in_session, TierPolicy(near_enabled=True), None
    )
    assert len(groups) == 1
    assert sorted(groups[0].picture_ids) == sorted(ids[:2])
    assert groups[0].confidence == pytest.approx(63 / 64)
    assert groups[0].tier is DedupTier.NEAR


def test_capture_minute_buckets_catch_a_resize(server):
    """Different dimensions, same capture minute: still one bucket, still found."""
    ids = _seed(
        server,
        [
            {
                "perceptual_hash": PHASH_ZERO,
                "width": 4000,
                "height": 3000,
                "created_at": 0,
            },
            {
                "perceptual_hash": PHASH_ONE_BIT,
                "width": 2000,
                "height": 1500,
                "created_at": 5,
            },
        ],
    )
    groups = _run(
        server, tiers.find_near_groups_in_session, TierPolicy(near_enabled=True), None
    )
    assert len(groups) == 1
    assert sorted(groups[0].picture_ids) == sorted(ids)
    assert any(
        pill["against"] and pill["text"] == "Different resolution"
        for pill in groups[0].evidence
    )


def test_the_threshold_excludes_a_weaker_near_pair(server):
    _seed(
        server,
        [
            {"perceptual_hash": PHASH_ZERO, "width": 100, "height": 100},
            {"perceptual_hash": PHASH_FOUR_BITS, "width": 100, "height": 100},
        ],
    )
    # 4 bits apart is 0.9375: above the 0.90 default, below a 0.99 threshold.
    loose = _run(
        server, tiers.find_near_groups_in_session, TierPolicy(near_enabled=True), None
    )
    assert len(loose) == 1
    tight = _run(
        server,
        tiers.find_near_groups_in_session,
        TierPolicy(near_enabled=True, threshold=0.99),
        None,
    )
    assert tight == []


def test_a_group_is_judged_by_its_weakest_link(server):
    """A~B at 1 bit and B~C at 4 bits is one group whose confidence is the 4-bit edge."""
    ids = _seed(
        server,
        [
            {"perceptual_hash": PHASH_ZERO, "width": 100, "height": 100},
            {"perceptual_hash": PHASH_ONE_BIT, "width": 100, "height": 100},
            {"perceptual_hash": PHASH_FOUR_BITS, "width": 100, "height": 100},
        ],
    )
    groups = _run(
        server, tiers.find_near_groups_in_session, TierPolicy(near_enabled=True), None
    )
    assert len(groups) == 1
    assert sorted(groups[0].picture_ids) == sorted(ids)
    assert groups[0].confidence == pytest.approx(60 / 64)


def test_unparseable_perceptual_hashes_are_excluded_not_crashed(server):
    _seed(
        server,
        [
            {"perceptual_hash": PHASH_ZERO, "width": 100, "height": 100},
            {"perceptual_hash": "not-a-hash-xx", "width": 100, "height": 100},
            {"perceptual_hash": "abc", "width": 100, "height": 100},
        ],
    )
    assert (
        _run(
            server,
            tiers.find_near_groups_in_session,
            TierPolicy(near_enabled=True),
            None,
        )
        == []
    )


def test_a_signed_perceptual_hash_is_excluded_and_the_rest_still_group(server):
    """``int(..., 16)`` accepts a sign; the negative must not reach the uint64 pack."""
    ids = _seed(
        server,
        [
            {"perceptual_hash": PHASH_ZERO, "width": 100, "height": 100},
            {"perceptual_hash": PHASH_ONE_BIT, "width": 100, "height": 100},
            {"perceptual_hash": "-000000000000001", "width": 100, "height": 100},
        ],
    )
    groups = _run(
        server, tiers.find_near_groups_in_session, TierPolicy(near_enabled=True), None
    )
    assert [sorted(group.picture_ids) for group in groups] == [sorted(ids[:2])]


# ── tier 3: embedding ─────────────────────────────────────────────────────────


def test_embedding_tier_reuses_the_shipped_likeness_table(server):
    ids = _seed(server, [{}, {}, {}])

    def link(session):
        first, second = PictureLikeness.canon_pair(ids[0], ids[1])
        session.add(
            PictureLikeness(
                picture_id_a=first, picture_id_b=second, likeness=0.97, metric="test"
            )
        )
        session.commit()

    _run(server, link)
    policy = TierPolicy(near_enabled=True, embedding_enabled=True)
    groups = _run(server, tiers.find_embedding_groups_in_session, policy, None)
    assert len(groups) == 1
    assert sorted(groups[0].picture_ids) == sorted(ids[:2])
    assert groups[0].tier is DedupTier.EMBEDDING
    assert groups[0].confidence == pytest.approx(0.97)


# ── persistence, the queue and the counts ─────────────────────────────────────


def _scan(server, policy=None, scope=None):
    return _run(server, tiers.run_scan_now_in_session, policy or TierPolicy(), scope)


def test_a_rescan_refreshes_groups_instead_of_duplicating_them(server):
    _seed(
        server,
        [
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "aaa", "size_bytes": 100},
        ],
    )
    assert _scan(server)["unresolved_groups"] == 1
    _scan(server)
    rows = _run(server, lambda session: session.exec(select(DedupGroup)).all())
    assert len(rows) == 1


def test_the_queue_pages_by_confidence_descending(server):
    _seed(
        server,
        [
            # An exact pair (confidence 1.0). Separate folders so it does not
            # also land in a near bucket with the pairs below.
            {
                "pixel_sha": "aaa",
                "size_bytes": 100,
                "width": 10,
                "height": 10,
                "file_path": "/vault/x/1.png",
            },
            {
                "pixel_sha": "aaa",
                "size_bytes": 100,
                "width": 10,
                "height": 10,
                "file_path": "/vault/x/2.png",
            },
            # A 1-bit near pair (0.984).
            {
                "perceptual_hash": PHASH_ZERO,
                "width": 20,
                "height": 20,
                "file_path": "/vault/y/1.png",
            },
            {
                "perceptual_hash": PHASH_ONE_BIT,
                "width": 20,
                "height": 20,
                "file_path": "/vault/y/2.png",
            },
            # A 4-bit near pair (0.9375), in its own folder so it stays its own
            # group rather than chaining onto the pair above.
            {
                "perceptual_hash": PHASH_ZERO,
                "width": 30,
                "height": 30,
                "file_path": "/vault/z/1.png",
            },
            {
                "perceptual_hash": PHASH_FOUR_BITS,
                "width": 30,
                "height": 30,
                "file_path": "/vault/z/2.png",
            },
        ],
    )
    policy = TierPolicy(near_enabled=True)
    _scan(server, policy)
    page, total, _cursor = _run(server, tiers.page_queue_in_session, policy, None, 0, 2)
    assert total == 3
    assert [round(group["confidence"], 4) for group in page] == [
        1.0,
        round(63 / 64, 4),
    ]
    assert page[0]["tier"] == "exact"
    second, _total, _cursor = _run(
        server, tiers.page_queue_in_session, policy, None, 2, 2
    )
    assert len(second) == 1
    assert second[0]["confidence"] == pytest.approx(60 / 64)


def test_the_queue_carries_the_cover_and_both_evidence_layers(server):
    ids = _seed(
        server,
        [
            {
                "pixel_sha": "aaa",
                "size_bytes": 100,
                "score": 5,
                "tags": ["portrait", "outdoor"],
            },
            {"pixel_sha": "aaa", "size_bytes": 100},
        ],
    )
    _scan(server)
    page, _total, _cursor = _run(server, tiers.page_queue_in_session, None, None, 0, 10)
    group = page[0]
    assert group["cover_picture_id"] == ids[0]
    assert any(pill["text"] == "Identical file hash" for pill in group["why"])
    cover = next(c for c in group["candidates"] if c["picture_id"] == ids[0])
    assert cover["tag_count"] == 2
    assert any(p["text"] == "Preselected as cover" for p in cover["why"])
    assert any(p["text"].startswith("Most metadata") for p in cover["why"])


def test_the_scan_ranks_the_cover_on_smart_score_end_to_end(server):
    """The stored smart score drives the baked cover and member order, and the
    queue serves the ranking signals null-safe (the -1.0 failed sentinel and a
    missing Quality row both read as null on the wire)."""
    ids = _seed(
        server,
        [
            # Bigger, heavily tagged and starred - but the scorer rates it low.
            {
                "pixel_sha": "aaa",
                "size_bytes": 100,
                "width": 8000,
                "height": 6000,
                "score": 5,
                "smart_score": 2.0,
                "sharpness": -1.0,
                "tags": ["portrait", "outdoor"],
            },
            # Smaller but rated clearly better: the cover under the new rule.
            {
                "pixel_sha": "aaa",
                "size_bytes": 100,
                "width": 4000,
                "height": 3000,
                "smart_score": 4.5,
                "sharpness": 0.4,
            },
        ],
    )
    _scan(server)
    page, _total, _cursor = _run(server, tiers.page_queue_in_session, None, None, 0, 10)
    group = page[0]
    assert group["cover_picture_id"] == ids[1]
    # Members serialize in ranking order, cover first.
    assert [c["picture_id"] for c in group["candidates"]] == [ids[1], ids[0]]
    cover, other = group["candidates"]
    assert cover["smart_score"] == pytest.approx(4.5)
    assert cover["sharpness"] == pytest.approx(0.4)
    assert any(p["text"] == "Best smart score (4.5)" for p in cover["why"])
    assert any(p["text"] == "Sharpest copy" for p in cover["why"])
    assert any(p["text"] == "Preselected as cover" for p in cover["why"])
    assert other["smart_score"] == pytest.approx(2.0)
    assert other["sharpness"] is None, "-1.0 failed sentinel must serve null"
    assert any(
        p["against"] and p["text"] == "Lower smart score (2.0 vs 4.5)"
        for p in other["why"]
    )


def test_the_near_tier_is_hidden_until_it_is_switched_on(server):
    _seed(
        server,
        [
            {"perceptual_hash": PHASH_ZERO, "width": 100, "height": 100},
            {"perceptual_hash": PHASH_ONE_BIT, "width": 100, "height": 100},
        ],
    )
    near_policy = TierPolicy(near_enabled=True)
    _scan(server, near_policy)
    # Detected and stored, but the default policy shows only tier 1.
    assert _run(server, tiers.count_unresolved_in_session, TierPolicy(), None) == 0
    assert _run(server, tiers.count_unresolved_in_session, near_policy, None) == 1
    # The per-tier counts report the switched-off tier anyway, so the user can
    # see what enabling it would add.
    by_tier = _run(server, tiers.count_by_tier_in_session, TierPolicy(), None)
    assert by_tier["near"] == 1
    assert by_tier["exact"] == 0


def test_a_recorded_verdict_resolves_the_group_on_the_next_scan(server):
    _seed(
        server,
        [
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "aaa", "size_bytes": 100},
        ],
    )
    _scan(server)
    page, _total, _cursor = _run(server, tiers.page_queue_in_session, None, None, 0, 10)
    signature = page[0]["signature"]

    def record(session):
        session.add(
            DedupVerdict(
                signature=signature,
                verdict=VERDICT_KEEP_SEPARATE,
                picture_ids="[]",
                excluded_picture_ids="[]",
            )
        )
        session.commit()

    _run(server, record)
    # A rescan re-derives the same signature and never re-asks.
    _scan(server)
    assert _run(server, tiers.count_unresolved_in_session, None, None) == 0
    page, total, _cursor = _run(server, tiers.page_queue_in_session, None, None, 0, 10)
    assert page == [] and total == 0


def test_a_reopened_verdict_puts_the_group_back_in_the_queue(server):
    _seed(
        server,
        [
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "aaa", "size_bytes": 100},
        ],
    )
    _scan(server)
    page, _total, _cursor = _run(server, tiers.page_queue_in_session, None, None, 0, 10)
    signature = page[0]["signature"]

    def record_and_reopen(session):
        session.add(
            DedupVerdict(
                signature=signature,
                verdict=VERDICT_KEEP_SEPARATE,
                picture_ids="[]",
                excluded_picture_ids="[]",
                reopened_at=_BASE_TIME,
            )
        )
        session.commit()

    _run(server, record_and_reopen)
    _scan(server)
    assert _run(server, tiers.count_unresolved_in_session, None, None) == 1


def test_stale_groups_are_pruned_when_their_members_go_to_the_scrapheap(server):
    ids = _seed(
        server,
        [
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "aaa", "size_bytes": 100},
        ],
    )
    _scan(server)
    assert _run(server, tiers.count_unresolved_in_session, None, None) == 1

    def soft_delete(session):
        pic = session.get(Picture, ids[1])
        pic.deleted = True
        session.add(pic)
        session.commit()

    _run(server, soft_delete)
    assert _run(server, tiers.prune_stale_groups_in_session) == 1
    assert _run(server, tiers.count_unresolved_in_session, None, None) == 0


def _soft_delete(server, picture_id):
    """Move one picture to the scrapheap, the way ``DELETE /pictures/{id}`` does."""

    def go(session):
        picture = session.get(Picture, picture_id)
        picture.deleted = True
        session.add(picture)
        session.commit()

    _run(server, go)


def test_a_group_thinned_below_two_leaves_the_queue_at_once(server):
    """Not "on the next verdict or scan": immediately, with no prune in between.

    The owner's report: scrapheap one of a pair and the queue kept the row,
    drawn as a lone picture beside an empty slot.
    """
    ids = _seed(
        server,
        [
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "aaa", "size_bytes": 100},
        ],
    )
    _scan(server)
    page, total, _cursor = _run(server, tiers.page_queue_in_session)
    assert len(page) == 1 and total == 1

    _soft_delete(server, ids[1])

    page, total, _cursor = _run(server, tiers.page_queue_in_session)
    assert page == []
    assert total == 0
    assert _run(server, tiers.count_unresolved_in_session, None, None) == 0
    assert _run(server, tiers.count_by_tier_in_session, None, None)["exact"] == 0
    # The row itself is still there, unpruned: it is the READ that filters, so
    # restoring the picture must bring the group straight back.
    assert (
        len(_run(server, lambda session: session.exec(select(DedupGroup)).all())) == 1
    )


def test_a_restored_member_brings_its_group_back_without_a_rescan(server):
    ids = _seed(
        server,
        [
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "aaa", "size_bytes": 100},
        ],
    )
    _scan(server)
    signature = _run(server, tiers.page_queue_in_session)[0][0]["signature"]
    _soft_delete(server, ids[1])
    assert _run(server, tiers.page_queue_in_session)[0] == []

    def restore(session):
        picture = session.get(Picture, ids[1])
        picture.deleted = False
        session.add(picture)
        session.commit()

    _run(server, restore)
    page, total, _cursor = _run(server, tiers.page_queue_in_session)
    assert [group["signature"] for group in page] == [signature]
    assert total == 1
    assert _run(server, tiers.count_unresolved_in_session, None, None) == 1


def test_a_group_with_two_live_members_left_keeps_its_place_in_the_queue(server):
    """Over-filtering is its own regression: three minus one is still a decision.

    The row stays, and every number on it counts the LIVE members only, the
    stored ``member_count`` still says three until the next prune.
    """
    ids = _seed(
        server,
        [
            {"pixel_sha": "bbb", "size_bytes": 100},
            {"pixel_sha": "bbb", "size_bytes": 100},
            {"pixel_sha": "bbb", "size_bytes": 100},
        ],
    )
    _scan(server)
    _soft_delete(server, ids[2])

    page, total, _cursor = _run(server, tiers.page_queue_in_session)
    assert len(page) == 1
    assert total == 1
    group = page[0]
    assert group["member_count"] == 2
    assert sorted(c["picture_id"] for c in group["candidates"]) == sorted(ids[:2])
    assert group["cover_picture_id"] in ids[:2]
    stored = _run(
        server,
        lambda session: session.exec(select(DedupGroup)).first().member_count,
    )
    assert stored == 3, "the payload's count is live even while the stored one lags"


def test_large_group_evidence_tracks_the_live_member_count(server):
    """The warning and header must never report two different group sizes.

    The evidence is stored at scan time, but the queue filters scrapheaped
    members live. A group that shrinks back under the policy limit must lose
    its stale size warning rather than saying both "2 pictures" and "3
    pictures" in the same row.
    """
    ids = _seed(
        server,
        [
            {"pixel_sha": "large", "size_bytes": 100},
            {"pixel_sha": "large", "size_bytes": 100},
            {"pixel_sha": "large", "size_bytes": 100},
        ],
    )
    policy = TierPolicy(max_group_size=2)
    _scan(server, policy)

    page, _total, _cursor = _run(server, tiers.page_queue_in_session, policy)
    assert page[0]["member_count"] == 3
    assert any("(3 pictures)" in pill["text"] for pill in page[0]["why"])

    _soft_delete(server, ids[2])
    page, _total, _cursor = _run(server, tiers.page_queue_in_session, policy)
    assert page[0]["member_count"] == 2
    assert not any(
        pill["text"].startswith("Unusually large group") for pill in page[0]["why"]
    )


def test_a_thinned_group_still_lists_on_the_decided_page(server):
    """The way back must survive the scrapheap.

    The open queue drops a group that no longer poses a decision, but the
    decided page keeps it: the verdict already happened and "clear this
    decision" is the only route back to it.
    """
    ids = _seed(
        server,
        [
            {"pixel_sha": "ccc", "size_bytes": 100},
            {"pixel_sha": "ccc", "size_bytes": 100},
        ],
    )
    _scan(server)
    signature = _run(server, tiers.page_queue_in_session)[0][0]["signature"]
    _run(server, verdicts.apply_keep_separate_in_session, signature, None)
    _soft_delete(server, ids[1])

    page, total, _cursor = _run(
        server, tiers.page_queue_in_session, None, None, 0, 20, None, True
    )
    assert [group["signature"] for group in page] == [signature]
    assert total == 1
    assert page[0]["member_count"] == 1, "live members only, even here"


# ── decisions outlive prune and retirement ───────────────────────────────────


def _decide_the_only_group(server):
    """Keep the one open group's pictures separate; return its signature."""
    signature = _run(server, tiers.page_queue_in_session)[0][0]["signature"]
    _run(server, verdicts.apply_keep_separate_in_session, signature, None)
    return signature


def _stored_groups(server):
    """``{signature: (resolved, sorted member ids)}`` for every stored group."""

    def read(session):
        return {
            row.signature: (row.resolved, _stored_members(session, row.id))
            for row in session.exec(select(DedupGroup)).all()
        }

    return _run(server, read)


def _decided_signatures(server):
    page, total, _cursor = _run(
        server, tiers.page_queue_in_session, None, None, 0, 20, None, True
    )
    assert total == len(page)
    return [group["signature"] for group in page]


def test_a_decided_group_thinned_to_one_keeps_its_decision(server):
    """Keep-cover-only leaves one live copy; the decision must stay clearable.

    Both deleters used to drop it: the prune (fewer than two live members) and
    the completed scan's retirement (its signature is no longer found).
    """
    ids = _seed(server, [{"pixel_sha": "ddd", "size_bytes": 100}] * 3)
    _run_requested_scan(server)
    signature = _decide_the_only_group(server)
    _soft_delete(server, ids[1])
    _soft_delete(server, ids[2])

    assert _run(server, tiers.prune_stale_groups_in_session) == 0
    rescan = _run_requested_scan(server)

    assert rescan["status"] == "complete"
    assert rescan["retired_groups"] == 0
    assert _stored_groups(server) == {signature: (True, sorted(ids))}
    assert _decided_signatures(server) == [signature]
    assert _run(server, tiers.count_unresolved_in_session, None, None) == 0


def test_a_decided_group_goes_once_a_purge_leaves_it_one_member(server):
    """The decided exemption ends where the pictures do."""
    ids = _seed(server, [{"pixel_sha": "ddd", "size_bytes": 100}] * 2)
    _scan(server)
    _decide_the_only_group(server)

    def purge(session):
        session.delete(session.get(Picture, ids[1]))
        session.commit()

    _run(server, purge)
    assert _run(server, tiers.prune_stale_groups_in_session) == 1
    assert _stored_groups(server) == {}


def test_a_decision_covers_its_pictures_when_some_of_them_leave(server):
    """A subset of a decided group is not a new question.

    The Scrapheap takes one copy, so the global rescan finds the other three
    under a new signature; a scoped scan sees only the scope's two. Neither is
    asked again, and the decided group keeps its place.
    """
    ids = _seed(server, [{"pixel_sha": "eee", "size_bytes": 100}] * 4)
    _run_requested_scan(server)
    signature = _decide_the_only_group(server)
    _soft_delete(server, ids[3])

    _run_requested_scan(server)
    assert _stored_groups(server) == {signature: (True, sorted(ids))}

    def add_set(session):
        picture_set = PictureSet(name="Part of the group")
        session.add(picture_set)
        session.flush()
        for picture_id in ids[:2]:
            session.add(
                PictureSetMember(set_id=int(picture_set.id), picture_id=picture_id)
            )
        session.commit()
        return int(picture_set.id)

    scope = DedupScope(scope_type=ScopeType.SET, scope_id=str(_run(server, add_set)))
    _run_requested_scan(server, None, scope)

    assert _stored_groups(server) == {signature: (True, sorted(ids))}
    assert _decided_signatures(server) == [signature]
    assert _run(server, tiers.count_unresolved_in_session, None, None) == 0


def test_a_new_copy_asks_a_decided_group_again(server):
    """Members changed: a copy the decision never saw joins the group."""
    ids = _seed(server, [{"pixel_sha": "fff", "size_bytes": 100}] * 2)
    _run_requested_scan(server)
    signature = _decide_the_only_group(server)
    ids += _seed(server, [{"pixel_sha": "fff", "size_bytes": 100}])

    _run_requested_scan(server)

    stored = _stored_groups(server)
    assert signature not in stored
    assert list(stored.values()) == [(False, sorted(ids))]
    assert _decided_signatures(server) == []
    assert _run(server, tiers.count_unresolved_in_session, None, None) == 1


def test_retirement_never_takes_a_decided_group(server):
    """Not when the reader sees the decision, nor when it lands after the read."""
    ids = _seed(server, [{"pixel_sha": "ggg", "size_bytes": 100}] * 2)
    _run_requested_scan(server)
    scan_id = _run(server, tiers.scan_progress_in_session, None)["scan_id"]
    every_tier = {"exact", "near", "embedding"}
    nothing_found = {tier: set() for tier in every_tier}
    [group_id] = _run(
        server,
        tiers.obsolete_scan_group_ids_in_session,
        scan_id,
        nothing_found,
        every_tier,
    )

    signature = _decide_the_only_group(server)
    assert (
        _run(
            server,
            tiers.obsolete_scan_group_ids_in_session,
            scan_id,
            nothing_found,
            every_tier,
        )
        == []
    )

    def delete_and_commit(session):
        removed = tiers.delete_scan_groups_in_session(session, scan_id, [group_id])
        session.commit()
        return removed

    assert _run(server, delete_and_commit) == 0
    assert _stored_groups(server) == {signature: (True, sorted(ids))}


def _persist_exact(session, member_lists, scan_id=None, pill=None):
    """Persist one exact group per id list; return the count and the signatures.

    ``pill`` is appended to the first group's evidence, so a test can tell a
    refreshed row's evidence from the one it was created with.
    """
    groups = []
    for picture_ids in member_lists:
        members = tiers.load_candidates(session, picture_ids)
        groups.append(
            tiers.assemble_group(
                DedupTier.EXACT, 1.0, [members[pid] for pid in picture_ids]
            )
        )
    if pill is not None:
        groups[0].evidence.append(pill)
    unresolved = tiers.persist_groups_in_session(session, groups, scan_id)
    return unresolved, [group.signature for group in groups]


def _group_state(server, signature):
    """``(resolved, scan_id, evidence, member ids)`` of one stored group, or None."""

    def read(session):
        row = session.exec(
            select(DedupGroup).where(DedupGroup.signature == signature)
        ).first()
        if row is None:
            return None
        return (
            row.resolved,
            row.scan_id,
            json.loads(row.evidence),
            _stored_members(session, row.id),
        )

    return _run(server, read)


def test_a_stored_group_inside_a_decision_keeps_being_refreshed(server):
    """A decision holds back the groups it would create, not the rows that exist.

    The part's row was stored before the decision covering it was made. A
    rescan that finds the part again must refresh it (members, evidence and
    scan id) like any other row, or it keeps the members and evidence of the
    scan that created it and retirement reads it as an older scan's group.
    """
    ids = _seed(server, [{"pixel_sha": "hhh", "size_bytes": 100}] * 4)
    _unresolved, [whole, part] = _run(server, _persist_exact, [ids, ids[:2]])
    _run(server, verdicts.apply_keep_separate_in_session, whole, None)

    pill = {"text": "Refreshed by scan 7", "against": False}
    unresolved, [again] = _run(server, _persist_exact, [ids[2:]], 7, pill)

    # Same content keys, so the same signature, with today's member ids.
    assert again == part
    assert unresolved == 1
    resolved, scan_id, evidence, members = _group_state(server, part)
    assert (resolved, scan_id, members) == (False, 7, ids[2:])
    assert evidence[-1] == pill
    assert _stored_groups(server)[whole] == (True, ids)


def test_a_batch_that_refreshes_a_decision_does_not_delete_it(server):
    """A decided group found again under its own signature survives its batch.

    Another group of the same tier in the same batch adds a picture to it,
    which on its own would supersede the decision. The batch also refreshes
    the decision itself, so it must stay. No shipped scan builds such a batch:
    every batch is one tier's groups, and a tier's groups never share a
    picture (exact groups are one content key each, near and embedding groups
    are the components of one forest). The rule is pinned through a direct
    persist call so a change to either side cannot delete a decision silently.
    """
    ids = _seed(server, [{"pixel_sha": "iii", "size_bytes": 100}] * 3)
    _unresolved, [decided] = _run(server, _persist_exact, [ids[:2]])
    _run(server, verdicts.apply_keep_separate_in_session, decided, None)

    unresolved, [same, grown] = _run(server, _persist_exact, [ids[:2], ids], 9)

    assert same == decided
    assert unresolved == 1
    assert _group_state(server, decided)[0:2] == (True, 9)
    assert _stored_groups(server) == {
        decided: (True, ids[:2]),
        grown: (False, ids),
    }


def test_a_decks_depth_in_the_queue_follows_a_scrapheaped_stack_member(server):
    """A deleted member must not leave a hole in the deck's depth.

    The deck's depth is the STACK's live member count, so scrapheaping a member
    the group never even names still has to shrink it, otherwise the row
    promises to move a picture that is in the Scrapheap.
    """
    ids = _seed(
        server,
        [
            {"pixel_sha": "stack-leader", "size_bytes": 10},
            {"pixel_sha": "stack-b", "size_bytes": 11},
            {"pixel_sha": "shared", "size_bytes": 100},
            {"pixel_sha": "shared", "size_bytes": 100},
        ],
    )
    stack_id = _stack(server, ids[:3])
    group = _only_group(server)
    assert group["stacks"][str(stack_id)]["member_count"] == 3

    # A sibling the group never names: it is not a candidate, but it IS depth.
    _soft_delete(server, ids[1])
    page, _total, _cursor = _run(server, tiers.page_queue_in_session)
    assert len(page) == 1
    deck = page[0]["stacks"][str(stack_id)]
    assert deck["member_count"] == 2
    assert deck["leader_picture_id"] == ids[0]
    assert deck["matched_picture_ids"] == [ids[2]]


def test_a_deck_whose_leader_is_scrapheaped_reports_the_next_live_leader(server):
    """The face has to move: a deck drawn from a scrapheaped leader is the
    404-thumbnail the queue's empty placeholder was made of."""
    ids = _seed(
        server,
        [
            {"pixel_sha": "stack-leader", "size_bytes": 10},
            {"pixel_sha": "stack-b", "size_bytes": 11},
            {"pixel_sha": "shared", "size_bytes": 100},
            {"pixel_sha": "shared", "size_bytes": 100},
        ],
    )
    stack_id = _stack(server, ids[:3])
    assert _only_group(server)["stacks"][str(stack_id)]["leader_picture_id"] == ids[0]

    _soft_delete(server, ids[0])
    page, _total, _cursor = _run(server, tiers.page_queue_in_session)
    deck = page[0]["stacks"][str(stack_id)]
    assert deck["member_count"] == 2
    assert deck["leader_picture_id"] == ids[1]


def test_deck_facts_are_read_by_stack_id_and_leave_the_scrapheap_out(
    server, key_list_plan, monkeypatch
):
    """A page touching two stacks reads their members through
    ``ix_picture_stack_id``, never by walking every kept picture, and the
    scrapheaped members still count for nothing: not depth, not the face."""
    ids = _seed(
        server,
        [
            # Stack A: a scrapheaped leader, a live sibling, the duplicate.
            {"pixel_sha": "a-leader", "deleted": True},
            {"pixel_sha": "a-b"},
            {"pixel_sha": "shared", "size_bytes": 100},
            # Stack B: a live leader, the duplicate, a scrapheaped sibling.
            {"pixel_sha": "b-leader"},
            {"pixel_sha": "shared", "size_bytes": 100},
            {"pixel_sha": "b-gone", "deleted": True},
            # Stack C: nothing left outside the Scrapheap.
            {"pixel_sha": "c-a", "deleted": True},
            {"pixel_sha": "c-b", "deleted": True},
        ],
    )
    stack_a = _stack(server, ids[0:3])
    stack_b = _stack(server, ids[3:6])
    stack_c = _stack(server, ids[6:8])
    _run(server, tiers.run_scan_now_in_session, TierPolicy(), None)

    engine = server.vault.db._engine
    marker = "FROM picture \nWHERE picture.stack_id IN (?"
    with key_list_plan(engine, marker, "ix_picture_stack_id") as plans:
        page, _total, _cursor = _run(server, tiers.page_queue_in_session)
    assert len(plans) == 1, plans

    assert len(page) == 1, page
    decks = page[0]["stacks"]
    assert decks[str(stack_a)]["member_count"] == 2
    assert decks[str(stack_a)]["leader_picture_id"] == ids[1]
    assert decks[str(stack_b)]["member_count"] == 2
    assert decks[str(stack_b)]["leader_picture_id"] == ids[3]

    for chunk in (None, 1):
        if chunk is not None:
            monkeypatch.setattr(tiers, "ID_CHUNK", chunk)
        facts = _run(server, tiers.load_stack_facts, [stack_c, stack_b, None, stack_a])
        assert sorted(facts) == sorted([stack_a, stack_b])
        assert facts[stack_a].member_ids == (ids[1], ids[2])
        assert facts[stack_b].member_ids == (ids[3], ids[4])
    assert _run(server, tiers.load_stack_facts, []) == {}


def test_scoped_counts_are_reported_per_scope(server):
    ids = _seed(
        server,
        [
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "ccc", "size_bytes": 100},
            {"pixel_sha": "ccc", "size_bytes": 100},
        ],
    )
    _scan(server)

    def add_set(session):
        picture_set = PictureSet(name="Scope")
        session.add(picture_set)
        session.commit()
        session.refresh(picture_set)
        for picture_id in ids[:2]:
            session.add(
                PictureSetMember(set_id=int(picture_set.id), picture_id=picture_id)
            )
        session.commit()
        return int(picture_set.id)

    set_id = _run(server, add_set)
    scopes = [
        DedupScope(),
        DedupScope(scope_type=ScopeType.SET, scope_id=str(set_id)),
    ]
    counts = _run(server, tiers.scope_counts_in_session, scopes, None)
    assert counts[0]["key"] == "global"
    assert counts[0]["unresolved_groups"] == 2
    assert counts[1]["key"] == f"set:{set_id}"
    assert counts[1]["unresolved_groups"] == 1
    # Each scope carries its own tier split, and its exact figure is what an
    # Auto-stack run started in that scope would plan: the scoped queue's
    # button offers it, so it must not be the vault's.
    assert counts[0]["by_tier"]["exact"] == 2
    assert counts[1]["by_tier"] == {"exact": 1, "near": 0, "embedding": 0}
    for row, scope in zip(counts, scopes):
        planned = _run(server, verdicts.plan_auto_stack_in_session, scope)
        assert row["by_tier"]["exact"] == len(planned)


def test_a_scope_id_is_required_for_a_non_global_scope():
    with pytest.raises(ValueError, match="scope_id is required"):
        DedupScope(scope_type=ScopeType.PROJECT)
    assert DedupScope(scope_type=ScopeType.GLOBAL, scope_id="ignored").key == "global"


# (tier, stored confidence, resolved, how the members stand, first member in the
# set). Written directly rather than scanned so a row can hold a confidence no
# scan writes: an exact group below the threshold is the one row on which
# "exact is counted in full" and "the threshold applies" disagree.
_COUNTED_GROUPS = [
    (DedupTier.EXACT, 1.0, False, "live", True),
    (DedupTier.EXACT, 0.8, False, "live", True),
    (DedupTier.NEAR, 63 / 64, False, "live", False),
    (DedupTier.NEAR, 0.875, False, "live", True),
    (DedupTier.EMBEDDING, 0.97, False, "live", True),
    (DedupTier.EMBEDDING, 0.7, False, "live", False),
    (DedupTier.EXACT, 1.0, True, "live", True),
    (DedupTier.NEAR, 0.99, False, "one-scrapheaped", True),
    (DedupTier.EMBEDDING, 0.99, False, "stacked-together", True),
]

_COUNT_POLICIES = [
    pytest.param(TierPolicy(), id="exact-only"),
    pytest.param(TierPolicy(near_enabled=True), id="near"),
    pytest.param(
        TierPolicy(near_enabled=True, embedding_enabled=True), id="near-and-embedding"
    ),
    pytest.param(TierPolicy(threshold=tiers.MIN_THRESHOLD), id="exact-only-lowest"),
    pytest.param(
        TierPolicy(near_enabled=True, embedding_enabled=True, threshold=0.95),
        id="near-and-embedding-at-0.95",
    ),
    pytest.param(TierPolicy(threshold=tiers.MAX_THRESHOLD), id="exact-only-highest"),
    pytest.param(
        TierPolicy(
            near_enabled=True, embedding_enabled=True, threshold=tiers.MIN_THRESHOLD
        ),
        id="near-and-embedding-lowest",
    ),
    pytest.param(
        TierPolicy(near_enabled=True, threshold=tiers.MAX_THRESHOLD),
        id="near-highest",
    ),
    # A threshold equal to a stored confidence: the group on it must count.
    pytest.param(TierPolicy(threshold=0.8), id="exact-only-on-a-stored-confidence"),
    pytest.param(
        TierPolicy(near_enabled=True, embedding_enabled=True, threshold=0.875),
        id="near-and-embedding-on-a-stored-confidence",
    ),
]

# Only the live-groups test groups member rows by group and filters the groups
# with HAVING, so this text marks a statement that evaluates it.
_LIVE_GROUPS_SQL = "GROUP BY dedupgroupmember.group_id HAVING"


@pytest.fixture(scope="module")
def counted_library():
    """One library holding every kind of group the duplicate counts tell apart.

    Yields ``(server, set_id)``; the set holds the first member of the groups
    marked for it in :data:`_COUNTED_GROUPS`. Read-only for its tests.
    """
    temp_dir = tempfile.TemporaryDirectory()
    config_path = os.path.join(temp_dir.name, "server-config.json")
    with open(config_path, "w") as fh:
        fh.write(json.dumps({"port": 8000, "disable_background_workers": True}))
    Server.DEFAULT_FORCE_CPU = True
    srv = Server(config_path)
    try:
        specs = []
        for index, (_tier, _confidence, _resolved, standing, _in_set) in enumerate(
            _COUNTED_GROUPS
        ):
            specs.append({"file_path": f"/vault/counted_{index}_a.png"})
            specs.append(
                {
                    "file_path": f"/vault/counted_{index}_b.png",
                    "deleted": standing == "one-scrapheaped",
                }
            )
        ids = _seed(srv, specs)
        for index, group_spec in enumerate(_COUNTED_GROUPS):
            if group_spec[3] == "stacked-together":
                _stack(srv, ids[2 * index : 2 * index + 2])

        def write_groups(session):
            picture_set = PictureSet(name="Counted")
            session.add(picture_set)
            session.flush()
            for index, (tier, confidence, resolved, _standing, in_set) in enumerate(
                _COUNTED_GROUPS
            ):
                members = ids[2 * index : 2 * index + 2]
                group = DedupGroup(
                    signature=f"test-counted-{index}",
                    tier=tier.value,
                    confidence=confidence,
                    member_count=len(members),
                    resolved=resolved,
                )
                session.add(group)
                session.flush()
                for position, picture_id in enumerate(members):
                    session.add(
                        DedupGroupMember(
                            group_id=int(group.id),
                            picture_id=picture_id,
                            position=position,
                        )
                    )
                if in_set:
                    session.add(
                        PictureSetMember(
                            set_id=int(picture_set.id), picture_id=members[0]
                        )
                    )
            session.commit()
            return int(picture_set.id)

        yield srv, _run(srv, write_groups)
    finally:
        srv.close()
        temp_dir.cleanup()
        gc.collect()


def _counted_scope(set_id, scoped):
    if not scoped:
        return None
    return DedupScope(scope_type=ScopeType.SET, scope_id=str(set_id))


def test_the_badge_honours_the_tier_switches_and_the_split_does_not(
    counted_library,
):
    """Pins what the counted library holds, so the comparisons below compare
    real numbers. Exact only: the badge is the one exact group at or above the
    threshold; the split also shows the exact row below it and the near and
    embedding groups that pass. Resolved, single-survivor and stacked-together
    groups count nowhere."""
    server, set_id = counted_library
    assert _run(
        server, tiers.count_unresolved_and_by_tier_in_session, TierPolicy(), None
    ) == (1, {"exact": 2, "near": 1, "embedding": 1})
    assert _run(
        server,
        tiers.count_unresolved_and_by_tier_in_session,
        TierPolicy(),
        _counted_scope(set_id, True),
    ) == (1, {"exact": 2, "near": 0, "embedding": 1})


@pytest.mark.parametrize("scoped", [False, True], ids=["global", "set"])
@pytest.mark.parametrize("policy", _COUNT_POLICIES)
def test_one_read_gives_the_badge_and_the_split_the_separate_counts_give(
    counted_library, policy, scoped
):
    server, set_id = counted_library
    scope = _counted_scope(set_id, scoped)
    separate = (
        _run(server, tiers.count_unresolved_in_session, policy, scope),
        _run(server, tiers.count_by_tier_in_session, policy, scope),
    )
    combined = _run(
        server, tiers.count_unresolved_and_by_tier_in_session, policy, scope
    )
    assert combined == separate


@pytest.mark.parametrize(
    "policy",
    [
        pytest.param(TierPolicy(), id="exact-only"),
        pytest.param(
            TierPolicy(near_enabled=True, embedding_enabled=True, threshold=0.95),
            id="near-and-embedding-at-0.95",
        ),
    ],
)
def test_the_counts_response_tests_live_groups_once_for_the_badge_and_the_split(
    counted_library, policy
):
    """The counts are re-read after every verdict, and the live-groups test
    (members, stack units, locks) is their expensive part: the badge and the
    tier split share one evaluation of it."""
    server, _set_id = counted_library
    statements = []

    def record(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(" ".join(statement.split()))

    engine = server.vault.db._engine
    event.listen(engine, "before_cursor_execute", record)
    try:
        body = tiers.counts_response(server.vault, policy, [])
    finally:
        event.remove(engine, "before_cursor_execute", record)

    live_group_reads = [s for s in statements if _LIVE_GROUPS_SQL in s]
    assert len(live_group_reads) == 1, live_group_reads
    assert body["unresolved_groups"] == _run(
        server, tiers.count_unresolved_in_session, policy, None
    )
    assert body["by_tier"] == _run(server, tiers.count_by_tier_in_session, policy, None)


# ── scan progress ─────────────────────────────────────────────────────────────


def test_an_equivalent_active_scan_request_is_coalesced_without_restarting(server):
    first = _run(server, tiers.request_scan_in_session, TierPolicy(), None)
    assert first["status"] == "pending"
    assert first["scope_key"] == "global"
    second = _run(server, tiers.request_scan_in_session, TierPolicy(), None)
    assert second["scan_id"] == first["scan_id"]
    assert second["started_at"] == first["started_at"]
    assert second["updated_at"] == first["updated_at"]


def test_a_different_policy_cannot_overwrite_an_active_scan(server):
    first = _run(server, tiers.request_scan_in_session, TierPolicy(), None)
    with pytest.raises(tiers.DedupScanBusyError) as raised:
        _run(
            server,
            tiers.request_scan_in_session,
            TierPolicy(near_enabled=True),
            None,
        )
    assert raised.value.active_scan["scan_id"] == first["scan_id"]
    assert raised.value.active_scan["tiers"] == ["exact"]
    persisted = _run(server, tiers.scan_progress_in_session, None)
    assert persisted["tiers"] == ["exact"]
    assert persisted["status"] == "pending"


def test_scan_slices_yield_the_writer_between_near_buckets(server, monkeypatch):
    """An interactive callback queued after bucket 1 runs before bucket 2."""
    _seed(
        server,
        [
            {"perceptual_hash": PHASH_ZERO},
            {"perceptual_hash": PHASH_ONE_BIT},
        ],
    )
    progress = _run(
        server,
        tiers.request_scan_in_session,
        TierPolicy(near_enabled=True),
        None,
    )
    task = DedupScanTask(server.vault.db, progress["scan_id"])
    order: list[str] = []
    queued_interactive = threading.Event()
    original_bucket_slice = DedupScanTask._run_near_bucket_slice

    def marked_bucket_slice(session, *args):
        order.append(f"bucket-{args[3]}")
        return original_bucket_slice(session, *args)

    monkeypatch.setattr(
        DedupScanTask, "_run_near_bucket_slice", staticmethod(marked_bucket_slice)
    )
    original_run_low_slice = task._run_low_slice

    def run_low_slice(func, *args):
        result = original_run_low_slice(func, *args)
        if func is marked_bucket_slice and not queued_interactive.is_set():
            queued_interactive.set()
            server.vault.db.submit_task(
                lambda _session: order.append("interactive"),
                priority=DBPriority.IMMEDIATE,
            )
        return result

    monkeypatch.setattr(task, "_run_low_slice", run_low_slice)
    task._run_task()
    bucket_positions = [i for i, item in enumerate(order) if item.startswith("bucket-")]
    assert len(bucket_positions) >= 2, order
    assert bucket_positions[0] < order.index("interactive") < bucket_positions[1]


def test_sliced_scan_preserves_cross_bucket_chaining_and_restart(server):
    ids = _seed(
        server,
        [
            {
                "perceptual_hash": PHASH_ZERO,
                "width": 100,
                "height": 100,
                "file_path": "/a/one.png",
            },
            {
                "perceptual_hash": PHASH_ONE_BIT,
                "width": 100,
                "height": 100,
                "file_path": "/b/two.png",
            },
            {
                "perceptual_hash": PHASH_FOUR_BITS,
                "width": 200,
                "height": 200,
                "file_path": "/b/three.png",
            },
        ],
    )
    progress = _run(
        server,
        tiers.request_scan_in_session,
        TierPolicy(near_enabled=True),
        None,
    )

    def mark_interrupted(session, scan_id):
        scan = session.get(DedupScan, scan_id)
        scan.status = SCAN_RUNNING
        session.add(scan)
        session.commit()

    _run(server, mark_interrupted, progress["scan_id"])
    _run(
        server,
        DedupScanTask._mark_pending_after_cancel,
        progress["scan_id"],
    )
    pending = _run(server, DedupScanTask.find_pending_scan)
    assert pending.id == progress["scan_id"]
    assert pending.status == SCAN_PENDING
    resumed_task = DedupScanFinder(server.vault.db).find_task()
    assert resumed_task is not None
    assert resumed_task.params["scan_id"] == progress["scan_id"]
    DedupScanTask(server.vault.db, progress["scan_id"])._run_task()
    rows = _run(server, lambda session: session.exec(select(DedupGroup)).all())
    matching = [row for row in rows if row.tier == "near" and row.member_count == 3]
    assert len(matching) == 1
    members = _run(
        server,
        lambda session: session.exec(
            select(tiers.DedupGroupMember.picture_id).where(
                tiers.DedupGroupMember.group_id == matching[0].id
            )
        ).all(),
    )
    assert sorted(members) == sorted(ids)


def test_task_runner_shutdown_stops_scan_after_current_slice():
    """Shutdown lets one active callback finish but submits no later slice."""
    slice_started = threading.Event()
    release_slice = threading.Event()
    calls: list[str] = []

    class FakeDatabase:
        def run_task(self, func, *args, priority):
            calls.append(func.__name__)
            if func.__name__ == "_start_scan_slice":
                slice_started.set()
                assert release_slice.wait(timeout=2)
                return {
                    "policy": TierPolicy(near_enabled=True).as_dict(),
                    "scope_type": "global",
                    "scope_id": None,
                    "total_pictures": 100,
                }
            if func.__name__ == "_mark_pending_after_cancel":
                assert priority == DBPriority.IMMEDIATE
                return None
            raise AssertionError(f"shutdown submitted later slice {func.__name__}")

    task = DedupScanTask(FakeDatabase(), scan_id=7)
    runner = TaskRunner(name="dedup-shutdown-test", num_workers=1)
    shutdown = threading.Thread(target=runner.stop)
    runner.start()
    runner.submit(task)
    try:
        assert slice_started.wait(timeout=1)

        shutdown.start()
        # stop() must wait for the callback that currently owns the DB session.
        shutdown.join(timeout=0.05)
        assert shutdown.is_alive()
    finally:
        # Inside the try, so an assertion that fires still releases the paused
        # slice instead of leaving a runner worker parked on
        # release_slice.wait(2) and stop() running into the next test.
        release_slice.set()
        if shutdown.ident is not None:
            shutdown.join(timeout=1)
        else:  # the slice never began, so nothing else will stop the runner
            runner.stop()

    assert not shutdown.is_alive()
    assert calls == ["_start_scan_slice", "_mark_pending_after_cancel"]
    assert task.result == {
        "scan_id": 7,
        "status": SCAN_PENDING,
        "cancelled": True,
    }


def test_legacy_exact_plus_embedding_scan_has_defined_phase_progress(monkeypatch):
    """A persisted pre-validator tier combination must not unbind total_phases."""

    class LegacyPolicy:
        def __init__(self, **values):
            self.near_enabled = bool(values.get("near_enabled"))
            self.embedding_enabled = bool(values.get("embedding_enabled"))
            self.threshold = float(values.get("threshold", 0.9))
            self.min_group_size = int(values.get("min_group_size", 2))
            self.max_group_size = int(values.get("max_group_size", 24))

    monkeypatch.setattr(dedup_scan_task_module, "TierPolicy", LegacyPolicy)

    class FakeDatabase:
        def run_task(self, func, *args, priority):
            if func.__name__ == "_start_scan_slice":
                return {
                    "policy": {
                        "near_enabled": False,
                        "embedding_enabled": True,
                        "threshold": 0.9,
                        "min_group_size": 2,
                        "max_group_size": 24,
                    },
                    "scope_type": "global",
                    "scope_id": None,
                    "total_pictures": 1,
                }
            if func.__name__ == "_begin_phase_slice":
                return None
            if func.__name__ == "_embedding_edge_page_slice":
                return {"edges": [], "cursor": (-1, -1), "done": True}
            if func.__name__ == "_finish_scan_slice":
                assert task.task_progress() == (2, 3)
                return {
                    "scan_id": 1,
                    "scope": "global",
                    "total_pictures": 1,
                    "groups_found": 0,
                }
            raise AssertionError(f"unexpected slice {func.__name__}")

        def run_immediate_read_task(self, func, *args):
            if func.__name__ in ("_exact_keys_read", "_obsolete_groups_read"):
                return []
            # The scope's id filter is a whole-scope read: a reader's job.
            if func.__name__ == "_embedding_scope_read":
                return None
            raise AssertionError(f"unexpected read {func.__name__}")

    task = DedupScanTask(FakeDatabase(), scan_id=1)
    assert task._run_task()["scan_id"] == 1
    assert task.task_progress() == (3, 3)


def test_near_scan_with_no_buckets_keeps_finalisation_remaining():
    class FakeDatabase:
        def run_task(self, func, *args, priority):
            if func.__name__ == "_start_scan_slice":
                return {
                    "policy": TierPolicy(near_enabled=True).as_dict(),
                    "scope_type": "global",
                    "scope_id": None,
                    "total_pictures": 0,
                }
            if func.__name__ in ("_begin_phase_slice", "_publish_bucket_total_slice"):
                return None
            if func.__name__ == "_finish_scan_slice":
                assert task.task_progress() == (2, 3)
                return {
                    "scan_id": 1,
                    "scope": "global",
                    "total_pictures": 0,
                    "groups_found": 0,
                }
            raise AssertionError(f"unexpected slice {func.__name__}")

        def run_immediate_read_task(self, func, *args):
            if func.__name__ in ("_exact_keys_read", "_obsolete_groups_read"):
                return []
            # Building the buckets reads the whole scope: a reader's job.
            if func.__name__ == "_near_buckets_read":
                return []
            raise AssertionError(f"unexpected read {func.__name__}")

    task = DedupScanTask(FakeDatabase(), scan_id=1)
    task._run_task()
    assert task.task_progress() == (3, 3)


def test_scan_progress_for_an_unscanned_scope_is_idle_not_an_error(server):
    progress = _run(server, tiers.scan_progress_in_session, None)
    assert progress["status"] == "idle"
    assert progress["total_pictures"] == 0


# ── R1: the group signature must be injective over groups ─────────────────────


def test_two_groups_sharing_a_digest_but_not_a_size_stay_distinct(server):
    """Regression for the CSO's E1: the signature needs the ``size_bytes`` co-key.

    ``pixel_sha`` is a *sampled* digest above 128 KiB, which is exactly why tier 1
    detects on ``(pixel_sha, size_bytes)``. Identity that dropped the size made
    two distinct exact groups collapse onto one signature, and all three
    consequences were silent: one group vanished from the queue via the
    upsert-on-signature, a keep-separate on the survivor resolved both file sets,
    and a stack verdict's write target depended on scan order rather than on what
    the user saw.
    """
    ids = _seed(
        server,
        [
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "aaa", "size_bytes": 999},
            {"pixel_sha": "aaa", "size_bytes": 999},
        ],
    )
    groups = _run(server, tiers.find_exact_groups_in_session, None)
    assert len(groups) == 2
    assert sorted(sorted(g.picture_ids) for g in groups) == [
        sorted(ids[:2]),
        sorted(ids[2:]),
    ]
    # The whole point: two groups, two signatures.
    assert len({g.signature for g in groups}) == 2

    # ...and therefore two persisted rows, not one silently overwriting the other.
    _scan(server)
    rows = _run(
        server,
        lambda session: sorted(
            (row.signature, row.member_count)
            for row in session.exec(select(DedupGroup)).all()
        ),
    )
    assert len(rows) == 2
    assert _run(server, tiers.count_unresolved_in_session, None, None) == 2


def test_a_verdict_on_one_group_does_not_resolve_its_same_digest_twin(server):
    """The second half of E1: consent must not leak across file sets."""
    _seed(
        server,
        [
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "aaa", "size_bytes": 100},
            {"pixel_sha": "aaa", "size_bytes": 999},
            {"pixel_sha": "aaa", "size_bytes": 999},
        ],
    )
    _scan(server)
    page, total, _cursor = _run(server, tiers.page_queue_in_session, None, None, 0, 10)
    assert total == 2
    _run(server, verdicts.apply_keep_separate_in_session, page[0]["signature"], None)
    # The other group is still waiting for its own decision.
    assert _run(server, tiers.count_unresolved_in_session, None, None) == 1
    remaining, _total, _cursor = _run(
        server, tiers.page_queue_in_session, None, None, 0, 10
    )
    assert len(remaining) == 1
    assert remaining[0]["signature"] == page[1]["signature"]

    # And a rescan does not resurrect the decided one or silence the open one.
    _scan(server)
    assert _run(server, tiers.count_unresolved_in_session, None, None) == 1


def test_content_key_carries_the_size_co_key():
    """Unit-level pin on the identity format itself."""
    member = tiers.CandidateMember(id=1, pixel_sha="aaa", size_bytes=100)
    twin = tiers.CandidateMember(id=2, pixel_sha="aaa", size_bytes=999)
    assert member.content_key == "aaa:100"
    assert member.content_key != twin.content_key
    assert tiers.group_signature([member.content_key]) != tiers.group_signature(
        [twin.content_key]
    )
    # An unhashed picture still falls back to its id.
    assert tiers.CandidateMember(id=7).content_key == "id:7"


# ── tier precedence on upsert ────────────────────────────────────────────────


def _group_row(server, signature):
    return _run(
        server,
        lambda session: session.exec(
            select(DedupGroup).where(DedupGroup.signature == signature)
        ).first(),
    )


def test_a_near_scan_does_not_downgrade_an_exact_group(server):
    """QA blocker 2, the two-scan repro.

    A byte-identical pair is also perceptually identical, so every near-enabled
    scan rediscovers it as a ``near`` group with the same signature. The upsert
    wrote ``row.tier`` unconditionally, so the pair was demoted to ``near`` - and
    a ``near`` group is invisible in the exact-only default queue *and*
    ineligible for ``POST /dedup/auto-stack``, which only ever acts on ``exact``.
    """
    _seed(
        server,
        [
            {
                "pixel_sha": "same",
                "size_bytes": 100,
                "perceptual_hash": PHASH_ZERO,
            },
            {
                "pixel_sha": "same",
                "size_bytes": 100,
                "perceptual_hash": PHASH_ZERO,
            },
        ],
    )
    # Scan 1: exact only.
    _run(server, tiers.run_scan_now_in_session, TierPolicy(), None)
    rows = _run(server, lambda session: session.exec(select(DedupGroup)).all())
    assert len(rows) == 1
    signature = rows[0].signature
    assert rows[0].tier == DedupTier.EXACT.value
    exact_confidence = rows[0].confidence

    # Scan 2: the user turns tier 2 on.
    _run(
        server,
        tiers.run_scan_now_in_session,
        TierPolicy(near_enabled=True),
        None,
    )
    row = _group_row(server, signature)
    assert row.tier == DedupTier.EXACT.value, "exact must never be downgraded"
    assert row.confidence == exact_confidence

    # And the pair is still where the exact-only default queue and auto-stack
    # both look for it.
    page, _total, _cursor = _run(
        server, tiers.page_queue_in_session, TierPolicy(), None, 0, 10
    )
    assert [group["tier"] for group in page] == [DedupTier.EXACT.value]


def test_the_upsert_takes_the_stronger_tier_in_either_arrival_order(server):
    """Precedence is a *maximum*, not a freeze on whatever landed first.

    Driven through :func:`persist_groups_in_session` directly, because a real
    scan cannot present the same signature under two tiers in the weak-then-
    strong order: the signature hashes the members' ``<pixel_sha>:<size_bytes>``
    content keys, so anything tier 1 can group tier 2 also sees, and the reverse
    change (two files becoming byte-identical) changes the content keys and
    therefore the signature. The precedence rule still has to hold both ways, or
    it is an ordering accident rather than a rule.
    """
    ids = _seed(
        server,
        [
            {"pixel_sha": "same", "size_bytes": 100},
            {"pixel_sha": "same", "size_bytes": 100},
        ],
    )

    def upsert(session, first, second):
        members = tiers.load_candidates(session, ids)
        ordered = [members[pid] for pid in ids]
        for tier, confidence in (first, second):
            tiers.persist_groups_in_session(
                session, [tiers.assemble_group(tier, confidence, ordered)]
            )
        return session.exec(select(DedupGroup)).all()

    strong = (DedupTier.EXACT, 1.0)
    weak = (DedupTier.NEAR, 0.93)

    rows = _run(server, upsert, strong, weak)
    assert [row.tier for row in rows] == [DedupTier.EXACT.value]
    assert rows[0].confidence == 1.0

    rows = _run(server, upsert, weak, strong)
    assert [row.tier for row in rows] == [DedupTier.EXACT.value]
    assert rows[0].confidence == 1.0


def _stored_members(session, group_id):
    return sorted(
        session.exec(
            select(tiers.DedupGroupMember.picture_id).where(
                tiers.DedupGroupMember.group_id == group_id
            )
        ).all()
    )


def test_the_set_based_upsert_keeps_every_per_group_rule(server):
    """One call inserts, refreshes, resolves and applies tier precedence per group.

    The upsert writes whole sets of rows at once; each group must still end up
    exactly as the one-row-at-a-time upsert left it.
    """
    ids = _seed(
        server,
        [
            {"pixel_sha": "new", "size_bytes": 100},
            {"pixel_sha": "new", "size_bytes": 100},
            {"pixel_sha": "moved", "size_bytes": 100},
            {"pixel_sha": "moved", "size_bytes": 100},
            {"pixel_sha": "moved", "size_bytes": 100},
            {"pixel_sha": "decided", "size_bytes": 100},
            {"pixel_sha": "decided", "size_bytes": 100},
            {"pixel_sha": "strong", "size_bytes": 100},
            {"pixel_sha": "strong", "size_bytes": 100},
            {"pixel_sha": "twice", "size_bytes": 100},
            {"pixel_sha": "twice", "size_bytes": 100},
        ],
    )
    exact, near = (DedupTier.EXACT, 1.0), (DedupTier.NEAR, 0.93)

    def build(session, specs):
        groups = []
        for (tier, confidence), picture_ids in specs:
            members = tiers.load_candidates(session, picture_ids)
            groups.append(
                tiers.assemble_group(
                    tier, confidence, [members[pid] for pid in picture_ids]
                )
            )
        return groups

    def prior(session):
        groups = build(session, [(exact, ids[2:4]), (exact, ids[7:9])])
        tiers.persist_groups_in_session(session, groups)
        return {
            group.signature: session.exec(
                select(DedupGroup.id).where(DedupGroup.signature == group.signature)
            ).one()
            for group in groups
        }

    prior_ids = _run(server, prior)

    def decide(session):
        signature = build(session, [(exact, ids[5:7])])[0].signature
        session.add(
            DedupVerdict(
                signature=signature,
                verdict=VERDICT_KEEP_SEPARATE,
                picture_ids="[]",
                excluded_picture_ids="[]",
            )
        )
        session.commit()

    _run(server, decide)

    def upsert(session, specs):
        groups = build(session, specs)
        unresolved = tiers.persist_groups_in_session(session, groups, 42)
        return unresolved, [group.signature for group in groups]

    unresolved, (new, moved, decided, strong) = _run(
        server,
        upsert,
        [
            (exact, ids[0:2]),
            # Same content keys as the stored group, so the same signature: the
            # row is refreshed in place with today's member ids.
            (exact, [ids[2], ids[4]]),
            (exact, ids[5:7]),
            (near, ids[7:9]),
        ],
    )
    assert unresolved == 3
    # One signature twice in one call: the later, weaker occurrence must not
    # undo the earlier, stronger one.
    _unresolved, (twice, _twice_again) = _run(
        server, upsert, [(exact, ids[9:11]), (near, ids[9:11])]
    )

    def stored(session):
        rows = {row.signature: row for row in session.exec(select(DedupGroup)).all()}
        return rows, {
            signature: _stored_members(session, row.id)
            for signature, row in rows.items()
        }

    rows, members = _run(server, stored)
    assert len(rows) == 5
    assert rows[decided].resolved is True
    assert not any(rows[sig].resolved for sig in (new, moved, strong, twice))
    assert all(row.scan_id == 42 for row in rows.values())

    assert rows[new].created_at is not None
    assert rows[new].member_count == 2
    assert members[new] == ids[0:2]

    assert rows[moved].id == prior_ids[moved]
    assert rows[moved].member_count == 2
    assert members[moved] == [ids[2], ids[4]]

    assert rows[strong].id == prior_ids[strong]
    assert (rows[strong].tier, rows[strong].confidence) == ("exact", 1.0)
    assert (rows[twice].tier, rows[twice].confidence) == ("exact", 1.0)
    assert members[twice] == ids[9:11]


def test_prune_keeps_exactly_the_groups_with_two_live_members(server):
    """A group survives a prune only while two of its members are live.

    Member rows of a pruned group go with it through their cascading key.
    """
    ids = _seed(
        server,
        [
            {"pixel_sha": "a", "size_bytes": 100},
            {"pixel_sha": "b", "size_bytes": 100, "deleted": True},
            {"pixel_sha": "c", "size_bytes": 100},
            {"pixel_sha": "d", "size_bytes": 100},
            {"pixel_sha": "e", "size_bytes": 100},
            {"pixel_sha": "f", "size_bytes": 100},
            {"pixel_sha": "g", "size_bytes": 100, "deleted": True},
        ],
    )
    layout = {
        "no-members": [],
        "one-live": [ids[0], ids[1]],
        "two-live": [ids[2], ids[3]],
        "two-of-three-live": [ids[4], ids[5], ids[6]],
    }

    def seed_groups(session):
        for signature, picture_ids in layout.items():
            group = DedupGroup(
                signature=signature,
                tier="exact",
                confidence=1.0,
                member_count=len(picture_ids),
            )
            session.add(group)
            session.flush()
            for position, picture_id in enumerate(picture_ids):
                session.add(
                    tiers.DedupGroupMember(
                        group_id=int(group.id),
                        picture_id=picture_id,
                        position=position,
                    )
                )
        session.commit()

    _run(server, seed_groups)
    assert _run(server, tiers.prune_stale_groups_in_session) == 2

    def survivors(session):
        rows = session.exec(select(DedupGroup)).all()
        member_group_ids = set(
            session.exec(select(tiers.DedupGroupMember.group_id)).all()
        )
        return (
            {row.signature: _stored_members(session, row.id) for row in rows},
            member_group_ids == {row.id for row in rows},
        )

    remaining, members_match_groups = _run(server, survivors)
    assert remaining == {
        "two-live": [ids[2], ids[3]],
        "two-of-three-live": [ids[4], ids[5], ids[6]],
    }
    assert members_match_groups, "a pruned group left member rows behind"
    assert _run(server, tiers.prune_stale_groups_in_session) == 0


# ── keyset cursor ────────────────────────────────────────────────────────────


def test_a_cursor_round_trips_its_position_exactly():
    """Float equality drives the tie-break branch, so the encoding must be exact."""
    for confidence in (1.0, 0.9, 0.65, 0.123456789012345, 0.0):
        cursor = tiers.encode_queue_cursor(confidence, 4242)
        assert tiers.decode_queue_cursor(cursor) == (confidence, 4242)


def test_a_tampered_cursor_is_refused_rather_than_reinterpreted():
    for bad in ("", "AAAA", "not base64!", tiers.encode_queue_cursor(1.0, 1)[:-4]):
        with pytest.raises(tiers.DedupCursorError):
            tiers.decode_queue_cursor(bad)
    # A cursor from a future encoding version is refused, not misread.
    import base64

    forged = base64.urlsafe_b64encode(b"9|1.0|1").decode().rstrip("=")
    with pytest.raises(tiers.DedupCursorError):
        tiers.decode_queue_cursor(forged)


def test_a_decided_cursor_minted_before_the_upgrade_reads_as_utc():
    """An open Decided page may hold a cursor with a naive stamp; it must stay usable.

    A naive value would reach the UTCDateTime bind and raise, a 500 rather than a page.
    """
    import base64

    raw = f"{tiers.CURSOR_VERSION}|{tiers.DECIDED_CURSOR_KIND}|2026-07-30T12:28:53.123456|7"
    legacy = base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")
    stamp = datetime(2026, 7, 30, 12, 28, 53, 123456, tzinfo=timezone.utc)
    assert tiers.decode_decided_cursor(legacy) == (stamp, 7)
    fresh = tiers.encode_decided_cursor(stamp, 7)
    assert tiers.decode_decided_cursor(fresh) == (stamp, 7)
    assert tiers.decode_decided_cursor(fresh)[0].tzinfo is not None


def test_the_cursor_resumes_a_tied_confidence_run_without_gap_or_repeat(server):
    """Every exact group ties at the same confidence; the id breaks the tie."""
    for index in range(5):
        _seed(
            server,
            [
                {"pixel_sha": f"tie-{index}", "size_bytes": 100 + index},
                {"pixel_sha": f"tie-{index}", "size_bytes": 100 + index},
            ],
        )
    _run(server, tiers.run_scan_now_in_session, TierPolicy(), None)

    walked = []
    cursor = None
    for _ in range(10):
        page, total, cursor = _run(
            server, tiers.page_queue_in_session, None, None, 0, 2, cursor
        )
        assert total == 5
        walked.extend(group["signature"] for group in page)
        if cursor is None:
            break
    assert len(walked) == 5, walked
    assert len(set(walked)) == 5, "a tie must not be delivered twice"


# ── folder scope normalisation ───────────────────────────────────────────────


def test_a_folder_scope_that_normalises_to_empty_is_refused():
    """CSO W2: these all rstripped to "" and became a LIKE pattern of "%"."""
    for bad in ("/", "\\", "///", "\\\\", "/\\/"):
        with pytest.raises(ValueError):
            DedupScope(scope_type=ScopeType.FOLDER, scope_id=bad)


def test_a_folder_scope_is_normalised_to_one_scope_key():
    plain = DedupScope(scope_type=ScopeType.FOLDER, scope_id="/photos/2026")
    trailing = DedupScope(scope_type=ScopeType.FOLDER, scope_id="/photos/2026/")
    assert plain.key == trailing.key == "folder:/photos/2026"
    assert trailing.scope_id == "/photos/2026"


def test_a_folder_scope_is_its_folder_and_below_never_a_prefix_sibling(server):
    """A folder scope holds the folder and its sub-folders, nothing else.

    It used to be ``file_path LIKE '<folder>%'``: no separator, so
    ``/lib/photos`` took ``/lib/photos-old`` and ``/lib/photosX``, and SQLite's
    LIKE folds ASCII case, so on Linux and macOS it also took ``/lib/PHOTOS``,
    a different folder there. Both the predicate and the per-row probe are
    checked, through the exact tier the scan runs.
    """
    same = {"pixel_sha": "aaa", "size_bytes": 100}
    paths = [
        "/lib/photos/a.png",
        "/lib/photos/2020/b.png",
        "/lib/photos-old/c.png",
        "/lib/photosX/d.png",
        "/lib/photos0/e.png",
        "/lib/photos.png",
        "/lib/PHOTOS/f.png",
    ]
    ids = _seed(server, [{**same, "file_path": path} for path in paths])
    # Imported from the folder: in it by the import arm, wherever it is stored.
    (imported,) = _seed(
        server, [{**same, "file_path": "g.png", "import_source_folder": "/lib/photos"}]
    )
    expected = {ids[0], ids[1], imported}
    if PATHS_FOLD_CASE:
        expected.add(ids[6])

    scope = DedupScope(ScopeType.FOLDER, "/lib/photos/")

    def selected(session, clause):
        return set(session.exec(select(Picture.id).where(clause)).all())

    assert _run(server, selected, scope.picture_predicate()) == expected
    assert _run(server, selected, scope.picture_probe()) == expected
    groups = _run(server, tiers.find_exact_groups_in_session, scope)
    assert [set(group.picture_ids) for group in groups] == [expected]


def test_a_folder_scope_takes_like_wildcards_literally(server):
    """``%`` and ``_`` in a folder name are the characters, not wildcards."""
    ids = _seed(
        server,
        [
            {"file_path": "/lib/pho_os/a.png"},
            {"file_path": "/lib/photos/b.png"},
            {"file_path": "/lib/pho%os/c.png"},
            {"file_path": "/lib/phXtos/d.png"},
        ],
    )

    def selected(session, clause):
        return set(session.exec(select(Picture.id).where(clause)).all())

    underscore = DedupScope(ScopeType.FOLDER, "/lib/pho_os")
    percent = DedupScope(ScopeType.FOLDER, "/lib/pho%os")
    assert _run(server, selected, underscore.picture_predicate()) == {ids[0]}
    assert _run(server, selected, percent.picture_predicate()) == {ids[2]}


# ── stack units: the deck's depth is the STACK's, not the group's ─────────────


def _stack(server, picture_ids, thumbnails=None):
    """Put *picture_ids* in one stack, in order, and return the stack id.

    ``picture_ids[0]`` becomes the leader (``stack_position`` 0). *thumbnails*
    optionally maps a picture id to a ``(width, height)`` pair so the leader's
    cache-buster token is a real value rather than the unprocessed ``"0"``.
    """

    def build(session):
        stack = PictureStack(name=None)
        session.add(stack)
        session.commit()
        session.refresh(stack)
        for position, picture_id in enumerate(picture_ids):
            picture = session.get(Picture, int(picture_id))
            picture.stack_id = int(stack.id)
            picture.stack_position = position
            size = (thumbnails or {}).get(int(picture_id))
            if size is not None:
                picture.thumbnail_width, picture.thumbnail_height = size
            session.add(picture)
        session.commit()
        return int(stack.id)

    return _run(server, build)


def _lock_set(server, name, picture_ids):
    """Create a LOCKED picture set containing *picture_ids*; return its id."""

    def build(session):
        picture_set = PictureSet(name=name, locked=True)
        session.add(picture_set)
        session.commit()
        session.refresh(picture_set)
        for picture_id in picture_ids:
            session.add(
                PictureSetMember(set_id=int(picture_set.id), picture_id=int(picture_id))
            )
        session.commit()
        return int(picture_set.id)

    return _run(server, build)


def _only_group(server):
    """Scan, page the queue, and return the single group it serves."""
    _run(server, tiers.run_scan_now_in_session, TierPolicy(), None)
    page, _total, _cursor = _run(server, tiers.page_queue_in_session)
    assert len(page) == 1, page
    return page[0]


def test_a_group_naming_one_member_of_a_four_stack_reports_the_stacks_depth(server):
    """The measured majority case: 36 of 116 stack-touching groups name ONE
    member of a stack. The deck must still stand for the whole stack, or the
    row draws a 4-deep stack as one picture and then silently moves four."""
    ids = _seed(
        server,
        [
            # The stack: leader, two siblings the group never names, and the
            # member that is a byte-identical duplicate of the loose picture.
            {"pixel_sha": "stack-leader", "size_bytes": 10},
            {"pixel_sha": "stack-b", "size_bytes": 11},
            {"pixel_sha": "stack-c", "size_bytes": 12},
            {"pixel_sha": "shared", "size_bytes": 100},
            # The loose picture it duplicates.
            {"pixel_sha": "shared", "size_bytes": 100},
        ],
    )
    stack_id = _stack(server, ids[:4], thumbnails={ids[0]: (1024, 768)})

    group = _only_group(server)
    assert sorted(c["picture_id"] for c in group["candidates"]) == sorted(ids[3:])

    stacks = group["stacks"]
    assert list(stacks) == [str(stack_id)], "keyed by stack id, as a string"
    deck = stacks[str(stack_id)]
    assert deck["stack_id"] == stack_id
    # The stack's REAL depth, not the one member of it that is in the group.
    assert deck["member_count"] == 4
    assert deck["matched_picture_ids"] == [ids[3]]
    # The face is the stack's leader, which is NOT the matched member.
    assert deck["leader_picture_id"] == ids[0]
    assert deck["leader_picture_id"] not in deck["matched_picture_ids"]
    # The size, then the file: a reused picture id gets a URL of its own.
    assert deck["leader_thumbnail_version"].startswith("1024x768-")
    assert deck["stackable"] is True
    assert deck["blocked_by_sets"] == []
    # Eager count and leader, LAZY members: the members are never inlined.
    assert "members" not in deck
    assert "member_ids" not in deck


def test_a_group_naming_a_whole_two_stack_reports_two_and_two(server):
    """When the group does name every member, depth and matched agree."""
    ids = _seed(
        server,
        [
            {"pixel_sha": "shared", "size_bytes": 100},
            {"pixel_sha": "shared", "size_bytes": 100},
            {"pixel_sha": "shared", "size_bytes": 100},
        ],
    )
    stack_id = _stack(server, ids[:2])

    group = _only_group(server)
    deck = group["stacks"][str(stack_id)]
    assert deck["member_count"] == 2
    assert deck["matched_picture_ids"] == sorted(ids[:2])
    assert deck["leader_picture_id"] == ids[0]
    # The loose third picture is its own unit and contributes no stacks entry.
    assert len(group["stacks"]) == 1


def test_a_deck_with_one_locked_member_is_unstackable_at_unit_level(server):
    """A stack cannot be partially stacked, so ONE frozen member freezes the
    deck: including a member the group never names, because a locked set
    freezes a whole stack.

    Three units keep the group in the queue at all: a frozen deck plus two loose
    pictures still leaves two stackable units, which is what
    ``_live_groups_filter`` requires.
    """
    ids = _seed(
        server,
        [
            {"pixel_sha": "stack-leader", "size_bytes": 10},
            {"pixel_sha": "shared", "size_bytes": 100},
            {"pixel_sha": "shared", "size_bytes": 100},
            {"pixel_sha": "shared", "size_bytes": 100},
        ],
    )
    stack_id = _stack(server, ids[:2])
    # The LEADER is locked, and the leader is not in the group at all.
    set_id = _lock_set(server, "Frozen", [ids[0]])

    group = _only_group(server)
    deck = group["stacks"][str(stack_id)]
    assert deck["matched_picture_ids"] == [ids[1]], "only the sibling is in the group"
    assert deck["member_count"] == 2
    assert deck["stackable"] is False
    assert deck["blocked_by_sets"] == [{"id": set_id, "name": "Frozen"}]
    # The per-candidate value it rolls up says the same thing, and the two loose
    # units are untouched: over-blocking would be its own regression.
    by_id = {c["picture_id"]: c for c in group["candidates"]}
    assert by_id[ids[1]]["stackable"] is False
    assert by_id[ids[2]]["stackable"] is True
    assert by_id[ids[3]]["stackable"] is True


def test_the_deck_rollup_costs_one_query_for_the_whole_page(server):
    """The page resolves stacks once, not once per group (no N+1).

    Five groups, each touching its own stack. ``load_stack_facts`` must be
    called exactly once for the page, with every stack id in it.
    """
    stack_ids = []
    for index in range(5):
        ids = _seed(
            server,
            [
                {"pixel_sha": f"lead-{index}", "size_bytes": 10 + index},
                {"pixel_sha": f"shared-{index}", "size_bytes": 100 + index},
                {"pixel_sha": f"shared-{index}", "size_bytes": 100 + index},
            ],
        )
        stack_ids.append(_stack(server, ids[:2]))
    _run(server, tiers.run_scan_now_in_session, TierPolicy(), None)

    calls = []
    original = tiers.load_stack_facts

    def counting(session, ids):
        materialised = sorted({int(i) for i in ids})
        calls.append(materialised)
        return original(session, materialised)

    tiers.load_stack_facts = counting
    try:
        page, _total, _cursor = _run(server, tiers.page_queue_in_session)
    finally:
        tiers.load_stack_facts = original

    assert len(page) == 5
    assert len(calls) == 1, f"one batched resolve per page, got {len(calls)}"
    assert calls[0] == sorted(stack_ids)


def test_the_leader_ranking_matches_the_grids(server):
    """An unpositioned stack still names a leader, by the same rule the grid
    uses: position (NULLs last), then score, then newest capture, then id."""
    ids = _seed(
        server,
        [
            {"score": 1, "created_at": 0},
            {"score": 5, "created_at": 10},
            {"score": 5, "created_at": 20},
        ],
    )

    def unposition(session):
        stack = PictureStack(name=None)
        session.add(stack)
        session.commit()
        session.refresh(stack)
        for picture_id in ids:
            picture = session.get(Picture, picture_id)
            picture.stack_id = int(stack.id)
            picture.stack_position = None
            session.add(picture)
        session.commit()
        return int(stack.id)

    stack_id = _run(server, unposition)
    facts = _run(server, tiers.load_stack_facts, [stack_id])[stack_id]
    # Highest score wins; the newer capture breaks the tie between the two 5s.
    assert facts.leader_picture_id == ids[2]
    assert facts.member_ids == (ids[2], ids[1], ids[0])
    assert facts.member_count == 3


def test_a_scrapheaped_member_is_not_part_of_the_stacks_depth(server):
    """``member_count`` is the LIVE member count: a deck must not promise to
    move a picture that is already in the scrapheap."""
    ids = _seed(
        server,
        [
            {"pixel_sha": "a", "size_bytes": 10},
            {"pixel_sha": "b", "size_bytes": 11},
            {"pixel_sha": "c", "size_bytes": 12, "deleted": True},
        ],
    )
    stack_id = _stack(server, ids)
    facts = _run(server, tiers.load_stack_facts, [stack_id])[stack_id]
    assert facts.member_count == 2
    assert facts.member_ids == (ids[0], ids[1])


# ── the lazy half: one stack's members, paged ────────────────────────────────


def test_stack_members_pages_in_canonical_order(server):
    ids = _seed(server, [{"pixel_sha": f"m{index}"} for index in range(5)])
    stack_id = _stack(server, ids, thumbnails={ids[0]: (640, 480)})

    first = _run(server, tiers.stack_members_in_session, stack_id, 0, 2)
    assert first["stack_id"] == stack_id
    assert first["member_count"] == 5
    assert first["leader_picture_id"] == ids[0]
    assert first["leader_thumbnail_version"].startswith("640x480-")
    assert [m["picture_id"] for m in first["members"]] == ids[:2]
    assert [m["position"] for m in first["members"]] == [0, 1]
    assert [m["is_leader"] for m in first["members"]] == [True, False]
    assert first["next_offset"] == 2
    assert first["stackable"] is True
    assert first["blocked_by_sets"] == []
    # The tile fields are the queue candidate's, unchanged, so the expansion
    # strip reuses the row's tile.
    assert "thumbnail_version" in first["members"][0]
    assert "smart_score" in first["members"][0]

    last = _run(server, tiers.stack_members_in_session, stack_id, 4, 2)
    assert [m["picture_id"] for m in last["members"]] == [ids[4]]
    assert [m["position"] for m in last["members"]] == [4]
    assert last["next_offset"] is None


def test_stack_members_clamps_the_page_size(server):
    ids = _seed(server, [{"pixel_sha": "one"}, {"pixel_sha": "two"}])
    stack_id = _stack(server, ids)
    page = _run(
        server,
        tiers.stack_members_in_session,
        stack_id,
        0,
        tiers.MAX_STACK_MEMBER_PAGE_SIZE * 10,
    )
    assert page["limit"] == tiers.MAX_STACK_MEMBER_PAGE_SIZE


def test_stack_members_rolls_the_lock_up_over_the_whole_stack(server):
    """Page 2 must not report a different stackability from page 1, so the unit
    rollup is taken over every member, not over the page."""
    ids = _seed(server, [{"pixel_sha": f"m{index}"} for index in range(4)])
    stack_id = _stack(server, ids)
    set_id = _lock_set(server, "Frozen", [ids[3]])

    for offset in (0, 2):
        page = _run(server, tiers.stack_members_in_session, stack_id, offset, 2)
        assert page["stackable"] is False, offset
        assert page["blocked_by_sets"] == [{"id": set_id, "name": "Frozen"}]
        # A locked set freezes the whole stack, so every member is out.
        assert all(member["stackable"] is False for member in page["members"])


def test_stack_members_is_none_for_a_stack_with_no_live_members(server):
    """The route turns this into a 404 rather than an empty stack that looks
    like it exists."""
    assert _run(server, tiers.stack_members_in_session, 987654, 0, 10) is None
    ids = _seed(server, [{"pixel_sha": "gone", "deleted": True}])
    stack_id = _stack(server, ids)
    assert _run(server, tiers.stack_members_in_session, stack_id, 0, 10) is None


# ── release-candidate scan generation and bounded-work regressions ───────────


def _run_requested_scan(server, policy=None, scope=None):
    progress = _run(
        server,
        tiers.request_scan_in_session,
        policy or TierPolicy(),
        scope,
    )
    return DedupScanTask(server.vault.db, progress["scan_id"])._run_task()


def test_completed_rescan_retires_obsolete_exact_evidence(server):
    ids = _seed(
        server,
        [
            {"pixel_sha": "same", "size_bytes": 100},
            {"pixel_sha": "same", "size_bytes": 100},
        ],
    )
    first = _run_requested_scan(server)
    assert first["status"] == "complete"
    assert len(_run(server, lambda s: s.exec(select(DedupGroup)).all())) == 1

    def make_distinct(session):
        picture = session.get(Picture, ids[1])
        picture.pixel_sha = "different"
        session.add(picture)
        session.commit()

    _run(server, make_distinct)
    second = _run_requested_scan(server)

    assert second["status"] == "complete"
    assert second["retired_groups"] == 1
    assert _run(server, lambda s: s.exec(select(DedupGroup)).all()) == []


def test_failed_and_cancelled_rescans_preserve_prior_complete_evidence(
    server, monkeypatch
):
    ids = _seed(
        server,
        [
            {"pixel_sha": "same", "size_bytes": 100},
            {"pixel_sha": "same", "size_bytes": 100},
        ],
    )
    _run_requested_scan(server)

    def make_distinct(session):
        picture = session.get(Picture, ids[1])
        picture.pixel_sha = "different"
        session.add(picture)
        session.commit()

    _run(server, make_distinct)
    cancelled = _run(server, tiers.request_scan_in_session, TierPolicy(), None)
    _run(server, DedupScanTask._start_scan_slice, cancelled["scan_id"])
    assert _run(server, tiers.scan_progress_in_session, None)["phase"] == "exact"
    _run(server, DedupScanTask._mark_pending_after_cancel, cancelled["scan_id"])
    assert len(_run(server, lambda s: s.exec(select(DedupGroup)).all())) == 1
    # A pending scan is not in any tier: the banner would name a step that is
    # not running.
    assert _run(server, tiers.scan_progress_in_session, None)["phase"] is None

    # Allow a fresh request, then fail before a successful finalisation. Neither
    # path is allowed to erase the last complete generation's queue evidence.
    # The row also carries a tier from whatever wrote it last, which the
    # request must not hand on to the run it queues.
    def make_complete(session):
        scan = session.get(DedupScan, cancelled["scan_id"])
        scan.status = "complete"
        scan.phase = "near"
        session.add(scan)
        session.commit()

    _run(server, make_complete)
    failed = _run(server, tiers.request_scan_in_session, TierPolicy(), None)
    assert failed["phase"] is None

    def fail_exact(_session, _scope, _shard=None):
        raise RuntimeError("forced exact scan failure")

    monkeypatch.setattr(tiers, "find_exact_keys_in_session", fail_exact)
    with pytest.raises(RuntimeError, match="forced exact scan failure"):
        DedupScanTask(server.vault.db, failed["scan_id"])._run_task()
    progress = _run(server, tiers.scan_progress_in_session, None)
    assert progress["status"] == "failed"
    assert progress["phase"] is None
    assert len(_run(server, lambda s: s.exec(select(DedupGroup)).all())) == 1


def test_scan_retirement_covers_all_complete_tiers_but_not_sibling_evidence(server):
    def seed_generations(session):
        first = DedupScan(scope_key="set:1", scope_type="set", scope_id="1")
        sibling = DedupScan(scope_key="set:2", scope_type="set", scope_id="2")
        session.add(first)
        session.add(sibling)
        session.flush()
        owned = [
            DedupGroup(
                signature=f"owned-stale-{tier}",
                tier=tier,
                confidence=1.0,
                member_count=2,
                scan_id=int(first.id),
            )
            for tier in ("exact", "near", "embedding")
        ]
        protected = DedupGroup(
            signature="sibling-current",
            tier="exact",
            confidence=1.0,
            member_count=2,
            scan_id=int(sibling.id),
        )
        session.add_all(owned)
        session.add(protected)
        session.commit()
        return int(first.id)

    scan_id = _run(server, seed_generations)

    def retire_and_commit(session):
        removed = tiers.retire_obsolete_scan_groups_in_session(
            session,
            scan_id,
            {"exact": set(), "near": set(), "embedding": set()},
            {"exact", "near", "embedding"},
        )
        session.commit()
        return removed

    removed = _run(server, retire_and_commit)
    remaining = _run(
        server, lambda s: [row.signature for row in s.exec(select(DedupGroup)).all()]
    )
    assert removed == 3
    assert remaining == ["sibling-current"]


def test_incomplete_tier_is_omitted_from_generation_retirement(server):
    def seed_partial_generation(session):
        scan = DedupScan(scope_key="global", scope_type="global")
        session.add(scan)
        session.flush()
        session.add(
            DedupGroup(
                signature="prior-complete-near",
                tier="near",
                confidence=0.95,
                member_count=2,
                scan_id=int(scan.id),
            )
        )
        session.commit()
        return int(scan.id)

    scan_id = _run(server, seed_partial_generation)

    def retire_complete_tiers(session):
        removed = tiers.retire_obsolete_scan_groups_in_session(
            session,
            scan_id,
            {"exact": set(), "near": set()},
            {"exact"},
        )
        session.commit()
        return removed

    assert _run(server, retire_complete_tiers) == 0
    assert _run(
        server, lambda s: [row.signature for row in s.exec(select(DedupGroup)).all()]
    ) == ["prior-complete-near"]


def test_exact_only_scan_does_not_depend_on_embedding_work(server):
    _run(server, tiers.request_scan_in_session, TierPolicy(), None)
    finder = DedupScanFinder(server.vault.db)
    assert finder.depends_on() == [TaskType.PIXEL_SHA]


def test_4001_member_bucket_keeps_the_boundary_member(monkeypatch):
    rows = [(picture_id, 1, None, None, None) for picture_id in range(1, 4002)]
    monkeypatch.setattr(tiers, "_bucket_rows", lambda _session, _scope: rows)

    buckets = tiers.build_near_buckets(object(), DedupScope())

    assert [len(bucket.picture_ids) for bucket in buckets] == [4000, 2]
    assert buckets[0].picture_ids[-1] == buckets[1].picture_ids[0] == 4000
    assert set().union(*(set(bucket.picture_ids) for bucket in buckets)) == set(
        range(1, 4002)
    )
    assert all(bucket.oversized for bucket in buckets)


def test_pair_cap_marks_scan_partial_and_preserves_near_evidence(server, monkeypatch):
    _seed(
        server,
        [
            {"perceptual_hash": PHASH_ZERO},
            {"perceptual_hash": PHASH_ZERO},
            {"perceptual_hash": PHASH_ZERO},
        ],
    )
    monkeypatch.setattr(tiers, "MAX_PAIRS_PER_BUCKET", 1)

    summary = _run_requested_scan(server, TierPolicy(near_enabled=True))
    progress = _run(server, tiers.scan_progress_in_session, None)

    assert summary["status"] == "partial"
    assert progress["status"] == "partial"
    assert "pair cap" in progress["error"]
    assert any(
        row.tier == "near"
        for row in _run(server, lambda s: s.exec(select(DedupGroup)).all())
    )


# ── tier 1 in bounded writer slices ──────────────────────────────────────────


def _stored_memberships(server):
    """Every stored group's member ids, each sorted, in order of lowest member."""

    def read(session):
        by_group: dict[int, list[int]] = {}
        for group_id, picture_id in session.exec(
            select(tiers.DedupGroupMember.group_id, tiers.DedupGroupMember.picture_id)
        ).all():
            by_group.setdefault(int(group_id), []).append(int(picture_id))
        return sorted(sorted(picture_ids) for picture_ids in by_group.values())

    return _run(server, read)


def _rehash(server, picture_id, pixel_sha):
    """Give one picture new content, as a re-import over the same row would."""

    def go(session):
        picture = session.get(Picture, picture_id)
        picture.pixel_sha = pixel_sha
        session.add(picture)
        session.commit()

    _run(server, go)


def test_the_exact_tier_holds_the_writer_one_page_at_a_time(server, monkeypatch):
    """Tier 1 never holds the database writer for the whole library.

    The start slice only prunes, counts and marks the scan running. The keys are
    read once on a concurrent reader, and so is each page's picture count; then
    each page is its own writer slice, publishes the running counts, and lets an
    interactive callback queued after it run before the next page.
    """
    _seed(
        server,
        [{"pixel_sha": f"sha-{index // 2}", "size_bytes": 100} for index in range(14)],
    )
    monkeypatch.setattr(tiers, "EXACT_PAGE_MAX_KEYS", 3)
    scan_id = _run(server, tiers.request_scan_in_session, TierPolicy(), None)["scan_id"]
    task = DedupScanTask(server.vault.db, scan_id)
    order: list[str] = []
    published: list[tuple[int, int]] = []
    original_persist = tiers.persist_groups_in_session

    def counted_persist(session, groups, scan_id=None):
        order.append(f"persist:{len(groups)}")
        return original_persist(session, groups, scan_id)

    monkeypatch.setattr(tiers, "persist_groups_in_session", counted_persist)

    def scan_counts(session):
        scan = session.get(DedupScan, scan_id)
        return scan.groups_found, scan.scanned_pictures

    original_low_slice = task._run_low_slice
    original_read_slice = task._run_read_slice

    def run_low_slice(func, *args):
        order.append(f"writer:{func.__name__}")
        result = original_low_slice(func, *args)
        if func.__name__ == "_persist_exact_page_slice":
            published.append(_run(server, scan_counts))
            if len(published) == 1:
                server.vault.db.submit_task(
                    lambda _session: order.append("interactive"),
                    priority=DBPriority.IMMEDIATE,
                )
        return result

    def run_read_slice(read, *args):
        order.append(f"reader:{read.__name__}")
        return original_read_slice(read, *args)

    monkeypatch.setattr(task, "_run_low_slice", run_low_slice)
    monkeypatch.setattr(task, "_run_read_slice", run_read_slice)
    summary = task._run_task()

    assert summary["status"] == "complete"
    assert summary["groups_found"] == 7
    assert order[: tiers.EXACT_KEY_SHARD_COUNT + 2] == [
        "writer:_start_scan_slice",
        *["reader:_exact_keys_read"] * tiers.EXACT_KEY_SHARD_COUNT,
        "reader:_exact_progress_read",
    ], order
    persists = [i for i, item in enumerate(order) if item.startswith("persist:")]
    assert [order[i] for i in persists] == ["persist:3", "persist:3", "persist:1"]
    assert persists[0] < order.index("interactive") < persists[1], order
    # The running counts after each page: every picture below the next page's
    # first key is settled, and the last page settles the whole scope.
    assert published == [(3, 6), (6, 12), (7, 14)]
    assert order[-2:] == [
        "reader:_obsolete_groups_read",
        "writer:_finish_scan_slice",
    ], order
    assert len(_stored_memberships(server)) == 7


# Three duplicated keys, a page each at EXACT_PAGE_MAX_KEYS = 1, with pictures
# that have no duplicate between them and a scrapheaped one before the first
# frontier. Live ids in order: a a . b b . c c plus one more single at the end.
_SETTLED_SPECS = [
    {"pixel_sha": "a", "perceptual_hash": PHASH_FAR},
    {"pixel_sha": "a", "perceptual_hash": PHASH_FAR},
    {"pixel_sha": "gone", "perceptual_hash": PHASH_FAR, "deleted": True},
    {"pixel_sha": "single-1", "perceptual_hash": PHASH_ZERO},
    {"pixel_sha": "b", "perceptual_hash": PHASH_FAR},
    {"pixel_sha": "b", "perceptual_hash": PHASH_FAR},
    {"pixel_sha": "single-2", "perceptual_hash": PHASH_ZERO},
    {"pixel_sha": "c", "perceptual_hash": PHASH_FAR},
    {"pixel_sha": "c", "perceptual_hash": PHASH_FAR},
    {"pixel_sha": "single-3", "perceptual_hash": PHASH_ZERO},
]
# Live pictures below b's first id: a, a, single-1. Below c's: those plus b, b,
# single-2. The last page settles the scope. Counting only the members of the
# keys persisted so far would read 2, 4, 9 instead.
_SETTLED_AFTER_PAGES = [3, 6, 9]


def _scan_snapshot(server, scope):
    """The scope's scan progress as the API serialises it."""
    progress = _run(server, tiers.scan_progress_in_session, scope)
    return {
        "phase": progress["phase"],
        "scanned": progress["scanned_pictures"],
        "total": progress["total_pictures"],
        "buckets": progress["scanned_buckets"],
        "total_buckets": progress["total_buckets"],
    }


def test_every_tier_publishes_its_own_progress_as_it_runs(server, monkeypatch):
    """The banner can count every tier of a running scan, never "0 of N".

    The exact tier publishes the pictures it has settled after every page, in
    a scan that goes on to the near tier as much as in an exact-only one. Each
    tier then names itself in ``phase`` with its counters reset, so one tier's
    count is never read on another's scale, and the reads that cover the whole
    scope (the near buckets, the embedding scope) run off the writer.
    """
    _seed(server, _SETTLED_SPECS)
    monkeypatch.setattr(tiers, "EXACT_PAGE_MAX_KEYS", 1)
    policy = TierPolicy(near_enabled=True, embedding_enabled=True)
    scope = DedupScope()
    scan_id = _run(server, tiers.request_scan_in_session, policy, scope)["scan_id"]
    task = DedupScanTask(server.vault.db, scan_id)
    writes: list[tuple[str, dict]] = []
    reads: list[str] = []
    original_low_slice = task._run_low_slice
    original_read_slice = task._run_read_slice

    def run_low_slice(func, *args):
        result = original_low_slice(func, *args)
        writes.append((func.__name__, _scan_snapshot(server, scope)))
        return result

    def run_read_slice(read, *args):
        reads.append(read.__name__)
        return original_read_slice(read, *args)

    monkeypatch.setattr(task, "_run_low_slice", run_low_slice)
    monkeypatch.setattr(task, "_run_read_slice", run_read_slice)
    summary = task._run_task()

    assert summary["status"] == "complete"
    assert summary["total_pictures"] == 9
    assert [name for name, _ in writes][:4] == [
        "_start_scan_slice",
        "_persist_exact_page_slice",
        "_persist_exact_page_slice",
        "_persist_exact_page_slice",
    ]
    assert writes[0][1]["phase"] == "exact"
    assert writes[0][1]["scanned"] == 0
    assert [
        (snapshot["phase"], snapshot["scanned"], snapshot["total"])
        for name, snapshot in writes
        if name == "_persist_exact_page_slice"
    ] == [("exact", settled, 9) for settled in _SETTLED_AFTER_PAGES]

    phases = [snapshot for name, snapshot in writes if name == "_begin_phase_slice"]
    assert phases == [
        {"phase": "near", "scanned": 0, "total": 9, "buckets": 0, "total_buckets": 0},
        {
            "phase": "embedding",
            "scanned": 0,
            "total": 9,
            "buckets": 0,
            "total_buckets": 0,
        },
    ]
    near = [
        snapshot
        for name, snapshot in writes
        if name in ("_publish_bucket_total_slice", "_run_near_bucket_slice")
    ]
    bucket_total = near[0]["total_buckets"]
    assert bucket_total > 0
    assert [(s["phase"], s["buckets"], s["total_buckets"]) for s in near] == [
        ("near", index, bucket_total) for index in range(bucket_total + 1)
    ]
    assert writes[-1][0] == "_finish_scan_slice"
    assert writes[-1][1]["phase"] is None
    assert writes[-1][1]["scanned"] == 9

    for whole_scope_read in ("_near_buckets_read", "_embedding_scope_read"):
        assert whole_scope_read in reads
    assert reads.index("_exact_progress_read") < reads.index("_near_buckets_read")


def test_each_exact_page_is_built_from_the_pictures_as_they_are_now(
    server, monkeypatch
):
    """A picture scrapheaped after the keys were read is not stored as a member.

    The keys are read once per scan and each page re-reads its members, so a
    change in between gives the group as it is now, and the group it replaced
    is retired with the scan.
    """
    ids = _seed(
        server,
        [
            {"pixel_sha": "pair", "size_bytes": 100},
            {"pixel_sha": "pair", "size_bytes": 100},
            {"pixel_sha": "trio", "size_bytes": 100},
            {"pixel_sha": "trio", "size_bytes": 100},
            {"pixel_sha": "trio", "size_bytes": 100},
        ],
    )
    assert _run_requested_scan(server)["status"] == "complete"
    assert _stored_memberships(server) == [ids[0:2], ids[2:5]]
    original_page_exact_keys = tiers.page_exact_keys

    def scrapheap_then_page(keys, *args, **kwargs):
        _soft_delete(server, ids[4])
        return original_page_exact_keys(keys, *args, **kwargs)

    monkeypatch.setattr(tiers, "page_exact_keys", scrapheap_then_page)
    second = _run_requested_scan(server)

    assert second["status"] == "complete"
    assert second["groups_found"] == 2
    assert second["retired_groups"] == 1
    assert _stored_memberships(server) == [ids[0:2], ids[2:4]]


def test_a_cancel_after_retirements_are_chosen_retires_nothing(server, monkeypatch):
    """Retirement lands with the terminal status or not at all.

    The obsolete groups are chosen on a reader. A cancel that arrives then
    must leave the last complete scan's evidence in place and the scan
    restartable.
    """
    ids = _seed(
        server,
        [
            {"pixel_sha": "same", "size_bytes": 100},
            {"pixel_sha": "same", "size_bytes": 100},
        ],
    )
    _run_requested_scan(server)
    prior = _run(server, lambda s: [row.id for row in s.exec(select(DedupGroup))])
    assert len(prior) == 1
    _rehash(server, ids[1], "different")

    progress = _run(server, tiers.request_scan_in_session, TierPolicy(), None)
    task = DedupScanTask(server.vault.db, progress["scan_id"])
    chosen: list[int] = []
    original_choose = tiers.obsolete_scan_group_ids_in_session

    def choose_then_cancel(*args):
        chosen.extend(original_choose(*args))
        task.on_cancel()
        return list(chosen)

    monkeypatch.setattr(tiers, "obsolete_scan_group_ids_in_session", choose_then_cancel)
    summary = task._run_task()

    assert summary == {
        "scan_id": progress["scan_id"],
        "status": SCAN_PENDING,
        "cancelled": True,
    }
    assert chosen == prior
    assert _run(server, lambda s: [row.id for row in s.exec(select(DedupGroup))]) == (
        prior
    )
    assert _run(server, tiers.scan_progress_in_session, None)["status"] == SCAN_PENDING


def test_a_reader_step_honours_a_cancel_on_either_side():
    """A cancel before a reader step skips it; one during it stops the scan."""

    class FakeDatabase:
        def run_immediate_read_task(self, func, *args):
            return func(None, *args)

    task = DedupScanTask(FakeDatabase(), scan_id=1)
    reads: list[str] = []

    def read(_session, label):
        reads.append(label)
        return label

    def cancelling_read(_session, label):
        reads.append(label)
        task.on_cancel()
        return label

    assert task._run_read_slice(read, "before") == "before"
    with pytest.raises(dedup_scan_task_module._DedupScanCancelled):
        task._run_read_slice(cancelling_read, "during")
    with pytest.raises(dedup_scan_task_module._DedupScanCancelled):
        task._run_read_slice(read, "after")
    assert reads == ["before", "during"]


def test_a_group_another_scan_refreshes_before_the_final_slice_is_kept(
    server, monkeypatch
):
    """Ids chosen on the reader are deleted only while this scan still owns them.

    A sibling or global scan can refresh one of them between the read and the
    final writer slice; from then on the group is that scan's evidence.
    """
    ids = _seed(
        server,
        [
            {"pixel_sha": "same", "size_bytes": 100},
            {"pixel_sha": "same", "size_bytes": 100},
        ],
    )
    _run_requested_scan(server)
    _rehash(server, ids[1], "different")

    def add_other_scan(session):
        scan = DedupScan(scope_key="set:99", scope_type="set", scope_id="99")
        session.add(scan)
        session.commit()
        return int(scan.id)

    other_scan_id = _run(server, add_other_scan)

    def hand_over(session, group_ids):
        for group_id in group_ids:
            group = session.get(DedupGroup, group_id)
            group.scan_id = other_scan_id
            session.add(group)
        session.commit()

    progress = _run(server, tiers.request_scan_in_session, TierPolicy(), None)
    task = DedupScanTask(server.vault.db, progress["scan_id"])
    handed: list[int] = []
    original_read_slice = task._run_read_slice

    def run_read_slice(read, *args):
        result = original_read_slice(read, *args)
        if read.__name__ == "_obsolete_groups_read":
            handed.extend(result)
            _run(server, hand_over, result)
        return result

    monkeypatch.setattr(task, "_run_read_slice", run_read_slice)
    summary = task._run_task()

    assert len(handed) == 1
    assert summary["status"] == "complete"
    assert summary["retired_groups"] == 0
    assert _run(
        server, lambda s: [(row.id, row.scan_id) for row in s.exec(select(DedupGroup))]
    ) == [(handed[0], other_scan_id)]


def test_a_scan_counts_the_live_pictures_in_its_scope(server):
    ids = _seed(
        server,
        [
            {},
            {},
            {},
            {"deleted": True},
            {},
            {},
        ],
    )

    def add_set(session):
        picture_set = PictureSet(name="Counted")
        session.add(picture_set)
        session.commit()
        session.refresh(picture_set)
        for picture_id in ids[1:5]:
            session.add(
                PictureSetMember(set_id=int(picture_set.id), picture_id=picture_id)
            )
        session.commit()
        return int(picture_set.id)

    scope = DedupScope(scope_type=ScopeType.SET, scope_id=str(_run(server, add_set)))

    assert _run(server, DedupScanTask._count_live_pictures, scope) == 3
    assert _run(server, DedupScanTask._count_live_pictures, DedupScope()) == 5
    scan_id = _run(server, tiers.request_scan_in_session, TierPolicy(), scope)[
        "scan_id"
    ]
    assert _run(server, DedupScanTask._start_scan_slice, scan_id)["total_pictures"] == 3


def test_the_synchronous_scan_pages_tier_one_and_retires_what_it_lost(
    server, monkeypatch
):
    """``run_scan_in_session``, the end-to-end fixture's path, does the same work."""
    ids = _seed(
        server,
        [
            {"pixel_sha": "pair", "size_bytes": 100},
            {"pixel_sha": "pair", "size_bytes": 100},
            {"pixel_sha": "trio", "size_bytes": 100},
            {"pixel_sha": "trio", "size_bytes": 100},
            {"pixel_sha": "trio", "size_bytes": 100},
        ],
    )
    monkeypatch.setattr(tiers, "EXACT_PAGE_MAX_KEYS", 1)

    def scan(session):
        progress = tiers.request_scan_in_session(session, TierPolicy(), None)
        return DedupScanTask.run_scan_in_session(session, progress["scan_id"])

    first = _run(server, scan)
    assert (first["status"], first["total_pictures"], first["groups_found"]) == (
        "complete",
        5,
        2,
    )
    assert _stored_memberships(server) == [ids[0:2], ids[2:5]]

    _rehash(server, ids[1], "different")
    second = _run(server, scan)
    assert (second["status"], second["groups_found"], second["retired_groups"]) == (
        "complete",
        1,
        1,
    )
    assert _stored_memberships(server) == [ids[2:5]]


def test_the_synchronous_scan_publishes_the_same_progress(server, monkeypatch):
    """``run_scan_in_session`` counts each exact page and names each tier too."""
    _seed(server, _SETTLED_SPECS)
    monkeypatch.setattr(tiers, "EXACT_PAGE_MAX_KEYS", 1)
    seen: list[tuple[str, str | None, int]] = []
    original_persist = DedupScanTask._persist_exact_page_slice
    original_begin = DedupScanTask._begin_phase_slice

    def scan_row(session, scan_id):
        session.expire_all()
        return session.get(DedupScan, scan_id)

    def persist(session, scan_id, *args):
        result = original_persist(session, scan_id, *args)
        row = scan_row(session, scan_id)
        seen.append(("page", row.phase, row.scanned_pictures))
        return result

    def begin(session, scan_id, phase):
        original_begin(session, scan_id, phase)
        row = scan_row(session, scan_id)
        seen.append(("phase", row.phase, row.scanned_pictures))

    monkeypatch.setattr(
        DedupScanTask, "_persist_exact_page_slice", staticmethod(persist)
    )
    monkeypatch.setattr(DedupScanTask, "_begin_phase_slice", staticmethod(begin))

    def scan(session):
        progress = tiers.request_scan_in_session(
            session, TierPolicy(near_enabled=True, embedding_enabled=True), None
        )
        summary = DedupScanTask.run_scan_in_session(session, progress["scan_id"])
        return summary, scan_row(session, progress["scan_id"]).phase

    summary, phase_after = _run(server, scan)

    assert summary["status"] == "complete"
    assert seen == [
        *(("page", "exact", settled) for settled in _SETTLED_AFTER_PAGES),
        ("phase", "near", 0),
        ("phase", "embedding", 0),
    ]
    assert phase_after is None
