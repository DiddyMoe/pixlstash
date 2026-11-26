"""Unit tests for the tag scan's neighbour-vote arrays (no server needed)."""

import logging

import numpy as np

from pixlstash.services.tag_scan_service import _vote_arrays
from pixlstash.utils.near_neighbor import EMBEDDING_DIM


def _blob(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(EMBEDDING_DIM).astype(np.float32).tobytes()


def test_an_out_of_range_dhash_is_marked_invalid_not_raised(caplog):
    """A 17-digit, signed or ``0x`` dhash is not a dhash, though ``int`` parses it.

    One such row must leave the scan running: it is excluded from the twin vote,
    logged, and every other row keeps its value.
    """
    ids = [1, 2, 3, 4, 5]
    phashes = ["00000000000000ff", "1" + "0" * 16, "-1", None, "0x000000000000ff"]
    blobs = [_blob(i) for i in ids]
    with caplog.at_level(logging.WARNING):
        emb, phash_ints, valid_mask, has_literal = _vote_arrays(
            ids, blobs, phashes, {1}
        )
    assert phash_ints.dtype == np.uint64
    assert phash_ints.tolist() == [0xFF, 0, 0, 0, 0]
    assert valid_mask.tolist() == [True, False, False, False, False]
    assert has_literal.tolist() == [True, False, False, False, False]
    assert emb.shape == (5, EMBEDDING_DIM)
    flagged = [
        r.getMessage()
        for r in caplog.records
        if "unparseable perceptual_hash" in r.getMessage()
    ]
    assert flagged == [
        f"scan_tag: unparseable perceptual_hash {phashes[pid - 1]!r} for picture "
        f"{pid}; excluding from near-duplicate twin selection"
        for pid in (2, 3, 5)
    ]


def test_the_full_64_bit_range_round_trips():
    ids = [1, 2]
    phashes = ["ffffffffffffffff", "0000000000000000"]
    _, phash_ints, valid_mask, _ = _vote_arrays(
        ids, [_blob(1), _blob(2)], phashes, set()
    )
    assert phash_ints.tolist() == [(1 << 64) - 1, 0]
    assert valid_mask.tolist() == [True, True]
