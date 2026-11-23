"""Unit tests for the pure near-neighbour kernel helpers in
``pixlstash.utils.near_neighbor``.

These build their inputs directly (no DB, no models), so they isolate the perceptual-hash
near-duplicate twin selection that the tag scan uses to prefer an altered copy of a picture
over the CLIP-nearest opposite-labelled neighbour.
"""

import tracemalloc

import numpy as np

from pixlstash.utils.near_neighbor import (
    hamming_distance,
    knn_disagreement_with_neighbors,
    nearest_opposite_by_hamming,
)


def test_hamming_distance_basic():
    assert hamming_distance(0, 0) == 0
    assert hamming_distance(0b1011, 0b1110) == 2
    # Full 64-bit range round-trips (max unsigned 64-bit value).
    assert hamming_distance(0, (1 << 64) - 1) == 64


def test_returns_closest_opposite_in_threshold():
    """Picks the opposite-labelled near-dup with the SMALLEST Hamming distance."""
    base = 0b1010_1010
    phash = np.array(
        [
            base,  # i = 0 (the suspect)
            base ^ 0b0000_0011,  # j=1: 2 bits away, opposite label  <- expected
            base ^ 0b0000_0111,  # j=2: 3 bits away, opposite label
            base,  # j=3: 0 bits away but SAME label (ignored)
        ],
        dtype=np.uint64,
    )
    valid = np.ones(4, dtype=bool)
    has_tag = np.array([True, False, False, True], dtype=bool)

    j = nearest_opposite_by_hamming(phash, valid, has_tag, i=0, max_hamming=8)
    assert j == 1


def test_returns_minus_one_when_only_same_labelled():
    """All near-dups carry the SAME label → no genuine disagreement → -1."""
    base = 0b1111_0000
    phash = np.array([base, base ^ 0b1, base ^ 0b11], dtype=np.uint64)
    valid = np.ones(3, dtype=bool)
    has_tag = np.array([True, True, True], dtype=bool)

    j = nearest_opposite_by_hamming(phash, valid, has_tag, i=0, max_hamming=8)
    assert j == -1


def test_returns_minus_one_when_all_opposite_exceed_threshold():
    """Opposite-labelled candidates exist but are all beyond max_hamming → -1."""
    base = 0
    phash = np.array(
        [
            base,
            base ^ 0b1111_1111_1111,  # 12 bits away, opposite label
            base ^ 0b1111_1111_1111_1,  # 13 bits away, opposite label
        ],
        dtype=np.uint64,
    )
    valid = np.ones(3, dtype=bool)
    has_tag = np.array([True, False, False], dtype=bool)

    j = nearest_opposite_by_hamming(phash, valid, has_tag, i=0, max_hamming=8)
    assert j == -1
    # Loosening the threshold past 12 bits finds the closer one.
    j2 = nearest_opposite_by_hamming(phash, valid, has_tag, i=0, max_hamming=12)
    assert j2 == 1


def test_invalid_phash_rows_are_ignored_via_valid_mask():
    """Rows flagged invalid (no/garbled phash) never become the twin, even at distance 0."""
    base = 0b0101_0101
    phash = np.array([base, base, base ^ 0b11], dtype=np.uint64)
    valid = np.array([True, False, True], dtype=bool)  # row 1 invalid
    has_tag = np.array([True, False, False], dtype=bool)

    # Row 1 is distance 0 and opposite-labelled, but invalid → must be skipped for row 2.
    j = nearest_opposite_by_hamming(phash, valid, has_tag, i=0, max_hamming=8)
    assert j == 2


def test_returns_minus_one_when_suspect_phash_invalid():
    """If the suspect itself has no valid phash, there is nothing to match against."""
    phash = np.array([0, 1], dtype=np.uint64)
    valid = np.array([False, True], dtype=bool)
    has_tag = np.array([True, False], dtype=bool)

    j = nearest_opposite_by_hamming(phash, valid, has_tag, i=0, max_hamming=8)
    assert j == -1


def test_tie_broken_by_higher_clip_sim_then_lower_index():
    """Equal Hamming distance → higher CLIP cosine wins; absent sims → lower index."""
    base = 0
    phash = np.array(
        [
            base,
            base ^ 0b1,  # j=1: 1 bit away
            base ^ 0b10,  # j=2: 1 bit away (tie)
        ],
        dtype=np.uint64,
    )
    valid = np.ones(3, dtype=bool)
    has_tag = np.array([True, False, False], dtype=bool)

    # No tie-breaker similarity → deterministic lowest index.
    j = nearest_opposite_by_hamming(phash, valid, has_tag, i=0, max_hamming=8)
    assert j == 1

    # With similarity, the higher-cosine candidate (index 2) wins the tie.
    twin_sim = np.array([0.0, 0.10, 0.90], dtype=np.float32)
    j2 = nearest_opposite_by_hamming(
        phash, valid, has_tag, i=0, max_hamming=8, twin_sim=twin_sim
    )
    assert j2 == 2


def test_full_64bit_range_no_overflow():
    """uint64 dhash values near the top of the range XOR/popcount correctly."""
    top = (1 << 64) - 1
    phash = np.array([top, top ^ 0b111, top], dtype=np.uint64)
    valid = np.ones(3, dtype=bool)
    has_tag = np.array([True, False, True], dtype=bool)

    j = nearest_opposite_by_hamming(phash, valid, has_tag, i=0, max_hamming=8)
    assert j == 1


# ---------------------------------------------------------------------------
# knn_disagreement_with_neighbors: the neighbour-capture variant of the kernel.
# ---------------------------------------------------------------------------


def _unit_rows(vectors):
    emb = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def test_with_neighbors_matches_plain_kernel_and_orders_by_similarity():
    from pixlstash.utils.near_neighbor import (
        knn_disagreement,
        knn_disagreement_with_neighbors,
    )

    # Four points on two axes: 0 and 1 nearly parallel, 2 and 3 nearly parallel.
    emb = _unit_rows(
        [
            [1.0, 0.0, 0.0],
            [0.999, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.999, 0.0],
        ]
    )
    has_tag = np.array([True, False, True, False])

    pf1, ti1, ts1 = knn_disagreement(emb, has_tag, k=2)
    pf2, ti2, ts2, nbr = knn_disagreement_with_neighbors(emb, has_tag, k=2)
    np.testing.assert_allclose(pf1, pf2)
    np.testing.assert_array_equal(ti1, ti2)
    np.testing.assert_allclose(ts1, ts2)

    # Shape: one row per input, k columns; no self-references; descending sim.
    assert nbr.shape == (4, 2)
    for i in range(4):
        assert i not in nbr[i]
        sims = emb[i] @ emb[nbr[i]].T
        assert sims[0] >= sims[1]
    # Row 0's nearest neighbour is its near-parallel partner, row 1.
    assert nbr[0][0] == 1
    assert nbr[2][0] == 3


def test_with_neighbors_degenerate_sizes():
    from pixlstash.utils.near_neighbor import knn_disagreement_with_neighbors

    # n == 1: empty neighbour matrix, no crash.
    emb = _unit_rows([[1.0, 0.0]])
    pf, ti, ts, nbr = knn_disagreement_with_neighbors(emb, np.array([True]), k=5)
    assert nbr.shape == (1, 0)

    # k larger than n-1 clamps to n-1 columns.
    emb = _unit_rows([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    pf, ti, ts, nbr = knn_disagreement_with_neighbors(
        emb, np.array([True, False, False]), k=10
    )
    assert nbr.shape == (3, 2)
    assert (nbr >= 0).all()


# ---------------------------------------------------------------------------
# knn_disagreement_with_neighbors: outputs and scratch memory per block.
# ---------------------------------------------------------------------------

_OUTPUTS = ("pos_frac", "twin_idx", "twin_sim", "neighbor_idx")


def _reference_kernel(emb, has_tag, k, block):
    """The kernel's earlier, plainer loop: the oracle for its outputs."""
    n = emb.shape[0]
    pos_frac = np.zeros(n, dtype=np.float32)
    twin_idx = np.full(n, -1, dtype=np.int64)
    twin_sim = np.zeros(n, dtype=np.float32)
    if n <= 1:
        return pos_frac, twin_idx, twin_sim, np.empty((n, 0), dtype=np.int64)
    k = min(k, n - 1)
    neighbor_idx = np.full((n, k), -1, dtype=np.int64)
    y = has_tag.astype(np.float32)
    for start in range(0, n, block):
        end = min(start + block, n)
        sims = emb[start:end] @ emb.T
        for local, glob in enumerate(range(start, end)):
            sims[local, glob] = -np.inf
        top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        for local in range(end - start):
            glob = start + local
            nbr = top[local]
            w = np.clip(sims[local, nbr], 0.0, None)
            wsum = w.sum()
            pos_frac[glob] = float((w * y[nbr]).sum() / wsum) if wsum > 0 else 0.0
            order = np.argsort(-sims[local, nbr], kind="stable")
            neighbor_idx[glob] = nbr[order]
            opp = nbr[has_tag[nbr] != has_tag[glob]]
            if opp.size:
                best = int(opp[np.argmax(sims[local, opp])])
                twin_idx[glob] = best
                twin_sim[glob] = float(sims[local, best])
    return pos_frac, twin_idx, twin_sim, neighbor_idx


def _assert_same_bits(actual, expected, context):
    for name, got, want in zip(_OUTPUTS, actual, expected, strict=True):
        np.testing.assert_array_equal(got, want, err_msg=f"{name}, {context}")
        assert got.dtype == want.dtype, f"{name}, {context}"
        assert got.tobytes() == want.tobytes(), f"{name}, {context}"


def _random_rows(n, dim, seed):
    rng = np.random.default_rng(seed)
    return _unit_rows(rng.standard_normal((n, dim))), rng.random(n) < 0.3


def test_kernel_outputs_match_the_reference_loop_bit_for_bit():
    """At a given block height every output is the reference's, bit for bit,
    including exact ties (duplicates), a similarity of -1 and a zero vector."""
    emb, has_tag = _random_rows(1100, 32, seed=11)
    emb[1] = emb[0]
    emb[2] = emb[0]
    emb[3] = -emb[0]
    emb[4] = 0.0
    for block in (1024, 7):
        _assert_same_bits(
            knn_disagreement_with_neighbors(emb, has_tag, 12, block),
            _reference_kernel(emb, has_tag, 12, block),
            f"block={block}",
        )


def test_smallest_inputs_under_a_one_row_budget():
    for n, k in ((0, 5), (1, 5), (2, 5), (3, 1), (3, 10)):
        emb, has_tag = _random_rows(n, 4, seed=n)
        has_tag[:1] = True
        _assert_same_bits(
            knn_disagreement_with_neighbors(emb, has_tag, k, max_block_bytes=1),
            _reference_kernel(emb, has_tag, k, block=1),
            f"n={n}, k={k}",
        )


def test_a_budget_that_shrinks_the_block_moves_no_neighbour():
    """A smaller block may change a similarity in its last bit (the product is
    blocked differently), never which pictures are neighbours or twins."""
    n, k = 300, 12
    emb, has_tag = _random_rows(n, 32, seed=8)
    # Precondition: each row's top k + 1 similarities are far apart, so a
    # last-bit change cannot reorder them.
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    ranked = -np.sort(-sims, axis=1)[:, : k + 1]
    assert np.diff(ranked, axis=1).max() < -1e-5

    whole = knn_disagreement_with_neighbors(emb, has_tag, k)
    budgeted = knn_disagreement_with_neighbors(
        emb, has_tag, k, max_block_bytes=12 * n * 7
    )
    np.testing.assert_array_equal(budgeted[1], whole[1], err_msg="twin_idx")
    np.testing.assert_array_equal(budgeted[3], whole[3], err_msg="neighbor_idx")
    np.testing.assert_allclose(budgeted[0], whole[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(budgeted[2], whole[2], rtol=0, atol=1e-6)


def _block_heights(monkeypatch, emb, has_tag, **kwargs):
    heights = []
    real_argpartition = np.argpartition

    def recording_argpartition(a, *args, **kw):
        heights.append(a.shape)
        return real_argpartition(a, *args, **kw)

    monkeypatch.setattr(np, "argpartition", recording_argpartition)
    try:
        knn_disagreement_with_neighbors(emb, has_tag, 4, **kwargs)
    finally:
        monkeypatch.undo()
    return heights


def test_budget_sets_the_block_height(monkeypatch):
    """B = min(block, max(1, budget // (12 * n))): 12 bytes a similarity cell."""
    n = 60
    emb, has_tag = _random_rows(n, 8, seed=3)
    assert _block_heights(monkeypatch, emb, has_tag) == [(n, n)]
    assert (
        _block_heights(monkeypatch, emb, has_tag, max_block_bytes=12 * n * 5)
        == [(5, n)] * 12
    )
    assert (
        _block_heights(
            monkeypatch, emb, has_tag, max_block_bytes=12 * n * 5 + 12 * n - 1
        )
        == [(5, n)] * 12
    )
    assert _block_heights(monkeypatch, emb, has_tag, max_block_bytes=1) == [(1, n)] * n
    assert _block_heights(
        monkeypatch, emb, has_tag, block=16, max_block_bytes=10**12
    ) == [(16, n)] * 3 + [(12, n)]


def test_scratch_memory_stays_near_one_block():
    """One block is a float32 similarity matrix plus argpartition's index array,
    12 bytes a cell; the outputs add about half a byte a cell here. Holding a
    second similarity matrix or the previous block's index array is >= 16."""
    n, rows = 4096, 256
    emb, has_tag = _random_rows(n, 64, seed=9)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base, _ = tracemalloc.get_traced_memory()
        knn_disagreement_with_neighbors(emb, has_tag, 12, max_block_bytes=12 * n * rows)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert (peak - base) / (rows * n) < 14
