// Grid thumbnail batch - write results back by picture identity, not by slot.
//
// Symptom this guards: you add a penalised tag, the card moves to its new
// smart-score position correctly, but it never shows the problem indicator.
//
// fetchThumbnailsBatch snapshots `allGridImages.slice(start, end)` and then awaits
// the network. `penalised_tags` (what drives the indicator) only refreshes through
// that response. Meanwhile repositionImageBySmartScore → _spliceAndReinsert
// re-orders allGridImages *without* bumping thumbnailRequestEpoch, so the epoch
// guard in the batch cannot see the move. A positional write-back
// (`allGridImages[start + i] = img`) then lands the refreshed payload on whichever
// picture now occupies that slot: the tagged card keeps its stale tags (no
// indicator) and an unrelated card is corrupted with someone else's data.
//
// fetchThumbnailsBatch hands its rows to writeThumbnailBatch, tested here
// directly; ImageGridThumbnailVersion.test.js covers the batch end to end.

import { describe, it, expect } from "vitest";
import {
  loadedSlotRuns,
  writeThumbnailBatch,
} from "../../utils/thumbnailWriteback.js";

function writeBack(grid, batch, start) {
  writeThumbnailBatch(grid, batch, start);
  return grid;
}

// The grid as it looked when the batch was snapshotted.
function initialGrid() {
  return [
    { id: 1, penalised_tags: [] },
    { id: 2, penalised_tags: [] },
    { id: 3, penalised_tags: [] },
  ];
}

// The batch response: picture 2 came back carrying a freshly-applied problem tag.
function batchResponse() {
  return [
    { id: 1, penalised_tags: [] },
    { id: 2, penalised_tags: ["bad anatomy"] },
    { id: 3, penalised_tags: [] },
  ];
}

// Picture 2 gained a penalised tag, so its smart score dropped and the card was
// repositioned to the end while the batch was still in flight.
function gridAfterReposition() {
  return [
    { id: 1, penalised_tags: [] },
    { id: 3, penalised_tags: [] },
    { id: 2, penalised_tags: [] },
  ];
}

/** `rows`, behind a proxy that records every slot read. */
function watchedGrid(rows) {
  const reads = [];
  const grid = new Proxy(rows, {
    get(target, key, receiver) {
      if (typeof key === "string" && /^\d+$/.test(key)) reads.push(Number(key));
      return Reflect.get(target, key, receiver);
    },
  });
  return { grid, reads };
}

function pictures(count) {
  return Array.from({ length: count }, (_, i) => ({ id: i + 1, idx: i }));
}

function answered(rows) {
  return rows.map((row) => ({ ...row, thumbnail: `test-thumb-${row.id}` }));
}

describe("thumbnail batch write-back", () => {
  it("lands the problem indicator on the card that moved", () => {
    const result = writeBack(gridAfterReposition(), batchResponse(), 0);
    const byId = Object.fromEntries(result.map((img) => [img.id, img]));
    expect(byId[2].penalised_tags).toEqual(["bad anatomy"]);
    // And nobody else was given picture 2's tags.
    expect(byId[1].penalised_tags).toEqual([]);
    expect(byId[3].penalised_tags).toEqual([]);
  });

  it("keeps positional behaviour when nothing moved", () => {
    const result = writeBack(initialGrid(), batchResponse(), 0);
    expect(result.map((img) => img.id)).toEqual([1, 2, 3]);
    expect(result[1].penalised_tags).toEqual(["bad anatomy"]);
    expect(result.map((img) => img.idx)).toEqual([0, 1, 2]);
  });

  it("keeps a reposition made while the batch was out", () => {
    const before = gridAfterReposition().map((img) => img.id);
    expect(before).toEqual([1, 3, 2]);

    const result = writeBack(gridAfterReposition(), batchResponse(), 0);

    // Writing the pre-move snapshot back by slot reverts the grid to the order
    // the batch was built against, so the card the user just tagged jumps back
    // to where it was; merging by slot instead hands each card a neighbour's
    // fields.
    expect(result.map((img) => img.id)).toEqual([1, 3, 2]);
    expect(result.map((img) => img.penalised_tags)).toEqual([
      [],
      [],
      ["bad anatomy"],
    ]);
  });

  it("skips a picture that left the grid while the batch was out", () => {
    // Sliced from [1, 2, 3]; picture 2 has since collapsed into a stack.
    const card3 = { id: 3, penalised_tags: [] };
    const grid = [{ id: 1, penalised_tags: [] }, card3];
    const written = writeThumbnailBatch(grid, batchResponse().slice(0, 2), 0);

    expect(written.map(({ row }) => row.id)).toEqual([1]);
    expect(grid[1]).toBe(card3);
  });

  it("never looks past the batch's own slots when nothing moved", () => {
    const { grid, reads } = watchedGrid(pictures(1000));
    const batch = answered(pictures(1000).slice(100, 150));

    const written = writeThumbnailBatch(grid, batch, 100);

    expect(written).toHaveLength(50);
    expect(reads.filter((slot) => slot < 100 || slot >= 150)).toEqual([]);
  });

  it("scans the grid once however many rows of the batch moved", () => {
    const rows = pictures(1000);
    const batch = answered(rows.slice(0, 10));
    // Three of the batch's pictures were repositioned to the end of the grid.
    const movedIds = new Set([2, 5, 8]);
    const reordered = [
      ...rows.filter((row) => !movedIds.has(row.id)),
      ...rows.filter((row) => movedIds.has(row.id)),
    ];
    const { grid, reads } = watchedGrid(reordered);

    writeThumbnailBatch(grid, batch, 0);

    // Slot 500 is neither in the batch's window nor where a moved row landed:
    // only a scan of the whole grid reads it.
    expect(reads.filter((slot) => slot === 500)).toHaveLength(1);
    expect(grid.slice(997).map((card) => card.thumbnail)).toEqual([
      "test-thumb-2",
      "test-thumb-5",
      "test-thumb-8",
    ]);
  });

  it("lands a moved row on the first card with its id", () => {
    // An expanded stack repeats its leader's id on a member row.
    const grid = [{ id: 9 }, { id: 2 }, { id: 7 }, { id: 2 }];
    writeThumbnailBatch(grid, [{ id: 2, thumbnail: "test-thumb-2" }], 0);

    expect(grid[1].thumbnail).toBe("test-thumb-2");
    expect(grid[3].thumbnail).toBeUndefined();
  });

  it("skips a row that was still a placeholder when sliced", () => {
    const placeholder = { id: null, idx: 0 };
    const grid = [placeholder];
    const written = writeThumbnailBatch(
      grid,
      [{ id: null, idx: 0, thumbnail: null, faces: [] }],
      0,
    );

    expect(written).toEqual([]);
    expect(grid[0]).toBe(placeholder);
  });

  it("merges only the batch's fields into the card as it is now", () => {
    // Rated while the batch was out; the batch's copy predates the click.
    const grid = [{ id: 1, rating: 4, thumbnail: "test-thumb-old" }];
    writeThumbnailBatch(
      grid,
      [{ id: 1, rating: 2, thumbnail: null, penalised_tags: ["test-tag"] }],
      0,
    );

    expect(grid[0].rating).toBe(4);
    expect(grid[0].penalised_tags).toEqual(["test-tag"]);
    // No URL in the answer: the card keeps the one it has.
    expect(grid[0].thumbnail).toBe("test-thumb-old");
    expect(grid[0].score).toBe(0);
  });
});

describe("the slots a thumbnail batch counts as loaded", () => {
  it("leaves out a slot that was a placeholder when sliced", () => {
    const grid = [{ id: 1 }, { id: null }, { id: 3 }, { id: 4 }];
    const batch = grid.map((row) => ({ ...row }));

    expect(loadedSlotRuns(grid, batch, 0)).toEqual([
      [0, 1],
      [2, 4],
    ]);
  });

  it("leaves out a placeholder that has since been filled", () => {
    const grid = [{ id: 1 }, { id: 2 }];

    expect(loadedSlotRuns(grid, [{ id: 1 }, { id: null }], 0)).toEqual([
      [0, 1],
    ]);
  });

  it("leaves out a slot whose picture moved while the batch was out", () => {
    // Picture 1 was removed above, so 2 and 3 each moved up a slot.
    const grid = [{ id: 2 }, { id: 3 }];
    const batch = [{ id: 1 }, { id: 2 }, { id: 3 }];

    expect(loadedSlotRuns(grid, batch, 0)).toEqual([]);
  });

  it("counts from the slot the batch was sliced from", () => {
    const grid = pictures(5);

    expect(loadedSlotRuns(grid, grid.slice(2, 4), 2)).toEqual([[2, 4]]);
  });
});
