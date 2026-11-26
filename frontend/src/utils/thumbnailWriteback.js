// Writing a thumbnail batch back into the grid.
//
// A batch is sliced out of the grid, then awaits the network. While it is out
// the grid can be re-ordered without the batch knowing (a smart-score
// reposition moves a card that just gained a penalised tag), so a row is
// written back to the slot that holds its picture now, not to the slot it was
// sliced from.

// The card fields a thumbnail batch is the authority for; everything else on a
// card belongs to whoever last wrote it.
export const THUMBNAIL_BATCH_FIELDS = [
  "thumbnail",
  "thumbnail_width",
  "thumbnail_height",
  "square_crop_x",
  "square_crop_y",
  "square_crop_side",
  "faces",
  "detections",
  "penalised_tags",
];

/**
 * Map each picture id in `grid` to its slot. First occurrence wins: an
 * expanded stack can repeat a picture id, and the leader row is the one a
 * positional write would have targeted.
 *
 * @param {Array<{id?: unknown}|undefined>} grid
 * @returns {Map<string, number>}
 */
function slotsById(grid) {
  const slots = new Map();
  for (let i = 0; i < grid.length; i += 1) {
    const id = grid[i]?.id;
    if (id == null) continue;
    const key = String(id);
    if (!slots.has(key)) slots.set(key, i);
  }
  return slots;
}

/**
 * Write a thumbnail batch's rows into `grid`, in place, by picture identity.
 *
 * Row `i` of `batch` was sliced from slot `start + i`. It goes back there when
 * that slot still holds its picture (the common case, which never looks past
 * the batch's own window). Otherwise it goes to the slot its picture holds now,
 * found through an id -> slot map built once, on the first row that moved; a
 * picture no longer in the grid is skipped. A row without an id is skipped too:
 * it was a placeholder when sliced, and the slot may hold a loaded row by now.
 *
 * Each written card is the card as it is now with the batch's own fields
 * merged over it, so a change made while the batch was out (a rating, a tag, a
 * stack count) is kept. A row that came back without a thumbnail URL keeps the
 * card's, and the card keeps its score.
 *
 * @param {Array<object|undefined>} grid The grid's rows, written in place.
 * @param {Array<object>} batch The batch's rows, in slice order.
 * @param {number} start The slot the batch was sliced from.
 * @returns {Array<{row: object, slot: number}>} Each row written, with its slot.
 */
export function writeThumbnailBatch(grid, batch, start) {
  let slots = null;
  const written = [];
  for (let i = 0; i < batch.length; i += 1) {
    const row = batch[i];
    if (row.id == null) continue;
    let slot = start + i;
    if (String(grid[slot]?.id) !== String(row.id)) {
      slots ??= slotsById(grid);
      const moved = slots.get(String(row.id));
      // The picture left the grid (filtered out, or collapsed into a stack).
      if (moved === undefined) continue;
      slot = moved;
    }
    const current = grid[slot];
    const merged = { ...current, idx: slot };
    for (const field of THUMBNAIL_BATCH_FIELDS) merged[field] = row[field];
    merged.thumbnail = row.thumbnail ?? current?.thumbnail ?? null;
    merged.score = current?.score ?? 0;
    grid[slot] = merged;
    written.push({ row, slot });
  }
  return written;
}

/**
 * The slots a thumbnail batch can count as loaded, as `[start, end)` runs.
 *
 * Slot `start + i` counts only when row `i` of `batch` was a picture (a
 * placeholder has no id, so nothing was asked for it) and the slot still holds
 * that picture (a removal above can move the rows while the batch is out).
 * Counting a placeholder's slot made the fetch for the picture that later
 * landed there look done already, and its card stayed blank for good.
 *
 * @param {Array<{id?: unknown}|undefined>} grid The grid's rows now.
 * @param {Array<{id?: unknown}>} batch The batch's rows, in slice order.
 * @param {number} start The slot the batch was sliced from.
 * @returns {Array<[number, number]>} The runs, in slot order.
 */
export function loadedSlotRuns(grid, batch, start) {
  const runs = [];
  let runStart = null;
  for (let i = 0; i < batch.length; i += 1) {
    const id = batch[i]?.id;
    const held = id != null && String(grid[start + i]?.id) === String(id);
    if (held) {
      runStart ??= start + i;
    } else if (runStart != null) {
      runs.push([runStart, start + i]);
      runStart = null;
    }
  }
  if (runStart != null) runs.push([runStart, start + batch.length]);
  return runs;
}
