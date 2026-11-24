// Keep cover only over a selection of any size: one gesture, several requests.
//
// The server takes at most 2000 ids per call (`MAX_SELECTION_IDS` in
// `pixlstash/services/keep_cover_only_service.py`), and a collapse holds the
// library's single database writer for the whole call. A Ctrl+A over a stacked
// library names as many stacks as the library holds, so the selection is split
// into requests of at most `KEEP_COVER_ONLY_STACKS_PER_REQUEST` stacks. Three
// rules make the split invisible to the user:
//
//   * **The preview and the collapse use the same partition.** Request N of the
//     collapse sends the `picture_ids_moving` of preview N as its
//     `expected_picture_ids`, so each request acts on exactly what one preview
//     read described, and the server refuses (409, nothing written) a request
//     whose plan grew after the user read the figure.
//   * **Every collapse request carries the gesture's one batch id**, so however
//     many requests it took, the gesture is one history step and one Ctrl+Z.
//   * **Nothing is silently dropped.** The reports are merged field by field
//     into one report of the same shape, and a request that fails partway
//     raises `KeepCoverOnlyIncompleteError` carrying what already happened, so
//     the caller can say "500 of 1,200 done" instead of "it failed".
//
// See docs/design/keep-cover-only.md, "Selections larger than one request".

import { keepCoverOnly, previewKeepCoverOnly } from "../api/stacks";

/**
 * Stacks per request, for the preview and the collapse alike.
 *
 * Under the server's 2000-id cap with room to spare, and small enough that one
 * request holds the database writer for seconds, not the better part of the
 * client's 60 s request timeout. `tests/test_keep_cover_only.py` fails the
 * build if this ever exceeds the server's cap.
 */
export const KEEP_COVER_ONLY_STACKS_PER_REQUEST = 500;

// How each field of a report merges across requests. Every field the server
// returns is classified here, once: `tests/test_keep_cover_only.py` parses these
// arrays and fails the build when the response models gain a field nobody
// classified, because an unclassified field would vanish from a merged report
// without a sound.
//
//   counts   - per-selection integers: summed.
//   lists    - per-selection arrays: concatenated in request order.
//   settings - describe the library or the request, not the selection (the
//              retention window, the mode): the last report's value. Summing
//              `scrapheap_retention_days` would turn 30 days into 90.

/** Summed preview fields. */
export const KEEP_COVER_ONLY_PREVIEW_COUNTS = Object.freeze([
  "stacks_selected",
  "stacks_eligible",
  "stacks_skipped_locked",
  "stacks_skipped_character_on_copy",
  "stacks_skipped_single_member",
  "stacks_skipped_nothing_reproducible",
  "pictures_moving",
  "covers_kept",
  "covers_gaining_tags",
  "covers_gaining_score",
  "covers_gaining_metadata",
  "reference_folder_pictures_moving",
  "bytes_held_by_copies",
  "originals_deleted_from_disk",
  "pictures_staying_no_recipe",
  "pictures_staying_model_missing",
  "pictures_staying_no_thumbnail",
  "pictures_staying_ghost_not_kept",
]);

/** Concatenated preview fields. */
export const KEEP_COVER_ONLY_PREVIEW_LISTS = Object.freeze([
  "picture_ids_moving",
  "cover_picture_ids",
  "reference_folder_picture_ids_moving",
  "unknown_stack_ids",
  "stacks",
]);

/** Preview fields that describe the library or the request: last one wins. */
export const KEEP_COVER_ONLY_PREVIEW_SETTINGS = Object.freeze([
  "scrapheap_retention_days",
  "keep_recipes",
  "ghost_retention",
]);

/** Summed collapse-result fields. */
export const KEEP_COVER_ONLY_RESULT_COUNTS = Object.freeze([
  "stacks_collapsed",
  "pictures_moved",
  "covers_gaining_metadata",
  "tags_added",
  "scores_lifted",
  "reference_folder_pictures_moved",
  "originals_deleted_from_disk",
  "pictures_staying",
]);

/** Concatenated collapse-result fields. */
export const KEEP_COVER_ONLY_RESULT_LISTS = Object.freeze([
  "stack_ids_collapsed",
  "picture_ids_moved",
  "cover_picture_ids",
  "stacks_skipped_locked",
  "stacks_skipped_character_on_copy",
  "stacks_skipped_single_member",
  "stacks_skipped_nothing_reproducible",
  "unknown_stack_ids",
]);

/** Collapse-result fields that describe the request: last one wins. */
export const KEEP_COVER_ONLY_RESULT_SETTINGS = Object.freeze([
  "status",
  "keep_recipes",
  "ghost_retention",
  "batch_id",
]);

const PREVIEW_FIELDS = Object.freeze({
  counts: KEEP_COVER_ONLY_PREVIEW_COUNTS,
  lists: KEEP_COVER_ONLY_PREVIEW_LISTS,
  settings: KEEP_COVER_ONLY_PREVIEW_SETTINGS,
});

const RESULT_FIELDS = Object.freeze({
  counts: KEEP_COVER_ONLY_RESULT_COUNTS,
  lists: KEEP_COVER_ONLY_RESULT_LISTS,
  settings: KEEP_COVER_ONLY_RESULT_SETTINGS,
});

/**
 * A collapse that stopped partway through a multi-request selection.
 *
 * Earlier requests committed and are in the history under the gesture's batch
 * id; the failed one did not, unless its answer was lost on the way back
 * (`outcomeUnknown`), in which case nobody can say from here.
 */
export class KeepCoverOnlyIncompleteError extends Error {
  /**
   * @param {Object} details
   * @param {Object} details.result - the merged result of every request that
   *   answered, in the shape of one collapse response.
   * @param {number} details.stacksDone - stacks in the requests that answered.
   * @param {number} details.stacksTotal - stacks in the whole selection.
   * @param {number} details.operations - history steps those requests recorded.
   * @param {*} details.cause - the failed request's rejection.
   */
  constructor({ result, stacksDone, stacksTotal, operations, cause }) {
    super(
      `Keep cover only stopped after ${stacksDone} of ${stacksTotal} stacks`,
      { cause },
    );
    this.name = "KeepCoverOnlyIncompleteError";
    this.result = result;
    this.stacksDone = stacksDone;
    this.stacksTotal = stacksTotal;
    this.operations = operations;
    // No HTTP response means the request may still have reached the server and
    // committed (a timeout, a dropped connection): the grid must re-read what
    // is there rather than trust `result`.
    this.outcomeUnknown = !cause?.response;
  }
}

/**
 * Split a selection into the stack-id lists of its requests.
 *
 * Deduplicated in first-seen order, with anything that is not a positive
 * integer dropped, so a repeated or malformed id can neither cost a request nor
 * be counted twice.
 *
 * @param {Array<number|string>} stackIds
 * @param {number} [size=KEEP_COVER_ONLY_STACKS_PER_REQUEST]
 * @returns {number[][]} one list per request; `[]` for an empty selection.
 */
export function partitionKeepCoverOnlyStacks(
  stackIds,
  size = KEEP_COVER_ONLY_STACKS_PER_REQUEST,
) {
  const step = Math.floor(Number(size));
  if (!Number.isFinite(step) || step < 1) {
    throw new RangeError(`Request size must be a positive integer, got ${size}`);
  }
  const unique = [
    ...new Set(
      (Array.isArray(stackIds) ? stackIds : [])
        .map(Number)
        .filter((id) => Number.isInteger(id) && id > 0),
    ),
  ];
  const requests = [];
  for (let i = 0; i < unique.length; i += step) {
    requests.push(unique.slice(i, i + step));
  }
  return requests;
}

/**
 * Merge several reports of one shape into one.
 *
 * A field is present in the merge only when some report carried it, so a
 * merged report never claims a figure the server did not send.
 *
 * @param {Object[]} reports - in request order.
 * @param {{counts: string[], lists: string[], settings: string[]}} fields
 * @returns {Object}
 */
function mergeReports(reports, { counts, lists, settings }) {
  const merged = {};
  for (const key of counts) {
    if (!reports.some((report) => key in report)) continue;
    merged[key] = reports.reduce(
      (sum, report) => sum + (Number(report[key]) || 0),
      0,
    );
  }
  for (const key of lists) {
    if (!reports.some((report) => key in report)) continue;
    merged[key] = reports.flatMap((report) =>
      Array.isArray(report[key]) ? report[key] : [],
    );
  }
  for (const key of settings) {
    const carrier = reports.findLast((report) => key in report);
    if (carrier) merged[key] = carrier[key];
  }
  return merged;
}

/**
 * Merge preview reports into the one report the dialog renders.
 * @param {Object[]} reports
 * @returns {Object}
 */
export function mergeKeepCoverOnlyPreviews(reports) {
  return mergeReports(reports, PREVIEW_FIELDS);
}

/**
 * Merge collapse results into one result of the same shape.
 * @param {Object[]} results
 * @returns {Object}
 */
export function mergeKeepCoverOnlyResults(results) {
  return mergeReports(results, RESULT_FIELDS);
}

/**
 * History steps a list of collapse responses recorded.
 *
 * A request that collapsed nothing (every stack in it locked, say) records no
 * operation, so the count of steps is not the count of requests.
 *
 * @param {Object[]} results
 * @returns {number}
 */
function operationsRecorded(results) {
  return results.filter((result) => Number(result.stacks_collapsed) > 0).length;
}

/**
 * Preview a selection of any size.
 *
 * Requests go one at a time: the server serves them from one database, so
 * firing them together would only queue them there, and a sequential walk can
 * stop at the first abort.
 *
 * @param {Array<number|string>} stackIds
 * @param {Object} [options]
 * @param {AbortSignal} [options.signal] - abandons the walk, including the
 *   request in flight; rejects with the signal's reason.
 * @returns {Promise<{report: Object, requests: Array<{stackIds: number[],
 *   expectedPictureIds: number[]}>}>} the merged report the dialog renders, and
 *   the requests a confirm must send, each holding its own preview's moving ids.
 */
export async function previewKeepCoverOnlySelection(stackIds, { signal } = {}) {
  const partition = partitionKeepCoverOnlyStacks(stackIds);
  const reports = [];
  const requests = [];
  for (const ids of partition) {
    signal?.throwIfAborted();
    const report = await previewKeepCoverOnly({ stackIds: ids, signal });
    reports.push(report ?? {});
    requests.push({
      stackIds: ids,
      expectedPictureIds: Array.isArray(report?.picture_ids_moving)
        ? report.picture_ids_moving
        : [],
    });
  }
  return { report: mergeKeepCoverOnlyPreviews(reports), requests };
}

/**
 * Collapse a previewed selection, request by request, as one gesture.
 *
 * Never abortable: a collapse request that has been sent may commit whether or
 * not anybody waits for its answer, so the walk always runs to its end or to
 * its first failure.
 *
 * @param {Array<{stackIds: number[], expectedPictureIds: number[]}>} requests -
 *   from {@link previewKeepCoverOnlySelection}, unchanged.
 * @param {Object} options
 * @param {string} options.batchId - the gesture's `cli-…` batch id, sent with
 *   every request.
 * @param {Function} [options.onProgress] - called before the walk and after
 *   each request with `{ stacksDone, stacksTotal }`.
 * @returns {Promise<{result: Object, operations: number}>} the merged result,
 *   and the history steps the gesture recorded under its batch id: the receipt
 *   narrates one of them and says when there were more.
 * @throws {KeepCoverOnlyIncompleteError} when a request fails; it carries what
 *   the requests before it did.
 */
export async function keepCoverOnlySelection(
  requests,
  { batchId, onProgress } = {},
) {
  if (!batchId) {
    // Without one the server mints a batch per request and the gesture
    // becomes several undo steps, which is the failure this module exists to
    // prevent.
    throw new TypeError("keepCoverOnlySelection needs the gesture's batch id");
  }
  const stacksTotal = requests.reduce((n, r) => n + r.stackIds.length, 0);
  const results = [];
  let stacksDone = 0;
  onProgress?.({ stacksDone, stacksTotal });
  for (const request of requests) {
    try {
      results.push(
        (await keepCoverOnly({
          stackIds: request.stackIds,
          batchId,
          expectedPictureIds: request.expectedPictureIds,
        })) ?? {},
      );
    } catch (cause) {
      throw new KeepCoverOnlyIncompleteError({
        result: mergeKeepCoverOnlyResults(results),
        stacksDone,
        stacksTotal,
        operations: operationsRecorded(results),
        cause,
      });
    }
    stacksDone += request.stackIds.length;
    onProgress?.({ stacksDone, stacksTotal });
  }
  return {
    result: mergeKeepCoverOnlyResults(results),
    operations: operationsRecorded(results),
  };
}
