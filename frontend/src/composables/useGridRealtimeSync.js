// useGridRealtimeSync - owns the WebSocket → grid update decision table.
//
// App.vue keeps only the socket lifecycle (connect / reconnect / close /
// set_filters). Every incoming message is handed to `handleMessage(payload)`
// here, which decides between:
//   - suppressing an echo of this tab's own optimistic op,
//   - a targeted in-place op (insert / refresh / reposition / remove),
//   - raising one of the two pills (new pictures / sort changed externally),
//   - or, rarely, a logged full reload.
//
// The DECISION is synchronous and returns a `{ action, reason }` descriptor.
// The grid SIDE-EFFECTS it implies (insert / per-id refresh / remove / pill)
// are not applied inline: they are accumulated into per-category buffers and
// flushed once per short coalescing window (see the coalescer below). A burst
// of N foreign events therefore collapses into one batched insert, one batched
// per-id refresh pass, and one batched remove - not N fetch+rebuild cycles.
//
// All dependencies (stores, grid imperative API, predicates, scheduler, logger)
// are injected so the decision table can be unit-tested without a real grid or a
// live Pinia instance. The default `schedule` flushes synchronously so unit
// tests see side-effects immediately; App.vue passes a real debounce.

import { getCurrentScope, onScopeDispose, watch } from "vue";
import { onSessionReset } from "../utils/apiClient";

const TARGETED = "targeted";
const PILL = "pill";
const SUPPRESSED = "suppressed";
const RELOAD = "reload";
const DEFERRED = "deferred";
const IGNORED = "ignored";

// Above this many ids, the per-id targeted-update loop (one /metadata +
// thumbnail fetch each) becomes a fetch storm. A foreign-tab "set project on
// 400 pictures" or "apply scores" lands here; doing one reload (or deferring it
// under an open lightbox) is cheaper than hundreds of single-card refreshes.
// Applied to the COALESCED batch, not to a single event - a window that
// accumulates >50 distinct ids escalates to one reload.
const MAX_TARGETED_UPDATE = 50;

// Server-computed sort fields whose value the originating tab can only guess
// optimistically; even an own-origin echo for these needs a single-card
// reconcile under the matching sort so the optimistic guess can't diverge from
// server truth.
const SERVER_COMPUTED_SORT_FIELDS = new Set([
  "smart_score",
  "character_likeness",
]);

// "Card-content" fields change how a card *renders* (its overlays / icons) but
// never its sort or filter position - by construction they are not
// view-affecting, so they must never reshuffle the grid or raise a pill. The
// card still needs a targeted in-place metadata refresh so the new content shows
// without a full reload. `detections` is the canonical case: the grid draws
// object boxes from `img.detections` (ImageGrid.getDetectionBboxOverlays, gated
// on the `showDetections` prop), and a user-triggered Segment completes the same
// way for every tab regardless of who started it.
// `pixels` belongs here for the same reason: an in-place rotate (or the undo of
// one) rewrites the file and therefore the card's thumbnail, while its sort and
// filter position are untouched - a turned photo does not move in the grid. It
// needs one thing `detections` does not, handled at the branch below: the
// thumbnail URL lives on the batch-thumbnail endpoint, not on /metadata, so a
// card refresh alone repaints the pre-rotate bitmap.
// `faces` (Detect faces again) is the same case for the face boxes.
const CARD_CONTENT_FIELDS = new Set([
  "detections",
  "faces",
  "pixels",
  "orientation",
]);

// Stack membership. `stack_count` is the number of LIVE members a card's stack
// badge renders, and it is the one facet neither of the branches above can
// repair: it is derived per stack by the listing endpoint rather than stored on
// the picture, so it is absent from the `/pictures/{id}/metadata` read that
// `refreshGridImage` performs. A "Keep cover only" that moves four of five
// members changes exactly this and nothing else about the cover, which is how a
// collapsed stack kept rendering "5" around a picture that was on its own.
//
// Normally it is not a sort field, so it gets a targeted facet refresh for
// every origin. The one exception is STACK_UPDATED_AT: the membership edit that
// produced this event also advanced the stack timestamp, so that active sort
// requires a grid reorder rather than a badge-only refresh.
const STACK_FACET_FIELDS = new Set(["stack_count"]);

function isStackUpdatedSortValue(value) {
  return String(value || "").toUpperCase() === "STACK_UPDATED_AT";
}

function asPictureIds(payload) {
  return Array.isArray(payload?.picture_ids) ? payload.picture_ids : [];
}

function normaliseSource(payload) {
  // `picture_imported`'s source migrates from "user" to "ui"; accept both.
  const source = payload?.source;
  if (source === "ui" || source === "user") return "ui";
  if (source === "external") return "external";
  // No source → treat as external (the conservative default for unattributed
  // background work).
  return "external";
}

// The wire values for `change_kind`, mirroring the backend's
// `WsBroadcasterMixin.CHANGE_KINDS` allowlist. The two are ONE contract: the
// backend silently drops a kind missing from its tuple, and this function
// silently degrades a kind missing from this set to "updated" - which for a
// lifecycle change leaves a stale, 404-clickable card behind. Never add a value
// on one side only.
//
// `restored` is a scrapheap comeback (undo of a move-to-Scrapheap, or
// POST /pictures/scrapheap/restore). It is deliberately NOT `added`: both put a
// card back, but only `added` means "new to the vault", and the sidebar reads
// that difference - it raises its NEW marker for an import and must not for a
// picture that has been in the library all along.
const CHANGE_KINDS = new Set(["added", "updated", "removed", "restored"]);

function resolveChangeKind(payload) {
  const kind = payload?.change_kind;
  if (CHANGE_KINDS.has(kind)) return kind;
  // `picture_imported` is implicitly an addition.
  if (payload?.type === "picture_imported") return "added";
  return "updated";
}

// Default scheduler: run the flush synchronously. Keeps the composable's
// side-effects observable in the same tick for unit tests; App.vue overrides
// this with a real debounce so production coalesces across a window.
function createSyncScheduler() {
  return {
    schedule(flush) {
      flush();
    },
    cancel() {},
  };
}

/**
 * Positions (indices into `values`) of one longest strictly increasing run,
 * in order. Patience sorting, O(n log n).
 *
 * @param {number[]} values
 * @returns {number[]}
 */
function longestIncreasingRun(values) {
  const tails = []; // tails[k]: index of the smallest last value of a run of k + 1
  const previous = new Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    let lo = 0;
    let hi = tails.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (values[tails[mid]] < values[i]) lo = mid + 1;
      else hi = mid;
    }
    previous[i] = lo > 0 ? tails[lo - 1] : -1;
    tails[lo] = i;
  }
  const run = [];
  let k = tails.length ? tails[tails.length - 1] : -1;
  while (k >= 0) {
    run.push(k);
    k = previous[k];
  }
  return run.reverse();
}

/**
 * What a reconnect's gap changed in the grid, recovered from ids: the grid's
 * cards against a fresh read of one contiguous range of the same list
 * (`/pictures/stream?fields=id,stack_id&stack_leaders_only=true`). The server
 * does not replay the events it sent while the socket was down, so this is
 * the only account of them there is.
 *
 * A card is a stack or a lone picture. The grid's list can hold a stack's
 * members as well as its cover (a local reorder writes them all in), and the
 * read lists one row per stack, so the two are compared card by card. An
 * id-list view's read (a search) lists every member, as its load did: the
 * first row of a stack is the card, as the grid's collapse takes it. The
 * Scrapheap is the exception: the server lists every scrapped picture on its
 * own there, stack or not, and so does the grid, so each picture is a card.
 *
 * The cards both hold line the two lists up: the longest run of them still in
 * grid order stays put, and any other card both hold has moved. The first and
 * last of that run bound the stretch of the grid the read covers (or the
 * list's own ends, where the read reached them); a card inside it that the
 * read no longer lists is gone. A listed card the grid does not hold is new,
 * since a grid that has finished loading holds its whole list. A card whose
 * cover changed, or a picture that changed card (it joined or left a stack),
 * counts as moved, never as a removal and an arrival.
 *
 * @param {Object} read - what ImageGrid's `readViewForReconcile` resolves to.
 * @param {Array<{id: *, stackId: *}>} read.before - the grid's base list, in
 *   grid order, when the read went out.
 * @param {Array<{id: *, stackId: *}>} read.after - the same list when the
 *   answer came back.
 * @param {Array<{id: *, stack_id: *}>} read.answer - the server's rows.
 * @param {boolean} read.fromStart - the read began at the list's first row.
 * @param {boolean} read.toEnd - the read reached the list's last row.
 * @param {boolean} read.stacked - the list shows a stack as one card; false
 *   in the Scrapheap, which lists each picture on its own.
 * @param {Array<*>} [read.skipIds] - pictures the grid keeps on purpose
 *   although the server no longer lists them (ghosts of a send to the
 *   Scrapheap that can still be undone).
 * @returns {{gone: Array, added: Array, moved: Array, goneCards: Array}}
 *   picture ids, as the grid or the server gave them. `gone` names every row
 *   of a gone card; `goneCards` holds the same rows card by card, the shown
 *   one first, for `settleGoneCards`.
 */
export function diffViewAgainstIdRead({
  before,
  after,
  answer,
  fromStart,
  toEnd,
  stacked,
  skipIds = [],
}) {
  const key = (id) => String(id);
  const cardKey = (id, stackId) =>
    !stacked || stackId == null
      ? `picture:${key(id)}`
      : `stack:${key(stackId)}`;
  // A picture this tab took out or put in while the read was out is that
  // change's, and the answer may have been read either side of it.
  const beforeIds = new Set(before.map((row) => key(row.id)));
  const afterIds = new Set(after.map((row) => key(row.id)));
  const skip = new Set(skipIds.map(key));
  for (const id of beforeIds) if (!afterIds.has(id)) skip.add(id);
  for (const id of afterIds) if (!beforeIds.has(id)) skip.add(id);

  // The grid's cards in order; a card's first row is the one it shows.
  const cards = [];
  const cardAt = new Map();
  const gridIds = new Set();
  for (const row of after) {
    if (skip.has(key(row.id))) continue;
    gridIds.add(key(row.id));
    const ck = cardKey(row.id, row.stackId);
    const at = cardAt.get(ck);
    if (at === undefined) {
      cardAt.set(ck, cards.length);
      cards.push({ key: ck, id: row.id, ids: [row.id] });
    } else {
      cards[at].ids.push(row.id);
    }
  }
  const listed = [];
  const listedKeys = new Set();
  const listedIds = new Set();
  for (const pic of answer) {
    if (pic?.id == null || skip.has(key(pic.id))) continue;
    const ck = cardKey(pic.id, pic.stack_id);
    listedIds.add(key(pic.id));
    if (listedKeys.has(ck)) continue;
    listedKeys.add(ck);
    listed.push({ key: ck, id: pic.id });
  }

  const common = listed.filter((card) => cardAt.has(card.key));
  const run = longestIncreasingRun(common.map((card) => cardAt.get(card.key)));

  if (!run.length && !(fromStart && toEnd) && cards.length) {
    // Nothing both hold lines the read up with the grid, so where it sits in
    // the grid is unknown and no card can be judged gone: the view changed
    // past what ids can place. Offer the refresh, not a pill of thousands.
    return {
      gone: [],
      added: [],
      moved: listed.length
        ? listed.map((card) => card.id)
        : cards.map((card) => card.id),
      goneCards: [],
    };
  }

  const gone = [];
  const goneCards = [];
  const added = [];
  const moved = [];
  const inRun = new Set(run.map((i) => common[i].key));
  for (const card of common) {
    const shown = cards[cardAt.get(card.key)];
    if (!inRun.has(card.key) || key(shown.id) !== key(card.id)) {
      moved.push(card.id);
    }
  }
  for (const card of listed) {
    if (cardAt.has(card.key)) continue;
    if (gridIds.has(key(card.id))) moved.push(card.id);
    else added.push(card.id);
  }

  const from = fromStart || !run.length ? 0 : cardAt.get(common[run[0]].key);
  const to =
    toEnd || !run.length
      ? cards.length - 1
      : cardAt.get(common[run[run.length - 1]].key);
  for (let i = from; i <= to; i += 1) {
    const card = cards[i];
    if (listedKeys.has(card.key)) continue;
    if (card.ids.some((id) => listedIds.has(key(id)))) moved.push(card.id);
    else {
      gone.push(...card.ids);
      goneCards.push(card.ids);
    }
  }
  return { gone, added, moved: uniqueIds(moved), goneCards };
}

// A picture that joined a stack as its cover is moved twice over: once as the
// stack's new face, once as the lone card it no longer is.
function uniqueIds(ids) {
  return [...new Map(ids.map((id) => [String(id), id])).values()];
}

/**
 * Settle the cards a reconnect read called gone when that read covered only
 * part of a long view. Inside the stretch it covered, a card it does not list
 * has either left the view or moved out of that stretch, and ids alone cannot
 * tell which: a second read of the same view, narrowed to those cards' ids,
 * can. A card any of whose pictures it lists is still in the view, so it has
 * moved; the rest are gone.
 *
 * @param {{gone: Array, added: Array, moved: Array, goneCards: Array<Array>}} diff
 *   `diffViewAgainstIdRead`'s answer.
 * @param {Set<string>} listedIds - the ids, as strings, that the narrowed
 *   read lists.
 * @returns {{gone: Array, added: Array, moved: Array}}
 */
export function settleGoneCards({ added, moved, goneCards }, listedIds) {
  const gone = [];
  const settledMoved = [...moved];
  for (const ids of goneCards) {
    if (ids.some((id) => listedIds.has(String(id)))) {
      settledMoved.push(ids[0]);
    } else {
      gone.push(...ids);
    }
  }
  return { gone, added, moved: uniqueIds(settledMoved) };
}

/**
 * @param {Object} deps
 * @param {() => string|null} deps.getMyClientId   Active tab's client id.
 * @param {Object} deps.grid                        Imperative grid API (the ImageGrid template-ref's exposed methods, or spies in tests). Expected: insertGridImagesById, refreshGridImage, refreshStackFacets, repositionImageByScore, repositionImageBySmartScore, removeImagesById, isImagesLoading.
 * @param {Object} deps.wsStore                     useWsStore instance (pill ids + setters).
 * @param {(fields: string[]) => boolean} deps.pictureChangeAffectsView
 * @param {() => string} deps.getSelectedSort       Active sort key (e.g. "SMART_SCORE", "DATE_TAKEN").
 * @param {Object} [deps.logger]                    console-like; defaults to console.
 * @param {() => void} [deps.reload]                Full-reload fallback.
 * @param {() => void} [deps.refreshSidebar]        Sidebar picture-count refresh.
 * @param {() => boolean} [deps.isScrapheapView]    True while the grid shows the Scrapheap; defaults to false.
 * @param {{schedule: (flush: () => void) => void, cancel: () => void}} [deps.scheduler]  Coalescing scheduler; defaults to synchronous flush (unit tests). App.vue passes a debounce.
 */
export function useGridRealtimeSync(deps) {
  const {
    getMyClientId,
    grid,
    wsStore,
    pictureChangeAffectsView,
    getSelectedSort,
    logger = console,
    reload = () => {},
    refreshSidebar = () => {},
    isScrapheapView = () => false,
    scheduler = createSyncScheduler(),
  } = deps;

  // --- Sidebar refresh held back by an upload ------------------------------
  // While this tab uploads, the import dialog drives the counts and a refresh
  // per imported picture would churn them, so handleMessage holds the refresh
  // back. A change from elsewhere in that time (another tab deletes, or sends
  // pictures to the Scrapheap) still moved the counts, and nothing else replays
  // it: the refresh is owed, and runs once when the upload ends. The value is
  // the flash it owes (any held refresh that would have flashed makes it
  // flash); null when none is owed.
  let owedSidebarFlash = null;
  watch(
    () => wsStore.isUploadInProgress,
    (uploading) => {
      if (uploading || owedSidebarFlash === null) return;
      const flash = owedSidebarFlash;
      owedSidebarFlash = null;
      refreshSidebar(flash);
    },
  );
  // Sign-out also ends an upload (the store drops the flag), but the counts
  // owed then belong to the session that is going away.
  const stopSessionReset = onSessionReset(() => {
    owedSidebarFlash = null;
  });
  if (getCurrentScope()) onScopeDispose(stopSessionReset);

  // --- Coalescing buffers -------------------------------------------------
  // Accumulated across one window, then applied once per category on flush.
  // Ordering correctness: a remove supersedes a prior add/update of the same id
  // (and vice-versa) by mutating these buffers at enqueue time, so an add→remove
  // of the same id inside the window nets out to a remove (and remove→add to an
  // add), never both.
  let addedIds = new Set();
  let restoredIds = new Set(); // scrapheap comebacks - inserted without the flash
  let updatedFields = new Map(); // id -> Set<field>
  let removedIds = new Set();
  let sawRemove = false; // a removal event arrived this window (even empty-id)
  let addPillIds = new Set(); // → "New pictures" pill
  let sortPillIds = new Set(); // → "View changed externally" pill
  let flushScheduled = false;

  function resetBuffers() {
    addedIds = new Set();
    restoredIds = new Set();
    updatedFields = new Map();
    removedIds = new Set();
    sawRemove = false;
    addPillIds = new Set();
    sortPillIds = new Set();
  }

  function ensureFlushScheduled() {
    if (flushScheduled) return;
    flushScheduled = true;
    scheduler.schedule(flushCoalesced);
  }

  function enqueueAdded(ids) {
    for (const id of ids) {
      removedIds.delete(id);
      restoredIds.delete(id);
      addedIds.add(id);
    }
    ensureFlushScheduled();
  }

  // A comeback, not an arrival. Buffered apart from `addedIds` for exactly one
  // reason: the insert must not fire the grid's new-picture highlight. That
  // flash means "this was not here before", and strobing a 40-picture undo with
  // it tells the user the wrong story about pictures they just asked to have
  // back. The confirmation is the receipt, which has already flipped to
  // "Undone: …" with a Redo.
  function enqueueRestored(ids) {
    for (const id of ids) {
      removedIds.delete(id);
      if (addedIds.has(id)) continue; // a genuine add in the same window wins
      restoredIds.add(id);
    }
    ensureFlushScheduled();
  }

  function enqueueUpdated(ids, fields) {
    for (const id of ids) {
      if (removedIds.has(id)) continue; // a pending remove wins
      let set = updatedFields.get(id);
      if (!set) {
        set = new Set();
        updatedFields.set(id, set);
      }
      for (const f of fields) set.add(f);
    }
    ensureFlushScheduled();
  }

  function enqueueRemoved(ids) {
    sawRemove = true;
    for (const id of ids) {
      addedIds.delete(id);
      restoredIds.delete(id);
      updatedFields.delete(id);
      removedIds.add(id);
    }
    ensureFlushScheduled();
  }

  function enqueueAddPill(ids) {
    for (const id of ids) addPillIds.add(id);
    ensureFlushScheduled();
  }

  function enqueueSortPill(ids) {
    for (const id of ids) sortPillIds.add(id);
    ensureFlushScheduled();
  }

  // Apply one window's worth of accumulated intents. Removes run first so a
  // late add/update for an id that was also removed can't resurrect a card the
  // server deleted; the enqueue-time bookkeeping above already keeps the three
  // buffers disjoint, this is just defensive ordering.
  function flushCoalesced() {
    flushScheduled = false;
    const removed = [...removedIds];
    const hadRemove = sawRemove;
    const added = [...addedIds];
    const restored = [...restoredIds];
    const updated = [...updatedFields.entries()];
    const addPill = [...addPillIds];
    const sortPill = [...sortPillIds];
    resetBuffers();

    // Apply removals whenever a removal event arrived, even with no ids: an
    // empty-id removal still signals the grid (matching the un-coalesced
    // contract) and the call is a harmless no-op there.
    if (hadRemove) {
      grid.removeImagesById?.(removed);
    }
    if (added.length) {
      grid.insertGridImagesById?.(added);
    }
    if (restored.length) {
      // Idempotent: ids still mounted as ghosts are already in the grid's base
      // list, so this is a no-op for them and only the ghost flag clears.
      grid.insertGridImagesById?.(restored, { highlight: false });
    }
    // The MAX_TARGETED_UPDATE escalation applies to the COALESCED batch: a
    // window that accreted >50 distinct per-id refreshes is a fetch storm, so
    // collapse it into one reload (or defer under an open lightbox) instead of
    // the per-id loop.
    if (updated.length > MAX_TARGETED_UPDATE) {
      reloadOrDefer("coalesced-updated-too-large");
    } else {
      for (const [id, fields] of updated) {
        applyTargetedUpdate(id, [...fields]);
      }
    }
    if (addPill.length) {
      wsStore.addPendingExternalImportIds?.(addPill);
    }
    if (sortPill.length) {
      wsStore.addSortChangedExternalIds?.(sortPill);
    }
  }

  function isLightboxOpen() {
    return grid.isLightboxOpen?.() === true;
  }

  // While the lightbox is open the grid sequence is frozen for
  // navigation (see Lightbox's frozen filmstrip). A pill would either flash
  // a "sort order changed" / "new pictures" prompt for the user's own in-lightbox
  // edits or reshuffle the grid under them. Instead, flag a deferred in-place
  // reconcile that ImageGrid.closeLightbox() runs on close (refetch → re-filter +
  // re-sort), and raise no pill. Returns true when the change was deferred.
  function deferWhileLightboxOpen() {
    if (!isLightboxOpen()) return false;
    grid.markLightboxDeferredRefresh?.();
    return true;
  }

  // A change we can't (or shouldn't) apply with a targeted op: an untargetable
  // empty-id add/update, or a batch too large for the per-id loop. Reload the
  // grid, or defer that reload to lightbox close so we never reshuffle the grid
  // under the lightbox. Returns a `{ action, reason }` descriptor.
  function reloadOrDefer(reason) {
    if (deferWhileLightboxOpen()) {
      logger.warn?.(
        `useGridRealtimeSync: deferring grid reload under open lightbox (${reason})`,
      );
      return { action: DEFERRED, reason: `${reason}-lightbox-deferred` };
    }
    logger.warn?.(`useGridRealtimeSync: full grid reload (${reason})`);
    reload();
    return { action: RELOAD, reason };
  }

  function isSmartScoreSort() {
    return String(getSelectedSort() || "").includes("SMART_SCORE");
  }

  function isLikenessSort() {
    const sort = String(getSelectedSort() || "");
    return sort.includes("CHARACTER_LIKENESS") || sort.includes("LIKENESS");
  }

  // True when one of the changed fields is a server-computed sort field that is
  // also the active sort - the only case where an own-origin echo still needs a
  // single-card reconcile.
  function fieldsAreActiveServerSort(fields) {
    if (!Array.isArray(fields) || !fields.length) return false;
    return fields.some((field) => {
      if (!SERVER_COMPUTED_SORT_FIELDS.has(field)) return false;
      if (field === "smart_score") return isSmartScoreSort();
      if (field === "character_likeness") return isLikenessSort();
      return false;
    });
  }

  // True when every changed field is a card-content-only field (and there is at
  // least one). The CARD_CONTENT_FIELDS are, by construction, not
  // view-affecting, so this can be handled uniformly for all origins with a
  // targeted in-place refresh.
  //
  // `file_path` joins them only beside `pixels`, and only where the view does
  // not select on it (anywhere but a folder view): a layout move is the same
  // card with a new thumbnail URL, which only the `pixels` branch re-reads. In
  // a folder view the move can take the card out of the view, so it keeps the
  // ordinary dispatch.
  function fieldsAreCardContentOnly(fields) {
    if (!Array.isArray(fields) || !fields.length) return false;
    const movedFileRepaints =
      fields.includes("pixels") && !pictureChangeAffectsView(["file_path"]);
    return fields.every(
      (field) =>
        CARD_CONTENT_FIELDS.has(field) ||
        (field === "file_path" && movedFileRepaints),
    );
  }

  // True when every changed field is a stack facet (and there is at least one).
  // Mixed fields deliberately fall through to the ordinary dispatch: a cover
  // that also gained a score needs the sort treatment its own announcement
  // carries, which is why the backend emits the stack change as an event of its
  // own rather than widening the metadata one.
  function fieldsAreStackFacetsOnly(fields) {
    if (!Array.isArray(fields) || !fields.length) return false;
    return fields.every((field) => STACK_FACET_FIELDS.has(field));
  }

  // Reconcile a single card under the active sort. The WS event never carries
  // the new server-computed value, so for smart-score we fetch-then-reposition
  // (refreshSmartScoreForImage does both with the true value); otherwise we
  // just refresh the card's metadata in place.
  function reconcileServerSortField(id, fields) {
    if (fields.includes("smart_score") && isSmartScoreSort()) {
      grid.refreshSmartScoreForImage?.(id);
      return;
    }
    grid.refreshGridImage?.(id);
  }

  function applyTargetedUpdate(id, fields) {
    if (fields.includes("smart_score") && isSmartScoreSort()) {
      // Fetch the true smart_score and reposition off it.
      grid.refreshSmartScoreForImage?.(id);
      return;
    }
    // The picture's own BYTES changed - an in-place rotate, or an undo/redo of
    // one arriving over the socket. The card is the same card, but its thumbnail
    // URL has to be re-read, and `refreshGridImage` cannot do it: the metadata
    // endpoint carries no thumbnail URL, so on its own it repaints the
    // pre-rotate bitmap. Same reasoning as `stack_count` - a listing-only value
    // a per-card metadata read cannot repair.
    if (fields.includes("pixels")) {
      void grid.applyRotatedCards?.([id]);
      return;
    }
    // Any other change (incl. a score change): refresh the card's metadata in
    // place. refreshGridImage re-fetches so the new value is reflected.
    grid.refreshGridImage?.(id);
  }

  // --- Scrapheap comeback (any origin) ------------------------------------
  // A `restored` event puts cards BACK. The insert is idempotent by
  // construction: `insertGridImagesById` skips ids already in
  // `lastFetchedGridImages`, so a tile still mounted as a ghost (its undo
  // receipt was live when the undo landed) is left exactly where it is and only
  // the ghost flag clears, while a tile whose ghost window had already elapsed -
  // or that was never ghosted, because the undo came from the toolbar, the
  // lightbox or Ctrl+Z long after the fact - is fetched and re-inserted at its
  // sorted position. One path, both cases, no refetch flash for the live one.
  /**
   * Put newly-imported pictures into the grid, for own-origin and foreign-ui
   * alike. Shared because an own-origin `added` is NOT a true echo: see
   * `handleOwnOrigin`.
   */
  function applyAdded(originLabel, pictureIds) {
    if (deferWhileLightboxOpen()) {
      return {
        action: TARGETED,
        reason: `${originLabel}-added-lightbox-deferred`,
      };
    }
    if (grid.isImagesLoading?.()) {
      // Streaming fetch owns allGridImages; defer to the pill.
      enqueueAddPill(pictureIds);
      return { action: PILL, reason: `${originLabel}-added-during-load` };
    }
    enqueueAdded(pictureIds);
    return { action: TARGETED, reason: `${originLabel}-added` };
  }

  function applyRestored(originLabel, pictureIds) {
    if (!pictureIds.length) {
      // "Restore everything" broadcasts no ids (see the restore endpoint), so
      // there is nothing to target; the per-id ops below would be silent no-ops
      // and the comeback would never show.
      return reloadOrDefer(`${originLabel}-restored-untargetable-empty-ids`);
    }
    // In the Scrapheap a restored picture has LEFT the view, so its card goes.
    // The insert below would be a no-op here (the id is still in the base
    // list), which is what kept a restore run's cards on screen until it ended.
    if (isScrapheapView()) {
      enqueueRemoved(pictureIds);
      return {
        action: TARGETED,
        reason: `${originLabel}-restored-left-scrapheap`,
      };
    }
    if (deferWhileLightboxOpen()) {
      // Same contract as an add (§9.1): never restructure the grid under the
      // frozen filmstrip. closeLightbox() reconciles.
      return {
        action: TARGETED,
        reason: `${originLabel}-restored-lightbox-deferred`,
      };
    }
    if (grid.isImagesLoading?.()) {
      // A streaming fetch owns `allGridImages` wholesale and an insert into it
      // would be clobbered. Raise the "view changed" pill rather than dropping
      // the comeback - deliberately NOT the "new pictures" pill, whose copy
      // would call a restored picture new.
      enqueueSortPill(pictureIds);
      return { action: PILL, reason: `${originLabel}-restored-during-load` };
    }
    enqueueRestored(pictureIds);
    return { action: TARGETED, reason: `${originLabel}-restored` };
  }

  // --- Echo of this tab's own optimistic op -------------------------------
  function handleOwnOrigin(payload, changeKind, fields, pictureIds) {
    // A Scrapheap run's events (`run_id` on the message) follow
    // no optimistic op: the run works on the server, slice by slice, and
    // nothing in this tab applied any of it. So none of them is an echo, and
    // each is applied as another tab's would be, as it arrives.
    if (payload?.run_id) {
      return handleForeignUi(payload, changeKind, fields, pictureIds);
    }
    // The one own-origin echo that must NOT be suppressed. Suppression assumes
    // an optimistic local op already applied the change, and for a scrapheap
    // undo there may be none: the tiles may have collapsed when the receipt
    // expired, or the undo may have come from the toolbar / Ctrl+Z / the
    // lightbox with no grid op at all. Suppressing it is what left the grid
    // showing the pre-undo state after an undo (the reported bug).
    if (changeKind === "restored") {
      return applyRestored("own-origin", pictureIds);
    }
    if (changeKind === "updated" && fieldsAreActiveServerSort(fields)) {
      // Optimistic guess for a server-computed sort field can diverge from
      // server truth - reconcile each card, never reload. Cap the per-id
      // fetch loop: a large own-origin batch reconcile becomes a fetch storm.
      // (Reconcile is its own immediate per-id pass, not the coalesced updated
      // buffer, because it dispatches refreshSmartScoreForImage rather than the
      // generic per-id refresh.)
      if (pictureIds.length > MAX_TARGETED_UPDATE) {
        return reloadOrDefer("own-origin-server-sort-reconcile-too-large");
      }
      for (const id of pictureIds) reconcileServerSortField(id, fields);
      return { action: TARGETED, reason: "own-origin-server-sort-reconcile" };
    }
    // The second own-origin echo that must NOT be suppressed, and for the same
    // reason as `restored` above: suppression assumes an optimistic local op
    // already applied the change, and for an add there cannot have been one.
    // The grid cannot insert a picture whose id the *server* assigns on commit.
    // A paste, or a drop outside the grid, goes through the sidebar importer,
    // which deliberately reports no per-file results because "the grid
    // refreshes off the WS broadcast" (ImageImporter.vue). Suppressing that
    // broadcast is exactly what left a pasted picture invisible until the view
    // was switched away and back.
    if (changeKind === "added") {
      return applyAdded("own-origin", pictureIds);
    }
    // A removal from the Scrapheap is a purge, and no purge removes a tile
    // optimistically: a run settles by re-reading the heap, and a request that
    // timed out after the server committed has no local step at all. So this
    // echo is the only thing that takes the destroyed pictures off the screen
    // as the server confirms them. Idempotent: an id no longer in the grid is a
    // no-op. Outside the Scrapheap the echo stays suppressed, because there it
    // follows a soft delete whose tiles the grid holds ghosted for the undo
    // window, and removing them would take the one-click undo with them.
    if (changeKind === "removed" && isScrapheapView()) {
      enqueueRemoved(pictureIds);
      return { action: TARGETED, reason: "own-origin-removed-scrapheap" };
    }
    return { action: SUPPRESSED, reason: "own-origin-echo" };
  }

  // --- Owner UI change from a different tab -------------------------------
  function handleForeignUi(payload, changeKind, fields, pictureIds) {
    if (changeKind === "removed") {
      enqueueRemoved(pictureIds);
      return { action: TARGETED, reason: "foreign-ui-removed" };
    }
    // Another owner tab undid a scrapheap move (or hit Restore). Targeted, like
    // its `added` sibling - but never the "new pictures" pill, whose copy would
    // call a picture that has been here all along new.
    if (changeKind === "restored") {
      return applyRestored("foreign-ui", pictureIds);
    }
    // An empty-id add/update can't be targeted (e.g. restore-all broadcasts
    // change_kind:"added" with picture_ids:[]). The per-id ops below would be
    // silent no-ops, so other owner tabs would never reflect the change. Reload
    // (or defer under an open lightbox) rather than no-op.
    if (!pictureIds.length) {
      return reloadOrDefer("foreign-ui-untargetable-empty-ids");
    }
    if (changeKind === "added") {
      return applyAdded("foreign-ui", pictureIds);
    }
    // updated
    if (!pictureChangeAffectsView(fields)) {
      return { action: IGNORED, reason: "foreign-ui-updated-irrelevant" };
    }
    // A single event already over the cap is a fetch storm on its own (foreign
    // apply-scores / set-project on hundreds of pictures): escalate immediately
    // to one reload rather than queuing hundreds of per-id refreshes.
    if (pictureIds.length > MAX_TARGETED_UPDATE) {
      return reloadOrDefer("foreign-ui-updated-too-large");
    }
    // Otherwise queue the per-id refreshes. Each id becomes one /metadata +
    // thumbnail fetch on flush; the COALESCED batch is also capped at
    // MAX_TARGETED_UPDATE (see flushCoalesced), so many sub-cap events whose
    // distinct ids sum past 50 still escalate to a single reload there.
    enqueueUpdated(pictureIds, fields);
    return { action: TARGETED, reason: "foreign-ui-updated" };
  }

  // --- Change from outside the UI -----------------------------------------
  function handleExternal(payload, changeKind, fields, pictureIds) {
    if (changeKind === "removed") {
      // Never leave a stale 404-clickable card; remove silently.
      enqueueRemoved(pictureIds);
      return { action: TARGETED, reason: "external-removed" };
    }
    // A restore from outside this UI never auto-inserts under the user - the
    // external contract. It takes the "View changed externally" pill and NOT
    // the "New pictures" one: the pictures are coming back, not arriving, and
    // that pill's own copy would say otherwise.
    if (changeKind === "restored") {
      if (!pictureIds.length) {
        return reloadOrDefer("external-restored-untargetable-empty-ids");
      }
      // Leaving the Scrapheap moves nothing under the user that a removal
      // does not, and an external removal is applied silently too.
      if (isScrapheapView()) {
        enqueueRemoved(pictureIds);
        return { action: TARGETED, reason: "external-restored-left-scrapheap" };
      }
      if (deferWhileLightboxOpen()) {
        return {
          action: TARGETED,
          reason: "external-restored-lightbox-deferred",
        };
      }
      enqueueSortPill(pictureIds);
      return { action: PILL, reason: "external-restored" };
    }
    // An empty-id add/update can't be targeted (e.g. a restore-all broadcast, or
    // an unattributed bulk change with no ids). The pill / per-id branches below
    // would no-op on [], so the change would be lost. Reload (or defer under an
    // open overlay) rather than silently drop it.
    if (!pictureIds.length) {
      return reloadOrDefer("external-untargetable-empty-ids");
    }
    if (changeKind === "added") {
      if (deferWhileLightboxOpen()) {
        return { action: TARGETED, reason: "external-added-lightbox-deferred" };
      }
      enqueueAddPill(pictureIds);
      return { action: PILL, reason: "external-added" };
    }
    // updated
    if (pictureChangeAffectsView(fields)) {
      if (deferWhileLightboxOpen()) {
        // The classic case: the user's own tag edit kicks off a background
        // smart_score recompute that arrives origin-less (external) and would
        // raise the "sort order changed" pill. Defer it to lightbox close.
        return {
          action: TARGETED,
          reason: "external-updated-sort-affecting-lightbox-deferred",
        };
      }
      // Would reshuffle the grid - raise the pill instead of moving cards under
      // the user.
      enqueueSortPill(pictureIds);
      return { action: PILL, reason: "external-updated-sort-affecting" };
    }
    // The changed fields are known and invisible to the current sort/filter
    // (the only way to reach here - empty/unknown fields make affectsView true).
    // The classic case is a background `smart_score` recompute under a date
    // sort. Skip entirely, exactly as the old App.vue handler did: a per-id
    // refresh would fire a /metadata + thumbnail fetch for every affected card
    // in the view (a fetch storm on a full-library recompute) to update a value
    // that isn't even displayed under the current sort/filter.
    return { action: IGNORED, reason: "external-updated-invisible-field" };
  }

  // --- What a reconnect's gap changed ---------------------------------------
  /**
   * Apply `diffViewAgainstIdRead`'s account of the changes made while the
   * socket was down, on the external contract: gone pictures leave silently
   * (as an external removal does, lightbox or not), new ones go on the "New
   * pictures" pill and moved ones on the "View changed externally" pill. It
   * never reloads the grid. Returns a `{ action, reason }` descriptor.
   *
   * @param {{gone: Array, added: Array, moved: Array}} diff
   */
  function applyReconnectDiff({ gone, added, moved }) {
    // A picture this tab is already about to insert is not news.
    const arriving = added.filter(
      (id) => !addedIds.has(id) && !restoredIds.has(id),
    );
    if (!gone.length && !arriving.length && !moved.length) {
      return { action: IGNORED, reason: "reconnect-unchanged" };
    }
    if (gone.length) enqueueRemoved(gone);
    if (!arriving.length && !moved.length) {
      return { action: TARGETED, reason: "reconnect-removed" };
    }
    if (deferWhileLightboxOpen()) {
      return { action: TARGETED, reason: "reconnect-lightbox-deferred" };
    }
    if (arriving.length) enqueueAddPill(arriving);
    if (moved.length) enqueueSortPill(moved);
    return { action: PILL, reason: "reconnect-pills" };
  }

  /**
   * Apply a single picture-mutation event. Returns a `{ action, reason }`
   * descriptor (used by tests; App.vue ignores the return). The descriptor
   * reflects the DECISION; the implied grid side-effect is queued into the
   * coalescer and applied on the next window flush.
   */
  function handlePictureEvent(payload) {
    const pictureIds = asPictureIds(payload);
    if (!pictureIds.length && payload?.type !== "pictures_changed") {
      return { action: IGNORED, reason: "no-picture-ids" };
    }
    const myClientId = getMyClientId();
    const originClientId = payload?.origin_client_id ?? null;
    const source = normaliseSource(payload);
    const changeKind = resolveChangeKind(payload);
    const fields = Array.isArray(payload?.fields) ? payload.fields : [];

    // Card-content-only update (e.g. detections): the card's rendered content
    // changed but not its sort/filter position, so refresh each affected card in
    // place for ALL origins (own-origin echo, foreign UI, external) uniformly -
    // detection completion looks the same regardless of who triggered it. Never
    // a pill, never a reshuffle. Cap the per-id fetch loop and defer under an
    // open lightbox, like the other targeted branches.
    if (
      changeKind === "updated" &&
      fieldsAreCardContentOnly(fields) &&
      pictureIds.length
    ) {
      // A `pixels` change rewrote the FILE, so the card's shape and its bitmap
      // both move - from two different reads, which have to land together or the
      // tile turns twice on screen. `applyRotatedCards` owns both, so it
      // REPLACES the metadata refresh below rather than following it.
      //
      // NO `MAX_TARGETED_UPDATE` ESCALATION on this path. That cap is written
      // for the per-id `refreshGridImage` loop below - "one /metadata +
      // thumbnail fetch each" - and the applier is not that shape: one batched
      // `POST /pictures/thumbnails` for the whole set. Escalating it sent the
      // tab that ISSUED a 51-200 picture rotate (ROTATE_MAX_IDS is 200) into a
      // whole-library reload of a change it had already applied optimistically,
      // where before this branch existed its own echo was simply suppressed.
      if (fields.includes("pixels")) {
        // A TURN specifically - `orientation` rides with `pixels` only for a
        // rotate or its undo/redo. It runs under an open lightbox rather than
        // deferring: the applier is FIELDS-ONLY (see
        // ImageGrid.applyRotatedCards), nothing is inserted, removed or
        // reordered because a turned photo does not move in the grid, so there
        // is no restructuring to keep off the frozen filmstrip - and deferring
        // it queued a WHOLE-GRID refetch for `closeLightbox`, so every lightbox
        // session containing a rotate paid one.
        //
        // The other three producers of `pixels` (the thumbnail regeneration
        // task, the layout move route and its task) rewrite bytes without
        // turning anything. They are background work, not a gesture waiting to
        // land, and during an import they arrive in a steady stream - so they
        // keep the ordinary deferral rather than running the applier's forced
        // re-read and blocking decode under the open lightbox, once per batch.
        if (fields.includes("orientation")) {
          // ...unless a grid snapshot is already waiting for close. That branch
          // of `closeLightbox` assigns `pendingGridImages` wholesale and clears
          // the deferral flags, so it would discard this write and queue
          // nothing to repair it. Mark the deferral too and let close refetch.
          if (isLightboxOpen() && grid.hasPendingGridImages?.() === true) {
            grid.markLightboxDeferredRefresh?.();
            return {
              action: TARGETED,
              reason: "card-content-rotate-pending-snapshot-deferred",
            };
          }
          void grid.applyRotatedCards?.(pictureIds);
          return { action: TARGETED, reason: "card-content-rotate" };
        }
        if (deferWhileLightboxOpen()) {
          return {
            action: TARGETED,
            reason: "card-content-bytes-lightbox-deferred",
          };
        }
        void grid.applyRotatedCards?.(pictureIds);
        return { action: TARGETED, reason: "card-content-bytes" };
      }
      if (pictureIds.length > MAX_TARGETED_UPDATE) {
        return reloadOrDefer("card-content-refresh-too-large");
      }
      if (deferWhileLightboxOpen()) {
        // Only the GRID card refresh is deferred here. The open lightbox keeps
        // itself current off App.vue's `wsDetectionUpdate` signal, which it uses
        // to re-read /pictures/{id}/detections.
        return {
          action: TARGETED,
          reason: "card-content-refresh-lightbox-deferred",
        };
      }
      // `detections` leaves the file alone and takes the plain refresh.
      for (const id of pictureIds) grid.refreshGridImage?.(id);
      return { action: TARGETED, reason: "card-content-refresh" };
    }

    // Stack-membership-only update (a collapse, or the undo/redo of one): the
    // named pictures' cards render a new live member count and nothing else
    // moves. One batched read repairs every mounted member of the affected
    // stacks, so there is no per-id fetch loop to cap here. Uniform across
    // origins, never a pill, never a reshuffle.
    if (
      changeKind === "updated" &&
      fieldsAreStackFacetsOnly(fields) &&
      pictureIds.length
    ) {
      // Membership edits advance PictureStack.updated_at. Under the special
      // stack-time sort that changes the order of the whole deck list, so a
      // badge-only refresh would leave the rows in the old order. Re-read the
      // collapsed grid (or defer it under the frozen lightbox) instead.
      if (isStackUpdatedSortValue(getSelectedSort())) {
        return reloadOrDefer("stack-time-sort-changed");
      }
      if (deferWhileLightboxOpen()) {
        // §9.1: nothing mutates the grid under the frozen filmstrip.
        // closeLightbox() reconciles.
        return {
          action: TARGETED,
          reason: "stack-facet-refresh-lightbox-deferred",
        };
      }
      grid.refreshStackFacets?.(pictureIds);
      return { action: TARGETED, reason: "stack-facet-refresh" };
    }

    if (originClientId && myClientId && originClientId === myClientId) {
      return handleOwnOrigin(payload, changeKind, fields, pictureIds);
    }
    if (source === "ui") {
      return handleForeignUi(payload, changeKind, fields, pictureIds);
    }
    if (source === "external") {
      return handleExternal(payload, changeKind, fields, pictureIds);
    }

    // Unrecognised shape (e.g. bulk sort/filter-defining change) → rare,
    // logged full reload.
    logger.warn?.(
      "useGridRealtimeSync: falling back to full reload for unrecognised event",
      { type: payload?.type, source, changeKind },
    );
    reload();
    return { action: RELOAD, reason: "fallback" };
  }

  /**
   * Entry point for any /ws/updates message that concerns the picture grid.
   * App.vue routes the picture-change message types here and handles the
   * remaining (tags / descriptions / characters / snapshots / plugin) branches.
   */
  function handleMessage(payload) {
    const type = payload?.type;
    const isPictureChange =
      type === "pictures_changed" || type === "picture_imported";
    if (!isPictureChange) {
      return { action: IGNORED, reason: "not-a-picture-event" };
    }

    const affectsView = pictureChangeAffectsView(payload?.fields);

    // The sidebar picture-count only changes for adds/removes or changes that
    // affect the view; skip the churn for pure background recomputes that the
    // current sort/filter ignore (preserves the smart-score-under-date-sort
    // optimisation for the sidebar). refreshSidebar is itself debounced in
    // App.vue, so a burst already collapses to one count refresh.
    //
    // The boolean is the sidebar's FLASH (its NEW marker), which the sidebar
    // raises on any count that grew since the last fetch. A `restored` event
    // grows "All Pictures" exactly like an import does, so it must refresh the
    // counts but must NOT flash: the NEW marker means "this arrived while you
    // were not looking", and a picture you just pulled back out of the
    // Scrapheap yourself is neither new nor unseen. That mislabelling is the
    // reported bug, and it is why `restored` is a kind of its own rather than
    // the `added` the undo path used to send.
    const changeKind = resolveChangeKind(payload);
    // A moved file changes the sidebar's Folders counts, which are read from
    // the stored paths, even where it changes nothing in the grid.
    const touchesSidebar =
      affectsView ||
      (Array.isArray(payload?.fields) && payload.fields.includes("file_path")) ||
      changeKind === "added" ||
      changeKind === "removed" ||
      changeKind === "restored";
    if (touchesSidebar) {
      const flash = changeKind !== "restored";
      if (wsStore.isUploadInProgress) {
        owedSidebarFlash = owedSidebarFlash === true || flash;
      } else {
        refreshSidebar(flash);
      }
    }

    return handlePictureEvent(payload);
  }

  return {
    handleMessage,
    // Exposed for finer-grained tests and reuse.
    handlePictureEvent,
    applyReconnectDiff,
    // Force any pending coalesced side-effects to apply now (e.g. on unmount).
    flushNow: flushCoalesced,
  };
}
