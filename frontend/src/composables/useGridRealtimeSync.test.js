import { describe, it, expect, beforeEach, vi } from "vitest";
import { nextTick, reactive } from "vue";
import {
  diffViewAgainstIdRead,
  settleGoneCards,
  useGridRealtimeSync,
} from "./useGridRealtimeSync.js";
import { notifySessionReset } from "../utils/apiClient";

const MY_ID = "my-tab-uuid";
const OTHER_ID = "other-tab-uuid";

function makeHarness(overrides = {}) {
  const grid = {
    insertGridImagesById: vi.fn(),
    refreshGridImage: vi.fn(),
    refreshStackFacets: vi.fn(),
    refreshThumbnailUrls: vi.fn(),
    applyRotatedCards: vi.fn(),
    repositionImageByScore: vi.fn(),
    repositionImageBySmartScore: vi.fn(),
    refreshSmartScoreForImage: vi.fn(),
    removeImagesById: vi.fn(),
    isImagesLoading: vi.fn(() => false),
    isLightboxOpen: vi.fn(() => overrides.lightboxOpen === true),
    markLightboxDeferredRefresh: vi.fn(),
    hasPendingGridImages: vi.fn(() => overrides.pendingGridImages === true),
  };
  // Reactive, as the Pinia store is: the end of an upload is watched.
  const wsStore = reactive({
    isUploadInProgress: false,
    addPendingExternalImportIds: vi.fn(),
    addSortChangedExternalIds: vi.fn(),
  });
  const reload = vi.fn();
  const refreshSidebar = vi.fn();
  // Mirror App.vue's pictureChangeAffectsView: empty/absent fields => affects
  // view; otherwise depends on whether a field is sort-relevant.
  const selectedSort = { value: overrides.selectedSort ?? "DATE_TAKEN" };
  const pictureChangeAffectsView = vi.fn((fields) => {
    if (!Array.isArray(fields) || fields.length === 0) return true;
    return fields.some((f) => {
      if (f === "smart_score")
        return selectedSort.value.includes("SMART_SCORE");
      if (f === "character_likeness")
        return selectedSort.value.includes("CHARACTER_LIKENESS");
      // Detections are an opt-in overlay layer, never a sort/filter field -
      // mirror App.vue.pictureChangeFieldAffectsView.
      if (f === "detections") return false;
      if (f === "pixels") return false;
      // A moved file only matters to a folder view.
      if (f === "file_path") return overrides.folderView === true;
      return true; // unknown field assumed relevant
    });
  });

  const sync = useGridRealtimeSync({
    getMyClientId: () => MY_ID,
    grid,
    wsStore,
    pictureChangeAffectsView,
    getSelectedSort: () => selectedSort.value,
    logger: { warn: vi.fn() },
    reload,
    refreshSidebar,
    isScrapheapView: () => overrides.scrapheapView === true,
    scheduler: overrides.scheduler,
  });

  return { sync, grid, wsStore, reload, refreshSidebar, selectedSort };
}

describe("useGridRealtimeSync decision table", () => {
  let h;
  beforeEach(() => {
    h = makeHarness();
  });

  it("suppresses an echo of this tab's own optimistic op", () => {
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: MY_ID,
      picture_ids: [1, 2],
      change_kind: "updated",
      fields: ["tags"],
    });
    expect(res.action).toBe("suppressed");
    expect(h.grid.refreshGridImage).not.toHaveBeenCalled();
    expect(h.grid.removeImagesById).not.toHaveBeenCalled();
    expect(h.reload).not.toHaveBeenCalled();
  });

  it("reconciles own-origin smart_score echo under a smart-score sort (no reload)", () => {
    h.selectedSort.value = "SMART_SCORE";
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: MY_ID,
      picture_ids: [7],
      change_kind: "updated",
      fields: ["smart_score"],
    });
    expect(res.action).toBe("targeted");
    expect(h.grid.refreshSmartScoreForImage).toHaveBeenCalledWith(7);
    expect(h.reload).not.toHaveBeenCalled();
  });

  it("foreign-ui added -> inserts at sorted position + highlight", () => {
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [10, 11],
      change_kind: "added",
    });
    expect(res.action).toBe("targeted");
    expect(h.grid.insertGridImagesById).toHaveBeenCalledWith([10, 11]);
  });

  it("foreign-ui added during a streaming fetch -> defers to the pill", () => {
    h.grid.isImagesLoading.mockReturnValue(true);
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [10],
      change_kind: "added",
    });
    expect(res.action).toBe("pill");
    expect(h.grid.insertGridImagesById).not.toHaveBeenCalled();
    expect(h.wsStore.addPendingExternalImportIds).toHaveBeenCalledWith([10]);
  });

  it("foreign-ui updated (relevant fields) -> refreshGridImage per id", () => {
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [3, 4],
      change_kind: "updated",
      fields: ["tags"],
    });
    expect(res.action).toBe("targeted");
    expect(h.grid.refreshGridImage).toHaveBeenCalledWith(3);
    expect(h.grid.refreshGridImage).toHaveBeenCalledWith(4);
  });

  it("foreign-ui updated with sort-field change -> reposition", () => {
    h.selectedSort.value = "SMART_SCORE";
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [5],
      change_kind: "updated",
      fields: ["smart_score"],
    });
    expect(res.action).toBe("targeted");
    expect(h.grid.refreshSmartScoreForImage).toHaveBeenCalledWith(5);
  });

  it("foreign-ui updated with view-irrelevant fields -> ignored", () => {
    // smart_score field under a DATE sort does not affect the view.
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [6],
      change_kind: "updated",
      fields: ["smart_score"],
    });
    expect(res.action).toBe("ignored");
    expect(h.grid.refreshGridImage).not.toHaveBeenCalled();
  });

  it("foreign-ui removed -> removeImagesById", () => {
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [8, 9],
      change_kind: "removed",
    });
    expect(res.action).toBe("targeted");
    expect(h.grid.removeImagesById).toHaveBeenCalledWith([8, 9]);
  });

  it("external added -> New pictures pill", () => {
    const res = h.sync.handleMessage({
      type: "picture_imported",
      source: "external",
      origin_client_id: null,
      picture_ids: [20, 21],
    });
    expect(res.action).toBe("pill");
    expect(h.wsStore.addPendingExternalImportIds).toHaveBeenCalledWith([
      20, 21,
    ]);
    expect(h.grid.insertGridImagesById).not.toHaveBeenCalled();
  });

  it("external updated, sort-affecting -> Sort-order pill (no reshuffle)", () => {
    h.selectedSort.value = "SMART_SCORE";
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "external",
      origin_client_id: null,
      picture_ids: [30],
      change_kind: "updated",
      fields: ["smart_score"],
    });
    expect(res.action).toBe("pill");
    expect(h.wsStore.addSortChangedExternalIds).toHaveBeenCalledWith([30]);
    expect(h.grid.refreshGridImage).not.toHaveBeenCalled();
  });

  it("external updated, invisible field -> ignored (no fetch storm)", () => {
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "external",
      origin_client_id: null,
      picture_ids: [31],
      change_kind: "updated",
      fields: ["smart_score"], // under a DATE sort => invisible to the view
    });
    // A background recompute of a field that isn't displayed under the current
    // sort/filter must not trigger a per-card refetch across the whole view.
    expect(res.action).toBe("ignored");
    expect(h.grid.refreshGridImage).not.toHaveBeenCalled();
    expect(h.wsStore.addSortChangedExternalIds).not.toHaveBeenCalled();
  });

  it("external removed -> silent removal (never a 404 card)", () => {
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "external",
      origin_client_id: null,
      picture_ids: [40],
      change_kind: "removed",
    });
    expect(res.action).toBe("targeted");
    expect(h.grid.removeImagesById).toHaveBeenCalledWith([40]);
  });

  it("accepts legacy source 'user' as ui (transition compatibility)", () => {
    const res = h.sync.handleMessage({
      type: "picture_imported",
      source: "user",
      origin_client_id: OTHER_ID,
      picture_ids: [50],
    });
    // legacy 'user' from a different origin behaves as foreign-ui added.
    expect(res.action).toBe("targeted");
    expect(h.grid.insertGridImagesById).toHaveBeenCalledWith([50]);
  });

  // Expectation deliberately inverted. This asserted suppression, on the same
  // assumption the `restored` branch already had to walk back: that an
  // optimistic local op had applied the change. For an add there cannot have
  // been one -- the grid cannot insert a picture whose id the server assigns on
  // commit. A paste, or a drop outside the grid, goes through the sidebar
  // importer, which reports no per-file results because the grid is supposed to
  // refresh off this very broadcast. Suppressing it left a pasted picture
  // invisible until the view was switched away and back.
  it("own-origin import echo is applied, not suppressed", () => {
    const res = h.sync.handleMessage({
      type: "picture_imported",
      source: "ui",
      origin_client_id: MY_ID,
      picture_ids: [60],
    });
    expect(res.action).toBe("targeted");
    expect(res.reason).toBe("own-origin-added");
    expect(h.grid.insertGridImagesById).toHaveBeenCalledWith([60]);
  });

  it("still suppresses a genuine own-origin echo of an optimistic update", () => {
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: MY_ID,
      picture_ids: [61],
      change_kind: "updated",
      fields: ["tags"],
    });
    expect(res.action).toBe("suppressed");
  });

  it("does not refresh the sidebar for a view-irrelevant external update", () => {
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "external",
      origin_client_id: null,
      picture_ids: [70],
      change_kind: "updated",
      // smart_score under a DATE sort => no effect on the view: the event is
      // ignored entirely and the sidebar counts don't change.
      fields: ["smart_score"],
    });
    expect(res.action).toBe("ignored");
    expect(h.refreshSidebar).not.toHaveBeenCalled();
  });

  it("refreshes the sidebar for an affecting picture event", () => {
    h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [80],
      change_kind: "updated",
      fields: ["tags"],
    });
    expect(h.refreshSidebar).toHaveBeenCalledWith(true);
  });

  it("ignores non-picture event types", () => {
    const res = h.sync.handleMessage({ type: "characters_changed" });
    expect(res.action).toBe("ignored");
    expect(res.reason).toBe("not-a-picture-event");
  });
});

describe("useGridRealtimeSync - card-content-only updates (detections)", () => {
  it("own-origin detections echo -> targeted refresh per id (not suppressed)", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: MY_ID,
      picture_ids: [1, 2],
      change_kind: "updated",
      fields: ["detections"],
    });
    expect(res.action).toBe("targeted");
    expect(res.reason).toBe("card-content-refresh");
    expect(h.grid.refreshGridImage).toHaveBeenCalledWith(1);
    expect(h.grid.refreshGridImage).toHaveBeenCalledWith(2);
    expect(h.reload).not.toHaveBeenCalled();
  });

  it("external detections update -> targeted refresh per id (not ignored)", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "external",
      origin_client_id: null,
      picture_ids: [3, 4],
      change_kind: "updated",
      fields: ["detections"],
    });
    expect(res.action).toBe("targeted");
    expect(res.reason).toBe("card-content-refresh");
    expect(h.grid.refreshGridImage).toHaveBeenCalledWith(3);
    expect(h.grid.refreshGridImage).toHaveBeenCalledWith(4);
    expect(h.wsStore.addSortChangedExternalIds).not.toHaveBeenCalled();
    // A detections-only update is not view-affecting, so the sidebar count
    // must not churn.
    expect(h.refreshSidebar).not.toHaveBeenCalled();
  });

  it("foreign-ui detections update -> targeted refresh per id", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [5],
      change_kind: "updated",
      fields: ["detections"],
    });
    expect(res.action).toBe("targeted");
    expect(res.reason).toBe("card-content-refresh");
    expect(h.grid.refreshGridImage).toHaveBeenCalledWith(5);
  });

  it("detections update over the cap -> single reload (no per-id storm)", () => {
    const h = makeHarness();
    const manyIds = Array.from({ length: 51 }, (_, i) => i + 1);
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "external",
      origin_client_id: null,
      picture_ids: manyIds,
      change_kind: "updated",
      fields: ["detections"],
    });
    expect(res.action).toBe("reload");
    expect(res.reason).toBe("card-content-refresh-too-large");
    expect(h.reload).toHaveBeenCalledTimes(1);
    expect(h.grid.refreshGridImage).not.toHaveBeenCalled();
  });

  it("detections update while overlay open -> deferred, no immediate refresh", () => {
    const h = makeHarness({ lightboxOpen: true });
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: MY_ID,
      picture_ids: [6, 7],
      change_kind: "updated",
      fields: ["detections"],
    });
    expect(res.action).toBe("targeted");
    expect(res.reason).toBe("card-content-refresh-lightbox-deferred");
    expect(h.grid.refreshGridImage).not.toHaveBeenCalled();
    expect(h.grid.markLightboxDeferredRefresh).toHaveBeenCalledTimes(1);
  });

  it("mixed fields (detections + a view field) follow the OLD path, not the new branch", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [8],
      change_kind: "updated",
      // `project` is view-affecting -> this is NOT card-content-only, so it must
      // take the normal foreign-ui targeted-update path.
      fields: ["detections", "project"],
    });
    expect(res.action).toBe("targeted");
    expect(res.reason).toBe("foreign-ui-updated");
    expect(h.grid.refreshGridImage).toHaveBeenCalledWith(8);
  });

  it("a non-card-content field (smart_score) is not swallowed by the new branch", () => {
    const h = makeHarness({ selectedSort: "SMART_SCORE" });
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "external",
      origin_client_id: null,
      picture_ids: [9],
      change_kind: "updated",
      fields: ["smart_score"],
    });
    // Still the old external sort-affecting pill, not a card-content refresh.
    expect(res.action).toBe("pill");
    expect(res.reason).toBe("external-updated-sort-affecting");
    expect(h.wsStore.addSortChangedExternalIds).toHaveBeenCalledWith([9]);
  });
});

// The reported bug: "Keep cover only" collapsed a stack of five to its cover and
// the surviving cover went on rendering a stack badge of five, forever. The
// count is DERIVED per stack by the listing endpoint and is absent from the
// /pictures/{id}/metadata read `refreshGridImage` performs, so the per-card
// refresh every other branch uses cannot repair it, hence a branch, and a grid
// method, of its own. Both directions are covered: the collapse announces the
// covers, and the undo/redo announces the surviving members of every stack the
// lifecycle move touched.
describe("useGridRealtimeSync: stack-facet updates (stack_count)", () => {
  it("own-origin echo is NOT suppressed: the acting tab has no local count", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: MY_ID,
      picture_ids: [1, 2],
      change_kind: "updated",
      fields: ["stack_count"],
    });
    expect(res.action).toBe("targeted");
    expect(res.reason).toBe("stack-facet-refresh");
    // ONE batched read for the whole set, not one per card.
    expect(h.grid.refreshStackFacets).toHaveBeenCalledTimes(1);
    expect(h.grid.refreshStackFacets).toHaveBeenCalledWith([1, 2]);
    // Never the per-card path: it would fetch /metadata, which carries no count.
    expect(h.grid.refreshGridImage).not.toHaveBeenCalled();
    expect(h.reload).not.toHaveBeenCalled();
  });

  it("foreign-ui update converges the second tab", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [5],
      change_kind: "updated",
      fields: ["stack_count"],
    });
    expect(res.reason).toBe("stack-facet-refresh");
    expect(h.grid.refreshStackFacets).toHaveBeenCalledWith([5]);
    expect(h.grid.refreshGridImage).not.toHaveBeenCalled();
  });

  it("external update refreshes in place and raises no pill", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "external",
      origin_client_id: null,
      picture_ids: [6],
      change_kind: "updated",
      fields: ["stack_count"],
    });
    expect(res.reason).toBe("stack-facet-refresh");
    expect(h.grid.refreshStackFacets).toHaveBeenCalledWith([6]);
    // A badge is card content, never a reposition: nothing may reshuffle the
    // grid under the user for it.
    expect(h.wsStore.addSortChangedExternalIds).not.toHaveBeenCalled();
    expect(h.wsStore.addPendingExternalImportIds).not.toHaveBeenCalled();
  });

  it("reloads when stack membership changes the active stack-time order", () => {
    const h = makeHarness({ selectedSort: "STACK_UPDATED_AT" });
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: MY_ID,
      picture_ids: [6],
      change_kind: "updated",
      fields: ["stack_count"],
    });

    expect(res.action).toBe("reload");
    expect(res.reason).toBe("stack-time-sort-changed");
    expect(h.reload).toHaveBeenCalledTimes(1);
    expect(h.grid.refreshStackFacets).not.toHaveBeenCalled();
  });

  it("defers a stack-time reorder while the lightbox is open", () => {
    const h = makeHarness({
      selectedSort: "STACK_UPDATED_AT",
      lightboxOpen: true,
    });
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "external",
      picture_ids: [6],
      change_kind: "updated",
      fields: ["stack_count"],
    });

    expect(res.action).toBe("deferred");
    expect(res.reason).toBe("stack-time-sort-changed-lightbox-deferred");
    expect(h.reload).not.toHaveBeenCalled();
    expect(h.grid.markLightboxDeferredRefresh).toHaveBeenCalledTimes(1);
  });

  it("a large batch stays one read, never an escalated reload", () => {
    const h = makeHarness();
    const manyIds = Array.from({ length: 400 }, (_, i) => i + 1);
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: MY_ID,
      picture_ids: manyIds,
      change_kind: "updated",
      fields: ["stack_count"],
    });
    // The per-id cap exists to stop a fetch storm; there is no storm here, and
    // a reload would drop the ghosted copies the undo window is holding.
    expect(res.reason).toBe("stack-facet-refresh");
    expect(h.reload).not.toHaveBeenCalled();
    expect(h.grid.refreshStackFacets).toHaveBeenCalledTimes(1);
  });

  it("defers under an open overlay, like every other grid mutation", () => {
    const h = makeHarness({ lightboxOpen: true });
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: MY_ID,
      picture_ids: [7],
      change_kind: "updated",
      fields: ["stack_count"],
    });
    expect(res.action).toBe("targeted");
    expect(res.reason).toBe("stack-facet-refresh-lightbox-deferred");
    expect(h.grid.refreshStackFacets).not.toHaveBeenCalled();
    expect(h.grid.markLightboxDeferredRefresh).toHaveBeenCalledTimes(1);
  });

  it("mixed fields follow the OLD path: a lifted score still needs its sort", () => {
    const h = makeHarness({ selectedSort: "SMART_SCORE" });
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [8],
      change_kind: "updated",
      fields: ["stack_count", "smart_score"],
    });
    expect(res.reason).toBe("foreign-ui-updated");
    expect(h.grid.refreshStackFacets).not.toHaveBeenCalled();
  });

  it("a removal is never swallowed by the branch", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [9],
      change_kind: "removed",
      fields: ["stack_count"],
    });
    // The copies of a collapse are `removed`, and a vanished card must still
    // vanish; only the covers are `updated`.
    expect(res.reason).toBe("foreign-ui-removed");
    expect(h.grid.refreshStackFacets).not.toHaveBeenCalled();
  });
});

describe("useGridRealtimeSync - pills deferred while the lightbox is open", () => {
  it("external sort-affecting update defers to lightbox close, raises no pill", () => {
    const h = makeHarness({ selectedSort: "SMART_SCORE", lightboxOpen: true });
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "external",
      origin_client_id: null,
      picture_ids: [30],
      change_kind: "updated",
      fields: ["smart_score"],
    });
    expect(res.action).toBe("targeted");
    expect(res.reason).toBe("external-updated-sort-affecting-lightbox-deferred");
    expect(h.wsStore.addSortChangedExternalIds).not.toHaveBeenCalled();
    expect(h.grid.markLightboxDeferredRefresh).toHaveBeenCalledTimes(1);
  });

  it("external add defers to lightbox close, no New-pictures pill", () => {
    const h = makeHarness({ lightboxOpen: true });
    const res = h.sync.handleMessage({
      type: "picture_imported",
      source: "external",
      origin_client_id: null,
      picture_ids: [20, 21],
    });
    expect(res.action).toBe("targeted");
    expect(res.reason).toBe("external-added-lightbox-deferred");
    expect(h.wsStore.addPendingExternalImportIds).not.toHaveBeenCalled();
    expect(h.grid.markLightboxDeferredRefresh).toHaveBeenCalledTimes(1);
  });

  it("foreign-ui add defers to lightbox close instead of inserting live", () => {
    const h = makeHarness({ lightboxOpen: true });
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [10, 11],
      change_kind: "added",
    });
    expect(res.action).toBe("targeted");
    expect(res.reason).toBe("foreign-ui-added-lightbox-deferred");
    expect(h.grid.insertGridImagesById).not.toHaveBeenCalled();
    expect(h.grid.markLightboxDeferredRefresh).toHaveBeenCalledTimes(1);
  });

  it("still raises the pill for the same event when no lightbox is open", () => {
    const h = makeHarness({ selectedSort: "SMART_SCORE", lightboxOpen: false });
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "external",
      origin_client_id: null,
      picture_ids: [30],
      change_kind: "updated",
      fields: ["smart_score"],
    });
    expect(res.action).toBe("pill");
    expect(h.wsStore.addSortChangedExternalIds).toHaveBeenCalledWith([30]);
    expect(h.grid.markLightboxDeferredRefresh).not.toHaveBeenCalled();
  });

  it("external removal is NOT deferred while lightbox open (no stale 404 card)", () => {
    const h = makeHarness({ lightboxOpen: true });
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "external",
      origin_client_id: null,
      picture_ids: [40],
      change_kind: "removed",
    });
    expect(res.action).toBe("targeted");
    expect(res.reason).toBe("external-removed");
    expect(h.grid.removeImagesById).toHaveBeenCalledWith([40]);
  });
});

describe("useGridRealtimeSync - empty-id untargetable changes (restore-all)", () => {
  it("empty-id external add -> reload when the lightbox is closed", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "external",
      origin_client_id: null,
      change_kind: "added",
      picture_ids: [],
    });
    expect(res.action).toBe("reload");
    expect(res.reason).toBe("external-untargetable-empty-ids");
    expect(h.reload).toHaveBeenCalledTimes(1);
    expect(h.wsStore.addPendingExternalImportIds).not.toHaveBeenCalled();
  });

  it("empty-id external add -> deferred (no reload) when the lightbox is open", () => {
    const h = makeHarness({ lightboxOpen: true });
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "external",
      origin_client_id: null,
      change_kind: "added",
      picture_ids: [],
    });
    expect(res.action).toBe("deferred");
    expect(res.reason).toBe("external-untargetable-empty-ids-lightbox-deferred");
    expect(h.reload).not.toHaveBeenCalled();
    expect(h.grid.markLightboxDeferredRefresh).toHaveBeenCalledTimes(1);
  });

  it("empty-id foreign-ui add -> reload (other tabs reflect a restore-all)", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      change_kind: "added",
      picture_ids: [],
    });
    expect(res.action).toBe("reload");
    expect(res.reason).toBe("foreign-ui-untargetable-empty-ids");
    expect(h.reload).toHaveBeenCalledTimes(1);
    expect(h.grid.insertGridImagesById).not.toHaveBeenCalled();
  });

  it("empty-id removal still removes silently (not reloaded)", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "external",
      origin_client_id: null,
      change_kind: "removed",
      picture_ids: [],
    });
    expect(res.action).toBe("targeted");
    expect(res.reason).toBe("external-removed");
    expect(h.reload).not.toHaveBeenCalled();
    expect(h.grid.removeImagesById).toHaveBeenCalledWith([]);
  });
});

describe("useGridRealtimeSync - batch update fetch-storm cap", () => {
  // Mirrors MAX_TARGETED_UPDATE in useGridRealtimeSync.js.
  const THRESHOLD = 50;
  const manyIds = Array.from({ length: THRESHOLD + 1 }, (_, i) => i + 1);
  const someIds = Array.from({ length: THRESHOLD }, (_, i) => i + 1);

  it("foreign-ui updated with >threshold ids -> single reload (no per-id loop)", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      change_kind: "updated",
      fields: ["score"],
      picture_ids: manyIds,
    });
    expect(res.action).toBe("reload");
    expect(res.reason).toBe("foreign-ui-updated-too-large");
    expect(h.reload).toHaveBeenCalledTimes(1);
    expect(h.grid.refreshGridImage).not.toHaveBeenCalled();
  });

  it("foreign-ui updated with exactly threshold ids -> per-id refresh", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      change_kind: "updated",
      fields: ["score"],
      picture_ids: someIds,
    });
    expect(res.action).toBe("targeted");
    expect(res.reason).toBe("foreign-ui-updated");
    expect(h.reload).not.toHaveBeenCalled();
    expect(h.grid.refreshGridImage).toHaveBeenCalledTimes(THRESHOLD);
  });

  it("foreign-ui updated with >threshold ids -> deferred when lightbox open", () => {
    const h = makeHarness({ lightboxOpen: true });
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      change_kind: "updated",
      fields: ["score"],
      picture_ids: manyIds,
    });
    expect(res.action).toBe("deferred");
    expect(res.reason).toBe("foreign-ui-updated-too-large-lightbox-deferred");
    expect(h.reload).not.toHaveBeenCalled();
    expect(h.grid.markLightboxDeferredRefresh).toHaveBeenCalledTimes(1);
    expect(h.grid.refreshGridImage).not.toHaveBeenCalled();
  });
});

// A manual scheduler accumulates flush callbacks and only runs them on
// `tick()`, so a burst of events stays buffered (as it would inside the real
// 200ms debounce in App.vue) until we flush - letting us assert coalescing.
function makeCoalescingHarness(overrides = {}) {
  const grid = {
    insertGridImagesById: vi.fn(),
    refreshGridImage: vi.fn(),
    refreshStackFacets: vi.fn(),
    refreshThumbnailUrls: vi.fn(),
    applyRotatedCards: vi.fn(),
    repositionImageByScore: vi.fn(),
    repositionImageBySmartScore: vi.fn(),
    refreshSmartScoreForImage: vi.fn(),
    removeImagesById: vi.fn(),
    isImagesLoading: vi.fn(() => false),
    isLightboxOpen: vi.fn(() => overrides.lightboxOpen === true),
    markLightboxDeferredRefresh: vi.fn(),
    hasPendingGridImages: vi.fn(() => overrides.pendingGridImages === true),
  };
  const wsStore = {
    isUploadInProgress: false,
    addPendingExternalImportIds: vi.fn(),
    addSortChangedExternalIds: vi.fn(),
  };
  const reload = vi.fn();
  const refreshSidebar = vi.fn();
  const selectedSort = { value: overrides.selectedSort ?? "DATE_TAKEN" };
  const pictureChangeAffectsView = vi.fn(() => true);

  // Window-style scheduler: the first event arms one pending flush; the manual
  // tick() runs it, mirroring the leading-edge fixed window in App.vue.
  let pending = null;
  const scheduler = {
    schedule: (flush) => {
      pending = flush;
    },
    cancel: () => {
      pending = null;
    },
  };
  const tick = () => {
    const f = pending;
    pending = null;
    if (f) f();
  };

  const sync = useGridRealtimeSync({
    getMyClientId: () => MY_ID,
    grid,
    wsStore,
    pictureChangeAffectsView,
    getSelectedSort: () => selectedSort.value,
    logger: { warn: vi.fn() },
    reload,
    refreshSidebar,
    isScrapheapView: () => overrides.scrapheapView === true,
    scheduler,
  });

  return { sync, grid, wsStore, reload, refreshSidebar, selectedSort, tick };
}

describe("useGridRealtimeSync - coalescing window", () => {
  it("100 foreign updates for one id -> a single refreshGridImage on flush", () => {
    const h = makeCoalescingHarness();
    for (let i = 0; i < 100; i++) {
      h.sync.handleMessage({
        type: "pictures_changed",
        source: "ui",
        origin_client_id: OTHER_ID,
        change_kind: "updated",
        fields: ["tags"],
        picture_ids: [42],
      });
    }
    // Nothing applied yet - still inside the window.
    expect(h.grid.refreshGridImage).not.toHaveBeenCalled();
    h.tick();
    // Deduped to the single distinct id.
    expect(h.grid.refreshGridImage).toHaveBeenCalledTimes(1);
    expect(h.grid.refreshGridImage).toHaveBeenCalledWith(42);
  });

  it("batched inserts collapse to one insertGridImagesById call", () => {
    const h = makeCoalescingHarness();
    h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      change_kind: "added",
      picture_ids: [1, 2],
    });
    h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      change_kind: "added",
      picture_ids: [3],
    });
    h.tick();
    expect(h.grid.insertGridImagesById).toHaveBeenCalledTimes(1);
    expect(h.grid.insertGridImagesById).toHaveBeenCalledWith([1, 2, 3]);
  });

  it("external adds raise the New-pictures pill once per window", () => {
    const h = makeCoalescingHarness();
    h.sync.handleMessage({
      type: "picture_imported",
      source: "external",
      origin_client_id: null,
      picture_ids: [10],
    });
    h.sync.handleMessage({
      type: "picture_imported",
      source: "external",
      origin_client_id: null,
      picture_ids: [11],
    });
    h.tick();
    expect(h.wsStore.addPendingExternalImportIds).toHaveBeenCalledTimes(1);
    expect(h.wsStore.addPendingExternalImportIds).toHaveBeenCalledWith([
      10, 11,
    ]);
  });

  it("an add then a remove of the same id nets to a remove", () => {
    const h = makeCoalescingHarness();
    h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      change_kind: "added",
      picture_ids: [5],
    });
    h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      change_kind: "removed",
      picture_ids: [5],
    });
    h.tick();
    expect(h.grid.removeImagesById).toHaveBeenCalledWith([5]);
    expect(h.grid.insertGridImagesById).not.toHaveBeenCalled();
  });

  it("coalesced per-id batch over MAX_TARGETED_UPDATE escalates to one reload", () => {
    const h = makeCoalescingHarness();
    // 51 distinct ids, one event each - each event is sub-cap, but the window's
    // distinct-id total crosses MAX_TARGETED_UPDATE (50).
    for (let id = 1; id <= 51; id++) {
      h.sync.handleMessage({
        type: "pictures_changed",
        source: "ui",
        origin_client_id: OTHER_ID,
        change_kind: "updated",
        fields: ["tags"],
        picture_ids: [id],
      });
    }
    h.tick();
    expect(h.reload).toHaveBeenCalledTimes(1);
    expect(h.grid.refreshGridImage).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// change_kind: "restored" - a scrapheap comeback is not an import
// ---------------------------------------------------------------------------
//
// A picture coming back out of the Scrapheap puts a card back, exactly as an
// import does, and for a while both said `added`. The difference the SPA acts
// on is that `added` means NEW TO THE VAULT: the sidebar answers it with its
// NEW marker and the grid answers it with the new-picture flash. Neither is
// true of a picture that has been in the library the whole time, which is why
// `restored` is a kind of its own.

describe("useGridRealtimeSync - restored (scrapheap comeback)", () => {
  function restored(overrides = {}) {
    return {
      type: "pictures_changed",
      source: "ui",
      change_kind: "restored",
      picture_ids: [3, 4],
      ...overrides,
    };
  }

  it("reinserts on this tab's OWN undo instead of suppressing the echo", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage(restored({ origin_client_id: MY_ID }));
    // The bug this fixes: suppressed as "my own optimistic op already did it",
    // when nothing local had. The grid kept showing the pre-undo state.
    expect(res.action).toBe("targeted");
    expect(h.grid.insertGridImagesById).toHaveBeenCalledWith([3, 4], {
      highlight: false,
    });
    expect(h.grid.removeImagesById).not.toHaveBeenCalled();
  });

  it("reinserts without the new-picture flash - it is a comeback, not an arrival", () => {
    const h = makeHarness();
    h.sync.handleMessage(restored({ origin_client_id: MY_ID }));
    const [, options] = h.grid.insertGridImagesById.mock.calls[0];
    expect(options).toEqual({ highlight: false });

    // …whereas a genuine import still flashes (no options at all).
    h.grid.insertGridImagesById.mockClear();
    h.sync.handleMessage({
      type: "picture_imported",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [9],
    });
    expect(h.grid.insertGridImagesById).toHaveBeenCalledWith([9]);
  });

  it("refreshes the sidebar WITHOUT raising its NEW marker", () => {
    const h = makeHarness();
    h.sync.handleMessage(restored({ origin_client_id: MY_ID }));
    // The counts really did change (All Pictures up, Scrapheap down), so the
    // sidebar must re-read - it just must not call it new.
    expect(h.refreshSidebar).toHaveBeenCalledWith(false);

    h.refreshSidebar.mockClear();
    h.sync.handleMessage({
      type: "picture_imported",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [9],
    });
    expect(h.refreshSidebar).toHaveBeenCalledWith(true);
  });

  it("applies another owner tab's restore in place", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage(restored({ origin_client_id: OTHER_ID }));
    expect(res.action).toBe("targeted");
    expect(h.grid.insertGridImagesById).toHaveBeenCalledWith([3, 4], {
      highlight: false,
    });
  });

  it("raises the view-changed pill for an external restore, never the new-pictures one", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage(
      restored({ source: "external", origin_client_id: null }),
    );
    expect(res.action).toBe("pill");
    expect(h.wsStore.addSortChangedExternalIds).toHaveBeenCalledWith([3, 4]);
    // "↑ 2 new pictures, click to load" would be a lie about these pictures.
    expect(h.wsStore.addPendingExternalImportIds).not.toHaveBeenCalled();
    expect(h.grid.insertGridImagesById).not.toHaveBeenCalled();
  });

  it("reloads when a restore-all names no ids", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage(
      restored({ origin_client_id: MY_ID, picture_ids: [] }),
    );
    // `POST /pictures/scrapheap/restore` with no subset broadcasts an empty id
    // list; a per-id insert would silently do nothing.
    expect(res.action).toBe("reload");
    expect(h.reload).toHaveBeenCalled();
  });

  it("defers under an open lightbox rather than restructuring the filmstrip", () => {
    const h = makeHarness({ lightboxOpen: true });
    const res = h.sync.handleMessage(restored({ origin_client_id: MY_ID }));
    expect(res.action).toBe("targeted");
    expect(h.grid.markLightboxDeferredRefresh).toHaveBeenCalled();
    expect(h.grid.insertGridImagesById).not.toHaveBeenCalled();
  });

  it("falls back to the view-changed pill while a streaming fetch owns the grid", () => {
    const h = makeHarness();
    h.grid.isImagesLoading.mockReturnValue(true);
    const res = h.sync.handleMessage(restored({ origin_client_id: MY_ID }));
    expect(res.action).toBe("pill");
    expect(h.wsStore.addSortChangedExternalIds).toHaveBeenCalledWith([3, 4]);
    expect(h.wsStore.addPendingExternalImportIds).not.toHaveBeenCalled();
  });

  it("still treats a re-scrapheap on redo as a removal", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      change_kind: "removed",
      picture_ids: [3, 4],
    });
    expect(res.action).toBe("targeted");
    expect(h.grid.removeImagesById).toHaveBeenCalledWith([3, 4]);
  });

  it("degrades an unknown kind to updated, exactly as before", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      change_kind: "resurrected",
      fields: ["tags"],
      picture_ids: [3],
    });
    expect(res.reason).toBe("foreign-ui-updated");
  });
});

describe("useGridRealtimeSync - a rotate's pixels changed", () => {
  // The card's thumbnail URL comes from the batch-thumbnail endpoint, never
  // from /metadata, so refreshGridImage alone repaints the pre-rotate bitmap.
  // The forward rotate hides this because the tab that issued it refreshes the
  // URL itself; an undo arrives over the socket with no such local hook, which
  // is the bug this covers.
  it("own-origin undo echo -> refreshes the thumbnail URL, not just metadata", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: MY_ID,
      picture_ids: [11, 12],
      change_kind: "updated",
      fields: ["orientation", "pixels"],
    });
    expect(res.action).toBe("targeted");
    expect(res.reason).toBe("card-content-rotate");
    // One applier, not a metadata refresh followed by a thumbnail refresh: the
    // tile's shape and its bitmap have to land in the same frame or the picture
    // turns twice on screen. Asserting the old pair is asserting the bug.
    expect(h.grid.applyRotatedCards).toHaveBeenCalledWith([11, 12]);
    expect(h.grid.refreshGridImage).not.toHaveBeenCalled();
    expect(h.grid.refreshThumbnailUrls).not.toHaveBeenCalled();
  });

  it("a foreign tab's rotate also repaints here", () => {
    const h = makeHarness();
    h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [13],
      change_kind: "updated",
      fields: ["orientation", "pixels"],
    });
    expect(h.grid.applyRotatedCards).toHaveBeenCalledWith([13]);
  });

  it("never reloads the grid or raises a pill - a turned photo does not move", () => {
    const h = makeHarness();
    h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [14],
      change_kind: "updated",
      fields: ["orientation", "pixels"],
    });
    expect(h.reload).not.toHaveBeenCalled();
    expect(h.wsStore.addSortChangedExternalIds).not.toHaveBeenCalled();
  });

  // The applier is fields-only, so unlike the metadata refresh its `detections`
  // sibling takes, it has nothing to keep off the frozen filmstrip. Deferring it
  // queued a whole-grid refetch for every lightbox close after a rotate.
  it("repaints the card under an open overlay instead of deferring", () => {
    const h = makeHarness({ lightboxOpen: true });
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [16],
      change_kind: "updated",
      fields: ["orientation", "pixels"],
    });
    expect(res.reason).toBe("card-content-rotate");
    expect(h.grid.applyRotatedCards).toHaveBeenCalledWith([16]);
    expect(h.grid.markLightboxDeferredRefresh).not.toHaveBeenCalled();
    expect(h.reload).not.toHaveBeenCalled();
  });

  // ...but not when a fetch has already parked a whole list for close.
  // `closeLightbox` assigns `pendingGridImages` wholesale and clears the
  // deferral flags with it, so an in-place write made now is discarded with
  // nothing queued to repair it. Defer instead and let close refetch.
  it("defers under an open lightbox when a grid snapshot is waiting for close", () => {
    const h = makeHarness({ lightboxOpen: true, pendingGridImages: true });
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [17],
      change_kind: "updated",
      fields: ["orientation", "pixels"],
    });
    expect(res.reason).toBe("card-content-rotate-pending-snapshot-deferred");
    expect(h.grid.applyRotatedCards).not.toHaveBeenCalled();
    expect(h.grid.markLightboxDeferredRefresh).toHaveBeenCalledTimes(1);
  });

  // ROTATE_MAX_IDS is 200 and MAX_TARGETED_UPDATE is 50. That cap is written
  // for the per-id `refreshGridImage` loop - "one /metadata + thumbnail fetch
  // each" - and the applier is one batched thumbnails POST instead, so it does
  // not belong on this path. Escalating sent the tab that ISSUED the rotate
  // into a whole-library reload of a change it had already applied.
  it("does not escalate a bulk turn to a grid reload", () => {
    const h = makeHarness();
    const ids = Array.from({ length: 80 }, (_, i) => 100 + i);
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: MY_ID,
      picture_ids: ids,
      change_kind: "updated",
      fields: ["orientation", "pixels"],
    });
    expect(res.reason).toBe("card-content-rotate");
    expect(h.grid.applyRotatedCards).toHaveBeenCalledWith(ids);
    expect(h.reload).not.toHaveBeenCalled();
    expect(h.grid.markLightboxDeferredRefresh).not.toHaveBeenCalled();
  });

  // A byte rewrite that turned nothing: the thumbnail regeneration task, and
  // the layout move (`["file_path", "pixels"]`, card content too outside a
  // folder view - see below). The card's bitmap still has
  // to be re-read - the metadata endpoint carries no thumbnail URL - but this
  // is background work rather than a gesture waiting to land, so under an open
  // lightbox it keeps the ordinary deferral rather than running the applier's
  // forced re-read and blocking decode once per batch for a whole import.
  it("defers a non-turn byte rewrite under an open lightbox", () => {
    const h = makeHarness({ lightboxOpen: true });
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      picture_ids: [18, 19],
      change_kind: "updated",
      fields: ["pixels"],
    });
    expect(res.reason).toBe("card-content-bytes-lightbox-deferred");
    expect(h.grid.applyRotatedCards).not.toHaveBeenCalled();
    expect(h.grid.markLightboxDeferredRefresh).toHaveBeenCalledTimes(1);
  });

  it("still repaints a non-turn byte rewrite with no overlay open", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      picture_ids: [18, 19],
      change_kind: "updated",
      fields: ["pixels"],
    });
    expect(res.reason).toBe("card-content-bytes");
    expect(h.grid.applyRotatedCards).toHaveBeenCalledWith([18, 19]);
    expect(h.grid.refreshGridImage).not.toHaveBeenCalled();
  });

  // The reported bug: a workflow run's output is filed, a face assignment
  // makes its folder untrue, and LayoutMoveTask moves it with no origin. Under
  // a plain view that is the same card with a new thumbnail URL, not a view
  // change, so it repaints in place and raises no pill.
  it("repaints a background layout move in place, without the pill", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      picture_ids: [1368],
      change_kind: "updated",
      fields: ["file_path", "pixels"],
    });
    expect(res.reason).toBe("card-content-bytes");
    expect(h.grid.applyRotatedCards).toHaveBeenCalledWith([1368]);
    expect(h.wsStore.addSortChangedExternalIds).not.toHaveBeenCalled();
    expect(h.reload).not.toHaveBeenCalled();
    // The sidebar's Folders counts are read from the paths, so they re-read.
    expect(h.refreshSidebar).toHaveBeenCalledTimes(1);
  });

  // A path change with no `pixels` has nothing to repaint the thumbnail with
  // on the card path, so it is not treated as card content.
  it("does not take a bare file_path change as card content", () => {
    const h = makeHarness();
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      picture_ids: [1368],
      change_kind: "updated",
      fields: ["file_path"],
    });
    expect(res.reason).toBe("external-updated-invisible-field");
    expect(h.grid.refreshGridImage).not.toHaveBeenCalled();
  });

  // The control: in a folder view the move can take the card out of the view,
  // so an external one still offers the refresh rather than moving it silently.
  it("keeps the pill for a background layout move in a folder view", () => {
    const h = makeHarness({ folderView: true });
    const res = h.sync.handleMessage({
      type: "pictures_changed",
      picture_ids: [1368],
      change_kind: "updated",
      fields: ["file_path", "pixels"],
    });
    expect(res.reason).toBe("external-updated-sort-affecting");
    expect(h.wsStore.addSortChangedExternalIds).toHaveBeenCalledWith([1368]);
    expect(h.grid.applyRotatedCards).not.toHaveBeenCalled();
  });

  it("leaves the thumbnail alone for a change that did not touch the file", () => {
    const h = makeHarness();
    h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [15],
      change_kind: "updated",
      fields: ["detections"],
    });
    // `detections` leaves the FILE alone, so it keeps the plain metadata
    // refresh and must not take the rotate applier's decode-then-commit path.
    expect(h.grid.refreshGridImage).toHaveBeenCalledWith(15);
    expect(h.grid.applyRotatedCards).not.toHaveBeenCalled();
  });
});

// A purge from the Scrapheap is never applied optimistically: a run settles by
// re-reading the heap, and a purge request that timed out after the server
// committed has no local step at all. So this tab's own `removed` echo is what
// takes the destroyed pictures off the screen. Outside the Scrapheap the echo
// follows a soft delete whose tiles stay ghosted for the undo window, so there
// it must stay suppressed.
describe("useGridRealtimeSync - this tab's own removals", () => {
  function removed(overrides = {}) {
    return {
      type: "pictures_changed",
      source: "ui",
      origin_client_id: MY_ID,
      change_kind: "removed",
      picture_ids: [1, 2],
      ...overrides,
    };
  }

  it("removes the purged pictures from the Scrapheap as the server confirms them", () => {
    const h = makeCoalescingHarness({ scrapheapView: true });
    const res = h.sync.handleMessage(removed());
    expect(res).toEqual({
      action: "targeted",
      reason: "own-origin-removed-scrapheap",
    });
    h.tick();
    expect(h.grid.removeImagesById).toHaveBeenCalledWith([1, 2]);
    expect(h.reload).not.toHaveBeenCalled();
  });

  it("keeps the soft-deleted tiles outside the Scrapheap, where the undo window holds them", () => {
    const h = makeCoalescingHarness();
    const res = h.sync.handleMessage(removed());
    expect(res).toEqual({ action: "suppressed", reason: "own-origin-echo" });
    h.tick();
    expect(h.grid.removeImagesById).not.toHaveBeenCalled();
  });

  it("lets a restore that lands in the same window win over the removal before it", () => {
    // Outside the Scrapheap a restore brings the picture back, so of a delete
    // and its undo in one window the undo is what the grid must end on.
    const h = makeCoalescingHarness();
    const other = { origin_client_id: OTHER_ID };
    h.sync.handleMessage(removed({ ...other, picture_ids: [5] }));
    h.sync.handleMessage(
      removed({ ...other, change_kind: "restored", picture_ids: [5] }),
    );
    h.tick();
    for (const [ids] of h.grid.removeImagesById.mock.calls) {
      expect(ids).not.toContain(5);
    }
    expect(h.grid.insertGridImagesById).toHaveBeenCalledWith([5], {
      highlight: false,
    });
  });
});

describe("useGridRealtimeSync - a restored picture leaves the Scrapheap", () => {
  function restored(overrides = {}) {
    return {
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      change_kind: "restored",
      picture_ids: [3],
      ...overrides,
    };
  }

  it("takes another tab's restored card out of the Scrapheap and inserts nothing", () => {
    const h = makeCoalescingHarness({ scrapheapView: true });
    const res = h.sync.handleMessage(restored());
    expect(res).toEqual({
      action: "targeted",
      reason: "foreign-ui-restored-left-scrapheap",
    });
    h.tick();
    expect(h.grid.removeImagesById).toHaveBeenCalledWith([3]);
    expect(h.grid.insertGridImagesById).not.toHaveBeenCalled();
    expect(h.reload).not.toHaveBeenCalled();
  });

  it("takes this tab's own restored card out of the Scrapheap too", () => {
    const h = makeCoalescingHarness({ scrapheapView: true });
    const res = h.sync.handleMessage(restored({ origin_client_id: MY_ID }));
    expect(res.reason).toBe("own-origin-restored-left-scrapheap");
    h.tick();
    expect(h.grid.removeImagesById).toHaveBeenCalledWith([3]);
    expect(h.grid.insertGridImagesById).not.toHaveBeenCalled();
  });

  it("takes an external restore out of the Scrapheap instead of raising the pill", () => {
    const h = makeCoalescingHarness({ scrapheapView: true });
    const res = h.sync.handleMessage(
      restored({ source: "external", origin_client_id: null }),
    );
    expect(res).toEqual({
      action: "targeted",
      reason: "external-restored-left-scrapheap",
    });
    h.tick();
    expect(h.grid.removeImagesById).toHaveBeenCalledWith([3]);
    expect(h.wsStore.addSortChangedExternalIds).not.toHaveBeenCalled();
  });

  it("ends a removal and a restore of one picture in one window with it gone", () => {
    const h = makeCoalescingHarness({ scrapheapView: true });
    h.sync.handleMessage(
      restored({ change_kind: "removed", picture_ids: [5] }),
    );
    h.sync.handleMessage(restored({ picture_ids: [5] }));
    h.tick();
    const removedIds = h.grid.removeImagesById.mock.calls.flatMap(
      ([ids]) => ids,
    );
    expect(removedIds).toContain(5);
    expect(h.grid.insertGridImagesById).not.toHaveBeenCalled();
  });
});

describe("useGridRealtimeSync - a Scrapheap run's events (run_id)", () => {
  function runEvent(overrides = {}) {
    return {
      type: "pictures_changed",
      source: "ui",
      origin_client_id: MY_ID,
      run_id: "run-1",
      run_kind: "restore",
      change_kind: "removed",
      picture_ids: [7, 8],
      ...overrides,
    };
  }

  it("applies this tab's own run removal outside the Scrapheap, where no op ghosted it", () => {
    const h = makeCoalescingHarness();
    const res = h.sync.handleMessage(runEvent({ run_kind: "undo" }));
    expect(res).toEqual({ action: "targeted", reason: "foreign-ui-removed" });
    h.tick();
    expect(h.grid.removeImagesById).toHaveBeenCalledWith([7, 8]);
  });

  // A restore run's slices and an undo's comebacks put cards back as a
  // comeback, never with the new-picture flash.
  it("puts a run's restored pictures back without the new-picture highlight", () => {
    for (const runKind of ["restore", "undo"]) {
      const h = makeCoalescingHarness();
      const res = h.sync.handleMessage(
        runEvent({ run_kind: runKind, change_kind: "restored" }),
      );
      expect(res).toEqual({ action: "targeted", reason: "foreign-ui-restored" });
      h.tick();
      expect(h.grid.insertGridImagesById).toHaveBeenCalledWith([7, 8], {
        highlight: false,
      });
    }
  });

  it("takes a restore run's pictures off the Scrapheap as they leave it", () => {
    const h = makeCoalescingHarness({ scrapheapView: true });
    const res = h.sync.handleMessage(runEvent({ change_kind: "restored" }));
    expect(res.reason).toBe("foreign-ui-restored-left-scrapheap");
    h.tick();
    expect(h.grid.removeImagesById).toHaveBeenCalledWith([7, 8]);
    expect(h.grid.insertGridImagesById).not.toHaveBeenCalled();
  });

  it("still suppresses this tab's own removal echo when no run sent it", () => {
    const h = makeCoalescingHarness();
    const event = runEvent();
    delete event.run_id;
    delete event.run_kind;
    const res = h.sync.handleMessage(event);
    expect(res).toEqual({ action: "suppressed", reason: "own-origin-echo" });
    h.tick();
    expect(h.grid.removeImagesById).not.toHaveBeenCalled();
  });

  it("reads run_id at the top of the message, not inside data", () => {
    const h = makeCoalescingHarness();
    const event = runEvent();
    delete event.run_id;
    const res = h.sync.handleMessage({ ...event, data: { run_id: "run-1" } });
    expect(res.action).toBe("suppressed");
  });
});

describe("useGridRealtimeSync - a sidebar refresh held back by an upload", () => {
  const foreignRemoval = (id) => ({
    type: "pictures_changed",
    source: "ui",
    origin_client_id: OTHER_ID,
    picture_ids: [id],
    change_kind: "removed",
  });
  const foreignRestore = (id) => ({
    type: "pictures_changed",
    source: "ui",
    origin_client_id: OTHER_ID,
    picture_ids: [id],
    change_kind: "restored",
  });

  it("runs once when the upload ends, for every change it held back", async () => {
    const h = makeHarness();
    h.wsStore.isUploadInProgress = true;
    await nextTick();
    // Another tab deletes two pictures and restores one mid-upload.
    h.sync.handleMessage(foreignRemoval(7));
    h.sync.handleMessage(foreignRestore(8));
    h.sync.handleMessage(foreignRemoval(9));
    await nextTick();
    expect(h.refreshSidebar).not.toHaveBeenCalled();

    h.wsStore.isUploadInProgress = false;
    await nextTick();
    expect(h.refreshSidebar).toHaveBeenCalledTimes(1);
    // A deletion moved the counts the way any change does: it flashes.
    expect(h.refreshSidebar).toHaveBeenCalledWith(true);

    // Owed once, paid once: the next upload, with nothing held, owes nothing.
    h.wsStore.isUploadInProgress = true;
    await nextTick();
    h.wsStore.isUploadInProgress = false;
    await nextTick();
    expect(h.refreshSidebar).toHaveBeenCalledTimes(1);
  });

  it("keeps a held restore from flashing", async () => {
    const h = makeHarness();
    h.wsStore.isUploadInProgress = true;
    await nextTick();
    h.sync.handleMessage(foreignRestore(8));
    h.wsStore.isUploadInProgress = false;
    await nextTick();
    expect(h.refreshSidebar).toHaveBeenCalledTimes(1);
    expect(h.refreshSidebar).toHaveBeenCalledWith(false);
  });

  it("owes nothing when no change touched the counts", async () => {
    const h = makeHarness();
    h.wsStore.isUploadInProgress = true;
    await nextTick();
    h.sync.handleMessage({
      type: "pictures_changed",
      source: "external",
      origin_client_id: null,
      picture_ids: [70],
      change_kind: "updated",
      fields: ["smart_score"],
    });
    h.wsStore.isUploadInProgress = false;
    await nextTick();
    expect(h.refreshSidebar).not.toHaveBeenCalled();
  });

  it("drops what it owes at sign-out, which also ends the upload", async () => {
    const h = makeHarness();
    h.wsStore.isUploadInProgress = true;
    await nextTick();
    h.sync.handleMessage(foreignRemoval(7));
    // As useWsStore does in its own session-reset handler.
    notifySessionReset("logout");
    h.wsStore.isUploadInProgress = false;
    await nextTick();
    expect(h.refreshSidebar).not.toHaveBeenCalled();
  });
});

// The grid's base list: a number is a lone picture, [id, stackId] a stack row.
function gridRows(...specs) {
  return specs.map((spec) =>
    Array.isArray(spec)
      ? { id: spec[0], stackId: String(spec[1]) }
      : { id: spec, stackId: null },
  );
}

// The stream read's rows, as the server sends them.
function readRows(...specs) {
  return specs.map((spec) =>
    Array.isArray(spec)
      ? { id: spec[0], stack_id: spec[1] }
      : { id: spec, stack_id: null },
  );
}

// A read of the whole list that saw no local change while it was out, in a
// view that shows a stack as one card.
function wholeRead(grid, answer, extra = {}) {
  return {
    before: grid,
    after: grid,
    answer,
    fromStart: true,
    toEnd: true,
    stacked: true,
    ...extra,
  };
}

describe("diffViewAgainstIdRead - what a reconnect's gap changed", () => {
  it("finds nothing when the read matches the grid", () => {
    const grid = gridRows(1, 2, [3, 30], 4);
    expect(
      diffViewAgainstIdRead(wholeRead(grid, readRows(1, 2, [3, 30], 4))),
    ).toEqual({ gone: [], added: [], moved: [], goneCards: [] });
  });

  it("names a card the read no longer lists as gone", () => {
    const grid = gridRows(1, 2, 3, 4, 5);
    expect(
      diffViewAgainstIdRead(wholeRead(grid, readRows(1, 2, 4, 5))),
    ).toEqual({ gone: [3], added: [], moved: [], goneCards: [[3]] });
  });

  it("judges only the stretch of the grid the read covers", () => {
    // The read covers pictures 4..7; 1-3 and 8-10 lie outside it, and their
    // absence from the answer says nothing about them.
    const grid = gridRows(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    expect(
      diffViewAgainstIdRead({
        ...wholeRead(grid, readRows(4, 5, 7)),
        fromStart: false,
        toEnd: false,
      }),
    ).toEqual({ gone: [6], added: [], moved: [], goneCards: [[6]] });
  });

  it("judges the grid's head and tail when the read reached the list's ends", () => {
    const grid = gridRows(1, 2, 3, 4, 5);
    expect(diffViewAgainstIdRead(wholeRead(grid, readRows(2, 3, 4)))).toEqual(
      { gone: [1, 5], added: [], moved: [], goneCards: [[1], [5]] },
    );
  });

  it("names a listed picture the grid does not hold as new", () => {
    const grid = gridRows(1, 2, 3);
    expect(
      diffViewAgainstIdRead(wholeRead(grid, readRows(1, 9, 2, 3))),
    ).toEqual({ gone: [], added: [9], moved: [], goneCards: [] });
  });

  it("names the fewest pictures that explain a reorder as moved", () => {
    const grid = gridRows(1, 2, 3, 4, 5, 6);
    expect(
      diffViewAgainstIdRead(wholeRead(grid, readRows(1, 2, 5, 3, 4, 6))),
    ).toEqual({ gone: [], added: [], moved: [5], goneCards: [] });
  });

  it("calls a stack whose cover changed moved, not a removal and an arrival", () => {
    const grid = gridRows(1, [2, 20], 3);
    expect(
      diffViewAgainstIdRead(wholeRead(grid, readRows(1, [5, 20], 3))),
    ).toEqual({ gone: [], added: [], moved: [5], goneCards: [] });
  });

  it("compares a stack whose members the grid's list holds as one card", () => {
    // A local reorder writes a stack's members into the base list; the read
    // lists the stack once, by its cover.
    const grid = gridRows(1, [2, 20], [3, 20], [4, 20], 5);
    expect(
      diffViewAgainstIdRead(wholeRead(grid, readRows(1, [2, 20], 5))),
    ).toEqual({ gone: [], added: [], moved: [], goneCards: [] });
    expect(diffViewAgainstIdRead(wholeRead(grid, readRows(1, 5)))).toEqual({
      gone: [2, 3, 4],
      added: [],
      moved: [],
      goneCards: [[2, 3, 4]],
    });
  });

  it("calls a picture that joined a stack as its cover moved, once", () => {
    const grid = gridRows(1, 2, [3, 30]);
    expect(
      diffViewAgainstIdRead(wholeRead(grid, readRows(1, [2, 30]))),
    ).toEqual({ gone: [], added: [], moved: [2], goneCards: [] });
  });

  it("calls a lone picture that became a new stack's cover moved, not new", () => {
    const grid = gridRows(1, 2, 3);
    expect(
      diffViewAgainstIdRead(wholeRead(grid, readRows(1, [2, 40], 3))),
    ).toEqual({ gone: [], added: [], moved: [2], goneCards: [] });
  });

  it("in the Scrapheap, compares each picture of a stack on its own", () => {
    // The Scrapheap lists a scrapped stack's members one by one, and so does
    // its grid: a member that left, or arrived, is a card of its own.
    const grid = gridRows(1, [2, 20], [3, 20], 4);
    const scrapheap = { stacked: false };
    expect(
      diffViewAgainstIdRead(
        wholeRead(grid, readRows(1, [2, 20], 4), scrapheap),
      ),
    ).toEqual({ gone: [3], added: [], moved: [], goneCards: [[3]] });
    expect(
      diffViewAgainstIdRead(
        wholeRead(grid, readRows(1, [2, 20], [3, 20], [5, 20], 4), scrapheap),
      ),
    ).toEqual({ gone: [], added: [5], moved: [], goneCards: [] });
  });

  it("leaves alone a picture the grid keeps on purpose (a ghost)", () => {
    const grid = gridRows(1, 2, 3);
    expect(
      diffViewAgainstIdRead(
        wholeRead(grid, readRows(1, 3), { skipIds: ["2"] }),
      ),
    ).toEqual({ gone: [], added: [], moved: [], goneCards: [] });
  });

  it("leaves alone a picture this tab took out or put in during the read", () => {
    // The read was answered before this tab removed 2 and inserted 7.
    expect(
      diffViewAgainstIdRead({
        before: gridRows(1, 2, 3, 4),
        after: gridRows(1, 3, 7, 4),
        answer: readRows(1, 2, 3, 4),
        fromStart: true,
        toEnd: true,
        stacked: true,
      }),
    ).toEqual({ gone: [], added: [], moved: [], goneCards: [] });
  });

  it("with nothing in common, judges a whole-list read card by card", () => {
    const grid = gridRows(1, 2);
    expect(diffViewAgainstIdRead(wholeRead(grid, readRows(3, 4)))).toEqual({
      gone: [1, 2],
      added: [3, 4],
      moved: [],
      goneCards: [[1], [2]],
    });
  });

  it("with nothing in common, removes nothing on a partial read and offers the refresh", () => {
    const grid = gridRows(1, 2);
    expect(
      diffViewAgainstIdRead({
        ...wholeRead(grid, readRows(3, 4)),
        toEnd: false,
      }),
    ).toEqual({ gone: [], added: [], moved: [3, 4], goneCards: [] });
  });

  it("takes a search's first row of a stack as its card, as the grid's collapse does", () => {
    // An id-list view (a text search) lists every member of a stack, and so
    // does the grid's list behind its one card.
    const grid = gridRows(1, [2, 20], [3, 20], 4);
    expect(
      diffViewAgainstIdRead(wholeRead(grid, readRows(1, [2, 20], [3, 20], 4))),
    ).toEqual({ gone: [], added: [], moved: [], goneCards: [] });
    expect(
      diffViewAgainstIdRead(wholeRead(grid, readRows(1, [3, 20], [2, 20], 4))),
    ).toEqual({ gone: [], added: [], moved: [3], goneCards: [] });
  });
});

describe("settleGoneCards - a partial read's gone cards, asked again by id", () => {
  // Pictures 1..10 in the grid; the read covered 4..7 and no longer lists 6.
  function partialDiff(grid = gridRows(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)) {
    return diffViewAgainstIdRead({
      ...wholeRead(grid, readRows(4, 5, 7)),
      fromStart: false,
      toEnd: false,
    });
  }

  it("calls a card the view still lists moved, out of the stretch read", () => {
    expect(settleGoneCards(partialDiff(), new Set(["6"]))).toEqual({
      gone: [],
      added: [],
      moved: [6],
    });
  });

  it("calls a card the view no longer lists gone", () => {
    expect(settleGoneCards(partialDiff(), new Set())).toEqual({
      gone: [6],
      added: [],
      moved: [],
    });
  });

  it("keeps a stack any of whose pictures the view lists, by the card it shows", () => {
    const diff = partialDiff(
      gridRows(1, 2, 3, 4, 5, [6, 60], [61, 60], 7, [8, 80], [81, 80], 9),
    );
    expect(diff.goneCards).toEqual([[6, 61]]);
    expect(settleGoneCards(diff, new Set(["61"]))).toEqual({
      gone: [],
      added: [],
      moved: [6],
    });
    expect(settleGoneCards(diff, new Set(["81"]))).toEqual({
      gone: [6, 61],
      added: [],
      moved: [],
    });
  });

  it("keeps the diff's own news and names a picture moved once", () => {
    const diff = {
      gone: [6],
      added: [9],
      moved: [6, 5],
      goneCards: [[6]],
    };
    expect(settleGoneCards(diff, new Set(["6"]))).toEqual({
      gone: [],
      added: [9],
      moved: [6, 5],
    });
  });
});

describe("useGridRealtimeSync - applyReconnectDiff", () => {
  it("takes gone pictures out of the grid, without a reload or a pill", () => {
    const h = makeHarness();
    const res = h.sync.applyReconnectDiff({ gone: [3], added: [], moved: [] });
    expect(res).toEqual({ action: "targeted", reason: "reconnect-removed" });
    expect(h.grid.removeImagesById).toHaveBeenCalledWith([3]);
    expect(h.wsStore.addPendingExternalImportIds).not.toHaveBeenCalled();
    expect(h.wsStore.addSortChangedExternalIds).not.toHaveBeenCalled();
    expect(h.reload).not.toHaveBeenCalled();
  });

  it("puts new pictures on the New pictures pill and moves on the View changed pill", () => {
    const h = makeHarness();
    const res = h.sync.applyReconnectDiff({
      gone: [3],
      added: [9],
      moved: [5],
    });
    expect(res).toEqual({ action: "pill", reason: "reconnect-pills" });
    expect(h.grid.removeImagesById).toHaveBeenCalledWith([3]);
    expect(h.wsStore.addPendingExternalImportIds).toHaveBeenCalledWith([9]);
    expect(h.wsStore.addSortChangedExternalIds).toHaveBeenCalledWith([5]);
    expect(h.grid.insertGridImagesById).not.toHaveBeenCalled();
    expect(h.reload).not.toHaveBeenCalled();
  });

  it("under the lightbox, still takes gone pictures out but holds the pills for its close", () => {
    const h = makeHarness({ lightboxOpen: true });
    const res = h.sync.applyReconnectDiff({
      gone: [3],
      added: [9],
      moved: [5],
    });
    expect(res).toEqual({
      action: "targeted",
      reason: "reconnect-lightbox-deferred",
    });
    expect(h.grid.removeImagesById).toHaveBeenCalledWith([3]);
    expect(h.grid.markLightboxDeferredRefresh).toHaveBeenCalledTimes(1);
    expect(h.wsStore.addPendingExternalImportIds).not.toHaveBeenCalled();
    expect(h.wsStore.addSortChangedExternalIds).not.toHaveBeenCalled();
    expect(h.reload).not.toHaveBeenCalled();
  });

  it("does nothing when the gap changed nothing", () => {
    const h = makeHarness();
    const res = h.sync.applyReconnectDiff({ gone: [], added: [], moved: [] });
    expect(res).toEqual({ action: "ignored", reason: "reconnect-unchanged" });
    expect(h.grid.removeImagesById).not.toHaveBeenCalled();
    expect(h.grid.markLightboxDeferredRefresh).not.toHaveBeenCalled();
  });

  it("does not offer a picture this tab is already about to insert", () => {
    const flushes = [];
    const h = makeHarness({
      scheduler: { schedule: (flush) => flushes.push(flush), cancel() {} },
    });
    h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [9],
      change_kind: "added",
    });
    const res = h.sync.applyReconnectDiff({ gone: [], added: [9], moved: [] });
    expect(res).toEqual({ action: "ignored", reason: "reconnect-unchanged" });
    flushes.forEach((flush) => flush());
    expect(h.grid.insertGridImagesById).toHaveBeenCalledWith([9]);
    expect(h.wsStore.addPendingExternalImportIds).not.toHaveBeenCalled();
  });

  it("nor one it is already about to put back from the Scrapheap", () => {
    const flushes = [];
    const h = makeHarness({
      scheduler: { schedule: (flush) => flushes.push(flush), cancel() {} },
    });
    h.sync.handleMessage({
      type: "pictures_changed",
      source: "ui",
      origin_client_id: OTHER_ID,
      picture_ids: [4],
      change_kind: "restored",
    });
    const res = h.sync.applyReconnectDiff({ gone: [], added: [4], moved: [] });
    expect(res).toEqual({ action: "ignored", reason: "reconnect-unchanged" });
    flushes.forEach((flush) => flush());
    expect(h.grid.insertGridImagesById).toHaveBeenCalledWith([4], {
      highlight: false,
    });
    expect(h.wsStore.addPendingExternalImportIds).not.toHaveBeenCalled();
  });
});
