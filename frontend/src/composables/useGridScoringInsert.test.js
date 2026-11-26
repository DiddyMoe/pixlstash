// useGridScoring - insertGridImagesById puts each new row at its sorted place.
//
// The grid is in sort order, so the place is found by halving rather than by
// reading every row's key: a run restoring 1,000 pictures into a 50,000-row
// grid sorted by date parsed 50,000 dates per picture and held the page for
// seconds.

import { afterEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import { createPinia, setActivePinia } from "pinia";

import { useGridScoring } from "./useGridScoring.js";
import { useSortStore } from "../stores/useSortStore";
import { listPicturesByIds } from "../api/pictures";

vi.mock("../api/pictures", () => ({
  applyScores: vi.fn(() => Promise.resolve()),
  getGuestScores: vi.fn(),
  listPicturesByIds: vi.fn(),
  submitGuestScores: vi.fn(),
}));

const DAY = 24 * 60 * 60 * 1000;
const at = (day) => new Date(Date.UTC(2026, 0, 1) + day * DAY).toISOString();

function grid(rows, { descending = true } = {}) {
  setActivePinia(createPinia());
  const sortStore = useSortStore();
  sortStore.selectedSort = "DATE";
  sortStore.selectedDescending = descending;
  const lastFetchedGridImages = ref(rows);
  const scoring = useGridScoring({
    backendUrl: "",
    allGridImages: ref([]),
    lastFetchedGridImages,
    loadedRanges: ref([]),
    visibleStart: ref(0),
    visibleEnd: ref(0),
    renderBuffer: ref(0),
    imagesLoading: ref(false),
    lightboxOpen: ref(false),
    pendingLightboxGridRefresh: ref(false),
    preserveScrollOnNextFetch: ref(false),
    skipNextWsRefresh: ref(false),
    gridContainer: ref(null),
    guestSessionId: ref(null),
    guestConsentState: ref(null),
    guestScoreMap: ref(new Map()),
    guestConsentBannerVisible: ref(false),
    pendingGuestScoreIntent: ref(null),
    emit: vi.fn(),
    debouncedFetchAllGridImages: vi.fn(),
    fetchImageInfo: vi.fn(),
    rebuildGridImagesFromLastFetch: vi.fn(),
    triggerNewImageHighlight: vi.fn(),
    updateVisibleThumbnails: vi.fn(),
    removeImagesById: vi.fn(),
    rowFitsMediaFilter: () => true,
  });
  return { scoring, lastFetchedGridImages };
}

const ids = (rows) => rows.map((row) => row.id);

afterEach(() => {
  vi.restoreAllMocks();
});

describe("insertGridImagesById", () => {
  it("puts rows at their places in a newest-first grid, after the rows they tie with", async () => {
    const { scoring, lastFetchedGridImages } = grid([
      { id: 1, created_at: at(9) },
      { id: 2, created_at: at(7) },
      { id: 3, created_at: at(5) },
      { id: 4, created_at: at(3) },
    ]);
    listPicturesByIds.mockResolvedValue([
      { id: 10, created_at: at(10) },
      { id: 11, created_at: at(5) },
      { id: 12, created_at: at(6) },
      { id: 13, created_at: at(1) },
    ]);

    await scoring.insertGridImagesById([10, 11, 12, 13]);

    expect(ids(lastFetchedGridImages.value)).toEqual([
      10, 1, 2, 12, 3, 11, 4, 13,
    ]);
  });

  it("puts rows at their places in an oldest-first grid", async () => {
    const { scoring, lastFetchedGridImages } = grid(
      [
        { id: 1, created_at: at(3) },
        { id: 2, created_at: at(5) },
        { id: 3, created_at: at(7) },
      ],
      { descending: false },
    );
    listPicturesByIds.mockResolvedValue([
      { id: 10, created_at: at(1) },
      { id: 11, created_at: at(5) },
      { id: 12, created_at: at(9) },
    ]);

    await scoring.insertGridImagesById([10, 11, 12]);

    expect(ids(lastFetchedGridImages.value)).toEqual([10, 1, 2, 11, 3, 12]);
  });

  it("inserts a row the answer carries twice once", async () => {
    const { scoring, lastFetchedGridImages } = grid([
      { id: 1, created_at: at(9) },
      { id: 2, created_at: at(3) },
    ]);
    listPicturesByIds.mockResolvedValue([
      { id: 10, created_at: at(5) },
      { id: 10, created_at: at(5) },
    ]);

    await scoring.insertGridImagesById([10]);

    expect(ids(lastFetchedGridImages.value)).toEqual([1, 10, 2]);
  });

  it("reads a few dozen keys, not every row's, to place a row in a 50,000-row grid", async () => {
    const rows = Array.from({ length: 50000 }, (_, i) => ({
      id: i + 1,
      created_at: at(-i),
    }));
    const { scoring, lastFetchedGridImages } = grid(rows);
    listPicturesByIds.mockResolvedValue([
      { id: 60001, created_at: at(-60000) },
    ]);
    const parse = vi.spyOn(Date, "parse");

    await scoring.insertGridImagesById([60001]);

    expect(lastFetchedGridImages.value.at(-1).id).toBe(60001);
    expect(lastFetchedGridImages.value).toHaveLength(50001);
    expect(parse.mock.calls.length).toBeLessThan(40);
  });
});
