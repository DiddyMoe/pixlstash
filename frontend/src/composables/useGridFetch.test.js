import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { ref, reactive, effectScope, customRef, toRaw } from "vue";
import { setActivePinia, createPinia } from "pinia";
import { useSortStore } from "../stores/useSortStore.js";
import { useFilterStore } from "../stores/useFilterStore.js";
import { useSelectionStore } from "../stores/useSelectionStore.js";
import { useInboxStore } from "../stores/useInboxStore.js";
import { useDedupStore } from "../stores/useDedupStore.js";
import { useProjectStore } from "../stores/useProjectStore.js";
import { useGridFetch } from "./useGridFetch.js";
import { useSearchStore } from "../stores/useSearchStore.js";
import { SCRAPHEAP_PICTURES_ID } from "../stores/useViewStore";
import { filterChips } from "../utils/filterChips.js";
import { notifySessionReset, sessionContext } from "../utils/apiClient";
import {
  getPictureCount,
  getLikenessGroups,
  faceSearch,
  characterFaceSearch,
  setLikenessSearch,
  likenessSearch,
  listPicturesByIds,
  searchPictures,
  streamPictures,
} from "../api/pictures";

// Mock the pictures API module so streaming-path tests can control the count
// and stream responses. The lightbox-defer test below returns before any
// network call, so the mock is inert there.
vi.mock("../api/pictures", () => ({
  getPictureCount: vi.fn(),
  streamPictures: vi.fn(),
  getLikenessGroups: vi.fn(),
  faceSearch: vi.fn(),
  characterFaceSearch: vi.fn(),
  setLikenessSearch: vi.fn(),
  likenessSearch: vi.fn(),
  searchPictures: vi.fn(),
  listPicturesByIds: vi.fn(),
}));

// Each harness runs in its own effect scope, as it would inside ImageGrid's
// setup; stopping it is the component unmounting.
const harnessScopes = [];
afterEach(() => {
  while (harnessScopes.length) harnessScopes.pop().stop();
});

// Build a minimal harness for useGridFetch. Covers the lightbox-defer path
// (returns before any network call) and the streaming path (count + stream
// mocked above). Sort, selection and filter state come from the stores;
// `storeOverrides` writes them.
function makeHarness({
  lightboxOpen = false,
  selectedSort = "DATE_TAKEN",
  storeOverrides = {},
  extraRefs = {},
} = {}) {
  // The composable reads the query's filter, sort and selection facets
  // straight from the stores, so every harness needs a live Pinia.
  setActivePinia(createPinia());
  const startSmartScoreProgress = vi.fn();
  const completeSmartScoreProgress = vi.fn();

  const refs = {
    allGridImages: ref([]),
    lastFetchedGridImages: ref([]),
    scrollWrapper: ref(null),
    preserveScrollOnNextFetch: ref(false),
    pendingScrollTop: ref(null),
    lightboxOpen: ref(lightboxOpen),
    pendingGridImages: ref(null),
    pendingLightboxGridRefresh: ref(false),
    visibleStart: ref(0),
    visibleEnd: ref(0),
    divisibleViewWindow: ref(40),
    initialRender: ref(false),
    rowHeight: ref(128),
    sharedPictureIds: ref(new Set()),
    guestConsentState: ref(null),
    guestSessionId: ref(null),
    highlightNextFetch: ref(false),
    hasLoadedOnce: ref(false),
    previousImageIds: new Set(),
    normalizedSelectedCharacterIds: ref([]),
    normalizedSelectedSetIds: ref([]),
    hasSetSelection: ref(false),
    isSetOverlapView: ref(false),
    isMultiCharacterView: ref(false),
    primarySelectedSetId: ref(null),
    smartScoreProgress: reactive({ visible: false, percent: 0, message: "" }),
    exportProgress: reactive({ visible: false, percent: 0, message: "" }),
    reverseImageSearchPictureIds: ref([]),
    faceLikenessSearchFaceId: ref(null),
    faceSearchCharacter: ref(null),
    faceSearchRanked: ref(null),
    setSuggestSet: ref(null),
    setSuggestRanked: ref(null),
    textSearchResults: ref(null),
  };
  Object.assign(refs, extraRefs);

  const sortStore = useSortStore();
  sortStore.selectedSort = selectedSort;
  for (const [key, value] of Object.entries(storeOverrides)) {
    sortStore[key] = value;
  }

  const props = reactive({ backendUrl: "http://test" });

  const callbacks = {
    collapseStackImages: (x) => x,
    mapGridImages: (x) => x,
    syncExpandAllStacksFromFetchedImages: vi.fn(),
    refreshExpandedStacksAfterFetch: vi.fn(),
    resetThumbnailState: vi.fn(),
    triggerNewImageHighlight: vi.fn(),
    updateVisibleThumbnails: vi.fn(),
    fetchThumbnailsBatch: vi.fn(),
    startSmartScoreProgress,
    completeSmartScoreProgress,
    onGridFetchStart: vi.fn(),
    onGridVisibleMetadataReady: vi.fn(),
    onGridFetchDone: vi.fn(),
  };

  const scope = effectScope();
  harnessScopes.push(scope);
  const grid = scope.run(() => useGridFetch(refs, props, callbacks));
  return {
    grid,
    refs,
    callbacks,
    scope,
    startSmartScoreProgress,
    completeSmartScoreProgress,
  };
}

// A promise the test settles by hand, to hold a read in flight.
function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

// A read that never answers on its own and rejects the way Axios does once its
// signal is aborted.
function readUntilAborted(signal) {
  return new Promise((_, reject) => {
    signal.addEventListener("abort", () => {
      const error = new Error("canceled");
      error.name = "CanceledError";
      reject(error);
    });
  });
}

// The stream answers by view: the person-7 view has one picture, the library
// view two, so a row in the grid says which load wrote it.
function streamByView(query) {
  return Promise.resolve({
    pictures: query.includes("character_id=7")
      ? [{ id: 70 }]
      : [{ id: 1 }, { id: 2 }],
  });
}

describe("useGridFetch sort-progress lifecycle", () => {
  it("dismisses the sort progress bar when a sorted fetch is deferred for an open overlay", async () => {
    const { grid, refs, startSmartScoreProgress, completeSmartScoreProgress } =
      makeHarness({ lightboxOpen: true });

    await grid.fetchAllGridImages({ force: true, showProgress: true });

    // The bar was started…
    expect(startSmartScoreProgress).toHaveBeenCalledTimes(1);
    // …the refresh was deferred to lightbox-close…
    expect(refs.pendingLightboxGridRefresh.value).toBe(true);
    // …and crucially the bar was dismissed instead of being stranded forever.
    expect(completeSmartScoreProgress).toHaveBeenCalledTimes(1);
  });
});

describe("useGridFetch streaming path", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("sends sort, descending and reference_character_id on the count request for CHARACTER_LIKENESS", async () => {
    const pics = [{ id: 11 }, { id: 12 }, { id: 13 }];
    getPictureCount.mockResolvedValue({ count: pics.length });
    streamPictures.mockResolvedValue({ pictures: pics });

    const { grid } = makeHarness({
      selectedSort: "CHARACTER_LIKENESS",
      storeOverrides: {
        selectedSimilarityCharacter: "7",
        selectedDescending: true,
      },
    });

    await grid.fetchAllGridImages({ force: true });

    expect(getPictureCount).toHaveBeenCalledTimes(1);
    const countQuery = getPictureCount.mock.calls[0][0];
    // The count must run over the same row set as the stream: for
    // CHARACTER_LIKENESS the sort changes which rows exist at all.
    expect(countQuery).toContain("sort=CHARACTER_LIKENESS");
    expect(countQuery).toContain("descending=true");
    expect(countQuery).toContain("reference_character_id=7");
  });

  it("tells the stack fold to keep every row of the Scrapheap, and fold a person's", async () => {
    getPictureCount.mockResolvedValue({ count: 2 });
    streamPictures.mockResolvedValue({
      pictures: [
        { id: 1, stack_id: 7 },
        { id: 2, stack_id: 7 },
      ],
      done: true,
      next_offset: 2,
    });
    const { grid } = makeHarness();
    const selection = useSelectionStore();

    selection.selectedCharacter = String(SCRAPHEAP_PICTURES_ID);
    await grid.fetchAllGridImages({ force: true });
    expect(streamPictures.mock.calls[0][0]).toContain("only_deleted=true");
    expect(grid.loadFoldsStacks.value).toBe(false);

    selection.selectedCharacter = "7";
    await grid.fetchAllGridImages({ force: true });
    expect(grid.loadFoldsStacks.value).toBe(true);
  });

  it("issues a singleton character fetch while a duplicate scan is running", async () => {
    getPictureCount.mockResolvedValue({ count: 1 });
    streamPictures.mockResolvedValue({
      pictures: [{ id: 17, character_id: 7 }],
      done: true,
      next_offset: 1,
    });

    const { grid, refs, callbacks } = makeHarness();
    useSelectionStore().selectedCharacter = 7;
    // The duplicate store remains alive after leaving its destination. Its
    // progress must never become a fetch gate for the newly mounted grid.
    useDedupStore().scan = {
      status: "running",
      scanned: 5000,
      total: 12098,
    };

    await grid.fetchAllGridImages({ force: true });

    expect(getPictureCount).toHaveBeenCalledTimes(1);
    expect(getPictureCount.mock.calls[0][0]).toContain("character_id=7");
    expect(streamPictures).toHaveBeenCalledTimes(1);
    expect(streamPictures.mock.calls[0][0]).toContain("character_id=7");
    expect(refs.allGridImages.value.map((picture) => picture.id)).toEqual([17]);
    expect(callbacks.fetchThumbnailsBatch).toHaveBeenCalledWith(0, 1, {
      reason: "initial-visible-prefetch",
    });
  });

  it("narrows a person's count and stream to the Likeness cluster, never the menu's count base", async () => {
    getPictureCount.mockResolvedValue({ count: 1 });
    streamPictures.mockResolvedValue({
      pictures: [{ id: 17 }],
      done: true,
      next_offset: 1,
    });
    const { grid } = makeHarness();
    useSelectionStore().selectedCharacter = 7;
    useFilterStore().faceClusterFilter = {
      cluster: 2,
      strictness: "strict",
      threshold: 0.55,
      label: "cluster 2 · strict",
    };

    await grid.fetchAllGridImages({ force: true });

    const cluster = "face_cluster=2&face_cluster_threshold=0.55";
    expect(getPictureCount.mock.calls[0][0]).toContain(cluster);
    expect(streamPictures.mock.calls[0][0]).toContain(cluster);
    expect(grid.buildPictureIdsQueryParams()).toContain(cluster);
    expect(grid.buildFilterCountBaseQuery()).not.toContain("face_cluster");
  });

  it("trims trailing placeholders when the stream yields fewer rows than the count", async () => {
    const pics = [{ id: 21 }, { id: 22 }, { id: 23 }];
    // Count says 5, stream only ever delivers 3 - the last 2 cells would
    // otherwise remain permanent id-less spinners.
    getPictureCount.mockResolvedValue({ count: 5 });
    streamPictures.mockResolvedValue({ pictures: pics });

    const { grid, refs } = makeHarness({
      selectedSort: "CHARACTER_LIKENESS",
      storeOverrides: {
        selectedSimilarityCharacter: "7",
        selectedDescending: true,
      },
    });

    await grid.fetchAllGridImages({ force: true });

    expect(refs.allGridImages.value).toHaveLength(3);
    expect(refs.allGridImages.value.every((img) => img.id != null)).toBe(true);
    // visibleEnd was sized from the count (5) and must be clamped to the
    // trimmed length.
    expect(refs.visibleEnd.value).toBe(3);
    expect(refs.lastFetchedGridImages.value).toHaveLength(3);
  });

  // The Stacks segments moved the store and the grid refetched, but the default
  // grid path built its own param set and left stack_state out of it, so the
  // same rows came back and the control looked dead. Assert the param on both
  // the stream and the count: they must run over the same row set.
  it("sends stack_state on the stream and count requests when the Stacks filter is set", async () => {
    const pics = [{ id: 31 }, { id: 32 }];
    getPictureCount.mockResolvedValue({ count: pics.length });
    streamPictures.mockResolvedValue({ pictures: pics });

    const { grid } = makeHarness();
    useFilterStore().stackStateFilter = "unresolved";

    await grid.fetchAllGridImages({ force: true });

    expect(streamPictures).toHaveBeenCalled();
    expect(streamPictures.mock.calls[0][0]).toContain("stack_state=unresolved");
    expect(getPictureCount).toHaveBeenCalled();
    expect(getPictureCount.mock.calls[0][0]).toContain(
      "stack_state=unresolved",
    );
  });

  it("sends tag_state on the stream and count requests when an Untagged row is on", async () => {
    getPictureCount.mockResolvedValue({ count: 1 });
    streamPictures.mockResolvedValue({ pictures: [{ id: 41 }] });

    const { grid } = makeHarness();
    useFilterStore().tagStateFilter = "none";

    await grid.fetchAllGridImages({ force: true });

    expect(streamPictures.mock.calls[0][0]).toContain("tag_state=none");
    expect(getPictureCount.mock.calls[0][0]).toContain("tag_state=none");
  });

  // *Show all N pictures* on a workflow card (F7). The chip is set on one
  // screen and read on another, so the half that could rot in silence is this
  // one: the strip would go on showing "Workflow: X" over the whole library.
  it("sends workflow on the stream and count requests when a workflow chip is on", async () => {
    getPictureCount.mockResolvedValue({ count: 2 });
    streamPictures.mockResolvedValue({ pictures: [{ id: 51 }, { id: 52 }] });

    const { grid } = makeHarness();
    useFilterStore().workflowFilter = [{ id: "auto:abc123", name: "Cinematic" }];

    await grid.fetchAllGridImages({ force: true });

    expect(streamPictures.mock.calls[0][0]).toContain("workflow=auto%3Aabc123");
    expect(getPictureCount.mock.calls[0][0]).toContain("workflow=auto%3Aabc123");
  });

  it("sends one workflow param per ticked workflow, and no LoRA beside two (#1797)", async () => {
    getPictureCount.mockResolvedValue({ count: 2 });
    streamPictures.mockResolvedValue({ pictures: [{ id: 51 }, { id: 52 }] });

    const { grid } = makeHarness();
    useFilterStore().workflowFilter = [
      { id: "auto:a", name: "A", lora: "asset:feed", loraName: "Ada" },
      { id: "manual:b", name: "B" },
    ];

    await grid.fetchAllGridImages({ force: true });

    for (const url of [
      streamPictures.mock.calls[0][0],
      getPictureCount.mock.calls[0][0],
    ]) {
      expect(new URLSearchParams(url).getAll("workflow")).toEqual([
        "auto:a",
        "manual:b",
      ]);
      expect(url).not.toContain("workflow_lora");
    }
  });

  it("sends the workflow and the LoRA for the Workflow tab's *Show N*", async () => {
    getPictureCount.mockResolvedValue({ count: 1 });
    streamPictures.mockResolvedValue({ pictures: [{ id: 51 }] });

    const { grid } = makeHarness();
    useFilterStore().workflowFilter = [
      {
        id: "auto:core",
        lora: "asset:feed",
        name: "Ada in Cinematic",
      },
    ];

    await grid.fetchAllGridImages({ force: true });

    for (const url of [
      streamPictures.mock.calls[0][0],
      getPictureCount.mock.calls[0][0],
    ]) {
      expect(url).toContain("workflow=auto%3Acore");
      expect(url).toContain("workflow_lora=asset%3Afeed");
      expect(url).not.toContain("workflow_key");
      expect(url).not.toContain("workflow_stack");
    }
  });

  it("omits workflow once the chip is removed", async () => {
    getPictureCount.mockResolvedValue({ count: 1 });
    streamPictures.mockResolvedValue({ pictures: [{ id: 61 }] });

    const { grid } = makeHarness();
    const filterStore = useFilterStore();
    filterStore.workflowFilter = [{ id: "auto:abc123", name: "Cinematic" }];
    // What the chip's × does, through the same list the strip draws from.
    filterChips(filterStore)
      .find((chip) => chip.key === "workflow:auto:abc123")
      .remove();

    await grid.fetchAllGridImages({ force: true });

    expect(filterStore.workflowFilter).toEqual([]);
    expect(streamPictures.mock.calls[0][0]).not.toContain("workflow=");
    expect(getPictureCount.mock.calls[0][0]).not.toContain("workflow=");
  });

  // "No character" and "In no set" are independent (#1488): each alone narrows
  // the unassigned view to its own half, both together are the whole of it.
  it.each([
    [true, false, "unassigned_by=character"],
    [false, true, "unassigned_by=set"],
    [true, true, null],
  ])(
    "no character %s, in no set %s sends %s",
    async (noCharacter, noSet, expected) => {
      getPictureCount.mockResolvedValue({ count: 1 });
      streamPictures.mockResolvedValue({ pictures: [{ id: 71 }] });

      const { grid } = makeHarness();
      const filterStore = useFilterStore();
      filterStore.noCharacterFilter = noCharacter;
      filterStore.noSetFilter = noSet;

      await grid.fetchAllGridImages({ force: true });

      for (const url of [
        streamPictures.mock.calls[0][0],
        getPictureCount.mock.calls[0][0],
      ]) {
        expect(url).toContain("character_id=UNASSIGNED");
        if (expected) expect(url).toContain(expected);
        else expect(url).not.toContain("unassigned_by");
      }
    },
  );

  // "all" is the absence of the filter, not a value the backend knows.
  it("omits stack_state when the Stacks filter is on Any", async () => {
    getPictureCount.mockResolvedValue({ count: 1 });
    streamPictures.mockResolvedValue({ pictures: [{ id: 41 }] });

    const { grid } = makeHarness();
    useFilterStore().stackStateFilter = "all";

    await grid.fetchAllGridImages({ force: true });

    expect(streamPictures.mock.calls[0][0]).not.toContain("stack_state");
    expect(getPictureCount.mock.calls[0][0]).not.toContain("stack_state");
  });
});

describe("useGridFetch text search: narrowing to text matches (#1197)", () => {
  const ROWS = [
    { id: 1, text_match: true },
    { id: 2, text_match: false },
    { id: 3, text_match: true },
  ];

  beforeEach(() => {
    searchPictures.mockReset();
    searchPictures.mockResolvedValue(ROWS);
  });

  it("narrows the fetched rows to text matches without searching again", async () => {
    const { grid, refs } = makeHarness();
    const searchStore = useSearchStore();
    searchStore.searchQuery = "coffee";

    await grid.fetchAllGridImages({ force: true });
    expect(refs.allGridImages.value.map((r) => r.id)).toEqual([1, 2, 3]);
    expect(refs.textSearchResults.value.rows).toHaveLength(3);

    searchStore.textMatchesOnly = true;
    await grid.fetchAllGridImages({ textMatchRecut: true });
    expect(refs.allGridImages.value.map((r) => r.id)).toEqual([1, 3]);

    searchStore.textMatchesOnly = false;
    await grid.fetchAllGridImages({ textMatchRecut: true });
    expect(refs.allGridImages.value.map((r) => r.id)).toEqual([1, 2, 3]);
    expect(searchPictures).toHaveBeenCalledTimes(1);
  });

  it("any other fetch searches again", async () => {
    const { grid } = makeHarness();
    useSearchStore().searchQuery = "coffee";
    await grid.fetchAllGridImages({ force: true });
    await grid.fetchAllGridImages({ force: true });
    expect(searchPictures).toHaveBeenCalledTimes(2);
  });
});

describe("useGridFetch: the Inbox and Imported from", () => {
  it("asks for the Inbox's period, and follows it when it changes", () => {
    const { grid } = makeHarness();
    useSelectionStore().selectedFolderFilter = { inbox: true, label: "Inbox" };
    const listing = new URLSearchParams(grid.buildPictureIdsQueryParams());
    expect(listing.get("inbox")).toBe("14d");
    // The filter menu counts from the view, and the Inbox IS the view.
    const base = new URLSearchParams(grid.buildFilterCountBaseQuery());
    expect(base.get("inbox")).toBe("14d");
    useInboxStore().period = "90d";
    expect(
      new URLSearchParams(grid.buildPictureIdsQueryParams()).get("inbox"),
    ).toBe("90d");
  });

  it("asks for no Inbox outside it, whatever the period", () => {
    const { grid } = makeHarness();
    useInboxStore().period = "90d";
    const listing = new URLSearchParams(grid.buildPictureIdsQueryParams());
    expect(listing.has("inbox")).toBe(false);
  });

  it("narrows any view by Imported from, without moving the menu's base", () => {
    const { grid } = makeHarness();
    useFilterStore().importedFromFilter = "/home/me/renders";
    const listing = new URLSearchParams(grid.buildPictureIdsQueryParams());
    expect(listing.get("import_source_folder")).toBe("/home/me/renders");
    expect(listing.has("inbox")).toBe(false);
    // A filter, not the view: its row in the menu adds it to the base itself.
    const base = new URLSearchParams(grid.buildFilterCountBaseQuery());
    expect(base.has("import_source_folder")).toBe(false);
  });
});

describe("useGridFetch: the same-view key follows what the reads ask for", () => {
  afterEach(() => {
    sessionContext.value = null;
  });

  it.each([
    ["minScoreFilter", 3],
    ["maxScoreFilter", 4],
    ["unscoredOnlyFilter", true],
    ["smartScoreBucketFilter", "3-4"],
    ["resolutionBucketFilter", "lt1mp"],
    ["tagFilter", ["test-hat"]],
    ["tagRejectedFilter", ["test-hat"]],
    ["tagConfidenceAboveFilter", ["test-hat:0.80"]],
    ["tagConfidenceBelowFilter", ["test-hat:0.80"]],
    ["faceBboxFilter", "with_face"],
    ["sharedOnlyFilter", true],
  ])("changes when %s changes", (name, value) => {
    const { grid } = makeHarness();
    const before = grid.buildGridFetchKey();

    useFilterStore()[name] = value;

    expect(grid.buildGridFetchKey()).not.toBe(before);
  });

  it("changes when the selected sets move to another project", () => {
    const { grid, refs } = makeHarness();
    useSelectionStore().selectedSetIds = [3, 4];
    refs.hasSetSelection.value = true;
    refs.isSetOverlapView.value = true;
    refs.normalizedSelectedSetIds.value = [3, 4];
    const projectStore = useProjectStore();
    projectStore.projectViewMode = "project";
    projectStore.setProjectIds = { 3: 1, 4: 1 };
    const before = grid.buildGridFetchKey();

    projectStore.setProjectIds = { 3: 2, 4: 2 };

    expect(grid.buildGridFetchKey()).not.toBe(before);
  });

  it("changes with the session of a guest who declined scoring consent", () => {
    const { grid, refs } = makeHarness();
    sessionContext.value = { scope: "READ" };
    refs.guestConsentState.value = "rejected";
    refs.guestSessionId.value = "test-guest-session-1";
    const before = grid.buildGridFetchKey();

    refs.guestSessionId.value = "test-guest-session-2";

    expect(grid.buildGridFetchKey()).not.toBe(before);
  });

  describe("the key carries what the reads send", () => {
    beforeEach(() => {
      for (const read of [
        getPictureCount,
        streamPictures,
        getLikenessGroups,
        faceSearch,
        characterFaceSearch,
        setLikenessSearch,
        likenessSearch,
        searchPictures,
        listPicturesByIds,
      ]) {
        read.mockReset();
      }
      streamPictures.mockImplementation(streamByView);
    });

    it.each([
      [
        "stream count",
        () => {},
        () =>
          getPictureCount.mock.calls[0][0].replace(
            "stack_leaders_only=true",
            "",
          ),
      ],
      [
        "likeness groups",
        () => {
          useSortStore().selectedSort = "LIKENESS_GROUPS";
        },
        () => getLikenessGroups.mock.calls[0][1],
      ],
      [
        "text search",
        () => {
          useSearchStore().searchQuery = "coffee";
        },
        () => searchPictures.mock.calls[0][1].query,
      ],
    ])("carries the query the %s read sends", async (_read, arrange, sent) => {
      getPictureCount.mockResolvedValue({ count: 1 });
      getLikenessGroups.mockResolvedValue([{ id: 1 }]);
      searchPictures.mockResolvedValue([{ id: 1 }]);
      const { grid } = makeHarness();
      useFilterStore().minScoreFilter = 3;
      arrange();

      await grid.fetchAllGridImages({ force: true });

      const query = sent();
      expect(query).toContain("min_score=3");
      expect(Object.values(JSON.parse(grid.buildGridFetchKey()))).toContain(
        query,
      );
    });
  });
});

describe("useGridFetch: one load at a time", () => {
  beforeEach(() => {
    for (const read of [
      getPictureCount,
      streamPictures,
      getLikenessGroups,
      faceSearch,
      characterFaceSearch,
      setLikenessSearch,
      likenessSearch,
      searchPictures,
      listPicturesByIds,
    ]) {
      read.mockReset();
    }
    streamPictures.mockImplementation(streamByView);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shares the running read with a second call for the same view that joins it", async () => {
    const count = deferred();
    getPictureCount.mockReturnValueOnce(count.promise);
    const { grid, refs } = makeHarness();

    const first = grid.fetchAllGridImages();
    const second = grid.fetchAllGridImages({ join: true });
    expect(getPictureCount).toHaveBeenCalledTimes(1);

    count.resolve({ count: 2 });
    await Promise.all([first, second]);
    expect(getPictureCount).toHaveBeenCalledTimes(1);
    expect(streamPictures).toHaveBeenCalledTimes(1);
    expect(refs.allGridImages.value.map((img) => img.id)).toEqual([1, 2]);
  });

  it("never lets a superseded load write its rows, even within one millisecond", async () => {
    vi.spyOn(Date, "now").mockReturnValue(1_700_000_000_000);
    const libraryCount = deferred();
    getPictureCount
      .mockReturnValueOnce(libraryCount.promise)
      .mockResolvedValueOnce({ count: 1 });
    const { grid, refs } = makeHarness();

    const libraryLoad = grid.fetchAllGridImages();
    useSelectionStore().selectedCharacter = 7;
    await grid.fetchAllGridImages();
    expect(refs.allGridImages.value.map((img) => img.id)).toEqual([70]);

    // The library's count answers late, after the person view has loaded.
    libraryCount.resolve({ count: 2 });
    await libraryLoad;
    expect(refs.allGridImages.value.map((img) => img.id)).toEqual([70]);
    expect(streamPictures).toHaveBeenCalledTimes(1);
    expect(grid.lastFetchSuccess.value.key).toBe(grid.buildGridFetchKey());
  });

  it("cancels the running read when the view changes", async () => {
    getPictureCount
      .mockImplementationOnce((_query, { signal }) => readUntilAborted(signal))
      .mockResolvedValueOnce({ count: 1 });
    const { grid } = makeHarness();

    const libraryLoad = grid.fetchAllGridImages();
    const librarySignal = getPictureCount.mock.calls[0][1].signal;
    useSelectionStore().selectedCharacter = 7;
    const personLoad = grid.fetchAllGridImages();

    expect(librarySignal.aborted).toBe(true);
    expect(getPictureCount.mock.calls[1][1].signal.aborted).toBe(false);
    await Promise.all([libraryLoad, personLoad]);
  });

  it("a filter change during a load cancels it and never publishes its rows", async () => {
    const heldCount = deferred();
    getPictureCount
      .mockReturnValueOnce(heldCount.promise)
      .mockResolvedValueOnce({ count: 1 });
    streamPictures.mockImplementation((query) =>
      Promise.resolve({
        pictures: query.includes("min_score=5")
          ? [{ id: 50 }]
          : [{ id: 1 }, { id: 2 }],
      }),
    );
    const { grid, refs } = makeHarness();

    const unfiltered = grid.fetchAllGridImages();
    const unfilteredSignal = getPictureCount.mock.calls[0][1].signal;
    useFilterStore().minScoreFilter = 5;
    const filtered = grid.fetchAllGridImages({ force: true });

    expect(unfilteredSignal.aborted).toBe(true);
    expect(getPictureCount).toHaveBeenCalledTimes(2);
    expect(getPictureCount.mock.calls[1][0]).toContain("min_score=5");

    // The cancelled load's count answers anyway: its rows must not land.
    heldCount.resolve({ count: 2 });
    await Promise.all([unfiltered, filtered]);
    expect(streamPictures).toHaveBeenCalledTimes(1);
    expect(refs.allGridImages.value.map((img) => img.id)).toEqual([50]);
  });

  it("a filter change during a text search cancels it and never publishes its rows", async () => {
    const heldSearch = deferred();
    searchPictures
      .mockReturnValueOnce(heldSearch.promise)
      .mockResolvedValueOnce([{ id: 50 }]);
    const { grid, refs } = makeHarness();
    useSearchStore().searchQuery = "coffee";

    const unfiltered = grid.fetchAllGridImages();
    await vi.waitFor(() => expect(searchPictures).toHaveBeenCalledTimes(1));
    const unfilteredSignal = searchPictures.mock.calls[0][1].signal;
    useFilterStore().minScoreFilter = 5;
    const filtered = grid.fetchAllGridImages({ force: true });

    expect(unfilteredSignal.aborted).toBe(true);
    await vi.waitFor(() => expect(searchPictures).toHaveBeenCalledTimes(2));
    expect(searchPictures.mock.calls[1][1].query).toContain("min_score=5");

    // The cancelled search answers anyway: its rows must not land.
    heldSearch.resolve([{ id: 1 }, { id: 2 }]);
    await Promise.all([unfiltered, filtered]);
    expect(refs.allGridImages.value.map((img) => img.id)).toEqual([50]);
  });

  it("fails a cancelled load silently", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    getPictureCount
      .mockImplementationOnce((_query, { signal }) => readUntilAborted(signal))
      .mockResolvedValueOnce({ count: 1 });
    const { grid, refs } = makeHarness();

    const libraryLoad = grid.fetchAllGridImages();
    useSelectionStore().selectedCharacter = 7;
    const personLoad = grid.fetchAllGridImages();
    const outcome = await libraryLoad;
    await personLoad;

    expect(outcome?.error ?? null).toBe(null);
    expect(grid.imagesError.value).toBe(null);
    expect(grid.lastFetchError.value.key).toBe("");
    expect(refs.allGridImages.value.map((img) => img.id)).toEqual([70]);
    expect(warn).not.toHaveBeenCalledWith(
      "[useGridFetch] Grid fetch failed",
      expect.anything(),
      expect.anything(),
    );
  });

  it("queues exactly one re-read behind the running load for forced calls", async () => {
    const count = deferred();
    getPictureCount
      .mockReturnValueOnce(count.promise)
      .mockResolvedValue({ count: 2 });
    const { grid } = makeHarness();

    const plain = grid.fetchAllGridImages();
    const forcedA = grid.fetchAllGridImages({ force: true });
    const forcedB = grid.fetchAllGridImages({ force: true });
    expect(getPictureCount).toHaveBeenCalledTimes(1);

    count.resolve({ count: 2 });
    await Promise.all([plain, forcedA, forcedB]);
    expect(getPictureCount).toHaveBeenCalledTimes(2);
    expect(streamPictures).toHaveBeenCalledTimes(2);
    // The re-read started only once the first load had read its rows, so it
    // sees whatever changed while that load was running.
    expect(getPictureCount.mock.invocationCallOrder[1]).toBeGreaterThan(
      streamPictures.mock.invocationCallOrder[0],
    );
  });

  it("an unforced call during a load gets exactly one re-read after it", async () => {
    const count = deferred();
    getPictureCount
      .mockReturnValueOnce(count.promise)
      .mockResolvedValueOnce({ count: 1 });
    streamPictures
      .mockResolvedValueOnce({ pictures: [{ id: 1 }, { id: 2 }] })
      .mockResolvedValueOnce({ pictures: [{ id: 2 }] });
    const { grid, refs } = makeHarness();

    const load = grid.fetchAllGridImages();
    // Two refreshes for the same view arrive while the first read is out, as
    // after deleting a picture the read may already have counted.
    const refreshA = grid.fetchAllGridImages();
    const refreshB = grid.fetchAllGridImages();
    expect(getPictureCount).toHaveBeenCalledTimes(1);

    count.resolve({ count: 2 });
    await Promise.all([load, refreshA, refreshB]);
    expect(getPictureCount).toHaveBeenCalledTimes(2);
    expect(streamPictures).toHaveBeenCalledTimes(2);
    expect(refs.allGridImages.value.map((img) => img.id)).toEqual([2]);
  });

  it("re-reads after a set-suggestion load that seats the strength cut", async () => {
    const ranked = deferred();
    const row = (id, likeness) => ({
      picture_id: id,
      likeness,
      cohesion: 0.5,
      tags_matched: 0,
      tags_total: 0,
    });
    setLikenessSearch
      .mockReturnValueOnce(ranked.promise)
      .mockResolvedValue([row(1, 0.9)]);
    listPicturesByIds.mockResolvedValue([{ id: 1 }, { id: 2 }]);
    const { grid, refs } = makeHarness({
      extraRefs: { setSuggestThreshold: ref(null), setSuggestMinTags: ref(0) },
    });
    refs.setSuggestSet.value = { id: 6 };

    const load = grid.fetchAllGridImages({ force: true });
    const refresh = grid.fetchAllGridImages();
    ranked.resolve([row(1, 0.9), row(2, 0.8)]);
    await Promise.all([load, refresh]);

    expect(setLikenessSearch).toHaveBeenCalledTimes(2);
  });

  describe("just after a load completes", () => {
    beforeEach(() => {
      // The second call comes within the de-dup windows.
      vi.spyOn(Date, "now").mockReturnValue(1_700_000_000_000);
      getPictureCount
        .mockResolvedValueOnce({ count: 2 })
        .mockResolvedValueOnce({ count: 1 });
      streamPictures
        .mockResolvedValueOnce({ pictures: [{ id: 1 }, { id: 2 }] })
        .mockResolvedValueOnce({ pictures: [{ id: 2 }] });
    });

    it("an unforced call re-reads", async () => {
      const { grid, refs } = makeHarness();

      await grid.fetchAllGridImages();
      await grid.fetchAllGridImages();

      expect(getPictureCount).toHaveBeenCalledTimes(2);
      expect(refs.allGridImages.value.map((img) => img.id)).toEqual([2]);
    });

    it("a call that joins does not", async () => {
      const { grid, refs } = makeHarness();

      await grid.fetchAllGridImages();
      await grid.fetchAllGridImages({ join: true });

      expect(getPictureCount).toHaveBeenCalledTimes(1);
      expect(refs.allGridImages.value.map((img) => img.id)).toEqual([1, 2]);
    });
  });

  describe("just after a load fails", () => {
    beforeEach(() => {
      vi.spyOn(console, "warn").mockImplementation(() => {});
      vi.spyOn(Date, "now").mockReturnValue(1_700_000_000_000);
      getPictureCount
        .mockRejectedValueOnce(new Error("test-count-failed"))
        .mockResolvedValueOnce({ count: 2 });
    });

    it("an unforced call reads again", async () => {
      const { grid, refs } = makeHarness();

      await grid.fetchAllGridImages();
      await grid.fetchAllGridImages();

      expect(getPictureCount).toHaveBeenCalledTimes(2);
      expect(grid.imagesError.value).toBe(null);
      expect(refs.allGridImages.value.map((img) => img.id)).toEqual([1, 2]);
    });

    it("a call that joins does not", async () => {
      const { grid } = makeHarness();

      await grid.fetchAllGridImages();
      await grid.fetchAllGridImages({ join: true });

      expect(getPictureCount).toHaveBeenCalledTimes(1);
      expect(grid.imagesError.value).toBe("test-count-failed");
    });
  });

  it("drops a queued re-read when the view changes before it starts", async () => {
    const count = deferred();
    getPictureCount
      .mockReturnValueOnce(count.promise)
      .mockResolvedValue({ count: 1 });
    const { grid, refs } = makeHarness();

    const libraryLoad = grid.fetchAllGridImages();
    const forced = grid.fetchAllGridImages({ force: true });
    useSelectionStore().selectedCharacter = 7;
    const personLoad = grid.fetchAllGridImages();
    count.resolve({ count: 2 });
    await Promise.all([libraryLoad, forced, personLoad]);

    expect(getPictureCount).toHaveBeenCalledTimes(2);
    expect(getPictureCount.mock.calls[1][0]).toContain("character_id=7");
    expect(refs.allGridImages.value.map((img) => img.id)).toEqual([70]);
  });

  it("drops a queued re-read when the view moves on before its own load is asked for", async () => {
    const count = deferred();
    getPictureCount.mockReturnValueOnce(count.promise);
    const { grid } = makeHarness();

    const libraryLoad = grid.fetchAllGridImages();
    const forced = grid.fetchAllGridImages({ force: true });
    // The selection has changed, but the grid has not yet asked for that view.
    useSelectionStore().selectedCharacter = 7;
    count.resolve({ count: 2 });
    await Promise.all([libraryLoad, forced]);

    expect(getPictureCount).toHaveBeenCalledTimes(1);
  });

  it("drops a queued re-read on a session reset", async () => {
    const count = deferred();
    getPictureCount.mockReturnValueOnce(count.promise);
    const { grid } = makeHarness();

    const load = grid.fetchAllGridImages();
    const forced = grid.fetchAllGridImages({ force: true });
    notifySessionReset("test");
    count.resolve({ count: 2 });
    await Promise.all([load, forced]);

    expect(getPictureCount).toHaveBeenCalledTimes(1);
  });

  it("never lets a finished, superseded load clear the newer load's state", async () => {
    const libraryCount = deferred();
    const personCount = deferred();
    getPictureCount
      .mockReturnValueOnce(libraryCount.promise)
      .mockReturnValueOnce(personCount.promise);
    const { grid } = makeHarness();

    const libraryLoad = grid.fetchAllGridImages();
    useSelectionStore().selectedCharacter = 7;
    const personLoad = grid.fetchAllGridImages();
    libraryCount.resolve({ count: 2 });
    await libraryLoad;

    expect(grid.imagesLoading.value).toBe(true);
    expect(grid.gridReady.value).toBe(false);
    // The person view is still the running load: a call that joins shares it
    // rather than starting a load of its own.
    const joined = grid.fetchAllGridImages({ join: true });
    expect(getPictureCount).toHaveBeenCalledTimes(2);

    personCount.resolve({ count: 1 });
    await Promise.all([personLoad, joined]);
    expect(grid.imagesLoading.value).toBe(false);
  });

  it("cancels the running load on a session reset and reads afresh after it", async () => {
    getPictureCount
      .mockImplementationOnce((_query, { signal }) => readUntilAborted(signal))
      .mockResolvedValue({ count: 2 });
    const { grid, refs } = makeHarness();

    const load = grid.fetchAllGridImages();
    const signal = getPictureCount.mock.calls[0][1].signal;
    notifySessionReset("test");

    expect(signal.aborted).toBe(true);
    expect(grid.imagesLoading.value).toBe(false);
    await load;
    expect(refs.allGridImages.value).toEqual([]);
    expect(grid.imagesError.value).toBe(null);

    await grid.fetchAllGridImages();
    expect(getPictureCount).toHaveBeenCalledTimes(2);
    expect(refs.allGridImages.value.map((img) => img.id)).toEqual([1, 2]);
  });

  it("does not debounce the first read after a session reset", async () => {
    getPictureCount.mockResolvedValue({ count: 2 });
    const { grid } = makeHarness();

    await grid.fetchAllGridImages();
    notifySessionReset("test");
    await grid.fetchAllGridImages({ join: true });

    expect(getPictureCount).toHaveBeenCalledTimes(2);
  });

  it("does not record success for a load superseded while its stacks refresh", async () => {
    getPictureCount.mockResolvedValue({ count: 1 });
    searchPictures.mockResolvedValue([{ id: 5 }]);
    const { grid, refs, callbacks } = makeHarness();
    const stacks = deferred();
    callbacks.refreshExpandedStacksAfterFetch.mockReturnValueOnce(
      stacks.promise,
    );
    const searchStore = useSearchStore();
    const selection = useSelectionStore();
    const allPictures = selection.selectedCharacter;

    searchStore.searchQuery = "coffee";
    const searchLoad = grid.fetchAllGridImages();
    await vi.waitFor(() =>
      expect(callbacks.refreshExpandedStacksAfterFetch).toHaveBeenCalled(),
    );
    searchStore.searchQuery = "";
    selection.selectedCharacter = 7;
    await grid.fetchAllGridImages();
    stacks.resolve();
    await searchLoad;

    // Straight back to the search: it must read again, not be taken as
    // already on screen.
    selection.selectedCharacter = allPictures;
    searchStore.searchQuery = "coffee";
    await grid.fetchAllGridImages({ join: true });
    expect(searchPictures).toHaveBeenCalledTimes(2);
    expect(refs.allGridImages.value.map((img) => img.id)).toEqual([5]);
  });

  it("cancels the running load when the grid is torn down", async () => {
    getPictureCount.mockImplementationOnce((_query, { signal }) =>
      readUntilAborted(signal),
    );
    const { grid, scope } = makeHarness();

    const load = grid.fetchAllGridImages();
    scope.stop();

    expect(getPictureCount.mock.calls[0][1].signal.aborted).toBe(true);
    await load;
    expect(grid.imagesError.value).toBe(null);
  });

  it.each([
    [
      "stream",
      () => {
        // Large enough for a first batch, an early tail and a background batch.
        getPictureCount.mockResolvedValue({ count: 3000 });
      },
      () => {
        expect(streamPictures).toHaveBeenCalledTimes(3);
        return [
          getPictureCount.mock.calls[0][1],
          ...streamPictures.mock.calls.map((call) => call[1]),
        ];
      },
    ],
    [
      "likeness groups",
      () => {
        useSortStore().selectedSort = "LIKENESS_GROUPS";
      },
      () => [getLikenessGroups.mock.calls[0][2]],
    ],
    [
      "person suggestion",
      (refs) => {
        refs.faceSearchCharacter.value = { id: 4 };
      },
      () => [
        characterFaceSearch.mock.calls[0][1],
        listPicturesByIds.mock.calls[0][1],
      ],
    ],
    [
      "set suggestion",
      (refs) => {
        refs.setSuggestSet.value = { id: 6 };
      },
      () => [
        setLikenessSearch.mock.calls[0][1],
        listPicturesByIds.mock.calls[0][1],
      ],
    ],
    [
      "face likeness",
      (refs) => {
        refs.faceLikenessSearchFaceId.value = 9;
      },
      () => [faceSearch.mock.calls[0][1], listPicturesByIds.mock.calls[0][1]],
    ],
    [
      "reverse image",
      (refs) => {
        refs.reverseImageSearchPictureIds.value = [5];
      },
      () => [
        likenessSearch.mock.calls[0][1],
        listPicturesByIds.mock.calls[0][1],
      ],
    ],
    [
      "text search",
      () => {
        useSearchStore().searchQuery = "coffee";
      },
      () => [searchPictures.mock.calls[0][1]],
    ],
  ])("hands every %s read the load's abort signal", async (_mode, arrange, reads) => {
    getPictureCount.mockResolvedValue({ count: 1 });
    getLikenessGroups.mockResolvedValue([{ id: 1 }]);
    faceSearch.mockResolvedValue([{ picture_id: 1 }]);
    characterFaceSearch.mockResolvedValue([{ picture_id: 1 }]);
    setLikenessSearch.mockResolvedValue([{ picture_id: 1 }]);
    likenessSearch.mockResolvedValue([{ picture_id: 1 }]);
    listPicturesByIds.mockResolvedValue([{ id: 1 }]);
    searchPictures.mockResolvedValue([{ id: 1 }]);
    const { grid, refs } = makeHarness();
    arrange(refs);

    await grid.fetchAllGridImages({ force: true });

    for (const options of reads()) {
      expect(options.signal).toBeInstanceOf(AbortSignal);
    }
  });
});

describe("useGridFetch: a failed batch fails the load", () => {
  // 3000 pictures stream as a first batch, an early tail ending at the count,
  // and one background batch between them.
  const TOTAL = 3000;
  const isTail = ({ offset, batchLimit }) => offset + batchLimit === TOTAL;

  beforeEach(() => {
    getPictureCount.mockReset();
    streamPictures.mockReset();
    getPictureCount.mockResolvedValue({ count: TOTAL });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("reports a failed early tail as a failed load, not a shorter grid", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const tailError = new Error("test-tail-failed");
    streamPictures.mockImplementation((_query, batch) =>
      isTail(batch)
        ? Promise.reject(tailError)
        : Promise.resolve({ pictures: [{ id: batch.offset + 1 }] }),
    );
    const { grid } = makeHarness();

    const outcome = await grid.fetchAllGridImages();

    expect(outcome.success).toBe(false);
    expect(outcome.error).toBe(tailError);
    expect(grid.imagesError.value).toBe("test-tail-failed");
    expect(grid.lastFetchError.value.key).toBe(grid.buildGridFetchKey());
    expect(grid.lastFetchSuccess.value.key).not.toBe(grid.buildGridFetchKey());
  });

  it("reports the first batch's failure when the tail fails as well", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const firstError = new Error("test-first-failed");
    streamPictures.mockImplementation((_query, batch) =>
      Promise.reject(isTail(batch) ? new Error("test-tail-failed") : firstError),
    );
    const { grid } = makeHarness();

    const outcome = await grid.fetchAllGridImages();
    // Let the tail's rejection settle inside this test, so an unhandled one is
    // reported against it.
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(outcome.error).toBe(firstError);
    expect(grid.imagesError.value).toBe("test-first-failed");
  });
});

// The rows a stream request answers with: ids count up from 1 in server order,
// so a row's id says which slot it belongs in.
function streamedRows(offset, batchLimit) {
  return Array.from({ length: batchLimit }, (_, i) => ({ id: offset + i + 1 }));
}

// A grid list that counts the rows it costs: each row read or written in an
// array it holds, and each row of an array handed to it whole.
function countingListRef(ops) {
  const isRow = (key) => typeof key === "string" && /^\d+$/.test(key);
  const counted = (rows) =>
    reactive(
      new Proxy(rows, {
        get(target, key, receiver) {
          if (isRow(key)) ops.rows += 1;
          return Reflect.get(target, key, receiver);
        },
        set(target, key, value, receiver) {
          if (isRow(key)) ops.rows += 1;
          return Reflect.set(target, key, value, receiver);
        },
      }),
    );
  let held = counted([]);
  return customRef((track, trigger) => ({
    get() {
      track();
      return held;
    },
    set(next) {
      const rows = toRaw(next);
      ops.rows += rows.length;
      held = counted(rows);
      trigger();
    },
  }));
}

describe("useGridFetch: a streamed batch costs what it brings", () => {
  // 50,000 pictures stream as a 200-row first batch, a 200-row early tail and
  // ten background batches between them, the first nine of 5,000 rows.
  const TOTAL = 50_000;

  beforeEach(() => {
    getPictureCount.mockReset();
    streamPictures.mockReset();
    getPictureCount.mockResolvedValue({ count: TOTAL });
  });

  it("works on each background batch's own rows, not on everything already loaded", async () => {
    const ops = { rows: 0 };
    const background = [];
    streamPictures.mockImplementation((_query, { offset, batchLimit }) => {
      if (offset > 0 && offset + batchLimit < TOTAL) {
        background.push({ batchLimit, rowsBefore: ops.rows });
      }
      return Promise.resolve({ pictures: streamedRows(offset, batchLimit) });
    });
    const { grid, refs } = makeHarness({
      extraRefs: {
        allGridImages: countingListRef(ops),
        lastFetchedGridImages: countingListRef(ops),
      },
    });

    await grid.fetchAllGridImages({ force: true });

    // A batch's cost runs from its own request to the next one's.
    const costs = background
      .slice(1)
      .map((next, i) => next.rowsBefore - background[i].rowsBefore);
    expect(costs).toHaveLength(9);
    costs.forEach((cost, i) => {
      expect(cost, `rows per batch: ${costs.join(", ")}`).toBeLessThanOrEqual(
        8 * background[i].batchLimit,
      );
    });
    const ids = refs.allGridImages.value.map((row) => row.id);
    expect(ids).toEqual(Array.from({ length: TOTAL }, (_, i) => i + 1));
    expect(refs.lastFetchedGridImages.value).toHaveLength(TOTAL);
  });
});

describe("useGridFetch: the fetched rows stay in step with a streaming grid", () => {
  // 12,000 pictures stream as a 200-row first batch, a 200-row early tail and
  // background batches of 5,000, 5,000 and 1,600 rows between them.
  const TOTAL = 12_000;
  const TAIL_START = TOTAL - 200;

  beforeEach(() => {
    getPictureCount.mockReset();
    streamPictures.mockReset();
    getPictureCount.mockResolvedValue({ count: TOTAL });
  });

  // Both lists as plain rows, so a comparison sees each row's fields.
  function readLists(refs) {
    const gridRows = toRaw(refs.allGridImages.value).map((row) => toRaw(row));
    const fetchedRows = toRaw(refs.lastFetchedGridImages.value).map((row) =>
      toRaw(row),
    );
    return { gridRows, fetchedRows };
  }

  // Streams the view, reading both lists as each later background batch is
  // asked for (the batch before it fully landed) and once at the end.
  async function streamReading(refsWrite = () => {}) {
    const reads = [];
    let refs = null;
    streamPictures.mockImplementation((_query, { offset, batchLimit }) => {
      if (offset > 200 && offset < TAIL_START) {
        reads.push({ offset, ...readLists(refs) });
        if (offset === 5200) refsWrite(refs);
      }
      return Promise.resolve({ pictures: streamedRows(offset, batchLimit) });
    });
    const harness = makeHarness();
    refs = harness.refs;
    await harness.grid.fetchAllGridImages({ force: true });
    reads.push({ offset: null, ...readLists(refs) });
    return reads;
  }

  const loaded = (rows) => rows.filter((row) => row && row.id != null);

  it("fills the grid in place in server order, a placeholder in every slot still to come", async () => {
    const reads = await streamReading();

    const expectedIds = (loadedEnd) =>
      Array.from({ length: TOTAL }, (_, i) =>
        i < loadedEnd || i >= TAIL_START ? i + 1 : null,
      );
    expect(reads.map((read) => read.offset)).toEqual([5200, 10200, null]);
    for (const { offset, gridRows, fetchedRows } of reads) {
      const ids = gridRows.map((row) => row.id);
      expect(ids).toEqual(expectedIds(offset ?? TOTAL));
      expect(gridRows.every((row, i) => row.idx === i)).toBe(true);
      expect(fetchedRows).toEqual(loaded(gridRows));
    }
  });

  it.each([
    [
      "a delete takes a picture out of both lists",
      (refs) => {
        const kept = (row) => row.id !== 3;
        refs.allGridImages.value = refs.allGridImages.value.filter(kept);
        refs.lastFetchedGridImages.value =
          refs.lastFetchedGridImages.value.filter(kept);
      },
    ],
    [
      "a re-score rewrites a card in the grid alone",
      (refs) => {
        const next = refs.allGridImages.value.slice();
        next[1] = { ...next[1], score: 5 };
        refs.allGridImages.value = next;
      },
    ],
  ])(
    "keeps the fetched rows the grid's loaded rows after %s mid-stream",
    async (_case, write) => {
      const reads = await streamReading(write);

      expect(reads.map((read) => read.offset)).toEqual([5200, 10200, null]);
      for (const { gridRows, fetchedRows } of reads) {
        expect(fetchedRows).toEqual(loaded(gridRows));
      }
    },
  );
});
