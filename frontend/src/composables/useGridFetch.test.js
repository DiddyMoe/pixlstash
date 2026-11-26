import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { ref, reactive, effectScope, customRef, toRaw, nextTick } from "vue";
import { setActivePinia, createPinia } from "pinia";
import { useSortStore } from "../stores/useSortStore.js";
import { useFilterStore } from "../stores/useFilterStore.js";
import { useSelectionStore } from "../stores/useSelectionStore.js";
import { useInboxStore } from "../stores/useInboxStore.js";
import { useDedupStore } from "../stores/useDedupStore.js";
import { useProjectStore } from "../stores/useProjectStore.js";
import { useGridFetch } from "./useGridFetch.js";
import { useSearchStore } from "../stores/useSearchStore.js";
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
  callbackOverrides = {},
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
    forgetThumbnailRangesFrom: vi.fn(),
    // ImageGrid's hold scrolls; the harness has no view, so the change only.
    holdView: vi.fn((_isLeaving, change) => change()),
    fetchThumbnailsBatch: vi.fn(),
    startSmartScoreProgress,
    completeSmartScoreProgress,
    onGridFetchStart: vi.fn(),
    onGridVisibleMetadataReady: vi.fn(),
    onGridFetchDone: vi.fn(),
  };
  Object.assign(callbacks, callbackOverrides);

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

  // The stats rail's Untagged rows. The rail narrowed its own counts by
  // tag_state while the grid beside it went on listing every picture.
  it.each([
    ["stream", () => {}, () => streamPictures.mock.calls[0][0]],
    ["count", () => {}, () => getPictureCount.mock.calls[0][0]],
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
  ])(
    "sends tag_state on the %s request while an Untagged row is on",
    async (_read, arrange, sent) => {
      getPictureCount.mockResolvedValue({ count: 1 });
      streamPictures.mockResolvedValue({ pictures: [{ id: 41 }] });
      getLikenessGroups.mockResolvedValue([{ id: 41 }]);
      searchPictures.mockResolvedValue([{ id: 41 }]);
      const { grid } = makeHarness();
      useFilterStore().tagStateFilter = "unprocessable";
      arrange();

      await grid.fetchAllGridImages({ force: true });

      expect(sent()).toContain("tag_state=unprocessable");
    },
  );

  // The backend narrows by tag_state in every scope, the Inbox and the
  // unassigned view included, so an Untagged row narrows the grid there too.
  it.each([
    [
      "the Inbox",
      () => {
        useSelectionStore().selectedFolderFilter = {
          inbox: true,
          label: "Inbox",
        };
      },
      "inbox=14d",
    ],
    [
      "the unassigned view",
      () => {
        useFilterStore().noCharacterFilter = true;
      },
      "character_id=UNASSIGNED",
    ],
  ])(
    "sends tag_state with the scope's own parameter in %s",
    async (_scope, arrange, scoped) => {
      getPictureCount.mockResolvedValue({ count: 1 });
      streamPictures.mockResolvedValue({ pictures: [{ id: 41 }] });
      const { grid } = makeHarness();
      arrange();
      useFilterStore().tagStateFilter = "pending";

      await grid.fetchAllGridImages({ force: true });

      for (const url of [
        streamPictures.mock.calls[0][0],
        getPictureCount.mock.calls[0][0],
      ]) {
        expect(url).toContain(scoped);
        expect(url).toContain("tag_state=pending");
      }
    },
  );

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
  // The server narrows: the grid sends the switch and shows what comes back,
  // so an export of the view, which sends the same query, packages the same.
  const ROWS = [
    { id: 1, text_match: true },
    { id: 3, text_match: true },
  ];

  beforeEach(() => {
    searchPictures.mockReset();
    searchPictures.mockResolvedValue(ROWS);
  });

  function sentQuery(call) {
    return new URLSearchParams(searchPictures.mock.calls[call][1].query);
  }

  it("asks the search for text matches only while the switch is on", async () => {
    const { grid } = makeHarness();
    const searchStore = useSearchStore();
    searchStore.searchQuery = "coffee";

    await grid.fetchAllGridImages({ force: true });
    searchStore.textMatchesOnly = true;
    await grid.fetchAllGridImages();
    searchStore.textMatchesOnly = false;
    await grid.fetchAllGridImages();

    expect(searchPictures).toHaveBeenCalledTimes(3);
    expect(sentQuery(0).has("text_match_only")).toBe(false);
    expect(sentQuery(1).get("text_match_only")).toBe("true");
    expect(sentQuery(2).has("text_match_only")).toBe(false);
  });

  it("shows the rows the search answers, without narrowing them again", async () => {
    // A row the server sent stays even if it says no text match: the rule is
    // the server's alone, so the grid and the export cannot disagree.
    searchPictures.mockResolvedValue([...ROWS, { id: 2, text_match: false }]);
    const { grid, refs } = makeHarness();
    const searchStore = useSearchStore();
    searchStore.searchQuery = "coffee";
    // A new query resets the switch (useSearchStore), so it is turned on after.
    await nextTick();
    searchStore.textMatchesOnly = true;

    await grid.fetchAllGridImages({ force: true });

    expect(sentQuery(0).get("text_match_only")).toBe("true");
    expect(refs.allGridImages.value.map((r) => r.id)).toEqual([1, 3, 2]);
    expect(refs.textSearchResults.value.rows).toHaveLength(3);
  });

  it("sends the switch only with a text search", () => {
    const { grid } = makeHarness();
    useSearchStore().textMatchesOnly = true;
    expect(
      new URLSearchParams(grid.searchViewQueryParams()).has("text_match_only"),
    ).toBe(false);
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
    ["tagStateFilter", "pending"],
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

  it("keeps the new load's removals when the load it cancelled winds down", async () => {
    getPictureCount
      .mockImplementationOnce((_query, { signal }) => readUntilAborted(signal))
      .mockResolvedValueOnce({ count: 3 });
    const firstBatch = deferred();
    streamPictures.mockImplementation((_q, { offset, batchLimit }) =>
      offset === 0
        ? firstBatch.promise
        : Promise.resolve({
            pictures: [{ id: 70 }, { id: 72 }].slice(
              offset,
              offset + batchLimit,
            ),
          }),
    );
    const { grid, refs } = makeHarness();

    const libraryLoad = grid.fetchAllGridImages();
    useSelectionStore().selectedCharacter = 7;
    const personLoad = grid.fetchAllGridImages();
    // The cancelled library load has run its cleanup.
    await libraryLoad;
    await vi.waitFor(() => expect(streamPictures).toHaveBeenCalledTimes(1));

    // Picture 71 is deleted while the person view's first batch is out; the
    // batch was read before the delete committed and still lists it.
    grid.noteRemovedDuringStream([71]);
    firstBatch.resolve({ pictures: [{ id: 70 }, { id: 71 }, { id: 72 }] });
    await personLoad;

    expect(refs.allGridImages.value.map((img) => img.id)).toEqual([70, 72]);
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
  // three background batches between them. Each background request starts at
  // the last row already loaded (its anchor, which places the answer), so
  // they ask from 199, 5,198 and 10,197 and bring 4,999, 4,999 and 1,602
  // rows.
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
      if (offset > 199 && offset < TAIL_START) {
        reads.push({ offset, ...readLists(refs) });
        if (offset === 5198) refsWrite(refs);
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
    expect(reads.map((read) => read.offset)).toEqual([5198, 10197, null]);
    for (const { offset, gridRows, fetchedRows } of reads) {
      const ids = gridRows.map((row) => row.id);
      // Asked at its anchor, the last row loaded.
      expect(ids).toEqual(expectedIds(offset == null ? TOTAL : offset + 1));
      expect(gridRows.every((row, i) => row.idx === i)).toBe(true);
      expect(fetchedRows).toEqual(loaded(gridRows));
    }
  });

  it.each([
    [
      "a delete takes a picture out of both lists",
      // The grid is a row shorter from there on, and so is each later anchor.
      [5198, 10196, null],
      (refs) => {
        const kept = (row) => row.id !== 3;
        refs.allGridImages.value = refs.allGridImages.value.filter(kept);
        refs.lastFetchedGridImages.value =
          refs.lastFetchedGridImages.value.filter(kept);
      },
    ],
    [
      "a re-score rewrites a card in the grid alone",
      [5198, 10197, null],
      (refs) => {
        const next = refs.allGridImages.value.slice();
        next[1] = { ...next[1], score: 5 };
        refs.allGridImages.value = next;
      },
    ],
  ])(
    "keeps the fetched rows the grid's loaded rows after %s mid-stream",
    async (_case, offsets, write) => {
      const reads = await streamReading(write);

      expect(reads.map((read) => read.offset)).toEqual(offsets);
      for (const { gridRows, fetchedRows } of reads) {
        expect(fetchedRows).toEqual(loaded(gridRows));
      }
    },
  );
});

describe("useGridFetch: a removal while the grid streams in loses no row", () => {
  // 12,000 pictures: a 200-row first batch, a 200-row early tail and the
  // background batches between them. A picture leaves the view while a
  // request is out, the way ImageGrid.removeImagesById takes it out
  // mid-stream: its cell goes, every cell after it moves up a slot, and the
  // stream is told. The server answers the request from the list as it was
  // when asked (before the delete committed) or as it is when it answers.
  const TOTAL = 12_000;

  beforeEach(() => {
    getPictureCount.mockReset();
    streamPictures.mockReset();
    getPictureCount.mockResolvedValue({ count: TOTAL });
  });

  function removeMidStream(grid, refs, ids) {
    const gone = new Set(ids);
    grid.noteRemovedDuringStream(ids);
    const remaining = refs.allGridImages.value.filter((img) => !gone.has(img.id));
    refs.allGridImages.value = remaining.map((img, i) =>
      img.idx === i ? img : { ...img, idx: i },
    );
    refs.lastFetchedGridImages.value = refs.lastFetchedGridImages.value.filter(
      (img) => !gone.has(img.id),
    );
  }

  // The first batch has landed (the tail may not have).
  const firstBatchLanded = (refs) => refs.allGridImages.value[0]?.id === 1;

  /**
   * Streams the view, removing `ids` while the request `holds` picks is out,
   * and answering that request from the list `before` or `after` the delete.
   * The removal waits until `landed(refs)` holds.
   */
  async function streamWithRemoval({
    ids,
    answer,
    holds,
    landed = firstBatchLanded,
  }) {
    const removed = new Set();
    const serverIds = () =>
      Array.from({ length: TOTAL }, (_, i) => i + 1).filter(
        (id) => !removed.has(id),
      );
    const page = (offset, limit) =>
      serverIds()
        .slice(offset, offset + limit)
        .map((id) => ({ id }));
    const asked = [];
    const release = deferred();
    let heldAsked = false;
    streamPictures.mockImplementation((_q, { offset, batchLimit }) => {
      asked.push({ offset, batchLimit });
      if (!heldAsked && holds({ offset, batchLimit })) {
        heldAsked = true;
        const before = page(offset, batchLimit);
        return release.promise.then(() => ({
          pictures: answer === "before" ? before : page(offset, batchLimit),
        }));
      }
      return Promise.resolve({ pictures: page(offset, batchLimit) });
    });
    const { grid, refs } = makeHarness();
    const done = grid.fetchAllGridImages({ force: true });
    await vi.waitFor(() => expect(heldAsked).toBe(true));
    await vi.waitFor(() => expect(landed(refs)).toBe(true));
    for (const id of ids) removed.add(id);
    removeMidStream(grid, refs, ids);
    release.resolve();
    await done;
    return { refs, asked, serverIds: serverIds() };
  }

  const firstBackground = ({ offset, batchLimit }) =>
    offset > 0 && batchLimit === 5000;
  const tail = ({ offset }) => offset === TOTAL - 200;
  const firstBatch = ({ offset }) => offset === 0;
  // The count has answered and the placeholder grid is down.
  const placeholdersDown = (refs) => refs.allGridImages.value.length === TOTAL;

  function expectWholeView(refs, serverIds) {
    const ids = refs.allGridImages.value.map((img) => img.id);
    const holes = ids.flatMap((id, i) => (id == null ? [i] : []));
    expect(holes.slice(0, 5)).toEqual([]);
    expect(ids).toHaveLength(serverIds.length);
    expect(ids).toEqual(serverIds);
    expect(refs.allGridImages.value.every((img, i) => img.idx === i)).toBe(
      true,
    );
    expect(refs.lastFetchedGridImages.value.map((img) => img.id)).toEqual(
      serverIds,
    );
  }

  it.each([
    ["a loaded picture", [2], "before", firstBackground],
    ["a loaded picture", [2], "after", firstBackground],
    ["a picture whose row has not arrived", [7000], "before", firstBackground],
    ["a picture whose row has not arrived", [7000], "after", firstBackground],
    [
      "every loaded picture above the gap",
      Array.from({ length: 200 }, (_, i) => i + 1),
      "after",
      firstBackground,
    ],
    ["the last loaded picture, the anchor", [200], "after", firstBackground],
    ["a loaded picture while the tail is out", [5], "before", tail],
    ["a loaded picture while the tail is out", [5], "after", tail],
    ["a tail picture before it arrived", [11_900], "before", tail],
  ])(
    "streams every remaining picture in, once, after removing %s (answered %s the delete)",
    async (_what, ids, answer, holds) => {
      const { refs, serverIds } = await streamWithRemoval({
        ids,
        answer,
        holds,
      });
      expectWholeView(refs, serverIds);
    },
  );

  it("does not write a picture removed while the first batch was out, though the batch lists it", async () => {
    // A delete from another tab lands while the first batch is out: no card
    // of it is in the grid yet, so only the stream hears of it.
    const { refs, serverIds } = await streamWithRemoval({
      ids: [5],
      answer: "before",
      holds: firstBatch,
      landed: placeholdersDown,
    });
    expect(refs.allGridImages.value.slice(0, 6).map((img) => img.id)).toEqual(
      [1, 2, 3, 4, 6, 7],
    );
    expectWholeView(refs, serverIds);
  });

  it("asks for each row once, but for one anchor row per background batch, when nothing moves", async () => {
    const asked = [];
    streamPictures.mockImplementation((_q, { offset, batchLimit }) => {
      asked.push([offset, offset + batchLimit]);
      return Promise.resolve({ pictures: streamedRows(offset, batchLimit) });
    });
    const { grid, refs } = makeHarness();
    await grid.fetchAllGridImages({ force: true });

    const times = new Array(TOTAL).fill(0);
    for (const [start, end] of asked) {
      for (let i = start; i < Math.min(end, TOTAL); i += 1) times[i] += 1;
    }
    const twice = times.flatMap((n, i) => (n === 2 ? [i] : []));
    // The anchors (the last row of the first batch, then of each batch after)
    // and the tail's first row, which the last batch reads to see it is done.
    expect(twice).toEqual([199, 5198, 10197, 11800]);
    expect(times.every((n) => n === 1 || n === 2)).toBe(true);
    expectWholeView(
      refs,
      Array.from({ length: TOTAL }, (_, i) => i + 1),
    );
  });
});

describe("useGridFetch: a re-read scrolled past the first batch keeps the viewport's cards", () => {
  // A delete's refetch keeps the scroll position. At row 1,500 of 50,000 the
  // first batch (rows 0-199) does not reach the viewport, and the swap to the
  // fresh grid turned every cell on screen into a placeholder until the
  // background stream got there.
  const TOTAL = 50000;
  const VIEW = [1500, 1540];

  beforeEach(() => {
    getPictureCount.mockReset();
    streamPictures.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // The library loaded whole, every card with its thumbnail, scrolled to
  // VIEW; then a re-read that keeps the scroll, against `serverIds`: the
  // server's list, or a function that returns it as it is when a request is
  // answered. A held viewport request answers from the list as it was when
  // asked, the way a read that started before a delete committed does.
  async function rereadScrolledDeep(serverIds, { holdView = false } = {}) {
    const listNow =
      typeof serverIds === "function" ? serverIds : () => serverIds;
    const page = (ids, offset, limit) =>
      ids.slice(offset, offset + limit).map((id) => ({ id }));
    const all = Array.from({ length: TOTAL }, (_, i) => i + 1);
    getPictureCount.mockResolvedValueOnce({ count: TOTAL });
    streamPictures.mockImplementation((_q, { offset, batchLimit }) =>
      Promise.resolve({ pictures: page(all, offset, batchLimit) }),
    );
    const harness = makeHarness();
    const { grid, refs } = harness;
    await grid.fetchAllGridImages();
    refs.allGridImages.value = refs.allGridImages.value.map((row) => ({
      ...row,
      thumbnail: `thumb-${row.id}`,
    }));
    refs.visibleStart.value = VIEW[0];
    refs.visibleEnd.value = VIEW[1];
    refs.preserveScrollOnNextFetch.value = true;

    // The background stream is held at its first request, so what the grid
    // shows then is what the swap left on screen.
    const background = deferred();
    const view = deferred();
    let backgroundAsked = false;
    let viewAsked = false;
    getPictureCount.mockResolvedValueOnce({ count: listNow().length });
    streamPictures.mockReset();
    streamPictures.mockImplementation((_q, { offset, batchLimit }) => {
      const answer = () => ({ pictures: page(listNow(), offset, batchLimit) });
      if (offset === 199 && !backgroundAsked) {
        backgroundAsked = true;
        return background.promise.then(answer);
      }
      if (holdView && offset > 199 && offset < VIEW[0] && !viewAsked) {
        viewAsked = true;
        const asked = answer();
        return view.promise.then(() => asked);
      }
      return Promise.resolve(answer());
    });
    const reread = grid.fetchAllGridImages({ force: true });
    return {
      grid,
      refs,
      callbacks: harness.callbacks,
      reread,
      background,
      view,
      isAsked: () => backgroundAsked,
      isViewAsked: () => viewAsked,
    };
  }

  const viewCells = (refs) => refs.allGridImages.value.slice(...VIEW);

  // A delete while the re-read's rows are out, as removeImagesById takes it:
  // the stream is told, and any card of it in the grid goes, the cards after
  // it closing up.
  function removeDuringReread(grid, refs, ids) {
    const gone = new Set(ids);
    grid.noteRemovedDuringStream(ids);
    refs.allGridImages.value = refs.allGridImages.value
      .filter((img) => !gone.has(img.id))
      .map((img, i) => (img.idx === i ? img : { ...img, idx: i }));
    refs.lastFetchedGridImages.value = refs.lastFetchedGridImages.value.filter(
      (img) => !gone.has(img.id),
    );
  }

  function expectWhole(refs, serverIds) {
    const rows = refs.allGridImages.value;
    expect(rows.length).toBe(serverIds.length);
    expect(rows.every((row, i) => row.id === serverIds[i] && row.idx === i)).toBe(
      true,
    );
  }

  it("swaps the viewport's rows in with the first batch", async () => {
    const serverIds = Array.from({ length: TOTAL }, (_, i) => i + 1);
    const { refs, reread, background, isAsked } =
      await rereadScrolledDeep(serverIds);
    await vi.waitFor(() => expect(isAsked()).toBe(true));

    // Row 1,500 is a card through the swap, and keeps its thumbnail.
    expect(refs.allGridImages.value[1500].id).toBe(1501);
    expect(viewCells(refs).map((row) => row.id)).toEqual(
      serverIds.slice(...VIEW),
    );
    expect(viewCells(refs).every((row) => row.thumbnail === `thumb-${row.id}`)).toBe(
      true,
    );

    background.resolve();
    await reread;
    expectWhole(refs, serverIds);
  });

  it("shows the server's rows there after a delete above the viewport", async () => {
    const serverIds = Array.from({ length: TOTAL }, (_, i) => i + 1).filter(
      (id) => id !== 50,
    );
    const { refs, reread, background, isAsked } =
      await rereadScrolledDeep(serverIds);
    await vi.waitFor(() => expect(isAsked()).toBe(true));

    expect(viewCells(refs).map((row) => row.id)).toEqual(
      serverIds.slice(...VIEW),
    );

    background.resolve();
    await reread;
    expectWhole(refs, serverIds);
  });

  it("keeps the old cards until the viewport's rows have answered too", async () => {
    const serverIds = Array.from({ length: TOTAL }, (_, i) => i + 1);
    const { refs, reread, background, view } = await rereadScrolledDeep(
      serverIds,
      { holdView: true },
    );
    // The first batch and the tail have answered; the viewport's rows not.
    await vi.waitFor(() =>
      expect(
        streamPictures.mock.calls.some(([, { offset }]) => offset === 0),
      ).toBe(true),
    );
    await Promise.resolve();
    expect(refs.allGridImages.value[1500]).toMatchObject({
      id: 1501,
      thumbnail: "thumb-1501",
    });

    view.resolve();
    background.resolve();
    await reread;
    expectWhole(refs, serverIds);
  });

  it("leaves out a picture deleted while the viewport's rows were out, though they list it", async () => {
    const all = Array.from({ length: TOTAL }, (_, i) => i + 1);
    let server = all;
    const { grid, refs, reread, background, view, isAsked, isViewAsked } =
      await rereadScrolledDeep(() => server, { holdView: true });
    await vi.waitFor(() => expect(isViewAsked()).toBe(true));

    // Picture 1,510, on screen, is deleted. The viewport's answer was read
    // before the delete committed and still lists it.
    server = all.filter((id) => id !== 1510);
    removeDuringReread(grid, refs, [1510]);
    view.resolve();
    await vi.waitFor(() => expect(isAsked()).toBe(true));

    expect(viewCells(refs).map((row) => row.id)).toEqual(
      server.slice(...VIEW),
    );

    background.resolve();
    await reread;
    expectWhole(refs, server);
  });

  it("drops the cells deleted pictures left above the viewport through the grid's hold", async () => {
    // Pictures 1,001-1,003 are deleted after the viewport's rows landed and
    // before the background stream reached theirs. The run above the
    // viewport then has three cells the server no longer fills: they go, and
    // the viewport's rows move up three slots. That is above the view, so it
    // goes through the grid's hold, which scrolls with it.
    const all = Array.from({ length: TOTAL }, (_, i) => i + 1);
    const gone = [1001, 1002, 1003];
    let server = all;
    const { grid, refs, callbacks, reread, background, isAsked } =
      await rereadScrolledDeep(() => server);
    await vi.waitFor(() => expect(isAsked()).toBe(true));
    const viewStart = streamPictures.mock.calls
      .map(([, { offset }]) => offset)
      .find((offset) => offset > 199 && offset < VIEW[0]);
    expect(refs.allGridImages.value[viewStart].id).toBe(viewStart + 1);
    const left = [];
    callbacks.holdView.mockImplementation((isLeaving, change) => {
      refs.allGridImages.value.forEach((img, i) => {
        if (isLeaving(img, i)) left.push({ slot: i, id: img.id });
      });
      change();
    });

    server = all.filter((id) => !gone.includes(id));
    removeDuringReread(grid, refs, gone);
    background.resolve();
    await reread;

    // The three placeholders right above the viewport's first row.
    expect(left).toEqual(
      [3, 2, 1].map((back) => ({ slot: viewStart - back, id: null })),
    );
    expectWhole(refs, server);
  });
});

describe("useGridFetch: a re-read keeps the cards until its rows land", () => {
  // The refetch after a delete, a run's end or a realtime change re-reads the
  // view on screen. Laying placeholders down at its count blanked every card
  // for the first batch's round trip and reloaded each thumbnail after it.
  beforeEach(() => {
    getPictureCount.mockReset();
    streamPictures.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // Load the library with pictures 1-3, each card carrying its thumbnail as
  // the thumbnail batch would have left it.
  async function loadedLibrary() {
    getPictureCount.mockResolvedValueOnce({ count: 3 });
    streamPictures.mockResolvedValueOnce({
      pictures: [{ id: 1 }, { id: 2 }, { id: 3 }],
    });
    const harness = makeHarness();
    await harness.grid.fetchAllGridImages();
    harness.refs.allGridImages.value = harness.refs.allGridImages.value.map(
      (row) => ({ ...row, thumbnail: `thumb-${row.id}`, faces: [row.id] }),
    );
    harness.callbacks.resetThumbnailState.mockClear();
    return harness;
  }

  const ids = (refs) => refs.allGridImages.value.map((row) => row.id);

  it("shows the cards it has while a re-read of the same view waits for its rows", async () => {
    const { grid, refs, callbacks } = await loadedLibrary();
    const firstBatch = deferred();
    getPictureCount.mockResolvedValueOnce({ count: 2 });
    streamPictures.mockReturnValueOnce(firstBatch.promise);

    const reread = grid.fetchAllGridImages({ force: true });
    await vi.waitFor(() => expect(streamPictures).toHaveBeenCalledTimes(2));

    expect(ids(refs)).toEqual([1, 2, 3]);
    expect(refs.allGridImages.value[0].thumbnail).toBe("thumb-1");
    expect(callbacks.resetThumbnailState).not.toHaveBeenCalled();

    // Picture 2 has gone and 3 now sorts first: each card that stays keeps
    // its own thumbnail and faces, wherever it now sits.
    firstBatch.resolve({ pictures: [{ id: 3 }, { id: 1 }] });
    await reread;

    expect(ids(refs)).toEqual([3, 1]);
    expect(refs.allGridImages.value.map((row) => row.thumbnail)).toEqual([
      "thumb-3",
      "thumb-1",
    ]);
    expect(refs.allGridImages.value.map((row) => row.faces)).toEqual([
      [3],
      [1],
    ]);
    expect(callbacks.resetThumbnailState).toHaveBeenCalledTimes(1);
  });

  it("lays another view's placeholders down as soon as its count answers", async () => {
    const { grid, refs } = await loadedLibrary();
    const firstBatch = deferred();
    getPictureCount.mockResolvedValueOnce({ count: 1 });
    streamPictures.mockReturnValueOnce(firstBatch.promise);

    useSelectionStore().selectedCharacter = 7;
    const personLoad = grid.fetchAllGridImages();
    await vi.waitFor(() => expect(streamPictures).toHaveBeenCalledTimes(2));

    // None of the library's cards stands in for the person's view.
    expect(ids(refs)).toEqual([null]);

    firstBatch.resolve({ pictures: [{ id: 70 }] });
    await personLoad;
    expect(ids(refs)).toEqual([70]);
    expect(refs.allGridImages.value[0].thumbnail).toBeNull();
  });

  it("takes the cards away at once when a re-read finds the view empty", async () => {
    const { grid, refs } = await loadedLibrary();
    getPictureCount.mockResolvedValueOnce({ count: 0 });

    await grid.fetchAllGridImages({ force: true });

    expect(ids(refs)).toEqual([]);
    expect(streamPictures).toHaveBeenCalledTimes(1);
  });

  it("does not write back a picture deleted while the count was out", async () => {
    // The old cards stay on screen while the count is read, so a delete can
    // land then. The first batch was read before the delete committed and
    // still lists the picture.
    const { grid, refs } = await loadedLibrary();
    const count = deferred();
    getPictureCount.mockReturnValueOnce(count.promise);
    streamPictures.mockImplementation((_q, { offset, batchLimit }) =>
      Promise.resolve({
        pictures:
          offset === 0
            ? [{ id: 1 }, { id: 2 }, { id: 3 }]
            : [{ id: 1 }, { id: 3 }].slice(offset, offset + batchLimit),
      }),
    );

    const reread = grid.fetchAllGridImages({ force: true });
    await vi.waitFor(() => expect(getPictureCount).toHaveBeenCalledTimes(2));
    grid.noteRemovedDuringStream([2]);
    refs.allGridImages.value = refs.allGridImages.value
      .filter((row) => row.id !== 2)
      .map((row, i) => ({ ...row, idx: i }));
    refs.lastFetchedGridImages.value = refs.lastFetchedGridImages.value.filter(
      (row) => row.id !== 2,
    );
    count.resolve({ count: 3 });
    await reread;

    expect(ids(refs)).toEqual([1, 3]);
  });
});

describe("useGridFetch: only the load that set a scroll place puts it back", () => {
  // A re-read that keeps the scroll records where the view stood and puts it
  // back once its answer is in. The place belongs to that load: a load that
  // cancels it, or one for another view, must not inherit it.
  beforeEach(() => {
    searchPictures.mockReset();
    getPictureCount.mockReset();
    streamPictures.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // A scroller whose scrollTop writes are recorded; jsdom has no layout, so
  // the content is tall enough that no restore is clamped.
  function scrollView(refs, top) {
    const view = { top, writes: [] };
    refs.scrollWrapper.value = {
      get scrollTop() {
        return view.top;
      },
      set scrollTop(value) {
        view.top = value;
        view.writes.push(value);
      },
      scrollHeight: 1e7,
      clientHeight: 100,
      clientWidth: 400,
    };
    return view;
  }

  // A search answer that lands with the view scrolled to the top, the way a
  // replaced list can leave it: only a restore brings it back.
  function answerAtTop(view, rows) {
    return () => {
      view.top = 0;
      return Promise.resolve(rows);
    };
  }

  it("a load that does not keep the scroll never puts back the place of the re-read it cancels", async () => {
    const held = deferred();
    const { grid, refs } = makeHarness();
    const view = scrollView(refs, 1200);
    searchPictures
      .mockReturnValueOnce(held.promise)
      .mockImplementationOnce(answerAtTop(view, [{ id: 50 }]));
    useSearchStore().searchQuery = "coffee";

    refs.preserveScrollOnNextFetch.value = true;
    const reread = grid.fetchAllGridImages({ force: true });
    await vi.waitFor(() => expect(searchPictures).toHaveBeenCalledTimes(1));
    refs.preserveScrollOnNextFetch.value = false;
    useFilterStore().minScoreFilter = 5;
    await grid.fetchAllGridImages({ force: true });
    held.resolve([{ id: 1 }]);
    await reread;
    await nextTick();

    expect(refs.allGridImages.value.map((row) => row.id)).toEqual([50]);
    expect(view.writes).toEqual([]);
    expect(refs.pendingScrollTop.value).toBeNull();
  });

  it("puts back the place a refresh recorded, though the refresh cleared the flag before its re-read", async () => {
    const { grid, refs } = makeHarness();
    const view = scrollView(refs, 800);
    searchPictures.mockImplementationOnce(answerAtTop(view, [{ id: 1 }]));
    useSearchStore().searchQuery = "coffee";

    // The gridVersion refresh: record the place, clear the flag, re-read.
    refs.preserveScrollOnNextFetch.value = true;
    grid.recordScrollPlace();
    refs.preserveScrollOnNextFetch.value = false;
    await grid.fetchAllGridImages({ force: true });
    await nextTick();

    expect(view.writes).toEqual([800]);
    expect(refs.pendingScrollTop.value).toBeNull();
  });

  it("drops a recorded place once another view loads, so a return to its view does not jump there", async () => {
    const { grid, refs } = makeHarness();
    const view = scrollView(refs, 800);
    searchPictures
      .mockImplementationOnce(answerAtTop(view, [{ id: 7 }]))
      .mockImplementationOnce(answerAtTop(view, [{ id: 1 }]));
    useSearchStore().searchQuery = "coffee";
    grid.recordScrollPlace();

    useSearchStore().searchQuery = "tea";
    await grid.fetchAllGridImages({ force: true });
    await nextTick();
    expect(view.writes).toEqual([]);

    useSearchStore().searchQuery = "coffee";
    await grid.fetchAllGridImages({ force: true });
    await nextTick();
    expect(view.writes).toEqual([]);
    expect(refs.pendingScrollTop.value).toBeNull();
  });

  it("a session reset drops the place of the re-read it cancels, so the next session's first load does not put the view back there", async () => {
    const held = deferred();
    const { grid, refs } = makeHarness();
    const view = scrollView(refs, 900);
    searchPictures
      .mockReturnValueOnce(held.promise)
      .mockImplementationOnce(answerAtTop(view, [{ id: 2 }]));
    useSearchStore().searchQuery = "coffee";

    refs.preserveScrollOnNextFetch.value = true;
    const reread = grid.fetchAllGridImages({ force: true });
    await vi.waitFor(() => expect(searchPictures).toHaveBeenCalledTimes(1));
    refs.preserveScrollOnNextFetch.value = false;
    notifySessionReset("test");
    expect(refs.pendingScrollTop.value).toBeNull();

    await grid.fetchAllGridImages({ force: true });
    await nextTick();
    held.resolve([{ id: 1 }]);
    await reread;
    await nextTick();

    expect(view.writes).toEqual([]);
  });

  it("a session reset also drops the place a re-read queued behind the cancelled load was to put back", async () => {
    const held = deferred();
    const { grid, refs } = makeHarness();
    const view = scrollView(refs, 600);
    searchPictures
      .mockReturnValueOnce(held.promise)
      .mockImplementationOnce(answerAtTop(view, [{ id: 2 }]));
    useSearchStore().searchQuery = "coffee";

    // A plain load runs; a call that keeps the scroll queues a re-read
    // behind it, and the place it records is that re-read's, not the load's.
    const running = grid.fetchAllGridImages({ force: true });
    await vi.waitFor(() => expect(searchPictures).toHaveBeenCalledTimes(1));
    refs.preserveScrollOnNextFetch.value = true;
    const queued = grid.fetchAllGridImages({ force: true });
    refs.preserveScrollOnNextFetch.value = false;
    notifySessionReset("test");
    expect(refs.pendingScrollTop.value).toBeNull();

    held.resolve([{ id: 1 }]);
    await Promise.all([running, queued]);
    await grid.fetchAllGridImages({ force: true });
    await nextTick();

    expect(searchPictures).toHaveBeenCalledTimes(2);
    expect(view.writes).toEqual([]);
  });

  it("a re-read standing for several calls puts back the place one of them recorded", async () => {
    const first = deferred();
    const { grid, refs } = makeHarness();
    const view = scrollView(refs, 600);
    searchPictures
      .mockReturnValueOnce(first.promise)
      .mockImplementationOnce(answerAtTop(view, [{ id: 1 }]));
    useSearchStore().searchQuery = "coffee";

    const running = grid.fetchAllGridImages({ force: true });
    await vi.waitFor(() => expect(searchPictures).toHaveBeenCalledTimes(1));
    // Two more calls for the same view while it runs: one re-read follows,
    // started by the first, and the second, which keeps the scroll, joins it.
    const plain = grid.fetchAllGridImages({ force: true });
    refs.preserveScrollOnNextFetch.value = true;
    const keeping = grid.fetchAllGridImages({ force: true });
    first.resolve([{ id: 1 }]);
    await Promise.all([running, plain, keeping]);
    await nextTick();

    expect(searchPictures).toHaveBeenCalledTimes(2);
    // Put back once, at the end of the re-read whose answer moved the view;
    // the plain load before it holds no place and puts nothing back.
    expect(view.writes).toEqual([600]);
    expect(view.top).toBe(600);
  });

  it("a stream that keeps the scroll leaves no place behind", async () => {
    const { grid, refs } = makeHarness();
    scrollView(refs, 300);
    getPictureCount.mockResolvedValueOnce({ count: 2 });
    streamPictures.mockResolvedValueOnce({ pictures: [{ id: 1 }, { id: 2 }] });

    refs.preserveScrollOnNextFetch.value = true;
    await grid.fetchAllGridImages({ force: true });

    expect(refs.pendingScrollTop.value).toBeNull();
  });
});

describe("useGridFetch: the reconnect read of an id-list view", () => {
  beforeEach(() => {
    for (const read of [
      getLikenessGroups,
      searchPictures,
      likenessSearch,
      characterFaceSearch,
      setLikenessSearch,
      faceSearch,
      listPicturesByIds,
    ]) {
      read.mockReset();
    }
  });

  it("names the mode runGridLoad reads, the likeness groups first", () => {
    const { grid, refs } = makeHarness({
      extraRefs: { setSuggestThreshold: ref(0), setSuggestMinTags: ref(0) },
    });
    const sortStore = useSortStore();
    const searchStore = useSearchStore();
    expect(grid.currentFetchMode()).toBe("stream");
    refs.faceLikenessSearchFaceId.value = 9;
    expect(grid.currentFetchMode()).toBe("face-likeness-search");
    refs.setSuggestSet.value = { id: 4 };
    expect(grid.currentFetchMode()).toBe("set-suggest-search");
    refs.faceSearchCharacter.value = { id: 2 };
    expect(grid.currentFetchMode()).toBe("character-face-search");
    refs.reverseImageSearchPictureIds.value = [5];
    expect(grid.currentFetchMode()).toBe("reverse-image-search");
    searchStore.searchQuery = "  ";
    expect(grid.currentFetchMode()).toBe("reverse-image-search");
    searchStore.searchQuery = "coffee";
    expect(grid.currentFetchMode()).toBe("text-search");
    sortStore.selectedSort = "LIKENESS_GROUPS";
    expect(grid.currentFetchMode()).toBe("likeness-groups");
  });

  it("re-reads the likeness groups whole, with the signal", async () => {
    const { grid } = makeHarness({ selectedSort: "LIKENESS_GROUPS" });
    const signal = new AbortController().signal;
    getLikenessGroups.mockResolvedValue([{ id: 3 }, { id: 1 }]);
    expect(await grid.readIdListForReconcile({ signal })).toEqual([
      { id: 3 },
      { id: 1 },
    ]);
    expect(getLikenessGroups).toHaveBeenCalledTimes(1);
    expect(getLikenessGroups.mock.calls[0][2]).toEqual({ signal });
    expect(listPicturesByIds).not.toHaveBeenCalled();
  });

  it("re-reads a text search with its trimmed words and the view's query", async () => {
    const { grid } = makeHarness();
    useSearchStore().searchQuery = " coffee ";
    const signal = new AbortController().signal;
    searchPictures.mockResolvedValue([{ id: 2 }]);
    expect(await grid.readIdListForReconcile({ signal })).toEqual([{ id: 2 }]);
    expect(searchPictures).toHaveBeenCalledWith("coffee", {
      query: grid.searchViewQueryParams(),
      signal,
    });
  });

  it("reads a ranked search's rows by id, in rank order, less those not listed", async () => {
    const { grid, refs } = makeHarness();
    refs.reverseImageSearchPictureIds.value = [5];
    const signal = new AbortController().signal;
    likenessSearch.mockResolvedValue([
      { picture_id: 3 },
      { picture_id: 8 },
      { picture_id: 1 },
    ]);
    listPicturesByIds.mockResolvedValue([{ id: 1 }, { id: 3 }]);
    expect(await grid.readIdListForReconcile({ signal })).toEqual([
      { id: 3 },
      { id: 1 },
    ]);
    expect(likenessSearch).toHaveBeenCalledWith([5], { signal });
    expect(listPicturesByIds).toHaveBeenCalledWith([3, 8, 1], {
      fields: "grid",
      signal,
    });
  });

  it("cuts a set suggestion as its sliders cut it before reading rows", async () => {
    const { grid, refs } = makeHarness({
      extraRefs: { setSuggestThreshold: ref(0.5), setSuggestMinTags: ref(0) },
    });
    refs.setSuggestSet.value = { id: 4 };
    setLikenessSearch.mockResolvedValue([
      { picture_id: 3, likeness: 0.9 },
      { picture_id: 6, likeness: 0.2 },
    ]);
    listPicturesByIds.mockResolvedValue([{ id: 3 }]);
    expect(await grid.readIdListForReconcile()).toEqual([{ id: 3 }]);
    expect(listPicturesByIds.mock.calls[0][0]).toEqual([3]);
  });

  it("sends no by-id read for an empty ranked list", async () => {
    const { grid, refs } = makeHarness();
    refs.faceLikenessSearchFaceId.value = 9;
    faceSearch.mockResolvedValue([]);
    expect(await grid.readIdListForReconcile()).toEqual([]);
    expect(listPicturesByIds).not.toHaveBeenCalled();
  });

  it.each([
    ["the likeness-groups read", (refs, sortStore) => {
      sortStore.selectedSort = "LIKENESS_GROUPS";
      getLikenessGroups.mockResolvedValue({ detail: "x" });
    }],
    ["the text search", () => {
      useSearchStore().searchQuery = "coffee";
      searchPictures.mockResolvedValue(null);
    }],
    ["the character-face-search", (refs) => {
      refs.faceSearchCharacter.value = { id: 2 };
      characterFaceSearch.mockResolvedValue({});
    }],
    ["the by-ids read", (refs) => {
      refs.faceLikenessSearchFaceId.value = 9;
      faceSearch.mockResolvedValue([{ picture_id: 3 }]);
      listPicturesByIds.mockResolvedValue({ pictures: [] });
    }],
  ])("rejects when %s answers no list, rather than read every card gone", async (read, arrange) => {
    const { grid, refs } = makeHarness();
    arrange(refs, useSortStore());
    await expect(grid.readIdListForReconcile()).rejects.toThrow(
      `readIdListForReconcile: ${read} returned no list`,
    );
  });

  it("refuses the stream view, which is not an id-list view", async () => {
    const { grid } = makeHarness();
    await expect(grid.readIdListForReconcile()).rejects.toThrow(
      "readIdListForReconcile: the stream view is not an id-list view",
    );
  });
});

// The server lists every scrapped picture on its own row (only_deleted), and
// stack leaders only elsewhere. Each load records which, with its rows, so the
// load's fold and every later rebuild (useStackOrdering's collapseStackImages)
// follow one rule instead of each caller guessing it.
describe("useGridFetch: records whether the load folds stacks", () => {
  const SCRAPHEAP = "SCRAPHEAP";
  const heapRows = [
    { id: 1, stack_id: 7 },
    { id: 2, stack_id: 7 },
    { id: 3, stack_id: null },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    getPictureCount.mockResolvedValue({ count: heapRows.length });
    streamPictures.mockResolvedValue({
      pictures: heapRows,
      done: true,
      next_offset: heapRows.length,
    });
    searchPictures.mockReset();
    searchPictures.mockResolvedValue(heapRows);
  });

  // What the load's own fold saw: the rule must be in place before it runs.
  function harnessSeeingFold() {
    const seen = [];
    const harness = makeHarness({
      callbackOverrides: {
        collapseStackImages: (rows) => {
          seen.push(harness.grid.loadFoldsStacks.value);
          return rows;
        },
      },
    });
    return Object.assign(harness, { seen });
  }

  it("keeps every row of the Scrapheap's stream, and folds a person's", async () => {
    const { grid, refs } = makeHarness();
    const selection = useSelectionStore();
    selection.selectedCharacter = SCRAPHEAP;

    await grid.fetchAllGridImages({ force: true });

    expect(streamPictures.mock.calls[0][0]).toContain("only_deleted=true");
    expect(refs.lastFetchedGridImages.value.map((row) => row.id)).toEqual([
      1, 2, 3,
    ]);
    expect(grid.loadFoldsStacks.value).toBe(false);

    selection.selectedCharacter = "7";
    await grid.fetchAllGridImages({ force: true });

    expect(streamPictures.mock.calls[1][0]).not.toContain("only_deleted");
    expect(grid.loadFoldsStacks.value).toBe(true);
  });

  it("keeps every row of a search in the Scrapheap, and folds one elsewhere", async () => {
    const { grid, seen } = harnessSeeingFold();
    const selection = useSelectionStore();
    useSearchStore().searchQuery = "coffee";
    selection.selectedCharacter = SCRAPHEAP;

    await grid.fetchAllGridImages({ force: true });

    expect(searchPictures.mock.calls[0][1].query).toContain(
      "only_deleted=true",
    );
    expect(seen).toEqual([false]);
    expect(grid.loadFoldsStacks.value).toBe(false);

    selection.selectedCharacter = "7";
    await grid.fetchAllGridImages({ force: true });

    expect(searchPictures.mock.calls[1][1].query).not.toContain("only_deleted");
    expect(seen).toEqual([false, true]);
    expect(grid.loadFoldsStacks.value).toBe(true);
  });
});
