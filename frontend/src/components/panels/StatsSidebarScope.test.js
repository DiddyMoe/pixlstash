// The Details panel's view statistics follow the view on screen: a reply for a
// view the user has left is dropped, the old view's numbers never sit under
// the new one's title, the panel reads again when pictures come and go, an
// empty view says so, and the Untagged split filters the view.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { enableAutoUnmount, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { computed, effectScope, nextTick, reactive, ref } from "vue";

const getPictureStats = vi.fn();
vi.mock("../../api/pictures", () => ({
  getPictureStats: (...args) => getPictureStats(...args),
  getPictureMetadata: vi.fn(async (id) => ({ id, tags: [] })),
  listPictureFaces: vi.fn(async () => []),
  openPictureLocation: vi.fn(),
  pictureThumbnailUrl: (id) => `/thumbs/${id}.webp`,
  resetPicturesTags: vi.fn(async () => ({})),
  // The grid's reads, for the builder the parity check below compares with.
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
vi.mock("../../api/comfyui", () => ({
  getPictureRecipe: vi.fn(async () => ({
    reason: "no_prompt_chunk",
    available: false,
  })),
}));
vi.mock("../../api/config", () => ({ getUserConfig: vi.fn(async () => ({})) }));
vi.mock("../../api/users", () => ({ getPenalisedTags: vi.fn(async () => ({})) }));
vi.mock("../../api/taggers", () => ({ listTaggers: vi.fn(async () => ({})) }));
vi.mock("../../api/tags", () => ({
  bulkFetchTags: vi.fn(async () => []),
  listTags: vi.fn(async () => []),
  listTagPredictions: vi.fn(async () => ({ tag_predictions: [], meta: {} })),
  addPictureTag: vi.fn(),
  removePictureTag: vi.fn(),
  confirmTagPrediction: vi.fn(),
  rejectTagPrediction: vi.fn(),
}));

import StatsSidebar from "./StatsSidebar.vue";
import { useGridFetch } from "../../composables/useGridFetch";
import { useSelectionStore } from "../../stores/useSelectionStore";
import { useSidebarStore } from "../../stores/useSidebarStore";
import { useFilterStore } from "../../stores/useFilterStore";
import { useInboxStore } from "../../stores/useInboxStore";
import { useProjectStore } from "../../stores/useProjectStore";
import { useUserPrefsStore } from "../../stores/useUserPrefsStore";
import { useWsStore } from "../../stores/useWsStore";
import { useSortStore } from "../../stores/useSortStore";
import * as picturesApi from "../../api/pictures";
import {
  ALL_PICTURES_ID,
  SCRAPHEAP_PICTURES_ID,
} from "../../stores/useViewStore";
import { UNTAGGED_STATES } from "../../utils/untaggedStates";

enableAutoUnmount(afterEach);

const STUBS = {
  "v-icon": true,
  "v-progress-circular": true,
  Tooltip: { props: ["text"], template: '<span class="tip" :data-text="text" />' },
  StatsHistogram: true,
  LightboxDescriptionPanel: true,
  AddToEntityControl: true,
  TasksPanel: true,
  LightboxRecipePanel: true,
};

function statsBody(overrides = {}) {
  return {
    total: 111,
    tagged: 109,
    untagged: 2,
    untagged_breakdown: { pending: 2, unprocessable: 0, none: 0 },
    total_tags: 1987,
    avg_tags_per_image: 17.9,
    top_tags: [],
    ...overrides,
  };
}

const LIBRARY = statsBody();
const HEAP = statsBody({
  total: 7,
  tagged: 4,
  untagged: 3,
  untagged_breakdown: { pending: 1, unprocessable: 1, none: 1 },
  total_tags: 40,
  avg_tags_per_image: 5.7,
});

async function flush() {
  for (let i = 0; i < 4; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

/** The main stats reads (no `include`), in the order they were sent. */
function mainReads() {
  return getPictureStats.mock.calls.filter(([, params]) => !params);
}

/**
 * Hold every main stats read until the test answers it, so replies can land
 * in any order. `pending[qs]` answers the read sent with that query.
 */
function holdMainReads() {
  const pending = {};
  getPictureStats.mockImplementation((qs, params) => {
    if (params) return Promise.resolve(statsBody());
    return new Promise((resolve, reject) => {
      pending[qs] = { resolve, reject };
    });
  });
  return pending;
}

function mountPanel() {
  return mount(StatsSidebar, {
    props: { viewName: "All Pictures", pictures: [] },
    global: { stubs: STUBS },
  });
}

const tiles = (wrapper) =>
  wrapper.findAll(".stats-tile-value").map((t) => t.text());

async function openScrapheap() {
  useSelectionStore().selectedCharacter = String(SCRAPHEAP_PICTURES_ID);
  await nextTick();
}

beforeEach(() => {
  setActivePinia(createPinia());
  window.localStorage.clear();
  useSidebarStore().statsOpen = true;
  getPictureStats.mockReset().mockResolvedValue(LIBRARY);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("the panel's stats follow the view on screen", () => {
  it("drops the old view's reply when it lands after the new view's", async () => {
    const pending = holdMainReads();
    const wrapper = mountPanel();
    await flush();
    await openScrapheap();
    await flush();
    expect(Object.keys(pending).sort()).toEqual(["", "only_deleted=true"]);

    // The Scrapheap answers first, then the library's late reply arrives.
    pending["only_deleted=true"].resolve(HEAP);
    await flush();
    pending[""].resolve(LIBRARY);
    await flush();

    expect(tiles(wrapper)).toEqual(["7", "40", "5.7"]);
    expect(wrapper.find(".donut-legend").text()).toContain("Untagged 3");
  });

  it("drops the old view's error too", async () => {
    const pending = holdMainReads();
    const wrapper = mountPanel();
    await flush();
    await openScrapheap();
    pending["only_deleted=true"].resolve(HEAP);
    await flush();
    pending[""].reject(new Error("example-timeout"));
    await flush();

    expect(wrapper.find(".stats-error").exists()).toBe(false);
    expect(tiles(wrapper)).toEqual(["7", "40", "5.7"]);
  });

  it("shows the spinner, not the old view's numbers, while the new view reads", async () => {
    const pending = holdMainReads();
    const wrapper = mountPanel();
    await flush();
    pending[""].resolve(LIBRARY);
    await flush();
    expect(tiles(wrapper)).toEqual(["111", "1,987", "17.9"]);

    await openScrapheap();
    await flush();

    expect(wrapper.find(".stats-loading").exists()).toBe(true);
    expect(wrapper.findAll(".stats-tile")).toHaveLength(0);
    expect(wrapper.text()).not.toContain("1,987");
  });
});

describe("the panel reads again when pictures come and go", () => {
  it("re-reads 2 s after a pictures-changed message while it is open", async () => {
    mountPanel();
    await flush();
    const before = mainReads().length;

    vi.useFakeTimers();
    const ws = useWsStore();
    ws.wsPicturesChanged = { key: ws.wsPicturesChanged.key + 1 };
    await nextTick();
    vi.advanceTimersByTime(1999);
    expect(mainReads()).toHaveLength(before);
    vi.advanceTimersByTime(1);
    expect(mainReads()).toHaveLength(before + 1);
  });

  it("does not read while the panel is closed", async () => {
    mountPanel();
    await flush();
    const before = mainReads().length;

    vi.useFakeTimers();
    useSidebarStore().statsOpen = false;
    const ws = useWsStore();
    ws.wsPicturesChanged = { key: ws.wsPicturesChanged.key + 1 };
    await nextTick();
    vi.advanceTimersByTime(5000);
    expect(mainReads()).toHaveLength(before);
  });
});

describe("an empty view", () => {
  it("says there are no pictures instead of drawing a ring", async () => {
    getPictureStats.mockResolvedValue(
      statsBody({
        total: 0,
        tagged: 0,
        untagged: 0,
        untagged_breakdown: { pending: 0, unprocessable: 0, none: 0 },
        total_tags: 0,
        avg_tags_per_image: 0,
      }),
    );
    const wrapper = mountPanel();
    await flush();

    expect(wrapper.find(".stats-empty").text()).toBe("No pictures in this view");
    expect(wrapper.find(".stats-donut").exists()).toBe(false);
  });
});

describe("the Untagged split", () => {
  it("lists the reasons there are, each a filter on the view", async () => {
    getPictureStats.mockResolvedValue(HEAP);
    const wrapper = mountPanel();
    await flush();

    const rows = wrapper.findAll(".untagged-row");
    expect(rows.map((r) => r.attributes("data-tag-state"))).toEqual([
      "pending",
      "unprocessable",
      "none",
    ]);
    expect(rows.map((r) => r.find(".untagged-row-count").text())).toEqual([
      "1",
      "1",
      "1",
    ]);
    // The rail and the grid's chip word a state alike: both read the table.
    expect(rows.map((r) => r.find(".untagged-row-label").text())).toEqual(
      UNTAGGED_STATES.map((row) => row.label),
    );

    await rows[1].trigger("click");
    await flush();
    expect(useFilterStore().tagStateFilter).toBe("unprocessable");
    expect(mainReads().at(-1)[0]).toContain("tag_state=unprocessable");
    expect(
      wrapper.find('[data-tag-state="unprocessable"]').attributes("aria-pressed"),
    ).toBe("true");

    // The filtered row clears it.
    await wrapper.find('[data-tag-state="unprocessable"]').trigger("click");
    await flush();
    expect(useFilterStore().tagStateFilter).toBe(null);
    expect(mainReads().at(-1)[0]).not.toContain("tag_state");
  });

  it("leaves out a reason no picture has", async () => {
    const wrapper = mountPanel();
    await flush();
    expect(
      wrapper.findAll(".untagged-row").map((r) => r.attributes("data-tag-state")),
    ).toEqual(["pending"]);
  });
});

// The grid's query for the view in the stores: `useGridFetch`'s builder, fed
// the selection as ImageGrid.vue derives it (copied from its computeds).
function gridQuery() {
  const selection = useSelectionStore();
  const setIds = computed(() => {
    const ids = (selection.selectedSetIds || [])
      .map((id) => Number(id))
      .filter((id) => Number.isFinite(id) && id > 0);
    if (ids.length > 0) return Array.from(new Set(ids));
    const single = Number(selection.selectedSet);
    return Number.isFinite(single) && single > 0 ? [single] : [];
  });
  const characterIds = computed(() =>
    (selection.selectedCharacterIds || [])
      .map((id) => Number(id))
      .filter((id) => Number.isFinite(id) && id > 0)
      .sort((a, b) => a - b),
  );
  const deps = new Proxy(
    {
      normalizedSelectedSetIds: setIds,
      hasSetSelection: computed(() => setIds.value.length > 0),
      isSetOverlapView: computed(() => setIds.value.length > 1),
      primarySelectedSetId: computed(() => setIds.value[0] ?? null),
      normalizedSelectedCharacterIds: characterIds,
      isMultiCharacterView: computed(() => characterIds.value.length > 1),
      smartScoreProgress: reactive({ visible: false, percent: 0 }),
      exportProgress: reactive({ visible: false, percent: 0 }),
      previousImageIds: new Set(),
    },
    // Every other input is fetch state the query does not read.
    { get: (target, key) => (key in target ? target[key] : ref(null)) },
  );
  const callbacks = new Proxy({}, { get: () => vi.fn((x) => x) });
  const scope = effectScope();
  try {
    const grid = scope.run(() => useGridFetch(deps, reactive({}), callbacks));
    const params = new URLSearchParams(grid.buildPictureIdsQueryParams());
    // What does not change which pictures are counted (StatsSidebar.vue's
    // buildQueryParams names them).
    for (const key of [
      "sort",
      "descending",
      "fields",
      "query",
      "reference_character_id",
      "guest_session_id",
    ]) {
      params.delete(key);
    }
    return params.toString();
  } finally {
    scope.stop();
  }
}

const VIEWS = {
  "a set difference within one project": () => {
    useProjectStore().projectViewMode = "project";
    useProjectStore().selectedProjectId = 4;
    useProjectStore().setProjectIds = { 3: 9, 5: 9 };
    const selection = useSelectionStore();
    selection.selectedSetIds = [3, 5];
    selection.setMultiMode = "difference";
    selection.setDifferenceBaseId = 5;
    return ["project_id=9", "base_set_id=5"];
  },
  "sets across two projects": () => {
    useProjectStore().projectViewMode = "project";
    useProjectStore().setProjectIds = { 3: 9, 5: 8 };
    useSelectionStore().selectedSetIds = [3, 5];
    return ["set_ids=3"];
  },
  "one set picked as a list of one": () => {
    useSelectionStore().selectedSetIds = [6];
    return ["set_id=6"];
  },
  "several people in one project": () => {
    useProjectStore().projectViewMode = "project";
    useProjectStore().characterProjectIds = { 2: null, 7: null };
    useSelectionStore().selectedCharacterIds = [7, 2];
    return ["project_id=UNASSIGNED", "character_ids=2"];
  },
  "one person's likeness cluster": () => {
    useProjectStore().projectViewMode = "project";
    useProjectStore().selectedProjectId = 4;
    useSelectionStore().selectedCharacter = "7";
    useFilterStore().faceClusterFilter = { cluster: 2, threshold: 0.4 };
    return ["face_cluster=2", "project_id=4"];
  },
  "No character": () => {
    useFilterStore().noCharacterFilter = true;
    return ["character_id=UNASSIGNED", "unassigned_by=character"];
  },
  "No character and In no set": () => {
    useFilterStore().noCharacterFilter = true;
    useFilterStore().noSetFilter = true;
    return ["character_id=UNASSIGNED"];
  },
  "the Inbox": () => {
    useSelectionStore().selectedFolderFilter = { inbox: true };
    return [`inbox=${useInboxStore().period}`];
  },
  "the Scrapheap": () => {
    useSelectionStore().selectedCharacter = String(SCRAPHEAP_PICTURES_ID);
    return ["only_deleted=true"];
  },
  "every filter at once": () => {
    useSelectionStore().selectedFolderFilter = {
      referenceFolderId: 3,
      pathPrefix: "/lib/a",
    };
    useUserPrefsStore().applyTagFilter = true;
    Object.assign(useFilterStore(), {
      mediaTypeFilter: "images",
      comfyuiModelFilter: ["example-model"],
      comfyuiLoraFilter: ["example-lora"],
      workflowFilter: [{ id: "example-wf", lora: "example-lora" }],
      minScoreFilter: 2,
      maxScoreFilter: 4,
      unscoredOnlyFilter: true,
      favouriteOnlyFilter: true,
      smartScoreBucketFilter: "3-4",
      resolutionBucketFilter: "hd",
      tagFilter: ["cat"],
      tagRejectedFilter: ["dog"],
      tagConfidenceAboveFilter: ["cat:0.5"],
      tagConfidenceBelowFilter: ["dog:0.2"],
      importedFromFilter: "/import/b",
      faceBboxFilter: "with",
      tagStateFilter: "pending",
      impossibleSources: ["example-source"],
      stackStateFilter: "leaders",
      sharedOnlyFilter: true,
    });
    return [
      "comfyui_model=example-model",
      "workflow=example-wf",
      "apply_tag_filter=true",
      "reference_folder_id=3",
      "impossible_tag_source=example-source",
      "stack_state=leaders",
      "shared_only=true",
    ];
  },
};

describe("the panel counts the view the grid shows", () => {
  for (const [name, arrange] of Object.entries(VIEWS)) {
    it(`asks for the grid's view: ${name}`, async () => {
      useSelectionStore().selectedCharacter = ALL_PICTURES_ID;
      const mustCarry = arrange();
      mountPanel();
      await flush();

      const sent = mainReads().at(-1)[0];
      expect(sent).toBe(gridQuery());
      for (const part of mustCarry) {
        expect(sent.split("&")).toContain(part);
      }
    });
  }
});

// ---------------------------------------------------------------------------
// One view, one set of numbers. The count above the grid is the
// grid's stream count (`getPictureCount`, which the grid's placeholders and so
// FilterStrip's "N" are), the grid is the stream (`streamPictures`), and the
// Details panel reads `viewQueryParams`. The stream builds its query apart
// (`buildStreamSelectionSuffix`), so these hold the three alike over the views
// the counts disagreed in. The sidebar's place numbers (All media, Favourites,
// Scrapheap, Not in a project) are held to the same query in
// SideBarCategoryCounts.test.js; a folder has no sidebar number.

// What the stream sends that does not choose which pictures are in the view:
// how the grid lists them (the sort, the likeness reference the sort ranks by,
// the row shape) and the collapse of a stack to its leader, which the grid's
// count shares and the Details panel leaves out on purpose (its Pictures tile
// counts every picture in a stack).
const STREAM_ONLY = [
  "fields",
  "grid_lite",
  "stack_leaders_only",
  "sort",
  "descending",
  "reference_character_id",
];

const sortedPairs = (query) =>
  [...new URLSearchParams(query).entries()]
    .map(([key, value]) => `${key}=${value}`)
    .sort();

// The grid as ImageGrid.vue runs it, with the reads the count and the stream
// make. The selection is derived as `gridQuery` derives it.
function gridLoader() {
  const selection = useSelectionStore();
  const setIds = computed(() => {
    const ids = (selection.selectedSetIds || [])
      .map((id) => Number(id))
      .filter((id) => Number.isFinite(id) && id > 0);
    if (ids.length > 0) return Array.from(new Set(ids));
    const single = Number(selection.selectedSet);
    return Number.isFinite(single) && single > 0 ? [single] : [];
  });
  const characterIds = computed(() =>
    (selection.selectedCharacterIds || [])
      .map((id) => Number(id))
      .filter((id) => Number.isFinite(id) && id > 0)
      .sort((a, b) => a - b),
  );
  const deps = new Proxy(
    {
      normalizedSelectedSetIds: setIds,
      hasSetSelection: computed(() => setIds.value.length > 0),
      isSetOverlapView: computed(() => setIds.value.length > 1),
      primarySelectedSetId: computed(() => setIds.value[0] ?? null),
      normalizedSelectedCharacterIds: characterIds,
      isMultiCharacterView: computed(() => characterIds.value.length > 1),
      allGridImages: ref([]),
      lastFetchedGridImages: ref([]),
      lightboxOpen: ref(false),
      preserveScrollOnNextFetch: ref(false),
      visibleStart: ref(0),
      visibleEnd: ref(0),
      divisibleViewWindow: ref(40),
      initialRender: ref(false),
      rowHeight: ref(128),
      sharedPictureIds: ref(new Set()),
      highlightNextFetch: ref(false),
      hasLoadedOnce: ref(false),
      reverseImageSearchPictureIds: ref([]),
      smartScoreProgress: reactive({ visible: false, percent: 0 }),
      exportProgress: reactive({ visible: false, percent: 0 }),
      previousImageIds: new Set(),
    },
    { get: (target, key) => (key in target ? target[key] : ref(null)) },
  );
  const callbacks = new Proxy(
    // ImageGrid's hold scrolls; there is no view here, so the change only.
    { holdView: vi.fn((_isLeaving, change) => change()) },
    { get: (target, key) => target[key] ?? vi.fn((x) => x) },
  );
  const scope = effectScope();
  const grid = scope.run(() => useGridFetch(deps, reactive({}), callbacks));
  return { grid, scope };
}

const HIDDEN = { off: false, on: true };

const COUNTED_VIEWS = {
  "All media": () => {},
  "the Scrapheap": () => {
    useSelectionStore().selectedCharacter = String(SCRAPHEAP_PICTURES_ID);
  },
  "a folder": () => {
    useSelectionStore().selectedFolderFilter = {
      referenceFolderId: 3,
      pathPrefix: "/lib/a",
    };
  },
  "a folder in the Scrapheap": () => {
    useSelectionStore().selectedCharacter = String(SCRAPHEAP_PICTURES_ID);
    useSelectionStore().selectedFolderFilter = {
      referenceFolderId: 3,
      pathPrefix: "/lib/a",
    };
  },
  // A navigation with no id clears the view; in project mode the grid still
  // lists the project's pictures.
  "project mode with no view picked": () => {
    useProjectStore().projectViewMode = "project";
    useProjectStore().selectedProjectId = 4;
    useSelectionStore().selectedCharacter = null;
  },
};

describe("the count above the grid, the grid and the Details panel count one view", () => {
  for (const [name, arrange] of Object.entries(COUNTED_VIEWS)) {
    for (const [hidden, on] of Object.entries(HIDDEN)) {
      it(`${name}, hidden tags ${hidden}`, async () => {
        const pictures = picturesApi;
        pictures.getPictureCount.mockReset().mockResolvedValue({ count: 1 });
        pictures.streamPictures
          .mockReset()
          .mockResolvedValue({ pictures: [{ id: 1 }] });
        useSelectionStore().selectedCharacter = ALL_PICTURES_ID;
        useUserPrefsStore().applyTagFilter = on;
        useUserPrefsStore().hiddenTags = on ? ["example-hidden"] : [];
        arrange();

        const { grid, scope } = gridLoader();
        try {
          await grid.fetchAllGridImages({ force: true });
        } finally {
          scope.stop();
        }
        expect(pictures.getPictureCount).toHaveBeenCalledTimes(1);
        const header = pictures.getPictureCount.mock.calls[0][0];
        const stream = pictures.streamPictures.mock.calls[0][0];

        // The header counts the rows the stream lists.
        const streamRows = new URLSearchParams(stream);
        streamRows.delete("fields");
        streamRows.delete("grid_lite");
        expect(sortedPairs(header)).toEqual(sortedPairs(streamRows));

        // The stream's view is the Details panel's, but for the keys above.
        const view = gridQuery();
        const streamView = new URLSearchParams(stream);
        for (const key of STREAM_ONLY) {
          expect(new URLSearchParams(view).has(key)).toBe(false);
          streamView.delete(key);
        }
        expect(sortedPairs(streamView)).toEqual(sortedPairs(view));
        expect(new URLSearchParams(view).has("apply_tag_filter")).toBe(on);

        // And the Details panel sends that view.
        mountPanel();
        await flush();
        expect(mainReads().at(-1)[0]).toBe(view);
      });
    }
  }
});

describe("the stream's query names the view the Details panel counts", () => {
  for (const [name, arrange] of Object.entries(VIEWS)) {
    it(name, () => {
      useSelectionStore().selectedCharacter = ALL_PICTURES_ID;
      useSortStore().selectedSort = "DATE_TAKEN";
      arrange();
      const view = gridQuery();
      const { grid, scope } = gridLoader();
      let suffix;
      try {
        suffix = grid.buildStreamSelectionSuffix();
      } finally {
        scope.stop();
      }
      const streamView = new URLSearchParams(suffix);
      expect(streamView.get("sort")).toBe("DATE_TAKEN");
      for (const key of STREAM_ONLY) streamView.delete(key);
      expect(sortedPairs(streamView)).toEqual(sortedPairs(view));
    });
  }
});

describe("the Pictures tile says what it counts", () => {
  const tileTip = (wrapper) =>
    wrapper.findAll(".stats-tile")[0].find(".tip").attributes("data-text");

  it("names the pictures in a stack, which the grid shows once", async () => {
    const wrapper = mountPanel();
    await flush();
    expect(tileTip(wrapper)).toBe(
      "Every picture here, counting each picture in a stack.",
    );
  });

  it("names the subfolders with a folder open, which the grid leaves out", async () => {
    useSelectionStore().selectedFolderFilter = {
      referenceFolderId: 3,
      pathPrefix: "/lib/a",
    };
    const wrapper = mountPanel();
    await flush();
    expect(tileTip(wrapper)).toBe(
      "Every picture in this folder and its subfolders, counting each picture in a stack.",
    );
  });
});
