// The grid must not say "No pictures match the current filters" until a
// fetch has actually answered with nothing. A fetch still in flight, or one
// that failed, says so and offers no "Show all pictures".
//
// Mounts the real ImageGrid, following ImageGridLibraryEmptyState.test.js.

import { describe, it, expect, beforeEach, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { setActivePinia, createPinia } from "pinia";
import { useSelectionStore } from "../../stores/useSelectionStore.js";
import { useProjectStore } from "../../stores/useProjectStore.js";
import { useSortStore } from "../../stores/useSortStore.js";
import { useFilterStore } from "../../stores/useFilterStore.js";
import {
  isSupportedImageFile,
  isSupportedVideoFile,
} from "../../utils/media.js";
import { ref, nextTick } from "vue";

const apiGet = vi.fn();
const apiPost = vi.fn();
const apiPatch = vi.fn();
const apiPut = vi.fn();
const apiDelete = vi.fn();

// Async factory with a local `await import("vue")`: the store imports above
// pull in apiClient, so this factory runs BEFORE the file's own top-level `vue`
// import has initialised. Closing over that binding throws "Cannot access
// __vi_import__ before initialization".
vi.mock("../../utils/apiClient", async () => {
  const { ref: makeRef, computed: makeComputed } = await import("vue");
  const isAuthenticated = makeRef(true);
  const readOnly = makeRef(false);
  const sessionContext = makeRef({ scope: "ALL" });
  return {
    onSessionReset: () => () => {},
    apiClient: {
      get: (...args) => apiGet(...args),
      post: (...args) => apiPost(...args),
      patch: (...args) => apiPatch(...args),
      put: (...args) => apiPut(...args),
      delete: (...args) => apiDelete(...args),
    },
    activateShareToken: vi.fn(),
    appendShareToken: (url) => url,
    checkLoginStatus: vi.fn(),
    checkSession: vi.fn(),
    isAuthenticated,
    isReadOnly: makeComputed(() => readOnly.value),
    login: vi.fn(),
    logout: vi.fn(),
    sessionContext,
    setRequestClientId: vi.fn(),
    API_BASE_URL: "/api/v1",
    // Reached back through the module so a test can flip the session scope.
    __readOnly: readOnly,
  };
});

// The `vuetify/components` barrel pulls component CSS that Vitest cannot load
// from node_modules; hand back a trivial stub for whatever name is imported.
vi.mock("vuetify/components", async () => {
  const { vuetifyComponentStubs } = await import("../../testing/vuetifyStubs");
  return vuetifyComponentStubs();
});



// The media predicates pass through, wrapped so a test can count how often the
// grid runs its Images/Videos filter.
vi.mock("../../utils/media.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    isSupportedImageFile: vi.fn(actual.isSupportedImageFile),
    isSupportedVideoFile: vi.fn(actual.isSupportedVideoFile),
  };
});

vi.mock("vue-router", () => ({
  useRoute: () => ({ query: {}, params: {}, path: "/", name: "grid" }),
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    currentRoute: ref({ query: {} }),
  }),
}));

import ImageGrid from "./ImageGrid.vue";
import LibraryEmptyState from "./LibraryEmptyState.vue";

function mountGrid() {
  const selectionStore = useSelectionStore();
  const projectStore = useProjectStore();
  const sortStore = useSortStore();
  selectionStore.selectedCharacter = "ALL";
  selectionStore.selectedSet = null;
  selectionStore.selectedSetIds = [];
  projectStore.projectViewMode = "global";
  projectStore.selectedProjectId = null;
  sortStore.selectedSort = "DATE";
  sortStore.selectedDescending = true;
  return mount(ImageGrid, {
    shallow: true,
    global: {
      config: {
        compilerOptions: { isCustomElement: (tag) => tag.startsWith("v-") },
      },
    },
    props: { backendUrl: "/api/v1" },
  });
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const never = () => new Promise(() => {});

function answer({ count, stream, library = 5 }) {
  apiGet.mockImplementation((url) => {
    const u = String(url);
    if (u.includes("/summary")) {
      return Promise.resolve({ data: { image_count: library } });
    }
    if (u.includes("/pictures/count")) return count();
    if (u.includes("/pictures/stream")) return stream();
    return Promise.resolve({ data: [] });
  });
}

beforeEach(() => {
  setActivePinia(createPinia());
  apiGet.mockReset();
  apiPost.mockReset();
  apiPost.mockResolvedValue({ data: {} });
});

const COUNT_FIVE = () => Promise.resolve({ data: { count: 5 } });
const COUNT_ZERO = () => Promise.resolve({ data: { count: 0 } });
const DOWN = () => Promise.reject(new Error("Request failed with status code 503"));

/** Mount, let the 80 ms first-fetch timer and the 350 ms card delay pass. */
async function settle() {
  const wrapper = mountGrid();
  await wait(800);
  await nextTick();
  return wrapper;
}

const cardText = (wrapper) => wrapper.find(".empty-state").text();
const showAll = (wrapper) => wrapper.find('[data-testid="show-all-pictures"]');
const retry = (wrapper) => wrapper.find('[data-testid="grid-fetch-retry"]');

describe("while the grid's fetch is in flight", () => {
  it("says Loading and offers nothing", async () => {
    answer({ count: never, stream: never });
    const wrapper = await settle();

    expect(cardText(wrapper)).toContain("Loading…");
    expect(cardText(wrapper)).not.toContain("No pictures match");
    expect(showAll(wrapper).exists()).toBe(false);
    expect(retry(wrapper).exists()).toBe(false);
  });

  it("comes back after a reset clears the card mid-fetch", async () => {
    // `_resetGridState` and friends clear the delay flag while the fetch is
    // still out, so neither `imagesLoading` nor the count moves to re-arm it.
    answer({ count: never, stream: never });
    const wrapper = await settle();
    wrapper.vm.emptyStateDelayPassed = false;
    await wait(500);
    await nextTick();

    expect(cardText(wrapper)).toContain("Loading…");
  });

  it("does not call an empty library empty yet", async () => {
    answer({ count: never, stream: never, library: 0 });
    const wrapper = await settle();

    expect(wrapper.findComponent(LibraryEmptyState).exists()).toBe(false);
    expect(cardText(wrapper)).toContain("Loading…");
  });
});

describe("when the grid's fetch failed", () => {
  for (const [what, spec] of [
    ["count", { count: DOWN, stream: never }],
    ["stream", { count: COUNT_FIVE, stream: DOWN }],
  ]) {
    it(`says it could not load, not no-match, when the ${what} fails`, async () => {
      answer(spec);
      const wrapper = await settle();

      expect(cardText(wrapper)).toContain("Couldn't load pictures");
      expect(cardText(wrapper)).toContain("503");
      expect(cardText(wrapper)).not.toContain("No pictures match");
      expect(showAll(wrapper).exists()).toBe(false);
      expect(retry(wrapper).exists()).toBe(true);
    });
  }

  it("Try again fetches and then gives the real answer", async () => {
    answer({ count: DOWN, stream: never });
    const wrapper = await settle();
    answer({ count: COUNT_ZERO, stream: never });

    await retry(wrapper).trigger("click");
    await wait(50);
    await nextTick();

    expect(cardText(wrapper)).toContain("No pictures match the current filters");
    expect(showAll(wrapper).exists()).toBe(true);
  });

  it("still shows an empty library its way in", async () => {
    // The library-empty claim rests on the summary count, which answered.
    answer({ count: DOWN, stream: never, library: 0 });
    const wrapper = await settle();

    expect(wrapper.findComponent(LibraryEmptyState).exists()).toBe(true);
  });
});

describe("once the fetch has answered with nothing", () => {
  it("says no pictures match and offers Show all pictures", async () => {
    answer({ count: COUNT_ZERO, stream: never });
    const wrapper = await settle();

    expect(cardText(wrapper)).toContain("No pictures match the current filters");
    expect(cardText(wrapper)).not.toContain("Loading");
    expect(showAll(wrapper).exists()).toBe(true);
    expect(retry(wrapper).exists()).toBe(false);
  });
});

describe("the Images/Videos filter", () => {
  // Rows as the stream delivers them: a loaded row carries its file, a slot
  // the stream has not filled yet is a placeholder at its own idx.
  const picture = (idx) => ({ id: idx + 1, idx, name: `test-${idx}.png` });
  const video = (idx) => ({ id: idx + 1, idx, name: `test-${idx}.mp4` });
  const placeholder = (idx) => ({ id: null, idx });

  /** Mount with the fetch held open, then put `rows` in the grid. */
  async function mountWithRows(mode, rows) {
    answer({ count: never, stream: never });
    const wrapper = mountGrid();
    useFilterStore().mediaTypeFilter = mode;
    await wait(800);
    wrapper.vm.allGridImages = rows;
    await nextTick();
    return wrapper;
  }

  async function scrollTo(wrapper, start) {
    wrapper.vm.visibleStart = start;
    wrapper.vm.visibleEnd = start + 20;
    await nextTick();
  }

  it("filters the grid once, not again on every scroll", async () => {
    const rows = Array.from({ length: 300 }, (_, i) => picture(i));
    const wrapper = await mountWithRows("images", rows);
    await scrollTo(wrapper, 0);
    const filterCalls = isSupportedImageFile.mock.calls.length;
    expect(filterCalls).toBeGreaterThanOrEqual(rows.length);

    for (let step = 1; step <= 10; step += 1) {
      await scrollTo(wrapper, step * 20);
      // The window did move.
      expect(wrapper.vm.gridImagesToRender[0].idx).toBe(wrapper.vm.renderStart);
    }
    expect(isSupportedImageFile.mock.calls.length).toBe(filterCalls);
  });

  it("keeps the slots still loading, so every card sits at its own idx", async () => {
    const rows = Array.from({ length: 90 }, (_, i) =>
      i % 3 === 1 ? placeholder(i) : picture(i),
    );
    const wrapper = await mountWithRows("images", rows);
    await scrollTo(wrapper, 40);

    expect(wrapper.vm.filteredGridCount).toBe(rows.length);
    const start = wrapper.vm.renderStart;
    const rendered = wrapper.vm.gridImagesToRender;
    expect(rendered.length).toBeGreaterThan(0);
    rendered.forEach((img, i) => expect(img.idx).toBe(start + i));
  });

  it("still drops a loaded row of the other kind", async () => {
    const rows = [picture(0), video(1), placeholder(2), picture(3)];
    const wrapper = await mountWithRows("images", rows);
    await scrollTo(wrapper, 0);

    expect(wrapper.vm.filteredGridCount).toBe(3);
    expect(wrapper.vm.gridImagesToRender.map((img) => img.idx)).toEqual([
      0, 2, 3,
    ]);
  });

  it("keeps only videos in Videos", async () => {
    const rows = [picture(0), video(1), placeholder(2)];
    const wrapper = await mountWithRows("videos", rows);
    await scrollTo(wrapper, 0);

    expect(wrapper.vm.gridImagesToRender.map((img) => img.idx)).toEqual([1, 2]);
  });

  it("passes the grid through untouched in All", async () => {
    const rows = [picture(0), video(1), placeholder(2)];
    const wrapper = await mountWithRows("all", rows);
    isSupportedImageFile.mockClear();
    isSupportedVideoFile.mockClear();
    await scrollTo(wrapper, 0);

    expect(wrapper.vm.mediaFilteredGridImages).toBe(wrapper.vm.allGridImages);
    expect(wrapper.vm.filteredGridCount).toBe(3);
    expect(isSupportedImageFile).not.toHaveBeenCalled();
    expect(isSupportedVideoFile).not.toHaveBeenCalled();
  });
});

// A read the test answers by hand.
function held() {
  let resolve;
  const promise = new Promise((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

describe("a filter change while the grid loads", () => {
  // The first load's count is held. The stream answers with a low-score row
  // that a minimum score of 5 excludes.
  function holdCounts() {
    const counts = [];
    const first = held();
    const later = held();
    apiGet.mockImplementation((url, config) => {
      const u = String(url);
      if (u.includes("/summary")) {
        return Promise.resolve({ data: { image_count: 5 } });
      }
      if (u.includes("/pictures/count")) {
        counts.push({ url: u, config });
        return counts.length === 1 ? first.promise : later.promise;
      }
      if (u.includes("/pictures/stream")) {
        return Promise.resolve({
          data: { pictures: [{ id: 11, name: "test-low-score.png", score: 1 }] },
        });
      }
      return Promise.resolve({ data: [] });
    });
    return {
      counts,
      first,
      release() {
        first.resolve({ data: { count: 1 } });
        later.resolve({ data: { count: 0 } });
      },
    };
  }

  it("cancels the running load and asks for the filtered rows", async () => {
    const reads = holdCounts();
    const wrapper = mountGrid();
    try {
      await vi.waitFor(() => expect(reads.counts).toHaveLength(1));
      const unfilteredSignal = reads.counts[0].config?.signal;

      useFilterStore().minScoreFilter = 5;
      await nextTick();

      expect(unfilteredSignal?.aborted).toBe(true);
      expect(reads.counts).toHaveLength(2);
      expect(reads.counts[1].url).toContain("min_score=5");
    } finally {
      wrapper.unmount();
      reads.release();
    }
  });

  it("never shows the unfiltered rows while the filtered count is pending", async () => {
    const reads = holdCounts();
    const wrapper = mountGrid();
    try {
      await vi.waitFor(() => expect(reads.counts).toHaveLength(1));

      useFilterStore().minScoreFilter = 5;
      await nextTick();
      // The cancelled load's count answers anyway.
      reads.first.resolve({ data: { count: 1 } });
      await vi.waitFor(() => expect(reads.counts).toHaveLength(2));
      await wait(50);

      expect(wrapper.vm.allGridImages.some((img) => img.id === 11)).toBe(false);
    } finally {
      wrapper.unmount();
      reads.release();
    }
  });
});

describe("emptying the scrapheap refreshes the grid", () => {
  // Before the purge the view holds a picture the purge deletes (11) and a
  // protected one it keeps (22). Every read after the first sees only 22.
  const DELETED = { id: 11, name: "test-deleted.png" };
  const PROTECTED = { id: 22, name: "test-protected.png" };

  function answerAroundPurge({ firstStream }) {
    const reads = { counts: 0, streams: 0 };
    apiDelete.mockResolvedValue({
      data: { deleted_count: 1, skipped_protected: [PROTECTED.id] },
    });
    apiGet.mockImplementation((url) => {
      const u = String(url);
      if (u.includes("/summary")) {
        return Promise.resolve({ data: { image_count: 1 } });
      }
      if (u.includes("/pictures/count")) {
        reads.counts += 1;
        return Promise.resolve({ data: { count: reads.counts === 1 ? 2 : 1 } });
      }
      if (u.includes("/pictures/stream")) {
        reads.streams += 1;
        return reads.streams === 1
          ? firstStream()
          : Promise.resolve({ data: { pictures: [PROTECTED] } });
      }
      return Promise.resolve({ data: [] });
    });
    return reads;
  }

  async function purge(wrapper) {
    wrapper.vm.deleteForeverConfirmToken = "test-confirmation";
    await wrapper.vm.runEmptyScrapheap(false);
    expect(apiDelete).toHaveBeenCalledWith(
      "/pictures/scrapheap",
      expect.anything(),
    );
  }

  it("re-reads after a purge that lands while the grid is still reading", async () => {
    const firstStream = held();
    const reads = answerAroundPurge({ firstStream: () => firstStream.promise });
    const wrapper = mountGrid();
    try {
      await vi.waitFor(() => expect(reads.streams).toBe(1));
      await purge(wrapper);
      // The read that was out before the purge answers with what it saw.
      firstStream.resolve({ data: { pictures: [DELETED, PROTECTED] } });
      await vi.waitFor(() => expect(wrapper.vm.imagesLoading).toBe(false));

      expect(wrapper.vm.allGridImages.map((img) => img.id)).toEqual([22]);
      expect(reads.counts).toBe(2);
    } finally {
      wrapper.unmount();
      firstStream.resolve({ data: { pictures: [] } });
    }
  });

  for (const [when, gapMs] of [
    ["just after the grid finished loading", 0],
    ["1.5 s after the grid finished loading", 1500],
  ]) {
    it(`re-reads after a purge that lands ${when}`, async () => {
      const reads = answerAroundPurge({
        firstStream: () =>
          Promise.resolve({ data: { pictures: [DELETED, PROTECTED] } }),
      });
      const wrapper = mountGrid();
      try {
        await vi.waitFor(() => expect(reads.streams).toBe(1));
        await vi.waitFor(() => expect(wrapper.vm.imagesLoading).toBe(false));
        if (gapMs) await wait(gapMs);
        await purge(wrapper);
        await wait(300);
        await vi.waitFor(() => expect(wrapper.vm.imagesLoading).toBe(false));

        expect(wrapper.vm.allGridImages.map((img) => img?.id)).toEqual([22]);
        expect(reads.counts).toBe(2);
      } finally {
        wrapper.unmount();
      }
    });
  }
});
