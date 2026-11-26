// What the grid re-reads after the updates socket reconnects.
//
// The server does not replay the events it sent while the socket was down, so
// once the link holds again useUpdatesSocket asks the grid to re-read its own
// view's ids (`readViewForReconcile`) and diffs the answer against the grid.
// Pinned here is the read itself: the same stream query the grid loaded from,
// asking for ids alone, in one read of at most 5000 rows centred on the
// viewport, with a second read by id for the cards a partial read did not
// list; an id-list view's own read made again, whole; and no answer handed
// back when it can no longer be compared with what the grid shows (a load
// still running, a newer load, a view that now needs another kind of read, a
// setting changed while the read was out). What the diff makes of it is
// useGridRealtimeSync's, tested there.
//
// Mounts the real ImageGrid, as ImageGridRemovalAnchor.test.js does.

import { describe, it, expect, beforeEach, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { setActivePinia, createPinia } from "pinia";
import { useSelectionStore } from "../../stores/useSelectionStore.js";
import { useProjectStore } from "../../stores/useProjectStore.js";
import { useGridStore } from "../../stores/useGridStore.js";
import { useSearchStore } from "../../stores/useSearchStore.js";
import { useOperationStore } from "../../stores/useOperationStore.js";
import { useUserPrefsStore } from "../../stores/useUserPrefsStore.js";
import { useSortStore } from "../../stores/useSortStore.js";
import { useFilterStore } from "../../stores/useFilterStore.js";
import { SCRAPHEAP_PICTURES_ID } from "../../stores/useViewStore.js";
import { reactive, ref, nextTick } from "vue";
import {
  diffViewAgainstIdRead,
  settleGoneCards,
  useGridRealtimeSync,
} from "../../composables/useGridRealtimeSync.js";

const apiGet = vi.fn();
const apiPost = vi.fn();

// Async factory with a local `await import("vue")`: the store imports above
// pull in apiClient, so this factory runs BEFORE the file's own top-level `vue`
// import has initialised.
vi.mock("../../utils/apiClient", async () => {
  const { ref: makeRef, computed: makeComputed } = await import("vue");
  return {
    onSessionReset: () => () => {},
    apiClient: {
      get: (...args) => apiGet(...args),
      post: (...args) => apiPost(...args),
      patch: vi.fn(() => Promise.resolve({ data: {} })),
      put: vi.fn(() => Promise.resolve({ data: {} })),
      delete: vi.fn(() => Promise.resolve({ data: {} })),
    },
    activateShareToken: vi.fn(),
    appendShareToken: (url) => url,
    checkLoginStatus: vi.fn(),
    checkSession: vi.fn(),
    isAuthenticated: makeRef(true),
    isReadOnly: makeComputed(() => false),
    login: vi.fn(),
    logout: vi.fn(),
    sessionContext: makeRef({ scope: "ALL" }),
    setRequestClientId: vi.fn(),
    API_BASE_URL: "/api/v1",
    newOperationBatchId: () => "cli-test-gesture",
    operationBatchHeaders: (batchId) =>
      batchId ? { headers: { "X-Operation-Batch-Id": batchId } } : undefined,
  };
});

// The `vuetify/components` barrel pulls component CSS that Vitest cannot load
// from node_modules; hand back a trivial stub for whatever name is imported.
vi.mock("vuetify/components", async () => {
  const { vuetifyComponentStubs } = await import("../../testing/vuetifyStubs");
  return vuetifyComponentStubs();
});

vi.mock("vue-router", () => ({
  useRoute: () => ({ query: {}, params: {}, path: "/", name: "grid" }),
  useRouter: () => ({
    push: vi.fn(() => Promise.resolve()),
    replace: vi.fn(() => Promise.resolve()),
    currentRoute: ref({ query: {} }),
  }),
}));

import ImageGrid from "./ImageGrid.vue";

const URL_BASE = "http://grid.test";
const ID_READ = "/pictures/stream?fields=id,stack_id";

/** The view the grid loads: three cards, the middle one a stack's cover. */
const VIEW = Object.freeze([
  { id: 11, format: "png", stack_id: null },
  { id: 12, format: "png", stack_id: 5 },
  { id: 13, format: "png", stack_id: null },
]);

let wrapper = null;
/** Answers the reconnect's id read; a case replaces it. */
let answerIdRead = null;

function mountGrid(selectedCharacter) {
  const selectionStore = useSelectionStore();
  const projectStore = useProjectStore();
  selectionStore.selectedCharacter = selectedCharacter;
  selectionStore.selectedSet = null;
  selectionStore.selectedSetIds = [];
  projectStore.projectViewMode = "global";
  projectStore.selectedProjectId = null;
  return mount(ImageGrid, {
    shallow: true,
    global: {
      config: {
        compilerOptions: { isCustomElement: (tag) => tag.startsWith("v-") },
      },
    },
    props: { backendUrl: "/api/v1", thumbnailMode: "square" },
  });
}

/** The server: the count, the grid's stream read, and the id read. */
function serve(rows) {
  apiGet.mockImplementation((url, config) => {
    if (url.startsWith("/pictures/count")) {
      return Promise.resolve({ data: { count: rows.length } });
    }
    if (url.startsWith(ID_READ)) return answerIdRead(url, config);
    if (url.startsWith("/pictures/stream?")) {
      const params = new URL(url, URL_BASE).searchParams;
      const offset = Number(params.get("offset"));
      const limit = Number(params.get("batch_limit"));
      const pictures = rows.slice(offset, offset + limit);
      return Promise.resolve({
        data: {
          pictures,
          done: offset + limit >= rows.length,
          next_offset: offset + pictures.length,
        },
      });
    }
    return Promise.resolve({ data: { pictures: [], count: 0, total: 0 } });
  });
}

function idReads() {
  return apiGet.mock.calls.filter(([url]) => url.startsWith(ID_READ));
}

/** An id read that stays out until the case answers it. */
function holdIdRead() {
  const held = {};
  answerIdRead = () =>
    new Promise((resolve) => {
      held.answer = (data) => resolve({ data });
    });
  return held;
}

async function settle() {
  for (let i = 0; i < 4; i += 1) await nextTick();
}

async function loadedGrid(selectedCharacter = "ALL") {
  serve(VIEW);
  wrapper = mountGrid(selectedCharacter);
  await vi.waitFor(() => {
    expect(wrapper.vm.imagesLoading).toBe(false);
    expect(wrapper.vm.allGridImages.map((img) => img.id)).toEqual([
      11, 12, 13,
    ]);
  });
  await settle();
  return wrapper.vm;
}

/**
 * Refresh the same view through the gridVersion watcher, as a library event
 * does: it marks the grid not ready and starts a debounced re-read. The
 * watcher ignores a refresh within 1200ms of the last one, the mount
 * included, so the clock is moved past that while it runs.
 */
async function refreshSameView() {
  const realNow = Date.now.bind(Date);
  const clock = vi
    .spyOn(Date, "now")
    .mockImplementation(() => realNow() + 5000);
  try {
    useGridStore().refreshGridVersion();
    await nextTick();
  } finally {
    clock.mockRestore();
  }
}

/** A query's parameters, less the columns asked for and the window read. */
function selectionOf(url) {
  const params = new URLSearchParams(new URL(url, URL_BASE).searchParams);
  for (const key of ["fields", "grid_lite", "offset", "batch_limit"]) {
    params.delete(key);
  }
  return params.toString();
}

beforeEach(() => {
  // Unmount before the new pinia: the old grid's unmount hooks call store
  // actions, and every pinia action makes its own store's pinia the active
  // one again, so a later case would read the previous case's state.
  wrapper?.unmount();
  wrapper = null;
  setActivePinia(createPinia());
  apiGet.mockReset();
  apiPost.mockReset();
  apiPost.mockResolvedValue({ data: {} });
  answerIdRead = () =>
    Promise.resolve({
      data: {
        pictures: VIEW.map(({ id, stack_id }) => ({ id, stack_id })),
        done: true,
      },
    });
});

describe("ImageGrid: the reconnect's id read", () => {
  it("re-reads the loaded view as ids, with the grid's own query, in one read from the top", async () => {
    // A sort and a filter, so the view's own part of the query is not empty.
    useSortStore().selectedSort = "SMART_SCORE";
    useFilterStore().favouriteOnlyFilter = true;
    await loadedGrid();
    const gridRead = apiGet.mock.calls
      .map(([url]) => url)
      .find((url) => url.startsWith("/pictures/stream?fields=grid"));
    const answer = [
      { id: 11, stack_id: null },
      { id: 13, stack_id: null },
      { id: 14, stack_id: null },
    ];
    answerIdRead = () => Promise.resolve({ data: { pictures: answer, done: true } });
    const controller = new AbortController();

    // Through what the grid exposes, which is all the socket's template ref
    // reaches.
    const read = await wrapper.vm.$.exposed.readViewForReconcile({
      signal: controller.signal,
    });

    expect(idReads()).toHaveLength(1);
    const [url, config] = idReads()[0];
    expect(config.signal).toBe(controller.signal);
    const params = new URL(url, URL_BASE).searchParams;
    expect(params.get("stack_leaders_only")).toBe("true");
    expect(params.get("offset")).toBe("0");
    expect(params.get("batch_limit")).toBe("5000");
    expect(params.get("sort")).toBe("SMART_SCORE");
    expect(params.get("favourite")).toBe("1");
    expect(selectionOf(url)).toBe(selectionOf(gridRead));
    const rows = [
      { id: 11, stackId: null },
      { id: 12, stackId: "5" },
      { id: 13, stackId: null },
    ];
    expect(read).toEqual({
      before: rows,
      after: rows,
      answer,
      fromStart: true,
      toEnd: true,
      stacked: true,
      skipIds: [],
      readListed: null,
    });
  });

  it("in the Scrapheap, where every scrapped picture is its own tile, compares picture by picture", async () => {
    // The server lists a scrapped stack's members one by one there, so a
    // comparison by stack would miss a member that left or arrived.
    const vm = await loadedGrid(SCRAPHEAP_PICTURES_ID);
    const read = await vm.readViewForReconcile();
    const params = new URL(idReads()[0][0], URL_BASE).searchParams;
    expect(params.get("only_deleted")).toBe("true");
    expect(read.stacked).toBe(false);
  });

  it("passes the ghosted pictures on, to be left out of the comparison", async () => {
    const vm = await loadedGrid();
    useOperationStore().ghostPictureIds = [12];
    await settle();
    const read = await vm.readViewForReconcile();
    expect(read.skipIds).toEqual(["12"]);
  });

  it("reads the 5000 rows around the first card in view when the view is longer", async () => {
    const vm = await loadedGrid();
    const base = Array.from({ length: 6000 }, (_, i) => ({
      id: i + 1,
      format: "png",
      stack_id: null,
    }));
    // An open stack's member, which the base list does not hold, is the first
    // card in view: the read centres on the card above it, base row 2999.
    const member = { id: 9001, format: "png", stack_id: 77 };
    vm.lastFetchedGridImages = base;
    vm.allGridImages = [...base.slice(0, 3000), member, ...base.slice(3000)];
    await settle();
    vm.visibleStart = 3000;
    answerIdRead = (url) => {
      const offset = Number(new URL(url, URL_BASE).searchParams.get("offset"));
      return Promise.resolve({
        data: {
          pictures: base
            .slice(offset, offset + 5000)
            .map(({ id }) => ({ id, stack_id: null })),
          done: offset + 5000 >= base.length,
        },
      });
    };

    const read = await vm.readViewForReconcile();

    const params = new URL(idReads()[0][0], URL_BASE).searchParams;
    expect(params.get("offset")).toBe("499");
    expect(params.get("batch_limit")).toBe("5000");
    expect(read.fromStart).toBe(false);
    expect(read.toEnd).toBe(false);
    expect(read.before).toHaveLength(6000);

    // Near the end of the list the read ends at its last row.
    vm.visibleStart = 5800;
    const tail = await vm.readViewForReconcile();
    const tailParams = new URL(idReads()[1][0], URL_BASE).searchParams;
    expect(tailParams.get("offset")).toBe("1000");
    expect(tail.toEnd).toBe(true);
  });

  // This tab can change the grid while the read is out (its own delete, a
  // held removal), and the comparison sets those pictures aside, so it needs
  // the list as it stands when the answer lands beside the one it went out
  // against.
  it("hands back the grid as the read found it and as it stands when the answer lands", async () => {
    const vm = await loadedGrid();
    const held = holdIdRead();
    const reading = vm.readViewForReconcile();
    await settle();
    vm.removeImagesById([13]);
    await settle();
    held.answer({
      pictures: [
        { id: 11, stack_id: null },
        { id: 12, stack_id: 5 },
      ],
      done: true,
    });

    const read = await reading;
    expect(read.before.map((row) => row.id)).toEqual([11, 12, 13]);
    expect(read.after.map((row) => row.id)).toEqual([11, 12]);
  });

  it("compares nothing while a refresh waits to start, or while its load runs", async () => {
    const vm = await loadedGrid();
    // The refresh's count stays out, so its load keeps running.
    apiGet.mockImplementation((url, config) =>
      url.startsWith("/pictures/count")
        ? new Promise(() => {})
        : url.startsWith(ID_READ)
          ? answerIdRead(url, config)
          : Promise.resolve({ data: { pictures: [], count: 0 } }),
    );
    const epoch = vm.gridLoadEpoch;
    await refreshSameView();
    expect(vm.gridLoadEpoch).toBe(epoch);
    expect(vm.imagesLoading).toBe(false);
    expect(await vm.readViewForReconcile()).toBeNull();

    await vi.waitFor(() => expect(vm.imagesLoading).toBe(true), {
      timeout: 3000,
    });
    expect(await vm.readViewForReconcile()).toBeNull();
    expect(idReads()).toHaveLength(0);
  });

  // A failed load leaves a grid that is not the view's list.
  it("compares nothing after a load that failed", async () => {
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      apiGet.mockImplementation((url) => {
        if (url.startsWith("/pictures/count")) {
          return Promise.resolve({ data: { count: VIEW.length } });
        }
        if (url.startsWith("/pictures/stream?")) {
          return Promise.reject(new Error("stream down"));
        }
        return Promise.resolve({ data: { pictures: [], count: 0, total: 0 } });
      });
      wrapper = mountGrid("ALL");
      await vi.waitFor(() => {
        expect(wrapper.vm.imagesLoading).toBe(false);
        expect(wrapper.vm.imagesError).toBe("stream down");
      });
      await settle();

      expect(await wrapper.vm.readViewForReconcile()).toBeNull();
      expect(idReads()).toHaveLength(0);
    } finally {
      logged.mockRestore();
    }
  });

  it("drops an answer that a newer load of the same view overtook", async () => {
    const vm = await loadedGrid();
    const held = holdIdRead();
    const reading = vm.readViewForReconcile();
    await settle();
    expect(idReads()).toHaveLength(1);

    const epoch = vm.gridLoadEpoch;
    await refreshSameView();
    await vi.waitFor(
      () => {
        expect(vm.gridLoadEpoch).toBeGreaterThan(epoch);
        expect(vm.imagesLoading).toBe(false);
        expect(vm.allGridImages.map((img) => img.id)).toEqual([11, 12, 13]);
      },
      { timeout: 3000 },
    );
    held.answer({ pictures: [{ id: 11, stack_id: null }], done: true });

    expect(await reading).toBeNull();
  });

  // The tag filter is one of the settings that reach the grid through the
  // throttled gridVersion refresh, so it can change with no load started yet.
  it("drops an answer read under a setting that has since changed", async () => {
    const vm = await loadedGrid();
    const held = holdIdRead();
    const reading = vm.readViewForReconcile();
    await settle();
    expect(idReads()).toHaveLength(1);

    const epoch = vm.gridLoadEpoch;
    useUserPrefsStore().applyTagFilter = !useUserPrefsStore().applyTagFilter;
    await settle();
    expect(vm.gridLoadEpoch).toBe(epoch);
    held.answer({ pictures: [{ id: 11, stack_id: null }], done: true });

    expect(await reading).toBeNull();
  });

  // An empty answer would read as "every picture left"; a malformed one must
  // not be mistaken for it.
  it("fails on an answer with no pictures list", async () => {
    const vm = await loadedGrid();
    answerIdRead = () => Promise.resolve({ data: { detail: "upstream" } });
    await expect(vm.readViewForReconcile()).rejects.toThrow(
      "returned no pictures list",
    );
  });
});

/** The grid's load and the reconnect's read of a text search. */
function serveSearch(rows) {
  apiGet.mockImplementation((url) => {
    if (url.startsWith("/pictures/search?")) {
      return Promise.resolve({ data: rows() });
    }
    return Promise.resolve({ data: { pictures: [], count: 0, total: 0 } });
  });
}

function searchReads() {
  return apiGet.mock.calls.filter(([url]) => url.startsWith("/pictures/search?"));
}

describe("ImageGrid: the reconnect's read of an id-list view", () => {
  it("makes a search's own read again, whole, with the grid's own query", async () => {
    const vm = await loadedGrid();
    let found = [
      { id: 11, format: "png", stack_id: null },
      { id: 13, format: "png", stack_id: null },
    ];
    serveSearch(() => found);
    useSearchStore().searchQuery = "harbour at dusk";
    await vi.waitFor(() => {
      expect(vm.imagesLoading).toBe(false);
      expect(vm.allGridImages.map((img) => img.id)).toEqual([11, 13]);
    });
    await settle();
    const loadRead = searchReads().at(-1)[0];
    found = [
      { id: 13, format: "png", stack_id: null },
      { id: 14, format: "png", stack_id: null },
    ];
    const controller = new AbortController();

    const read = await wrapper.vm.$.exposed.readViewForReconcile({
      signal: controller.signal,
    });

    const [url, config] = searchReads().at(-1);
    expect(searchReads()).toHaveLength(2);
    expect(url).toBe(loadRead);
    expect(config.signal).toBe(controller.signal);
    expect(idReads()).toHaveLength(0);
    const rows = [
      { id: 11, stackId: null },
      { id: 13, stackId: null },
    ];
    expect(read).toEqual({
      before: rows,
      after: rows,
      answer: found,
      fromStart: true,
      toEnd: true,
      stacked: true,
      skipIds: [],
      readListed: null,
    });
  });

  // The server lists every scrapped picture on its own row, a search there
  // included; the load keeps them all, so the comparison goes picture by
  // picture as in the Scrapheap's stream.
  it("keeps a search's scrapped stack members apart in the Scrapheap", async () => {
    const vm = await loadedGrid(SCRAPHEAP_PICTURES_ID);
    const found = [
      { id: 11, format: "png", stack_id: 5 },
      { id: 12, format: "png", stack_id: 5 },
      { id: 13, format: "png", stack_id: null },
    ];
    serveSearch(() => found);
    useSearchStore().searchQuery = "harbour at dusk";
    await vi.waitFor(() => {
      expect(vm.imagesLoading).toBe(false);
      expect(searchReads().length).toBeGreaterThan(0);
      expect(vm.allGridImages.map((img) => img.id)).toEqual([11, 12, 13]);
    });
    await settle();
    expect(searchReads().at(-1)[0]).toContain("only_deleted=true");

    const read = await vm.readViewForReconcile();

    expect(read.stacked).toBe(false);
  });

  // An empty list would read as "every card left".
  it("fails on a search answer that is not a list", async () => {
    const vm = await loadedGrid();
    let answer = [{ id: 11, format: "png", stack_id: null }];
    serveSearch(() => answer);
    useSearchStore().searchQuery = "harbour at dusk";
    await vi.waitFor(() => {
      expect(vm.imagesLoading).toBe(false);
      expect(vm.allGridImages.map((img) => img.id)).toEqual([11]);
    });
    await settle();
    answer = { detail: "upstream" };
    await expect(vm.readViewForReconcile()).rejects.toThrow(
      "the text search returned no list",
    );
  });

  it("drops a search answer read under a setting that has since changed", async () => {
    const vm = await loadedGrid();
    serveSearch(() => [{ id: 11, format: "png", stack_id: null }]);
    useSearchStore().searchQuery = "harbour at dusk";
    await vi.waitFor(() => {
      expect(vm.imagesLoading).toBe(false);
      expect(vm.allGridImages.map((img) => img.id)).toEqual([11]);
    });
    await settle();
    let answer;
    apiGet.mockImplementation((url) =>
      url.startsWith("/pictures/search?")
        ? new Promise((resolve) => {
            answer = resolve;
          })
        : Promise.resolve({ data: { pictures: [], count: 0, total: 0 } }),
    );
    const reading = vm.readViewForReconcile();
    await settle();
    expect(searchReads()).toHaveLength(2);

    useUserPrefsStore().applyTagFilter = !useUserPrefsStore().applyTagFilter;
    await settle();
    expect(searchReads()).toHaveLength(2);
    answer({ data: [] });

    expect(await reading).toBeNull();
  });

  // The setting has changed and the load it starts has not begun: the grid
  // still holds the stream's list, which a search's answer cannot be compared
  // with.
  it("compares nothing when the view now needs another kind of read than its load made", async () => {
    const vm = await loadedGrid();
    useSearchStore().searchQuery = "harbour at dusk";
    const calls = apiGet.mock.calls.length;

    const reading = vm.readViewForReconcile();

    expect(apiGet.mock.calls.length).toBe(calls);
    expect(await reading).toBeNull();
  });
});

/** A grid whose base list is longer than one reconnect read. */
async function longGrid(selectedCharacter) {
  const vm = await loadedGrid(selectedCharacter);
  const base = Array.from({ length: 6000 }, (_, i) => ({
    id: i + 1,
    format: "png",
    stack_id: null,
  }));
  vm.lastFetchedGridImages = base;
  vm.allGridImages = base;
  await settle();
  vm.visibleStart = 3000;
  return { vm, base };
}

/**
 * The window read answers from `base`, less the ids in `left`; a read by id
 * lists `listed` of its ids.
 */
function answerWindowAndIds(base, listed, left = new Set()) {
  answerIdRead = (url) => {
    const params = new URL(url, URL_BASE).searchParams;
    const ids = params.getAll("id");
    if (ids.length) {
      return Promise.resolve({
        data: {
          pictures: ids
            .filter((id) => listed.has(id))
            .map((id) => ({ id: Number(id), stack_id: null })),
          done: true,
        },
      });
    }
    const offset = Number(params.get("offset"));
    return Promise.resolve({
      data: {
        pictures: base
          .slice(offset, offset + 5000)
          .filter(({ id }) => !left.has(id))
          .map(({ id }) => ({ id, stack_id: null })),
        done: offset + 5000 >= base.length,
      },
    });
  };
}

function idListReads() {
  return idReads().filter(([url]) =>
    new URL(url, URL_BASE).searchParams.has("id"),
  );
}

describe("ImageGrid: the read by id after a partial reconnect read", () => {
  it("asks the window read's own query which of the ids it still lists, a thousand at a time", async () => {
    const { vm, base } = await longGrid();
    answerWindowAndIds(base, new Set(["2", "1500"]));
    const read = await vm.readViewForReconcile();
    const windowRead = idReads()[0][0];
    expect(typeof read.readListed).toBe("function");
    const ids = Array.from({ length: 1500 }, (_, i) => i + 1);
    const controller = new AbortController();

    const listed = await read.readListed(ids, { signal: controller.signal });

    expect(listed).toEqual({ listedIds: new Set(["2", "1500"]) });
    const reads = idListReads();
    expect(reads).toHaveLength(2);
    const asked = reads.map(([url]) => new URL(url, URL_BASE).searchParams);
    expect(asked[0].getAll("id")).toEqual(ids.slice(0, 1000).map(String));
    expect(asked[1].getAll("id")).toEqual(ids.slice(1000).map(String));
    for (const [index, params] of asked.entries()) {
      expect(params.get("offset")).toBe("0");
      expect(params.get("batch_limit")).toBe("1000");
      expect(params.get("fields")).toBe("id,stack_id");
      expect(params.get("stack_leaders_only")).toBe("true");
      const selection = new URLSearchParams(selectionOf(reads[index][0]));
      selection.delete("id");
      expect(selection.toString()).toBe(selectionOf(windowRead));
      expect(reads[index][1].signal).toBe(controller.signal);
    }
  });

  it("answers nothing, and reads nothing, once the view has changed", async () => {
    const { vm, base } = await longGrid();
    answerWindowAndIds(base, new Set(["2"]));
    const read = await vm.readViewForReconcile();
    useUserPrefsStore().applyTagFilter = !useUserPrefsStore().applyTagFilter;
    await settle();

    expect(await read.readListed([2])).toBeNull();
    expect(idListReads()).toHaveLength(0);
  });

  it("drops the answer when the view changes while the read by id is out", async () => {
    const { vm, base } = await longGrid();
    answerWindowAndIds(base, new Set(["2"]));
    const read = await vm.readViewForReconcile();
    const held = holdIdRead();
    const reading = read.readListed([2]);
    await settle();
    expect(idListReads()).toHaveLength(1);
    useUserPrefsStore().applyTagFilter = !useUserPrefsStore().applyTagFilter;
    await settle();
    held.answer({ pictures: [{ id: 2, stack_id: null }], done: true });

    expect(await reading).toBeNull();
  });

  // The listing narrows by `id` under every sort, character likeness
  // included, so the read by id settles that sort's cards as it does any
  // other's: one it no longer lists leaves, one it still lists has moved.
  it("under the character-likeness sort, takes an unlisted card out and puts a listed one on the pill", async () => {
    useSortStore().selectedSort = "CHARACTER_LIKENESS";
    useSortStore().selectedSimilarityCharacter = 7;
    const { vm, base } = await longGrid();
    answerWindowAndIds(base, new Set(["3002"]), new Set([3001, 3002]));
    const read = await vm.readViewForReconcile();
    expect(new URL(idReads()[0][0], URL_BASE).searchParams.get("sort")).toBe(
      "CHARACTER_LIKENESS",
    );
    const diff = diffViewAgainstIdRead(read);
    expect(diff.goneCards).toEqual([[3001], [3002]]);

    const listed = await read.readListed(diff.goneCards.flat());

    const reads = idListReads();
    expect(reads).toHaveLength(1);
    const asked = new URL(reads[0][0], URL_BASE).searchParams;
    expect(asked.get("sort")).toBe("CHARACTER_LIKENESS");
    expect(asked.getAll("id")).toEqual(["3001", "3002"]);
    expect(listed).toEqual({ listedIds: new Set(["3002"]) });

    const grid = {
      removeImagesById: vi.fn(),
      insertGridImagesById: vi.fn(),
      isImagesLoading: vi.fn(() => false),
      isLightboxOpen: vi.fn(() => false),
      markLightboxDeferredRefresh: vi.fn(),
      hasPendingGridImages: vi.fn(() => false),
    };
    const wsStore = reactive({
      isUploadInProgress: false,
      addPendingExternalImportIds: vi.fn(),
      addSortChangedExternalIds: vi.fn(),
    });
    useGridRealtimeSync({
      getMyClientId: () => "test-tab",
      grid,
      wsStore,
      pictureChangeAffectsView: () => true,
      getSelectedSort: () => "CHARACTER_LIKENESS",
      logger: { warn: vi.fn() },
    }).applyReconnectDiff(settleGoneCards(diff, listed.listedIds));
    expect(grid.removeImagesById).toHaveBeenCalledWith([3001]);
    expect(wsStore.addSortChangedExternalIds).toHaveBeenCalledWith([3002]);
    expect(wsStore.addPendingExternalImportIds).not.toHaveBeenCalled();
  });

  it("fails on a read by id with no pictures list", async () => {
    const { vm, base } = await longGrid();
    answerWindowAndIds(base, new Set());
    const read = await vm.readViewForReconcile();
    answerIdRead = () => Promise.resolve({ data: { detail: "upstream" } });
    await expect(read.readListed([2])).rejects.toThrow(
      "returned no pictures list",
    );
  });
});
