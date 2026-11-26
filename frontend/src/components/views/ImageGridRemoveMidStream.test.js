// A delete while the grid is still streaming in.
//
// The stream lays out one placeholder cell per picture in the view, then writes
// each batch at its offset into allGridImages. removeImagesById used to rebuild
// the grid from the rows loaded so far, which dropped every unfilled cell: the
// later batches then landed past the grid's end and were lost, and the grid
// stopped at the stream offset. Mid-stream the removed rows now leave in place
// and the cells after them move up a slot.
//
// Mounts the real ImageGrid, as ImageGridSquareGeometry.test.js does.

import { describe, it, expect, beforeEach, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { setActivePinia, createPinia } from "pinia";
import { useSelectionStore } from "../../stores/useSelectionStore.js";
import { useProjectStore } from "../../stores/useProjectStore.js";
import { ref, nextTick } from "vue";

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

function mountGrid() {
  const selectionStore = useSelectionStore();
  const projectStore = useProjectStore();
  selectionStore.selectedCharacter = "ALL";
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
    props: { backendUrl: "/api/v1" },
  });
}

beforeEach(() => {
  setActivePinia(createPinia());
  apiGet.mockReset();
  apiPost.mockReset();
  apiGet.mockResolvedValue({ data: { pictures: [], count: 0, total: 0 } });
  apiPost.mockResolvedValue({ data: {} });
});

const row = (id, idx) => ({ id, idx, format: "png" });

/** A grid of `total` cells whose first `loaded` hold pictures 1, 2, 3, … */
function streamingGrid(vm, { total, loaded }) {
  const cells = Array.from({ length: total }, (_, i) =>
    i < loaded ? row(i + 1, i) : { id: null, idx: i },
  );
  vm.allGridImages = cells;
  vm.lastFetchedGridImages = cells.filter((img) => img.id != null);
}

describe("removing pictures while the grid streams in", () => {
  it("keeps the unfilled cells, one fewer per removed picture", async () => {
    const wrapper = mountGrid();
    await nextTick();
    const vm = wrapper.vm;
    streamingGrid(vm, { total: 10, loaded: 4 });

    vm.removeImagesById([2]);

    expect(vm.allGridImages.map((img) => img.id)).toEqual([
      1,
      3,
      4,
      null,
      null,
      null,
      null,
      null,
      null,
    ]);
    // Every cell sits at its own position, where the next batch's offset
    // writes: the server's offsets moved up with the removal too.
    expect(vm.allGridImages.map((img) => img.idx)).toEqual([
      0, 1, 2, 3, 4, 5, 6, 7, 8,
    ]);
    expect(vm.lastFetchedGridImages.map((img) => img.id)).toEqual([1, 3, 4]);
  });

  it("lays the grid out again from what loaded, once the stream is done", async () => {
    const wrapper = mountGrid();
    await nextTick();
    const vm = wrapper.vm;
    streamingGrid(vm, { total: 4, loaded: 4 });

    vm.removeImagesById([2]);

    expect(vm.allGridImages.map((img) => img.id)).toEqual([1, 3, 4]);
    expect(vm.allGridImages.map((img) => img.idx)).toEqual([0, 1, 2]);
  });

  it("pulls the visible window back inside a grid the removal shortened", async () => {
    const wrapper = mountGrid();
    await nextTick();
    const vm = wrapper.vm;
    streamingGrid(vm, { total: 6, loaded: 3 });
    vm.visibleStart = 0;
    vm.visibleEnd = 6;

    vm.removeImagesById([1, 2]);

    expect(vm.allGridImages).toHaveLength(4);
    expect(vm.visibleEnd).toBe(4);
  });
});

describe("removing a picture while a 12,000-picture view streams in", () => {
  // The real stream: a 200-row first batch, a 200-row early tail, then the
  // background batches. The first background request is held; the delete
  // commits on the server and the grid takes the picture out before it
  // answers.
  const TOTAL = 12_000;

  function streamHoldingFirstBackground() {
    const removed = new Set();
    const serverIds = () =>
      Array.from({ length: TOTAL }, (_, i) => i + 1).filter(
        (id) => !removed.has(id),
      );
    let release;
    const held = new Promise((resolve) => {
      release = resolve;
    });
    let heldAsked = false;
    apiGet.mockImplementation((url) => {
      const u = String(url);
      if (u.includes("/summary")) {
        return Promise.resolve({ data: { image_count: TOTAL } });
      }
      if (u.includes("/pictures/count")) {
        return Promise.resolve({ data: { count: TOTAL } });
      }
      if (u.includes("/pictures/stream")) {
        const params = new URL(u, "http://test").searchParams;
        const offset = Number(params.get("offset"));
        const limit = Number(params.get("batch_limit"));
        const page = () =>
          serverIds()
            .slice(offset, offset + limit)
            .map((id) => ({ id, name: `test-${id}.png` }));
        if (!heldAsked && offset > 0 && limit === 5000) {
          heldAsked = true;
          return held.then(() => ({ data: { pictures: page() } }));
        }
        return Promise.resolve({ data: { pictures: page() } });
      }
      return Promise.resolve({ data: [] });
    });
    return {
      heldAsked: () => heldAsked,
      commitDelete: (ids) => ids.forEach((id) => removed.add(id)),
      release: () => release(),
      serverIds,
    };
  }

  it.each([
    ["a loaded picture", 2],
    ["a picture whose row has not arrived", 7000],
  ])(
    "ends with every remaining picture in its place after removing %s",
    async (_what, id) => {
      const server = streamHoldingFirstBackground();
      const wrapper = mountGrid();
      try {
        await vi.waitFor(() => expect(server.heldAsked()).toBe(true));
        await vi.waitFor(() =>
          expect(wrapper.vm.allGridImages[0]?.id).toBe(1),
        );
        server.commitDelete([id]);
        wrapper.vm.removeImagesById([id]);
        server.release();
        await vi.waitFor(() => expect(wrapper.vm.imagesLoading).toBe(false));

        const ids = wrapper.vm.allGridImages.map((img) => img.id);
        expect(ids.filter((cell) => cell == null)).toHaveLength(0);
        expect(ids).toHaveLength(TOTAL - 1);
        expect(ids).toEqual(server.serverIds());
        expect(wrapper.vm.allGridImages.every((img, i) => img.idx === i)).toBe(
          true,
        );
      } finally {
        wrapper.unmount();
      }
    },
  );
});

// The Scrapheap lists every scrapped picture on its own row: the server never
// folds a stack there (the listing's `only_deleted` branch), and the load keeps
// the rows as they came. A removal there used to rebuild the grid folding each
// stack to one card, so 634 pictures showed as 627 after one removal.
describe("removing a picture in the Scrapheap", () => {
  const HEAP = [
    { id: 1, stack_id: 7, stack_position: 0, stack_count: 2 },
    { id: 2, stack_id: 7, stack_position: 1, stack_count: 2 },
    { id: 3, stack_id: 8, stack_position: 0, stack_count: 2 },
    { id: 4, stack_id: 8, stack_position: 1, stack_count: 2 },
    { id: 5, stack_id: null },
    { id: 6, stack_id: 9, stack_position: 0, stack_count: 2 },
    { id: 7, stack_id: 9, stack_position: 1, stack_count: 2 },
  ].map((pic) => ({ ...pic, format: "png", name: `test-${pic.id}.png` }));

  function serveHeap() {
    const streamQueries = [];
    apiGet.mockImplementation((url) => {
      const u = String(url);
      if (u.includes("/pictures/count")) {
        return Promise.resolve({ data: { count: HEAP.length } });
      }
      if (u.includes("/pictures/stream")) {
        streamQueries.push(u);
        const params = new URL(u, "http://test").searchParams;
        const offset = Number(params.get("offset")) || 0;
        const limit = Number(params.get("batch_limit")) || HEAP.length;
        return Promise.resolve({
          data: { pictures: HEAP.slice(offset, offset + limit) },
        });
      }
      return Promise.resolve({ data: [] });
    });
    return streamQueries;
  }

  it("shows the server's pictures minus the removed one", async () => {
    const streamQueries = serveHeap();
    const wrapper = mountGrid();
    try {
      useSelectionStore().selectedCharacter = "SCRAPHEAP";
      const server = HEAP.map((pic) => pic.id);
      await vi.waitFor(() =>
        expect(wrapper.vm.allGridImages.map((img) => img.id)).toEqual(server),
      );
      await vi.waitFor(() => expect(wrapper.vm.imagesLoading).toBe(false));
      expect(streamQueries.at(-1)).toContain("only_deleted=true");

      wrapper.vm.removeImagesById([2]);

      expect(wrapper.vm.allGridImages.map((img) => img.id)).toEqual(
        server.filter((id) => id !== 2),
      );
      expect(wrapper.vm.allGridImages.every((img, i) => img.idx === i)).toBe(
        true,
      );
    } finally {
      wrapper.unmount();
    }
  });
});
