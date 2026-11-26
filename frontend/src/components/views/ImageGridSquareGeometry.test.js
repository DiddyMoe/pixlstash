// Where the square grid's own code places a row, against the virtualizer.
//
// useVirtualScroll puts square row r at squareRowTop(r): a top spacer that is
// always rendered, plus the measured row-gap, plus r pitches. Two places in
// ImageGrid used to redo that maths from zero (`r * rowHeight`): the ghost
// collapse's scroll anchor, which picked a card one origin away from the top
// one on screen, and the thumbnail-size watcher, which then overwrote the
// virtualizer's visible range with its own, up to a row off.
//
// Mounts the real ImageGrid, as ImageGridScrapheapRun.test.js does.

import { describe, it, expect, beforeEach, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { setActivePinia, createPinia } from "pinia";
import { useSelectionStore } from "../../stores/useSelectionStore.js";
import { useProjectStore } from "../../stores/useProjectStore.js";
import { useGridStore } from "../../stores/useGridStore.js";
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
    props: { backendUrl: "/api/v1", thumbnailMode: "square" },
  });
}

/**
 * Give the scroll wrapper the geometry jsdom does not lay out. Each value is a
 * getter, read when the virtualizer reads it, so it can follow a row pitch that
 * changes on the way.
 */
function placeViewport(el, { scrollTop, clientHeight }) {
  Object.defineProperty(el, "scrollTop", {
    configurable: true,
    get: scrollTop,
    set: () => {},
  });
  Object.defineProperty(el, "clientHeight", {
    configurable: true,
    get: clientHeight,
  });
}

beforeEach(() => {
  setActivePinia(createPinia());
  apiGet.mockReset();
  apiPost.mockReset();
  apiGet.mockResolvedValue({ data: { pictures: [], count: 0, total: 0 } });
  apiPost.mockResolvedValue({ data: {} });
});

describe("the square grid's row geometry", () => {
  it("anchors the ghost collapse on a card's real top, below the top spacer", async () => {
    const wrapper = mountGrid();
    await nextTick();
    const cols = useGridStore().columns;
    const vm = wrapper.vm;

    expect(vm.gridItemTopOffset(0)).toBe(vm.squareRowTop(0));
    expect(vm.gridItemTopOffset(0)).toBeGreaterThan(0);
    expect(vm.gridItemTopOffset(2 * cols + 1)).toBe(vm.squareRowTop(2));
  });

  it("leaves the visible range where the virtualizer puts it after a thumbnail resize", async () => {
    const wrapper = mountGrid();
    await nextTick();
    const vm = wrapper.vm;
    const cols = useGridStore().columns;
    const before = vm.rowHeight;
    // Half a pixel past two pitches from the very top, in the pitch the resize
    // leaves: still inside row 1, whose top is one origin further down than
    // `1 * pitch`. A range taken from the top of the scroll area says row 2.
    placeViewport(vm.scrollWrapper, {
      scrollTop: () => 2 * vm.rowHeight + 0.5,
      clientHeight: () => vm.rowHeight,
    });

    useGridStore().thumbnailSize += 32;
    await nextTick();
    await nextTick();
    await nextTick();

    // The resize was re-measured, not left at the old pitch.
    expect(vm.rowHeight).not.toBe(before);
    expect(vm.visibleStart).toBe(cols);
    expect(vm.visibleEnd).toBe(3 * cols);
  });
});
