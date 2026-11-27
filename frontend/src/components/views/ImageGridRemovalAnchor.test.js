// A removal leaves the cards on screen where they are, and a removal while
// the view still streams in leaves the stream its place.
//
// Every removal goes through removeImagesById: a delete from another tab, a
// Scrapheap purge's echo, a ghost collapse, a filter the card stopped
// matching. The grid turns the browser's scroll anchoring off, so a removal
// above the viewport has to hold the view by hand.
//
// Mounts the real ImageGrid, as ImageGridLoadingState.test.js does.

import { describe, it, expect, beforeEach, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { setActivePinia, createPinia } from "pinia";
import { useSelectionStore } from "../../stores/useSelectionStore.js";
import { useProjectStore } from "../../stores/useProjectStore.js";
import { useGridStore } from "../../stores/useGridStore.js";
import { useSearchStore } from "../../stores/useSearchStore.js";
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

const ROWS = 400;
const VIEW_ROW = 200;

let wrapper = null;

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
  wrapper?.unmount();
  wrapper = null;
});

async function settle() {
  for (let i = 0; i < 4; i += 1) await nextTick();
}

/** Records scrollTop writes on the grid's scroller and fixes its height. */
function trackScroll(el, top, clientHeight) {
  const scroll = { top, writes: [] };
  Object.defineProperty(el, "scrollTop", {
    configurable: true,
    get: () => scroll.top,
    set: (value) => {
      scroll.top = value;
      scroll.writes.push(value);
    },
  });
  Object.defineProperty(el, "clientHeight", {
    configurable: true,
    get: () => clientHeight,
  });
  return scroll;
}

/** A loaded square grid of ROWS rows, scrolled 7 px into row VIEW_ROW. */
async function scrolledGrid() {
  wrapper = mountGrid();
  await vi.waitFor(() => expect(wrapper.vm.imagesLoading).toBe(false));
  await settle();
  const vm = wrapper.vm;
  const cols = useGridStore().columns;
  const cells = Array.from({ length: ROWS * cols }, (_, i) => ({
    id: i + 1,
    idx: i,
    format: "png",
    thumbnail: `/api/v1/pictures/thumbnails/${i + 1}.webp?v=test-1`,
  }));
  vm.allGridImages = cells;
  vm.lastFetchedGridImages = cells.slice();
  const scroll = trackScroll(
    vm.scrollWrapper,
    vm.squareRowTop(VIEW_ROW) + 7,
    4 * vm.rowHeight,
  );
  vm.recalculateVisibleRange();
  await settle();
  return { vm, cols, scroll };
}

/** The top of picture `id` against the view's top. */
function offsetOf(vm, scroll, id) {
  const index = vm.allGridImages.findIndex((img) => img.id === id);
  return vm.gridItemTopOffset(index) - scroll.top;
}

describe("a removal above the view holds the cards on screen", () => {
  it("keeps the top card on screen when a row above the view is removed", async () => {
    const { vm, cols, scroll } = await scrolledGrid();
    const topId = VIEW_ROW * cols + 1;
    const before = offsetOf(vm, scroll, topId);
    const startTop = scroll.top;

    // A row's worth of cells, spread over the rows above the view.
    vm.removeImagesById(Array.from({ length: cols }, (_, i) => 3 + 2 * i));
    await settle();

    expect(scroll.writes).toEqual([startTop - vm.rowHeight]);
    expect(offsetOf(vm, scroll, topId)).toBe(before);
    expect(vm.allGridImages[vm.visibleStart].id).toBe(topId);
  });

  it("does not hold the view when nothing above the top row leaves", async () => {
    const { vm, cols, scroll } = await scrolledGrid();

    vm.removeImagesById([(VIEW_ROW + 10) * cols + 1, (VIEW_ROW + 11) * cols]);
    await settle();

    expect(scroll.writes).toEqual([]);
  });

  it("keeps the top card on screen in a justified view", async () => {
    const WIDTH = 1200;
    const ASPECTS = [
      [3, 2],
      [2, 3],
      [1, 1],
      [16, 9],
      [4, 5],
    ];
    useGridStore().setThumbnailMode("justified");
    wrapper = mountGrid();
    await vi.waitFor(() => expect(wrapper.vm.imagesLoading).toBe(false));
    await settle();
    const vm = wrapper.vm;
    const cells = Array.from({ length: 3000 }, (_, i) => ({
      id: i + 1,
      idx: i,
      format: "png",
      width: ASPECTS[i % ASPECTS.length][0] * 100,
      height: ASPECTS[i % ASPECTS.length][1] * 100,
      thumbnail: `/api/v1/pictures/thumbnails/${i + 1}.webp?v=test-1`,
    }));
    vm.allGridImages = cells;
    vm.lastFetchedGridImages = cells.slice();
    vm.gridContainer.getBoundingClientRect = () => ({
      width: WIDTH,
      height: 0,
      top: 0,
      left: 0,
      right: WIDTH,
      bottom: 0,
    });
    vm.updateRowHeightFromGrid();
    await settle();
    const layout = vm.justifiedLayout;
    expect(layout?.rowHeights.length).toBeGreaterThan(200);
    const topRow = 150;
    const scroll = trackScroll(
      vm.scrollWrapper,
      layout.rowOffsets[topRow] + 7,
      800,
    );
    vm.recalculateVisibleRange();
    await settle();
    // The first card of the row partly scrolled off; the first one wholly on
    // screen is the next row's.
    const anchorId = vm.allGridImages[layout.rowStarts[topRow + 1]].id;
    const before = offsetOf(vm, scroll, anchorId);

    vm.removeImagesById([3, 11, 40]);
    await settle();

    expect(scroll.writes).toHaveLength(1);
    expect(offsetOf(vm, scroll, anchorId)).toBe(before);
  });
});

describe("a removal while the view streams in", () => {
  /**
   * Mounts a streaming view of `total` pictures and holds its first
   * background batch until the test lets it go. The server answers every
   * request from the list as it is when the answer leaves.
   */
  async function streamingGrid(total) {
    const deleted = new Set();
    const serverIds = () =>
      Array.from({ length: total }, (_, i) => i + 1).filter(
        (id) => !deleted.has(id),
      );
    const asked = [];
    let release;
    const held = new Promise((resolve) => {
      release = resolve;
    });
    apiGet.mockImplementation((url) => {
      const u = String(url);
      if (u.includes("/pictures/count")) {
        return Promise.resolve({ data: { count: total } });
      }
      if (u.includes("/pictures/stream")) {
        const params = new URL(u, "http://test").searchParams;
        const offset = Number(params.get("offset"));
        const limit = Number(params.get("batch_limit"));
        asked.push(offset);
        const answer = () => ({
          data: {
            pictures: serverIds()
              .slice(offset, offset + limit)
              .map((id) => ({ id, format: "png" })),
          },
        });
        // The first background batch: everything before it went at once.
        if (asked.length === (total > 1400 ? 3 : 2)) {
          return held.then(answer);
        }
        return Promise.resolve(answer());
      }
      return Promise.resolve({ data: [] });
    });
    wrapper = mountGrid();
    await vi.waitFor(() => expect(asked.length).toBe(total > 1400 ? 3 : 2));
    await settle();
    return { vm: wrapper.vm, deleted, serverIds, release };
  }

  async function streamEnded(vm) {
    await vi.waitFor(() => {
      expect(vm.allGridImages.every((img) => img.id != null)).toBe(true);
      expect(vm.imagesLoading).toBe(false);
    });
    await settle();
  }

  it("writes a batch that lands after a removal into the right slots", async () => {
    const { vm, deleted, serverIds, release } = await streamingGrid(300);

    // A loaded picture is deleted while the batch after it is out.
    deleted.add(5);
    vm.removeImagesById([5]);
    release();
    await streamEnded(vm);

    expect(vm.allGridImages.map((img) => img.id)).toEqual(serverIds());
  });

  it("does not repeat a row when a picture not loaded yet is removed", async () => {
    // Large enough for an early tail batch, which the background stream
    // fills up to.
    const { vm, deleted, serverIds, release } = await streamingGrid(1500);

    deleted.add(700);
    vm.removeImagesById([700]);
    release();
    await streamEnded(vm);

    expect(vm.allGridImages.map((img) => img.id)).toEqual(serverIds());
  });
});

describe("a user scroll while a re-read is out is not undone by its end", () => {
  // A re-read that keeps the scroll puts the view back where it stood when
  // it began. The user's own scroll drops that place; other input does not.
  async function heldReread() {
    useSearchStore().searchQuery = "cat";
    const { vm, cols, scroll } = await scrolledGrid();
    const rows = Array.from({ length: ROWS * cols }, (_, i) => ({
      id: i + 1,
      format: "png",
    }));
    let release;
    const held = new Promise((resolve) => {
      release = resolve;
    });
    let armed = false;
    let searchAsked = false;
    apiGet.mockImplementation((url) => {
      if (String(url).includes("/pictures/search")) {
        if (armed) {
          searchAsked = true;
          return held.then(() => ({ data: rows }));
        }
        return Promise.resolve({ data: rows });
      }
      return Promise.resolve({ data: { pictures: [], count: 0, total: 0 } });
    });
    // The restore is clamped to the content's height, which jsdom reports as 0.
    Object.defineProperty(vm.scrollWrapper, "scrollHeight", {
      configurable: true,
      get: () => 1e7,
    });
    const startTop = scroll.top;
    await vm.fetchAllGridImages({ force: true });
    await settle();
    scroll.top = startTop;
    scroll.writes.length = 0;
    armed = true;

    vm.preserveScrollOnNextFetch = true;
    const reread = vm.fetchAllGridImages({ force: true });
    await vi.waitFor(() => expect(searchAsked).toBe(true));
    vm.preserveScrollOnNextFetch = false;
    const movedTop = startTop + 50 * vm.rowHeight;
    const land = async () => {
      release();
      await reread;
      await settle();
    };
    return { scroll, startTop, movedTop, land };
  }

  const scroller = () => wrapper.find(".grid-scroll-wrapper");
  const pageDown = (target) =>
    target.dispatchEvent(
      new KeyboardEvent("keydown", { key: "PageDown", bubbles: true }),
    );

  it.each([
    ["the wheel", () => scroller().trigger("wheel")],
    ["a touch drag", () => scroller().trigger("touchmove")],
    ["a press on the scrollbar", () => scroller().trigger("pointerdown")],
    [
      "a middle-button press on a card",
      // Dispatched by hand: test-utils cannot set `button` on its event.
      () =>
        wrapper
          .find(".image-card")
          .element.dispatchEvent(
            new MouseEvent("pointerdown", { button: 1, bubbles: true }),
          ),
    ],
    ["a scroll key", () => pageDown(document.body)],
  ])("leaves the view where %s took it", async (_name, userScroll) => {
    const { scroll, movedTop, land } = await heldReread();

    await userScroll();
    scroll.top = movedTop;
    await land();

    expect(scroll.writes).toEqual([]);
  });

  it.each([
    ["no input at all", () => {}],
    [
      "a press on a card",
      () => wrapper.find(".image-card").trigger("pointerdown"),
    ],
    [
      "a scroll key typed in a field",
      () => {
        const field = document.createElement("input");
        document.body.appendChild(field);
        pageDown(field);
        field.remove();
      },
    ],
    [
      "a scroll key pressed in a dialog",
      () => {
        const dialog = document.createElement("div");
        dialog.innerHTML =
          '<div class="v-overlay__content"><button>OK</button></div>';
        document.body.appendChild(dialog);
        pageDown(dialog.querySelector("button"));
        dialog.remove();
      },
    ],
  ])("still puts the view back after %s", async (_name, input) => {
    const { scroll, startTop, movedTop, land } = await heldReread();

    await input();
    scroll.top = movedTop;
    await land();

    expect(scroll.top).toBe(startTop);
  });
});
