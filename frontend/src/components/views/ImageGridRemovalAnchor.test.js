// A removal leaves the cards on screen where they are.
//
// Every removal goes through removeImagesById: a delete from another tab, a
// Scrapheap purge's echo once per slice, a ghost collapse, a filter the card
// stopped matching. It used to renumber every row after the first removed one,
// and the card's key carried that number (`img-<id>-<idx>`), so every card
// after it was torn down and built again (thumbnail element, readiness, face
// and detection overlays) - once per slice for a whole purge. Rows removed
// above the viewport also dragged the view up under the user, because the grid
// turns the browser's own scroll anchoring off (`overflow-anchor: none`) and
// anchored only the ghost collapse by hand.
//
// Mounts the real ImageGrid, as ImageGridSquareGeometry.test.js does.

import { describe, it, expect, beforeEach, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { setActivePinia, createPinia } from "pinia";
import { useSelectionStore } from "../../stores/useSelectionStore.js";
import { useProjectStore } from "../../stores/useProjectStore.js";
import { useGridStore } from "../../stores/useGridStore.js";
import { useSearchStore } from "../../stores/useSearchStore.js";
import { useFilterStore } from "../../stores/useFilterStore.js";
import { useWsStore } from "../../stores/useWsStore.js";
import { SCRAPHEAP_PICTURES_ID } from "../../stores/useViewStore.js";
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

const ROWS = 400;
const VIEW_ROW = 200;

let wrapper = null;

function mountGrid(view = "ALL") {
  const selectionStore = useSelectionStore();
  const projectStore = useProjectStore();
  selectionStore.selectedCharacter = view;
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

/**
 * A loaded grid of ROWS rows, every card with its thumbnail, scrolled 7 px
 * into row VIEW_ROW. `placeholders` cells with no picture yet follow it, as
 * while the view streams in. The viewport is jsdom's, so its geometry is set
 * by hand; scrollTop writes are recorded.
 */
async function scrolledGrid({ placeholders = 0, view = "ALL" } = {}) {
  wrapper = mountGrid(view);
  await vi.waitFor(() => expect(wrapper.vm.imagesLoading).toBe(false));
  await settle();
  const vm = wrapper.vm;
  const cols = useGridStore().columns;
  const loaded = ROWS * cols;
  const cells = Array.from({ length: loaded + placeholders }, (_, i) =>
    i < loaded
      ? {
          id: i + 1,
          idx: i,
          format: "png",
          thumbnail: `/api/v1/pictures/thumbnails/${i + 1}.webp?v=test-1`,
        }
      : { id: null, idx: i },
  );
  vm.allGridImages = cells;
  vm.lastFetchedGridImages = cells.filter((img) => img.id != null);
  vm.loadedRanges = [[0, loaded]];

  const scroll = { top: vm.squareRowTop(VIEW_ROW) + 7, writes: [] };
  const el = vm.scrollWrapper;
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
    get: () => 4 * vm.rowHeight,
  });
  vm.recalculateVisibleRange();
  await settle();
  return { vm, cols, scroll };
}

/** The rendered cards, by picture id. */
function renderedCards() {
  const cards = new Map();
  for (const el of wrapper.element.querySelectorAll(
    ".image-card[data-picture-id]",
  )) {
    cards.set(el.getAttribute("data-picture-id"), el);
  }
  return cards;
}

function visibleIds(vm) {
  return vm.allGridImages
    .slice(vm.visibleStart, vm.visibleEnd)
    .map((img) => String(img.id));
}

/** The first card at or below the viewport's top, and its offset into it. */
function topCard(vm, scrollTop) {
  const index = vm.allGridImages.findIndex(
    (img, i) => img.id != null && vm.gridItemTopOffset(i) >= scrollTop,
  );
  return {
    id: vm.allGridImages[index].id,
    offset: vm.gridItemTopOffset(index) - scrollTop,
  };
}

function idsOfRows(cols, firstRow, rowCount) {
  return Array.from(
    { length: rowCount * cols },
    (_, i) => firstRow * cols + i + 1,
  );
}

/**
 * The cards that were on screen keep their own elements, the picture on
 * screen keeps its pixel, and the view shows the same pictures it did.
 */
async function expectViewportHeld(vm, scroll, before) {
  const after = renderedCards();
  for (const id of before.visible) {
    expect(after.get(id), `card ${id} was torn down`).toBe(
      before.cards.get(id),
    );
  }
  const anchorIndex = vm.allGridImages.findIndex(
    (img) => img.id === before.anchor.id,
  );
  expect(vm.gridItemTopOffset(anchorIndex) - scroll.top).toBe(
    before.anchor.offset,
  );
  vm.recalculateVisibleRange();
  await settle();
  expect(visibleIds(vm)).toEqual(before.visible);
}

function snapshot(vm, scroll) {
  return {
    cards: renderedCards(),
    visible: visibleIds(vm),
    anchor: topCard(vm, scroll.top),
  };
}

/** Where each card sits at `scrollTop`: its top against the view's, its column. */
function positions(vm, scrollTop, cols) {
  const pos = new Map();
  vm.allGridImages.forEach((img, i) => {
    if (img.id == null) return;
    pos.set(String(img.id), {
      y: vm.gridItemTopOffset(i) - scrollTop,
      col: i % cols,
    });
  });
  return pos;
}

/** The cards of `pos` on scrolledGrid's screen, four rows tall. */
function onScreen(vm, pos) {
  const viewH = 4 * vm.rowHeight;
  return new Map(
    [...pos].filter(([, p]) => p.y + vm.rowHeight > 0 && p.y < viewH),
  );
}

/**
 * How the cards that were on screen moved: held their place, slid along their
 * row, went to another row, or left the grid. `slotsMoved` is the most slots
 * any card moved, reading the grid row by row.
 */
function motion(before, after, cols, pitch) {
  const m = { same: 0, sideways: 0, rowJump: 0, gone: 0, slotsMoved: 0 };
  for (const [id, b] of before) {
    const a = after.get(id);
    if (!a) {
      m.gone += 1;
      continue;
    }
    const rows = Math.round((a.y - b.y) / pitch);
    m.slotsMoved = Math.max(m.slotsMoved, Math.abs(rows * cols + a.col - b.col));
    if (rows !== 0) m.rowJump += 1;
    else if (a.col !== b.col) m.sideways += 1;
    else m.same += 1;
  }
  return m;
}

/**
 * Remove `ids` from scrolledGrid's grid. `held` is how the cards on screen
 * moved with the view as the removal left it, `unheld` how they would have
 * moved had the scroll stayed where it was.
 */
async function removeFromScrolledGrid(ids, options) {
  const { vm, cols, scroll } = await scrolledGrid(options);
  const pitch = vm.rowHeight;
  const startTop = scroll.top;
  const before = onScreen(vm, positions(vm, startTop, cols));
  const cards = renderedCards();
  vm.removeImagesById(ids);
  await settle();
  return {
    vm,
    cols,
    scroll,
    pitch,
    startTop,
    cards,
    held: motion(before, positions(vm, scroll.top, cols), cols, pitch),
    unheld: motion(before, positions(vm, startTop, cols), cols, pitch),
  };
}

/** The first picture of the row the view's top edge is in. */
function topRowFirstId(vm, scrollTop, cols) {
  const row = Math.floor((scrollTop - vm.squareRowTop(0)) / vm.rowHeight);
  return vm.allGridImages[row * cols]?.id;
}

describe("a removal leaves the cards on screen in place", () => {
  it("names each card by its picture", async () => {
    const { vm } = await scrolledGrid();
    const cards = renderedCards();
    expect(cards.size).toBeGreaterThan(0);
    for (const id of visibleIds(vm)) expect(cards.has(id)).toBe(true);
  });

  it.each([
    ["the grid is loaded", 0],
    ["the grid still streams in", 500],
  ])(
    "holds the view when rows above it leave, while %s",
    async (_what, placeholders) => {
      const { vm, cols, scroll } = await scrolledGrid({ placeholders });
      // More rows than the render buffer above the window, so a window left at
      // its old index would hold none of the cards that were on screen.
      const removedRows = Math.ceil(vm.renderBuffer / cols) + 2;
      const before = snapshot(vm, scroll);

      vm.removeImagesById(idsOfRows(cols, 5, removedRows));
      await settle();

      expect(scroll.writes).toHaveLength(1);
      await expectViewportHeld(vm, scroll, before);
    },
  );

  it("holds the view across a Scrapheap purge's slices", async () => {
    // Emptying the Scrapheap purges in slices, and each slice's echo removes
    // its rows from the grid separately. The cards on screen stay the same
    // elements through every slice, and none of them fades in again.
    const { vm, cols, scroll } = await scrolledGrid({ view: SCRAPHEAP_PICTURES_ID });
    expect(vm.isScrapheapView).toBe(true);
    const before = snapshot(vm, scroll);
    // jsdom loads no image, so mark the cards on screen as faded in.
    for (const id of before.visible) vm.onThumbnailLoad(Number(id));
    await settle();

    for (const firstRow of [5, 25, 45]) {
      vm.removeImagesById(idsOfRows(cols, firstRow, 10));
      await settle();
    }

    expect(scroll.writes).toHaveLength(3);
    await expectViewportHeld(vm, scroll, before);
    for (const id of before.visible) {
      expect(vm.thumbnailLoadedMap[id], `card ${id} faded in again`).toBe(1);
    }
  });

  it("leaves the scroll alone when the rows that leave are below the view", async () => {
    const { vm, cols, scroll } = await scrolledGrid();
    const before = snapshot(vm, scroll);

    vm.removeImagesById(idsOfRows(cols, VIEW_ROW + 40, 3));
    await settle();

    expect(scroll.writes).toEqual([]);
    await expectViewportHeld(vm, scroll, before);
  });

  it("closes a gap on screen without moving the rows above it", async () => {
    const { vm, cols, scroll } = await scrolledGrid();
    const before = snapshot(vm, scroll);
    const onScreen = Number(before.visible[cols + 1]);

    vm.removeImagesById([onScreen]);
    await settle();

    expect(scroll.writes).toEqual([]);
    const after = renderedCards();
    expect(after.has(String(onScreen))).toBe(false);
    for (const id of before.visible) {
      if (id !== String(onScreen)) {
        expect(after.get(id), `card ${id} was torn down`).toBe(
          before.cards.get(id),
        );
      }
    }
  });

  it("keeps the cursor on its picture when rows above it leave", async () => {
    const { vm, cols, scroll } = await scrolledGrid();
    const cursorId = Number(snapshot(vm, scroll).visible[2]);
    vm.cursorIdx = cursorId - 1;

    vm.removeImagesById(idsOfRows(cols, 5, 3));
    await settle();

    expect(vm.allGridImages[vm.cursorIdx].id).toBe(cursorId);
    expect(
      wrapper.element
        .querySelector(".image-card-cursor")
        ?.getAttribute("data-picture-id"),
    ).toBe(String(cursorId));
  });

  it("focuses the card a key moved to, after rows above it left", async () => {
    const { vm, cols, scroll } = await scrolledGrid();
    const targetId = Number(snapshot(vm, scroll).visible[cols]);

    vm.removeImagesById(idsOfRows(cols, 5, 3));
    await settle();
    const focusSpy = vi.spyOn(HTMLElement.prototype, "focus");
    try {
      const index = vm.allGridImages.findIndex((img) => img.id === targetId);
      await vm.focusGridCursor(index);
      const focused = focusSpy.mock.contexts.at(-1);
      expect(focused?.getAttribute("data-picture-id")).toBe(String(targetId));
    } finally {
      focusSpy.mockRestore();
    }
  });
});

describe("a removal above the view moves the cards on screen the least", () => {
  // n cells out above the top row on screen move every card on screen n slots
  // back. Scrolling up round(n / cols) whole rows leaves each card fewer than
  // half a row of slots from where it was, so only the cards that wrap change
  // row. Holding one card's pixel instead dropped the rest of the screen a row
  // whenever n was not a whole number of rows.
  it.each([
    ["one cell", () => 1, 0],
    ["a row less one cell", (cols) => cols - 1, 1],
    ["a whole row", (cols) => cols, 1],
    ["a row and one cell", (cols) => cols + 1, 1],
  ])(
    "scrolls up by the whole rows that leave when %s above the view leaves",
    async (_what, count, rowsUp) => {
      const cols = useGridStore().columns;
      const n = count(cols);
      // Spread over the rows above, none in the top row on screen.
      const ids = Array.from({ length: n }, (_, i) => 3 + 2 * i);
      expect(Math.max(...ids)).toBeLessThanOrEqual(VIEW_ROW * cols);

      const { scroll, pitch, startTop, held, unheld, cards } =
        await removeFromScrolledGrid(ids);

      expect(scroll.writes).toEqual(rowsUp ? [startTop - rowsUp * pitch] : []);
      expect(held.gone).toBe(0);
      expect(held.slotsMoved).toBe(Math.abs(n - rowsUp * cols));
      expect(held.rowJump).toBeLessThanOrEqual(unheld.rowJump);
      // Each picture that stays on screen keeps its element.
      for (const [id, el] of renderedCards()) {
        if (cards.has(id)) expect(el, `card ${id} was torn down`).toBe(cards.get(id));
      }
    },
  );

  it("leaves the scroll alone when cards leave the row partly scrolled off", async () => {
    // The top row on screen is on screen, if partly: its gap closes in plain
    // sight, the rows above it do not move, and nothing jumps a row to keep a
    // card that left in place.
    const cols = useGridStore().columns;
    const ids = Array.from({ length: cols - 1 }, (_, i) => VIEW_ROW * cols + 1 + i);

    const { scroll, held } = await removeFromScrolledGrid(ids);

    expect(scroll.writes).toEqual([]);
    expect(held.gone).toBe(cols - 1);
  });

  it("does not go back to rows scrolled past when every card on screen leaves", async () => {
    const cols = useGridStore().columns;
    const { vm, scroll } = await removeFromScrolledGrid(
      idsOfRows(cols, VIEW_ROW, 5),
    );

    expect(scroll.writes).toEqual([]);
    // The rows below move up into the view: the first one not yet seen tops it.
    expect(topRowFirstId(vm, scroll.top, cols)).toBe((VIEW_ROW + 5) * cols + 1);
  });

  it("does not go back to rows scrolled past when the screen and a row above it leave", async () => {
    const cols = useGridStore().columns;
    const { vm, scroll, pitch, startTop } = await removeFromScrolledGrid([
      ...idsOfRows(cols, 5, 1),
      ...idsOfRows(cols, VIEW_ROW, 5),
    ]);

    expect(scroll.writes).toEqual([startTop - pitch]);
    expect(topRowFirstId(vm, scroll.top, cols)).toBe((VIEW_ROW + 5) * cols + 1);
    for (const id of visibleIds(vm)) {
      expect(Number(id), `picture ${id} was scrolled past`).toBeGreaterThan(
        VIEW_ROW * cols,
      );
    }
  });

  it("writes nothing when the rest of the grid from the top row on leaves", async () => {
    // The grid now ends above the view; the browser pulls the scroll back to
    // its new end, and there is no card left on screen to hold.
    const cols = useGridStore().columns;
    const { scroll } = await removeFromScrolledGrid([
      ...idsOfRows(cols, 5, 1),
      ...idsOfRows(cols, VIEW_ROW, ROWS - VIEW_ROW),
    ]);

    expect(scroll.writes).toEqual([]);
  });

  it("counts rows at the column count of the moment", async () => {
    const gridStore = useGridStore();
    const { vm, scroll, cols: startCols } = await scrolledGrid();
    gridStore.sizeLevel = 5;
    await settle();
    const cols = gridStore.columns;
    expect(cols).not.toBe(startCols);
    scroll.top = vm.squareRowTop(VIEW_ROW) + 7;
    scroll.writes.length = 0;
    vm.recalculateVisibleRange();
    await settle();
    const pitch = vm.rowHeight;
    const startTop = scroll.top;
    const before = onScreen(vm, positions(vm, startTop, cols));

    // Two rows' worth at the new count, which is not two rows at the old one.
    vm.removeImagesById(Array.from({ length: 2 * cols }, (_, i) => 3 + 2 * i));
    await settle();

    expect(scroll.writes).toEqual([startTop - 2 * pitch]);
    const held = motion(before, positions(vm, scroll.top, cols), cols, pitch);
    expect(held.same).toBe(before.size);
  });
});

describe("the stream's drop above the view moves the cards on screen the least", () => {
  // A re-read that keeps the scroll (a delete's refetch) deep in the grid
  // swaps the viewport's rows in with the first batch and streams the rows
  // above them after. A picture deleted before the stream reached its row
  // leaves a cell the server no longer fills, which the stream drops when its
  // rows reach the viewport's: every card on screen moves back a slot per
  // cell. The drop is a removal above the view like any other and is held
  // the same way. The stream's end leaves the scroll where it is; a re-read
  // that puts it back at its end is the next describe's.
  async function rereadDroppingAbove(removedCount) {
    const { vm, cols, scroll } = await scrolledGrid();
    const total = ROWS * cols;
    const deleted = new Set();
    const serverIds = () =>
      Array.from({ length: total }, (_, i) => i + 1).filter(
        (id) => !deleted.has(id),
      );
    let release;
    const held = new Promise((resolve) => {
      release = resolve;
    });
    // Armed for the re-read only: the load before it answers at once.
    let armed = false;
    let backgroundAsked = false;
    const offsets = [];
    apiGet.mockImplementation((url) => {
      const u = String(url);
      if (u.includes("/pictures/count")) {
        return Promise.resolve({ data: { count: total } });
      }
      if (u.includes("/pictures/stream")) {
        const params = new URL(u, "http://test").searchParams;
        const offset = Number(params.get("offset"));
        const limit = Number(params.get("batch_limit"));
        // The background stream's first request starts one row before the
        // first batch's end, the row it lines its answer up on.
        const firstLimit = offsets.find((o) => o.offset === 0)?.limit;
        const background =
          armed && !backgroundAsked && offset === firstLimit - 1;
        offsets.push({ offset, limit, background });
        // Read when the request is answered, the server's list as it is then.
        const page = () => ({
          data: {
            pictures: serverIds()
              .slice(offset, offset + limit)
              .map((id) => ({ id, format: "png" })),
          },
        });
        if (background) {
          backgroundAsked = true;
          return held.then(page);
        }
        return Promise.resolve(page());
      }
      return Promise.resolve({ data: { pictures: [], count: 0, total: 0 } });
    });
    // The re-read's end clamps the scroll it puts back to the content's
    // height, which jsdom reports as 0.
    Object.defineProperty(vm.scrollWrapper, "scrollHeight", {
      configurable: true,
      get: () => 1e7,
    });
    const pitch = vm.rowHeight;
    const startTop = scroll.top;

    // The cards on screen were read for this view: it loads once first, so
    // the re-read below is one of the same view, which keeps its cards.
    await vm.fetchAllGridImages({ force: true });
    await settle();
    expect(vm.allGridImages.map((img) => img.id)).toEqual(serverIds());
    scroll.top = startTop;
    scroll.writes.length = 0;
    vm.recalculateVisibleRange();
    await settle();
    offsets.length = 0;
    armed = true;

    vm.preserveScrollOnNextFetch = true;
    const reread = vm.fetchAllGridImages({ force: true });
    await vi.waitFor(() =>
      expect(backgroundAsked, JSON.stringify(offsets)).toBe(true),
    );
    // The viewport's rows went in with the first batch; the deleted pictures
    // sit between the first batch and them, so their rows have not arrived.
    // The viewport's request is the one starting between the first batch and
    // the row on top of the view (the early tail starts near the grid's end).
    const viewAsk = offsets.find(
      ({ offset, background }) =>
        !background && offset > 0 && offset < VIEW_ROW * cols,
    );
    expect(
      viewAsk,
      JSON.stringify({
        offsets,
        visibleStart: vm.visibleStart,
        visibleEnd: vm.visibleEnd,
        cols,
      }),
    ).toBeDefined();
    const viewStart = viewAsk.offset;
    const gone = Array.from({ length: removedCount }, (_, i) => 301 + i);
    expect(Math.max(...gone)).toBeLessThan(viewStart);
    expect(vm.allGridImages[gone[0] - 1].id).toBeNull();
    const before = onScreen(vm, positions(vm, scroll.top, cols));
    expect(before.size).toBeGreaterThan(0);

    gone.forEach((id) => deleted.add(id));
    vm.removeImagesById(gone);
    await settle();
    // Nothing on screen or above it has left the grid yet.
    expect(scroll.writes).toEqual([]);
    release();
    await reread;
    await settle();

    expect(vm.allGridImages.map((img) => img.id)).toEqual(serverIds());
    return {
      cols,
      pitch,
      startTop,
      scroll,
      held: motion(before, positions(vm, scroll.top, cols), cols, pitch),
    };
  }

  it("scrolls up a row when a row's worth of cells is dropped above the view", async () => {
    const cols = useGridStore().columns;
    const { scroll, pitch, startTop, held } = await rereadDroppingAbove(cols);

    expect(scroll.top).toBe(startTop - pitch);
    // Every card on screen is where it was.
    expect(held.same).toBeGreaterThan(0);
    expect(held.gone + held.sideways + held.rowJump).toBe(0);
  });

  it("leaves the scroll alone when one cell is dropped above the view", async () => {
    const { scroll, startTop, held } = await rereadDroppingAbove(1);

    expect(scroll.top).toBe(startTop);
    expect(held.gone).toBe(0);
    expect(held.slotsMoved).toBe(1);
  });

  it("leaves no place behind for a later load to scroll back to", async () => {
    // The stream never puts the scroll back. Had it kept the place where the
    // re-read began, a search opened afterwards, whose end does put the
    // scroll back, would open on that place instead of where the user is.
    const { scroll } = await rereadDroppingAbove(1);
    const vm = wrapper.vm;
    const cols = useGridStore().columns;
    apiGet.mockImplementation((url) => {
      if (String(url).includes("/pictures/search")) {
        const rows = Array.from({ length: ROWS * cols }, (_, i) => ({
          id: i + 1,
          format: "png",
        }));
        return Promise.resolve({ data: rows });
      }
      return Promise.resolve({ data: { pictures: [], count: 0, total: 0 } });
    });
    scroll.top = 0;
    scroll.writes.length = 0;

    useSearchStore().searchQuery = "cat";
    await vm.fetchAllGridImages({ force: true });
    await settle();

    expect(vm.allGridImages).toHaveLength(ROWS * cols);
    expect(scroll.top).toBe(0);
  });
});

describe("a re-read's end does not undo a hold made while it was out", () => {
  // A re-read that keeps the scroll and waits for its whole answer (a search
  // view's refetch) records where the view was and puts the scroll back there
  // once its answer is in. A delete that lands while it is out is held like
  // any removal above the view, and moves the recorded place with it, so the
  // answer does not put the view back a row and jump every card on screen.
  it("keeps the row a removal above the view scrolled to", async () => {
    useSearchStore().searchQuery = "cat";
    const { vm, cols, scroll } = await scrolledGrid();
    const total = ROWS * cols;
    const deleted = new Set();
    const serverRows = () =>
      Array.from({ length: total }, (_, i) => i + 1)
        .filter((id) => !deleted.has(id))
        .map((id) => ({ id, format: "png" }));
    let release;
    const held = new Promise((resolve) => {
      release = resolve;
    });
    // Armed for the re-read only: the load before it answers at once.
    let armed = false;
    let searchAsked = false;
    apiGet.mockImplementation((url) => {
      if (String(url).includes("/pictures/search")) {
        // Read when the request is answered, the server's list as it is then.
        if (armed) {
          searchAsked = true;
          return held.then(() => ({ data: serverRows() }));
        }
        return Promise.resolve({ data: serverRows() });
      }
      return Promise.resolve({ data: { pictures: [], count: 0, total: 0 } });
    });
    // The re-read's end clamps the scroll it puts back to the content's
    // height, which jsdom reports as 0.
    Object.defineProperty(vm.scrollWrapper, "scrollHeight", {
      configurable: true,
      get: () => 1e7,
    });
    const pitch = vm.rowHeight;
    const startTop = scroll.top;

    await vm.fetchAllGridImages({ force: true });
    await settle();
    expect(vm.allGridImages.map((img) => img.id)).toEqual(
      serverRows().map((row) => row.id),
    );
    scroll.top = startTop;
    scroll.writes.length = 0;
    vm.recalculateVisibleRange();
    await settle();
    armed = true;

    vm.preserveScrollOnNextFetch = true;
    const reread = vm.fetchAllGridImages({ force: true });
    await vi.waitFor(() => expect(searchAsked).toBe(true));
    const before = onScreen(vm, positions(vm, scroll.top, cols));
    expect(before.size).toBeGreaterThan(0);

    const gone = idsOfRows(cols, VIEW_ROW - 100, 1);
    gone.forEach((id) => deleted.add(id));
    vm.removeImagesById(gone);
    await settle();
    expect(scroll.writes).toEqual([startTop - pitch]);
    release();
    await reread;
    await settle();

    expect(vm.allGridImages.map((img) => img.id)).toEqual(
      serverRows().map((row) => row.id),
    );
    expect(scroll.top).toBe(startTop - pitch);
    const after = motion(before, positions(vm, scroll.top, cols), cols, pitch);
    expect(after.same).toBe(before.size);
  });
});

describe("a user scroll while a re-read is out is not undone by its end", () => {
  // A re-read that keeps the scroll puts the view back where it stood when
  // the re-read began. Someone who scrolled on meanwhile was pulled back
  // there when the answer landed. Their scroll drops the place; a scroll the
  // grid writes itself does not.
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
    // Where the user, or the grid, takes the view while the answer is out.
    const movedTop = startTop + 50 * vm.rowHeight;
    const land = async () => {
      release();
      await reread;
      await settle();
    };
    return { vm, scroll, startTop, movedTop, land };
  }

  const scroller = () => wrapper.find(".grid-scroll-wrapper");

  it.each([
    ["the wheel", () => scroller().trigger("wheel")],
    ["a touch drag", () => scroller().trigger("touchmove")],
    ["a press on the scrollbar", () => scroller().trigger("pointerdown")],
    [
      "a middle-button press on a card",
      // Dispatched by hand: test-utils cannot set `button` on the event it
      // builds.
      () =>
        wrapper
          .find(".image-card")
          .element.dispatchEvent(
            new MouseEvent("pointerdown", { button: 1, bubbles: true }),
          ),
    ],
    [
      "a scroll key",
      () =>
        document.body.dispatchEvent(
          new KeyboardEvent("keydown", { key: "PageDown", bubbles: true }),
        ),
    ],
    [
      // Only a key pressed inside the menu is the menu's.
      "a scroll key pressed outside an open menu",
      () => {
        const menu = document.createElement("div");
        menu.className = "v-overlay v-overlay--active v-menu";
        menu.innerHTML = '<div class="v-overlay__content"></div>';
        document.body.appendChild(menu);
        document.body.dispatchEvent(
          new KeyboardEvent("keydown", { key: "PageDown", bubbles: true }),
        );
        menu.remove();
      },
    ],
  ])("leaves the view where %s took it", async (_name, userScroll) => {
    const { scroll, movedTop, land } = await heldReread();

    await userScroll();
    scroll.top = movedTop;
    await land();

    expect(scroll.top).toBe(movedTop);
    expect(scroll.writes).toEqual([]);
  });

  it.each([
    ["no input at all", () => {}],
    ["a press on a card", () => wrapper.find(".image-card").trigger("pointerdown")],
    [
      "a scroll key typed in a field",
      () => {
        const field = document.createElement("input");
        document.body.appendChild(field);
        field.dispatchEvent(
          new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }),
        );
        field.remove();
      },
    ],
    [
      "a key that does not scroll",
      () =>
        document.body.dispatchEvent(
          new KeyboardEvent("keydown", { key: "Shift", bubbles: true }),
        ),
    ],
    [
      "a scroll key the open lightbox takes",
      () => {
        wrapper.vm.lightboxOpen = true;
        document.body.dispatchEvent(
          new KeyboardEvent("keydown", { key: "PageDown", bubbles: true }),
        );
        wrapper.vm.lightboxOpen = false;
      },
    ],
    [
      // A menu has no scrim and Vuetify lets its list's arrow keys bubble:
      // the arrow moves the menu's own list, not the grid behind it.
      "a scroll key pressed in an open menu",
      () => {
        const menu = document.createElement("div");
        menu.className = "v-overlay v-overlay--active v-menu";
        menu.innerHTML =
          '<div class="v-overlay__content"><div class="v-list">' +
          '<div class="v-list-item v-list-item--link" tabindex="-2">Item</div>' +
          "</div></div>";
        document.body.appendChild(menu);
        menu
          .querySelector(".v-list-item")
          .dispatchEvent(
            new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }),
          );
        menu.remove();
      },
    ],
    [
      "a scroll key pressed in a dialog",
      () => {
        const dialog = document.createElement("div");
        dialog.className = "v-overlay v-overlay--active v-dialog";
        dialog.innerHTML =
          '<div class="v-overlay__scrim"></div>' +
          '<div class="v-overlay__content"><button>OK</button></div>';
        document.body.appendChild(dialog);
        dialog
          .querySelector("button")
          .dispatchEvent(
            new KeyboardEvent("keydown", { key: "PageDown", bubbles: true }),
          );
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

  it("puts back the place a realtime refresh recorded once its debounced re-read lands", async () => {
    useSearchStore().searchQuery = "cat";
    const { vm, cols, scroll } = await scrolledGrid();
    const rows = Array.from({ length: ROWS * cols }, (_, i) => ({
      id: i + 1,
      format: "png",
    }));
    let searches = 0;
    apiGet.mockImplementation((url) => {
      if (String(url).includes("/pictures/search")) {
        searches += 1;
        // The answer replaces the list with the view at the top.
        scroll.top = 0;
        return Promise.resolve({ data: rows });
      }
      return Promise.resolve({ data: { pictures: [], count: 0, total: 0 } });
    });
    Object.defineProperty(vm.scrollWrapper, "scrollHeight", {
      configurable: true,
      get: () => 1e7,
    });
    const startTop = scroll.top;
    scroll.writes.length = 0;

    // The gridVersion watcher records the place and clears the flag before
    // its debounced re-read runs. It ignores a refresh within 1200ms of the
    // last one, the mount included, so the clock is moved past that.
    const realNow = Date.now.bind(Date);
    const clock = vi
      .spyOn(Date, "now")
      .mockImplementation(() => realNow() + 5000);
    try {
      vm.preserveScrollOnNextFetch = true;
      useGridStore().refreshGridVersion();
      await vi.waitFor(() => expect(searches).toBe(1), { timeout: 3000 });
      await vi.waitFor(() => expect(vm.imagesLoading).toBe(false));
      await settle();
    } finally {
      clock.mockRestore();
    }

    expect(vm.preserveScrollOnNextFetch).toBe(false);
    expect(scroll.top).toBe(startTop);
  });
});

describe("a removal above a justified view holds the card on top", () => {
  // Justified rows re-pack from the first cell that left, so no row count
  // holds; the topmost staying card on screen keeps its pixel instead.
  const WIDTH = 1200;
  const ASPECTS = [
    [3, 2],
    [2, 3],
    [1, 1],
    [16, 9],
    [4, 5],
  ];

  async function justifiedGrid() {
    useGridStore().setThumbnailMode("justified");
    wrapper = mountGrid();
    await vi.waitFor(() => expect(wrapper.vm.imagesLoading).toBe(false));
    await settle();
    const vm = wrapper.vm;
    const count = 3000;
    const cells = Array.from({ length: count }, (_, i) => ({
      id: i + 1,
      idx: i,
      format: "png",
      width: ASPECTS[i % ASPECTS.length][0] * 100,
      height: ASPECTS[i % ASPECTS.length][1] * 100,
      thumbnail: `/api/v1/pictures/thumbnails/${i + 1}.webp?v=test-1`,
    }));
    vm.allGridImages = cells;
    vm.lastFetchedGridImages = cells.slice();
    vm.loadedRanges = [[0, count]];
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
    const scroll = { top: layout.rowOffsets[topRow] + 7, writes: [] };
    const el = vm.scrollWrapper;
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
      get: () => 800,
    });
    vm.recalculateVisibleRange();
    await settle();
    return { vm, scroll, firstSlot: layout.rowStarts[topRow] };
  }

  /** Each card's top against the view's at `scrollTop`. */
  function tops(vm, scrollTop) {
    const pos = new Map();
    vm.allGridImages.forEach((img, i) => {
      pos.set(String(img.id), vm.gridItemTopOffset(i) - scrollTop);
    });
    return pos;
  }

  /** How many cards that were on screen moved by half a row or more. */
  function rowJumps(vm, before, after) {
    const half = Math.min(...vm.justifiedLayout.rowHeights) / 2;
    let jumps = 0;
    for (const [id, y] of before) {
      if (y + half * 2 <= 0 || y >= 800) continue;
      if (!after.has(id) || Math.abs(after.get(id) - y) >= half) jumps += 1;
    }
    return jumps;
  }

  it("puts the topmost card on screen back under its pixel", async () => {
    const { vm, scroll, firstSlot } = await justifiedGrid();
    const startTop = scroll.top;
    const before = tops(vm, startTop);
    const anchorId = String(vm.allGridImages[firstSlot].id);

    vm.removeImagesById([3, 11, 40]);
    await settle();

    expect(scroll.writes).toHaveLength(1);
    expect(tops(vm, scroll.top).get(anchorId)).toBe(before.get(anchorId));
    expect(rowJumps(vm, before, tops(vm, scroll.top))).toBeLessThanOrEqual(
      rowJumps(vm, before, tops(vm, startTop)),
    );
  });

  it("does not go back to rows scrolled past when every card on screen leaves", async () => {
    const { vm, scroll, firstSlot } = await justifiedGrid();
    const lastOnScreen = vm.allGridImages.findIndex(
      (_img, i) => vm.gridItemTopOffset(i) >= scroll.top + 800,
    );
    const ids = [3, 11, 40];
    for (let i = firstSlot; i < lastOnScreen; i += 1) ids.push(i + 1);

    vm.removeImagesById(ids);
    await settle();

    expect(scroll.writes).toEqual([]);
  });
});

describe("a removal keeps the thumbnails it can", () => {
  /** Every range the grid counts as loaded holds rows with their thumbnail. */
  function expectLoadedRangesHoldThumbnails(vm) {
    for (const [start, end] of vm.loadedRanges) {
      for (let i = start; i < Math.min(end, vm.allGridImages.length); i += 1) {
        const img = vm.allGridImages[i];
        if (img?.id == null) continue;
        expect(img.thumbnail, `row ${i} is counted loaded`).toBeTruthy();
      }
    }
  }

  it("does not count rows that moved up as loaded when they were not", async () => {
    const { vm, cols } = await scrolledGrid();
    // Rows from VIEW_ROW + 20 on never had their thumbnails fetched.
    const fetchedUpTo = (VIEW_ROW + 20) * cols;
    vm.allGridImages = vm.allGridImages.map((img, i) =>
      i < fetchedUpTo ? img : { ...img, thumbnail: null },
    );
    vm.lastFetchedGridImages = vm.allGridImages.slice();
    vm.loadedRanges = [[0, fetchedUpTo]];

    vm.removeImagesById(idsOfRows(cols, 5, 3));
    await settle();

    expectLoadedRangesHoldThumbnails(vm);
  });

  it("lands a thumbnail answer that was in flight when the rows moved", async () => {
    const { vm, cols } = await scrolledGrid();
    const start = (VIEW_ROW + 10) * cols;
    const end = start + cols;
    // No row from `start` on has its thumbnail yet, so the rows that move up
    // into the asked slots have none either: the answer must not count those
    // slots as loaded.
    vm.allGridImages = vm.allGridImages.map((img, i) =>
      i >= start ? { ...img, thumbnail: null, imported_at: null } : img,
    );
    vm.lastFetchedGridImages = vm.allGridImages.slice();
    vm.loadedRanges = [[0, start]];
    const askedIds = vm.allGridImages.slice(start, end).map((img) => img.id);
    let answer;
    apiPost.mockImplementation((url) =>
      String(url).includes("/pictures/thumbnails")
        ? new Promise((resolve) => {
            answer = resolve;
          })
        : Promise.resolve({ data: {} }),
    );
    const inFlight = vm.fetchThumbnailsBatch(start, end);
    await vi.waitFor(() => expect(answer).toBeTypeOf("function"));

    vm.removeImagesById(idsOfRows(cols, 5, 3));
    answer({
      data: Object.fromEntries(
        askedIds.map((id) => [
          String(id),
          { thumbnail: `/pictures/thumbnails/${id}.webp?v=test-2` },
        ]),
      ),
    });
    await inFlight;
    await settle();

    for (const id of askedIds) {
      const img = vm.allGridImages.find((row) => row.id === id);
      expect(img.thumbnail).toBe(
        `/api/v1/pictures/thumbnails/${id}.webp?v=test-2`,
      );
    }
    expectLoadedRangesHoldThumbnails(vm);
  });
});

describe("a tag edit made from this tab re-reads a tag-filtered search", () => {
  // Under a tag filter a tag edit can change which pictures the search
  // lists, so this tab's own edit re-reads it (other tabs' raise the pill).
  // The first edit of a session counts as much as any later one, from rows
  // down the view as from the top.
  async function taggedSearchRowsDown() {
    useSearchStore().searchQuery = "cat";
    useFilterStore().tagFilter = ["bc-big"];
    const { vm, cols, scroll } = await scrolledGrid();
    const rows = Array.from({ length: ROWS * cols }, (_, i) => ({
      id: i + 1,
      format: "png",
    }));
    const searches = [];
    apiGet.mockImplementation((url) => {
      if (String(url).includes("/pictures/search")) {
        searches.push(String(url));
        return Promise.resolve({ data: rows });
      }
      return Promise.resolve({ data: { pictures: [], count: 0, total: 0 } });
    });
    scroll.top = vm.squareRowTop(25) + 7;
    vm.recalculateVisibleRange();
    await settle();
    return { vm, searches };
  }

  it("re-reads the search once on the first edit, 25 rows down", async () => {
    const { vm, searches } = await taggedSearchRowsDown();

    useWsStore().wsTagUpdate = { key: 1, pictureIds: [5], external: false };

    await vi.waitFor(() => expect(searches).toHaveLength(1), {
      timeout: 3000,
    });
    await vi.waitFor(() => expect(vm.imagesLoading).toBe(false));
    // Past another 1s debounce: the edit asks for one read, not more.
    await new Promise((resolve) => setTimeout(resolve, 1300));
    expect(searches).toHaveLength(1);
    const params = new URL(searches[0], "http://grid.test").searchParams;
    expect(params.get("query")).toBe("cat");
    expect(params.getAll("tag")).toEqual(["bc-big"]);
  });

  it("leaves an edit from elsewhere to the pill", async () => {
    const { searches } = await taggedSearchRowsDown();

    useWsStore().wsTagUpdate = { key: 1, pictureIds: [5], external: true };

    await vi.waitFor(() =>
      expect(wrapper.emitted("flag-sort-changed") ?? []).toContainEqual([[5]]),
    );
    // Past the re-read's 1s debounce.
    await new Promise((resolve) => setTimeout(resolve, 1300));
    expect(searches).toHaveLength(0);
  });
});
