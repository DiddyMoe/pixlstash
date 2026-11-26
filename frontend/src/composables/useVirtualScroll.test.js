import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { ref, reactive, computed, nextTick } from "vue";
import { setActivePinia, createPinia } from "pinia";
import { useGridStore } from "../stores/useGridStore.js";
import { useVirtualScroll } from "./useVirtualScroll.js";
import { JUSTIFIED_ROW_GAP } from "./useJustifiedLayout.js";

// The composable reads the grid geometry (columns, size level, thumbnail mode
// and size, compact mode) from the grid store.
beforeEach(() => {
  setActivePinia(createPinia());
});

// Justified mode packs its rows from the aspect ratios of the images that have
// actually arrived, so `justifiedLayout` is null before the first batch and
// repacks every time the ratio list changes. `visibleStart`/`visibleEnd` are
// plain refs seeded by the fetch from a UNIFORM square-grid estimate, so once
// the real packed model exists the range has to be recomputed against it;
// otherwise the render window describes a geometry that never existed and the
// grid paints its cards outside the viewport until a scroll recomputes it.

const VIEWPORT_H = 600;
const CONTAINER_W = 1000;

function makeHarness() {
  const scrollWrapper = ref({
    scrollTop: 0,
    clientHeight: VIEWPORT_H,
    clientWidth: CONTAINER_W,
  });
  const gridContainer = ref({
    clientWidth: CONTAINER_W,
    getBoundingClientRect: () => ({ width: CONTAINER_W }),
  });
  // Grid geometry lives in the store. Size level 3 is the 6-column step, which
  // is the geometry these layout cases are written against.
  const gridStore = useGridStore();
  gridStore.sizeLevel = 3;
  gridStore.thumbnailSize = 256;
  gridStore.compactMode = false;
  gridStore.thumbnailMode = "justified";
  const props = reactive({});
  // The image list starts empty, exactly as it is when ImageGrid mounts.
  const aspectRatios = ref([]);
  const allGridImagesLength = computed(() => aspectRatios.value.length);
  const onVisibleRangeChange = vi.fn();

  const vs = useVirtualScroll(
    scrollWrapper,
    gridContainer,
    props,
    allGridImagesLength,
    {
      onVisibleRangeChange,
      getAspectRatios: () => aspectRatios.value,
    },
  );
  return { vs, aspectRatios, onVisibleRangeChange, scrollWrapper };
}

describe("useVirtualScroll justified layout", () => {
  it("recalculates the visible range when the layout packs for the first time", async () => {
    const { vs, aspectRatios, onVisibleRangeChange } = makeHarness();

    // Mount-time measure: width is known, but no images have arrived, so the
    // layout cannot pack and the range collapses to an empty window.
    vs.updateRowHeightFromGrid();
    expect(vs.visibleEnd.value).toBe(0);

    // The first batch lands: real aspect ratios, so the model packs. No scroll
    // happens, because the user has not touched the grid.
    aspectRatios.value = Array.from({ length: 120 }, () => 1.5);
    await nextTick();

    const autoEnd = vs.visibleEnd.value;

    // Ground truth: what the range is when computed against the packed model.
    onVisibleRangeChange.mockClear();
    vs.recalculateVisibleRange();
    const trueEnd = vs.visibleEnd.value;

    expect(trueEnd).toBeGreaterThan(0);
    expect(
      autoEnd,
      "visible range is stale after the justified layout packed: the grid " +
        "renders a window computed for a geometry that never existed, and only " +
        "a scroll repairs it",
    ).toBe(trueEnd);
  });

  it("recalculates the visible range when a repack moves every row boundary", async () => {
    const { vs, aspectRatios } = makeHarness();

    vs.updateRowHeightFromGrid();
    // Placeholder rows: ImageGrid seeds allGridImages with id-less entries whose
    // aspect ratio defaults to 1, so the first pack is all squares.
    aspectRatios.value = Array.from({ length: 120 }, () => 1);
    await nextTick();
    const squareEnd = vs.visibleEnd.value;
    expect(squareEnd).toBeGreaterThan(0);

    // Real ratios splice in: much wider images pack fewer rows into the same
    // viewport, so the visible item count must change with them.
    aspectRatios.value = Array.from({ length: 120 }, () => 3.2);
    await nextTick();
    const autoEnd = vs.visibleEnd.value;

    vs.recalculateVisibleRange();
    expect(autoEnd, "visible range is stale after a repack").toBe(
      vs.visibleEnd.value,
    );
    expect(autoEnd).not.toBe(squareEnd);
  });

  it("packs full rows inside the edge gutter on both sides", async () => {
    const { vs, aspectRatios } = makeHarness();
    vs.updateRowHeightFromGrid();
    aspectRatios.value = Array.from({ length: 40 }, () => 1.5);
    await nextTick();

    const layout = vs.justifiedLayout.value;
    let width = JUSTIFIED_ROW_GAP * (layout.rowStarts[1] - 1);
    for (let i = 0; i < layout.rowStarts[1]; i++) {
      width += layout.itemScaledWidths[i];
    }
    // The container's padding is one gap each side, which is where the
    // outermost tiles' selection ring is drawn; a row packed to the full box
    // would wrap its last card.
    expect(width).toBe(CONTAINER_W - 2 * JUSTIFIED_ROW_GAP);
  });

  it("scrolls a cursor row below the fold fully into view, ring included", async () => {
    const { vs, aspectRatios, scrollWrapper } = makeHarness();
    vs.updateRowHeightFromGrid();
    aspectRatios.value = Array.from({ length: 120 }, () => 1.5);
    await nextTick();

    const layout = vs.justifiedLayout.value;
    const row = 5;
    vs.scrollCursorIntoView(layout.rowStarts[row]);
    // The row paints one edge gutter below its offset and its ring reaches one
    // more past the card, so both must end inside the viewport.
    const ringBottom =
      layout.rowOffsets[row] + layout.rowHeights[row] + 2 * JUSTIFIED_ROW_GAP;
    expect(scrollWrapper.value.scrollTop + VIEWPORT_H).toBe(ringBottom);
  });
});

// The grid geometry now comes from the store rather than from props, and the
// arithmetic that turns it into a render window is silent when it goes wrong:
// an undefined column count makes every bound NaN, `renderEnd` collapses, and
// the grid paints nothing at all. That is not a hypothetical - it is what the
// Phase 3 store-direct migration did to ImageGrid, with the whole unit suite
// still green, because nothing here asserted that a non-empty list produces a
// non-empty window.
describe("useVirtualScroll render window (square mode)", () => {
  function squareHarness(itemCount) {
    const gridStore = useGridStore();
    gridStore.sizeLevel = 3; // the 6-column step
    gridStore.thumbnailMode = "square";
    gridStore.thumbnailSize = 256;
    gridStore.compactMode = false;

    const scrollWrapper = ref({
      scrollTop: 0,
      clientHeight: VIEWPORT_H,
      clientWidth: CONTAINER_W,
    });
    const gridContainer = ref({
      clientWidth: CONTAINER_W,
      getBoundingClientRect: () => ({ width: CONTAINER_W }),
      // No cards laid out: the row pitch stays at its estimate.
      querySelector: () => null,
    });
    const length = computed(() => itemCount);
    return useVirtualScroll(
      scrollWrapper,
      gridContainer,
      reactive({}),
      length,
      {
        onVisibleRangeChange: vi.fn(),
        afterRowHeightUpdate: vi.fn(),
        getAspectRatios: () => [],
      },
    );
  }

  it("renders a non-empty window for a non-empty grid", () => {
    const vs = squareHarness(120);
    vs.visibleStart.value = 0;
    vs.visibleEnd.value = 24;

    expect(Number.isFinite(vs.renderStart.value)).toBe(true);
    expect(Number.isFinite(vs.renderEnd.value)).toBe(true);
    expect(vs.renderEnd.value).toBeGreaterThan(0);
    expect(vs.renderEnd.value).toBeGreaterThan(vs.renderStart.value);
  });

  it("keeps the spacers finite, so the scroll height is never NaN", () => {
    const vs = squareHarness(120);
    vs.visibleStart.value = 24;
    vs.visibleEnd.value = 48;

    expect(Number.isFinite(vs.topSpacerHeight.value)).toBe(true);
    expect(Number.isFinite(vs.bottomSpacerHeight.value)).toBe(true);
  });

  it("still yields an empty window for an empty grid", () => {
    const vs = squareHarness(0);
    vs.visibleStart.value = 0;
    vs.visibleEnd.value = 0;

    expect(vs.renderEnd.value).toBe(0);
  });
});

// Square and compact rows used to be placed with a whole-pixel ESTIMATE of the
// row height, round((width - 4) / cols + 24), while the browser lays the rows
// out at their real pitch: a fractional fr column plus the card chrome plus
// the row-gap. Every render-window move then re-based the rendered block on
// the wrong pitch and the thumbnails jumped by a few pixels mid-scroll. These
// cases lay the composable's spacers out the way a CSS grid does and require
// every row to land at the same document offset whichever rows are rendered.
describe("useVirtualScroll square row pitch", () => {
  // Not a multiple of the column count, so the columns are fractional.
  const GRID_W = 1003;
  const N = 8000;

  // Chrome's geometry for the square grid: `cols` equal fractional columns,
  // a 4px row/column gap (0 in compact mode), and a card of column width plus
  // 26px of chrome (4 + 4 padding, 2px margin and the 24px info row).
  function squareGeometry(cols, compact) {
    const gap = compact ? 0 : 4;
    const colW = (GRID_W - gap * (cols - 1)) / cols;
    const track = compact ? colW : colW + 26;
    return { cols, gap, track, pitch: track + gap };
  }

  function harness({ n = N, compact = false, sizeLevel = 3 } = {}) {
    const store = useGridStore();
    store.sizeLevel = sizeLevel;
    store.thumbnailMode = "square";
    store.thumbnailSize = 256;
    store.compactMode = compact;
    // What the page has laid out right now. Tests swap it to model Vue
    // patching the DOM after a geometry change.
    const page = { geom: squareGeometry(store.columns, compact), cards: true };
    const card = {
      getBoundingClientRect: () => ({ height: page.geom.track }),
    };
    const grid = {
      clientWidth: GRID_W,
      getBoundingClientRect: () => ({ width: GRID_W }),
      querySelector: (selector) =>
        selector === ":scope > .image-card" && page.cards ? card : null,
      computedStyle: () => ({ display: "grid", rowGap: `${page.geom.gap}px` }),
    };
    vi.stubGlobal("getComputedStyle", (el) => el.computedStyle());
    const scrollWrapper = ref({
      scrollTop: 0,
      clientHeight: VIEWPORT_H,
      clientWidth: GRID_W,
    });
    const vs = useVirtualScroll(
      scrollWrapper,
      ref(grid),
      reactive({}),
      computed(() => n),
      {
        onVisibleRangeChange: vi.fn(),
        afterRowHeightUpdate: vi.fn(),
        getAspectRatios: () => [],
      },
    );
    vs.initialRender.value = false;
    vs.updateRowHeightFromGrid();
    return { vs, store, page, scrollWrapper, n };
  }

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  // Lays out [top spacer, rendered rows, bottom spacer] as a CSS grid does:
  // one row-gap between consecutive tracks, zero-height spacers not rendered.
  function layOut(vs, geom) {
    const { cols, gap, track } = geom;
    const top = vs.topSpacerHeight.value;
    const bottom = vs.bottomSpacerHeight.value;
    const renderedRows = Math.ceil(
      (vs.renderEnd.value - vs.renderStart.value) / cols,
    );
    const tracks = [
      ...(top > 0 ? [top] : []),
      ...Array(renderedRows).fill(track),
      ...(bottom > 0 ? [bottom] : []),
    ];
    return {
      firstRow: Math.floor(vs.renderStart.value / cols),
      firstRowTop: top > 0 ? top + gap : 0,
      total: tracks.reduce((a, b) => a + b, 0) + gap * (tracks.length - 1),
    };
  }

  function showRow(vs, cols, row) {
    vs.visibleStart.value = row * cols;
    vs.visibleEnd.value = (row + 4) * cols;
  }

  function expectRowsInPlace(vs, geom, n) {
    const laid = layOut(vs, geom);
    const rowTop = 1 + geom.gap + laid.firstRow * geom.pitch;
    const total =
      1 + geom.gap + Math.ceil(n / geom.cols) * geom.pitch - geom.gap;
    // Chrome positions boxes in 1/64 px units.
    expect(Math.abs(laid.firstRowTop - rowTop)).toBeLessThan(1 / 64);
    expect(Math.abs(laid.total - total)).toBeLessThan(1 / 64);
  }

  it("keeps every square row at one offset whatever the render window", () => {
    const { vs, page, n } = harness();
    for (const row of [0, 1, 380, 390, Math.ceil(n / page.geom.cols) - 2]) {
      showRow(vs, page.geom.cols, row);
      expectRowsInPlace(vs, page.geom, n);
    }
  });

  it("keeps every compact row at one offset whatever the render window", () => {
    const { vs, page, n } = harness({ compact: true });
    expect(page.geom.gap).toBe(0);
    for (const row of [0, 1, 380, 390]) {
      showRow(vs, page.geom.cols, row);
      expectRowsInPlace(vs, page.geom, n);
    }
  });

  it("measures the pitch from a grid with less than one full row", () => {
    const store = useGridStore();
    store.sizeLevel = 3;
    const n = Math.max(1, store.columns - 2);
    const { vs, page } = harness({ n });
    showRow(vs, page.geom.cols, 0);
    expect(vs.rowHeight.value).toBe(page.geom.pitch);
    expectRowsInPlace(vs, page.geom, n);
  });

  it("re-measures after a column change and keeps the top picture in place", async () => {
    const { vs, store, page, scrollWrapper, n } = harness();
    const before = page.geom;
    // Scrolled a quarter of the way into row 200.
    scrollWrapper.value.scrollTop =
      1 + before.gap + 200 * before.pitch + 0.25 * before.pitch;
    vs.recalculateVisibleRange();
    const topItem = 200 * before.cols;

    store.sizeLevel = 4;
    expect(store.columns).not.toBe(before.cols);
    // ImageGrid's columns watcher runs before the patch: the page still has
    // the old layout when it asks for a re-measure.
    vs.updateRowHeightFromGrid();
    vs.recalculateVisibleRange();
    // Vue patches the DOM to the new column count.
    const after = squareGeometry(store.columns, false);
    page.geom = after;
    await nextTick();
    await nextTick();
    await nextTick();

    expect(vs.rowHeight.value).toBe(after.pitch);
    expectRowsInPlace(vs, after, n);
    const into = scrollWrapper.value.scrollTop - 1 - after.gap;
    expect(Math.floor(into / after.pitch)).toBe(
      Math.floor(topItem / after.cols),
    );
    expect(into / after.pitch - Math.floor(into / after.pitch)).toBeCloseTo(
      0.25,
      6,
    );
    // The render window was recomputed for the restored position.
    expect(vs.visibleStart.value).toBe(
      Math.floor(topItem / after.cols) * after.cols,
    );
  });

  it("reports the rows the viewport really shows", () => {
    const { vs, page, scrollWrapper } = harness();
    const { cols, gap, pitch } = page.geom;
    // Half a pixel above the bottom of row 99's card: row 99 is visible.
    scrollWrapper.value.scrollTop = 1 + gap + 100 * pitch - gap - 0.5;
    vs.recalculateVisibleRange();
    expect(vs.visibleStart.value).toBe(99 * cols);
  });

  it("scrolls the cursor row exactly into view", () => {
    const { vs, page, scrollWrapper } = harness();
    const { cols, gap, pitch, track } = page.geom;
    vs.scrollCursorIntoView(50 * cols);
    expect(scrollWrapper.value.scrollTop).toBeCloseTo(
      1 + gap + 50 * pitch + track - VIEWPORT_H,
      6,
    );
    vs.scrollCursorIntoView(10 * cols);
    expect(scrollWrapper.value.scrollTop).toBeCloseTo(1 + gap + 10 * pitch, 6);
  });
});

// The render window used to follow the scroll through a 50 ms TRAILING
// debounce on one global `window` timer: while a fling kept firing scroll
// events it never moved, so the grid ran off its render buffer into blank
// spacer, and two grids scrolling at once cancelled each other's update.
describe("useVirtualScroll scroll tracking", () => {
  beforeEach(() => {
    vi.useFakeTimers({
      toFake: [
        "setTimeout",
        "clearTimeout",
        "requestAnimationFrame",
        "cancelAnimationFrame",
      ],
    });
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  function scroller() {
    const store = useGridStore();
    store.sizeLevel = 3;
    store.thumbnailMode = "square";
    store.thumbnailSize = 256;
    store.compactMode = false;
    const scrollWrapper = ref({
      scrollTop: 0,
      clientHeight: VIEWPORT_H,
      clientWidth: CONTAINER_W,
    });
    const onVisibleRangeChange = vi.fn();
    const vs = useVirtualScroll(
      scrollWrapper,
      ref({
        clientWidth: CONTAINER_W,
        getBoundingClientRect: () => ({ width: CONTAINER_W }),
        querySelector: () => null,
      }),
      reactive({}),
      computed(() => 100000),
      { onVisibleRangeChange, getAspectRatios: () => [] },
    );
    return { vs, scrollWrapper, onVisibleRangeChange };
  }

  it("moves the render window while the scroll is still going", () => {
    const { vs, scrollWrapper } = scroller();
    // A fling: a scroll event every frame, never a pause.
    for (let frame = 0; frame < 30; frame += 1) {
      scrollWrapper.value.scrollTop += 300;
      vs.onGridScroll();
      vi.advanceTimersByTime(16);
    }
    const scrolledRows = Math.floor(
      scrollWrapper.value.scrollTop / vs.rowHeight.value,
    );
    expect(vs.visibleStart.value).toBeGreaterThanOrEqual(
      (scrolledRows - 1) * useGridStore().columns,
    );
  });

  it("updates once per frame however many scroll events arrive", () => {
    const { vs, scrollWrapper, onVisibleRangeChange } = scroller();
    for (let i = 0; i < 5; i += 1) {
      scrollWrapper.value.scrollTop += 2000;
      vs.onGridScroll();
    }
    vi.advanceTimersByTime(16);
    expect(onVisibleRangeChange).toHaveBeenCalledTimes(1);
    // ...and from the last event's position.
    expect(vs.visibleStart.value).toBe(
      // Unmeasured, row 0 starts below the 1px top spacer.
      Math.floor((10000 - 1) / vs.rowHeight.value) * useGridStore().columns,
    );
  });

  it("tracks two grids' scrolls independently", () => {
    const a = scroller();
    const b = scroller();
    a.scrollWrapper.value.scrollTop = 5000;
    a.vs.onGridScroll();
    b.scrollWrapper.value.scrollTop = 9000;
    b.vs.onGridScroll();
    vi.advanceTimersByTime(100);
    expect(a.vs.visibleStart.value).toBeGreaterThan(0);
    expect(b.vs.visibleStart.value).toBeGreaterThan(a.vs.visibleStart.value);
  });
});
