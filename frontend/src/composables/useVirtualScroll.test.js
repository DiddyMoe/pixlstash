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

// The browser lays square rows out at their real pitch (a fractional column,
// the card chrome and the row-gap), so the spacers have to use that pitch, or
// every render-window move shifts the rendered rows by the difference.
describe("useVirtualScroll square row pitch", () => {
  // Not a multiple of the column count, so the columns are fractional.
  const GRID_W = 1003;
  const N = 8000;

  // Chrome's square grid: `cols` equal columns with a 4px gap, and a card of
  // column width plus 26px of chrome (padding, margin and the info row).
  function squareGeometry(cols) {
    const gap = 4;
    const colW = (GRID_W - gap * (cols - 1)) / cols;
    const track = colW + 26;
    return { cols, gap, track, pitch: track + gap };
  }

  function harness() {
    const store = useGridStore();
    store.sizeLevel = 3;
    store.thumbnailMode = "square";
    store.thumbnailSize = 256;
    store.compactMode = false;
    const geom = squareGeometry(store.columns);
    const card = { getBoundingClientRect: () => ({ height: geom.track }) };
    const grid = {
      clientWidth: GRID_W,
      getBoundingClientRect: () => ({ width: GRID_W }),
      querySelector: (selector) =>
        selector === ":scope > .image-card" ? card : null,
      computedStyle: () => ({ display: "grid", rowGap: `${geom.gap}px` }),
    };
    vi.stubGlobal("getComputedStyle", (el) => el.computedStyle());
    const vs = useVirtualScroll(
      ref({ scrollTop: 0, clientHeight: VIEWPORT_H, clientWidth: GRID_W }),
      ref(grid),
      reactive({}),
      computed(() => N),
      {
        onVisibleRangeChange: vi.fn(),
        afterRowHeightUpdate: vi.fn(),
        getAspectRatios: () => [],
      },
    );
    vs.initialRender.value = false;
    vs.updateRowHeightFromGrid();
    return { vs, geom };
  }

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  // Lays out [top spacer, rendered rows, bottom spacer] as a CSS grid does:
  // one row-gap between consecutive tracks, a zero-height spacer not rendered.
  function layOut(vs, { cols, gap, track }) {
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

  it("keeps every square row at one offset whatever the render window", () => {
    const { vs, geom } = harness();
    const { cols, gap, pitch } = geom;
    const total = 1 + gap + Math.ceil(N / cols) * pitch - gap;
    for (const row of [0, 1, 380, 390, Math.ceil(N / cols) - 2]) {
      vs.visibleStart.value = row * cols;
      vs.visibleEnd.value = (row + 4) * cols;
      const laid = layOut(vs, geom);
      // Row r sits at 1 + gap + r * pitch wherever the window starts; Chrome
      // positions boxes in 1/64 px units.
      const rowTop = 1 + gap + laid.firstRow * pitch;
      expect(Math.abs(laid.firstRowTop - rowTop)).toBeLessThan(1 / 64);
      expect(Math.abs(laid.total - total)).toBeLessThan(1 / 64);
    }
  });
});
