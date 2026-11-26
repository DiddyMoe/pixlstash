import {
  ref,
  computed,
  watch,
  nextTick,
  getCurrentScope,
  onScopeDispose,
} from "vue";
import {
  packJustifiedRows,
  rowAtOffset,
  rowOfIndex,
  JUSTIFIED_ROW_GAP,
} from "./useJustifiedLayout";
import { rowHeightForSizeLevel } from "../utils/thumbnailSizes";
import { useGridStore } from "../stores/useGridStore";

// Constants shared with ImageGrid.vue. Cut 25% alongside the ladder in
// thumbnailSizes.js (owner call, 2026-09-03).
const MIN_THUMBNAIL_SIZE = 96;
const MAX_THUMBNAIL_SIZE = 288;
const THUMBNAIL_INFO_ROW_HEIGHT = 24;
const VIEW_WINDOW = 100;
// Square mode always renders the top spacer, at least this tall, so the first
// rendered row sits one spacer plus one row-gap below the top of the grid.
const SQUARE_TOP_SPACER_MIN = 1;

/**
 * Manages viewport geometry, row height, and scroll position for the virtual
 * scrolling image grid.
 *
 * Two layout modes, selected by `gridStore.thumbnailMode`:
 * - `'square'` (default): the original uniform grid - `cols` items per row,
 *   every row `rowHeight` px tall, `index → row` is `floor(index / cols)`.
 * - `'justified'`: Google-Photos-style rows packed by `useJustifiedLayout`.
 *   Row heights are near-uniform but items-per-row varies, so all
 *   index↔row↔pixel arithmetic goes through the packed layout
 *   (`rowStarts` / `rowOffsets`) instead of column arithmetic. In both modes
 *   `visibleStart`/`visibleEnd` are ITEM indices - the contract the parent's
 *   thumbnail fetching relies on.
 *
 * @param {import('vue').Ref} scrollWrapper - Ref to the scrollable container element.
 * @param {import('vue').Ref} gridContainer - Ref to the inner grid element.
 * @param {object} props - Reactive props: columns, thumbnailSize, compactMode,
 *   thumbnailMode.
 * @param {import('vue').ComputedRef<number>} allGridImagesLength - Total count of grid images.
 * @param {object} [callbacks]
 * @param {Function} [callbacks.onVisibleRangeChange] - Called when the visible
 *   row range changes (triggers thumbnail fetch in the parent).
 * @param {Function} [callbacks.afterRowHeightUpdate] - Called after rowHeight is
 *   recalculated (refreshes thumbnail info text in the parent).
 * @param {Function} [callbacks.getAspectRatios] - Returns the aspect-ratio
 *   array (one entry per grid image, in grid order) used to pack justified
 *   rows. Required for `thumbnailMode === 'justified'`; unused otherwise.
 */
export function useVirtualScroll(
  scrollWrapper,
  gridContainer,
  props,
  allGridImagesLength,
  { onVisibleRangeChange, afterRowHeightUpdate, getAspectRatios } = {},
) {
  const gridStore = useGridStore();
  // ── Render buffer ──────────────────────────────────────────────────────────
  // During the initial render the buffer is 0 so the first paint is fast;
  // once the grid has rendered once the buffer expands to a full view-window
  // worth of items on each side.
  const divisibleViewWindow = computed(() => {
    const cols = gridStore.columns;
    return Math.ceil(VIEW_WINDOW / cols) * cols;
  });

  const initialRender = ref(true);

  const renderBuffer = computed(() =>
    initialRender.value ? 0 : divisibleViewWindow.value,
  );

  // ── Visible row tracking ───────────────────────────────────────────────────
  const visibleStart = ref(0);
  const visibleEnd = ref(0);

  // ── Justified layout ──────────────────────────────────────────────────────
  const isJustifiedMode = computed(
    () =>
      gridStore.thumbnailMode === "justified" &&
      typeof getAspectRatios === "function",
  );

  // Measured width of the grid container, updated by updateRowHeightFromGrid
  // (mount + ResizeObserver). Floored to a whole px so packed rows can never
  // exceed the real (possibly fractional) box and wrap unexpectedly.
  const justifiedContainerWidth = ref(0);

  // The packed row model. Pure arithmetic over the aspect-ratio list - no DOM
  // reads - and deterministic, so recomputing on resize/fetch cannot thrash
  // thumbnails. Null while unmeasured/empty (the parent falls back to the
  // uniform path until the first measure lands).
  const justifiedLayout = computed(() => {
    if (!isJustifiedMode.value) return null;
    const containerWidth = justifiedContainerWidth.value;
    if (!(containerWidth > 0)) return null;
    const aspectRatios = getAspectRatios();
    if (!Array.isArray(aspectRatios) || aspectRatios.length === 0) return null;
    // Target row height comes from the shared size level so the size slider
    // resizes justified rows exactly as it resizes square columns. The min/max
    // bounds flex around the target so leftover rows can still stretch/shrink
    // to fill the container width without jumping to a fixed global size.
    const targetRowHeight = rowHeightForSizeLevel(gridStore.sizeLevel);
    return packJustifiedRows({
      aspectRatios,
      containerWidth,
      targetRowHeight,
      gap: JUSTIFIED_ROW_GAP,
      rowExtraHeight: gridStore.compactMode ? 0 : THUMBNAIL_INFO_ROW_HEIGHT,
      minRowHeight: Math.round(targetRowHeight * 0.7),
      maxRowHeight: Math.round(targetRowHeight * 1.4),
    });
  });

  function hasJustifiedLayout() {
    return (
      isJustifiedMode.value &&
      justifiedLayout.value !== null &&
      justifiedLayout.value.rowHeights.length > 0
    );
  }

  // ── Row height ────────────────────────────────────────────────────────────
  // Square mode: rowHeight is the row PITCH, the distance from one row's top
  // to the next row's top (row track + CSS row-gap), measured from the
  // rendered grid by measureSquareRowPitch. The spacers, the visible range and
  // the cursor scroll all place row r at squareRowTop(r), so the rendered block
  // sits exactly where the spacers say it does and a render-window move never
  // shifts it. Until the first measurement (and in justified mode, where rows
  // come from the packed layout instead) it holds a whole-pixel estimate.
  const rowHeight = ref(
    Math.round(
      Math.min(
        MAX_THUMBNAIL_SIZE,
        Math.max(
          MIN_THUMBNAIL_SIZE,
          gridStore.thumbnailSize || MIN_THUMBNAIL_SIZE,
        ),
      ) + (gridStore.compactMode ? 0 : THUMBNAIL_INFO_ROW_HEIGHT),
    ),
  );
  // Measured CSS row-gap of the square grid, in px (0 until measured).
  const squareRowGap = ref(0);
  // True once rowHeight holds a measured square pitch rather than an estimate.
  let squarePitchMeasured = false;

  function squareRowTop(row) {
    return SQUARE_TOP_SPACER_MIN + squareRowGap.value + row * rowHeight.value;
  }

  // Reads the square-mode row pitch from the rendered grid. Cards have no
  // margin and stretch to their row track, and every row has the same track
  // height, so one card's height plus the grid's row-gap is the exact pitch,
  // including the engine's sub-pixel rounding of the fr columns (the
  // TanStack Virtual approach: the rendered element is the source of truth,
  // the estimate only seeds it). One card is enough, so a grid with less than
  // a full row rendered measures the same way. Returns null when there is
  // nothing laid out to measure (no card, a grid that is not displayed, or a
  // container that is not a CSS grid).
  function measureSquareRowPitch() {
    const grid = gridContainer.value;
    if (!grid) return null;
    const card = grid.querySelector(":scope > .image-card");
    if (!card) return null;
    const trackHeight = card.getBoundingClientRect().height;
    if (!(trackHeight > 0)) return null;
    const gridStyle = getComputedStyle(grid);
    // The pitch formula is a CSS Grid property. Right after a switch out of
    // justified mode the DOM can still be the flex layout; the post-flush
    // watcher measures again once it is a grid.
    if (gridStyle.display !== "grid") return null;
    const rowGapText = gridStyle.rowGap;
    const rowGap = parseFloat(rowGapText);
    if (!Number.isFinite(rowGap)) {
      console.error(
        "useVirtualScroll: the square grid's row-gap is not a length; keeping the previous row pitch",
        { rowGap: rowGapText, trackHeight, rowHeight: rowHeight.value },
      );
      return null;
    }
    return { pitch: trackHeight + rowGap, gap: rowGap };
  }

  // The viewport's position in square-row terms: the first item of the row at
  // the top edge, and how far into that row the edge sits (fraction of pitch).
  // Taken under the geometry that is about to change, restored under the new
  // one, so a resize, column change or compact toggle keeps the same pictures
  // at the top. The browser's scroll anchoring is off for the grid (see
  // ImageGrid.css), so nothing else does this.
  function captureSquareAnchor(cols) {
    const el = scrollWrapper.value;
    if (!el || !(el.scrollTop > 0)) return null;
    const into = Math.max(0, el.scrollTop - squareRowTop(0));
    const row = Math.floor(into / rowHeight.value);
    return {
      idx: row * cols,
      frac: (into - row * rowHeight.value) / rowHeight.value,
    };
  }

  function restoreSquareAnchor(anchor) {
    const el = scrollWrapper.value;
    if (!el || !anchor) return;
    const cols = Math.max(1, gridStore.columns || 1);
    const row = Math.floor(anchor.idx / cols);
    el.scrollTop = squareRowTop(row) + anchor.frac * rowHeight.value;
  }

  // Measures the square pitch and, when it (or the column count, given as
  // prevCols) changed, applies it. The scroll position is restored after the
  // DOM has the new spacers, so the browser cannot clamp it against the old
  // scroll height; the visible range is then recomputed. Returns whether a
  // measurement was available.
  function syncSquarePitch(prevCols) {
    const measured = measureSquareRowPitch();
    if (!measured) return false;
    const cols = Math.max(1, gridStore.columns || 1);
    const pitchChanged =
      !squarePitchMeasured ||
      measured.pitch !== rowHeight.value ||
      measured.gap !== squareRowGap.value;
    if (!pitchChanged && prevCols === cols) return true;
    const anchor = captureSquareAnchor(Math.max(1, prevCols || cols));
    rowHeight.value = measured.pitch;
    squareRowGap.value = measured.gap;
    squarePitchMeasured = true;
    if (!anchor) {
      recalculateVisibleRange();
      return true;
    }
    nextTick(() => {
      restoreSquareAnchor(anchor);
      recalculateVisibleRange();
    });
    return true;
  }

  function getGridColumnWidth() {
    const cols = Math.max(1, gridStore.columns || 1);
    const gridWidth =
      gridContainer.value?.clientWidth ?? scrollWrapper.value?.clientWidth ?? 0;
    if (!gridWidth) {
      return Math.min(
        MAX_THUMBNAIL_SIZE,
        Math.max(
          MIN_THUMBNAIL_SIZE,
          gridStore.thumbnailSize || MIN_THUMBNAIL_SIZE,
        ),
      );
    }
    const availableWidth = Math.max(0, gridWidth - 4);
    const rawWidth = availableWidth / cols;
    return Math.min(
      MAX_THUMBNAIL_SIZE,
      Math.max(1, rawWidth || MIN_THUMBNAIL_SIZE),
    );
  }

  function estimateRowHeight() {
    const columnWidth = getGridColumnWidth();
    const infoHeight = gridStore.compactMode ? 0 : THUMBNAIL_INFO_ROW_HEIGHT;
    return Math.round(columnWidth + infoHeight);
  }

  // Called on mount, from the grid's ResizeObserver and from ImageGrid's
  // layout watchers. Those watchers run before Vue patches the DOM, so in
  // square mode the measurement here can still see the old layout; the
  // post-flush watcher below measures again once the new layout is in place.
  function updateRowHeightFromGrid() {
    if (isJustifiedMode.value) {
      rowHeight.value = estimateRowHeight();
      squarePitchMeasured = false;
      remeasureJustifiedWidth();
    } else if (!syncSquarePitch(gridStore.columns) && !squarePitchMeasured) {
      // Nothing rendered to measure yet: seed with the estimate. A measured
      // pitch is kept instead, since the grid being briefly empty or hidden
      // says nothing about the row geometry.
      rowHeight.value = estimateRowHeight();
    }
    afterRowHeightUpdate?.();
  }

  // Re-measures the container and, when the width actually changed, re-packs
  // the row model.
  //
  // ANTI-JUMP: a repack moves every row boundary, so a naive repack would
  // leave scrollTop pointing at arbitrary different images and the viewport
  // would visibly jump on every window/sidebar resize. We therefore capture
  // the first item of the current top-visible row under the OLD layout before
  // changing the width, then restore scrollTop to that item's row offset in
  // the NEW layout - the user keeps looking at the same picture.
  function remeasureJustifiedWidth() {
    const rect = gridContainer.value?.getBoundingClientRect?.();
    // The container carries a JUSTIFIED_ROW_GAP edge gutter on each side
    // (ImageGrid's gridContainerStyle); rows pack inside it.
    const measured = Math.floor(
      (rect?.width ?? scrollWrapper.value?.clientWidth ?? 0) -
        2 * JUSTIFIED_ROW_GAP,
    );
    if (!(measured > 0) || measured === justifiedContainerWidth.value) return;

    const el = scrollWrapper.value;
    const prevLayout = hasJustifiedLayout() ? justifiedLayout.value : null;
    let anchorIdx = null;
    if (el && prevLayout && el.scrollTop > 0) {
      const topRow = rowAtOffset(prevLayout.rowOffsets, el.scrollTop);
      anchorIdx = prevLayout.rowStarts[topRow];
    }

    justifiedContainerWidth.value = measured;

    // Computeds are synchronous: reading justifiedLayout here already yields
    // the re-packed model for the new width.
    const newLayout = hasJustifiedLayout() ? justifiedLayout.value : null;
    if (el && anchorIdx !== null && newLayout) {
      const clampedIdx = Math.min(
        anchorIdx,
        newLayout.itemScaledWidths.length - 1,
      );
      const newRow = rowOfIndex(newLayout.rowStarts, Math.max(0, clampedIdx));
      el.scrollTop = newLayout.rowOffsets[newRow];
    }
    recalculateVisibleRange();
  }

  // ── Visible-range calculation (shared by scroll + immediate recalc) ───────
  // Returns ITEM indices in both modes: [start, end) is exactly the set of
  // items whose row intersects the viewport. This is the contract the parent's
  // updateVisibleThumbnails depends on - if these drifted from what is painted
  // the grid would show blank tiles.
  // `viewport` ({ scrollTop, clientHeight }) defaults to the live scroller.
  function computeVisibleRange(viewport = scrollWrapper.value) {
    const el = viewport;
    if (!el) return null;
    if (isJustifiedMode.value) {
      if (!hasJustifiedLayout()) return { start: 0, end: 0 };
      const layout = justifiedLayout.value;
      const firstRow = rowAtOffset(layout.rowOffsets, el.scrollTop);
      const lastRow = rowAtOffset(
        layout.rowOffsets,
        el.scrollTop + el.clientHeight - 1,
      );
      return {
        start: layout.rowStarts[firstRow],
        end: layout.rowStarts[lastRow + 1],
      };
    }
    const pitch = rowHeight.value;
    const intoGrid = el.scrollTop - squareRowTop(0);
    const cols = gridStore.columns;
    const firstVisibleRow = Math.max(0, intoGrid / pitch);
    const lastVisibleRow = Math.max(
      0,
      (intoGrid + el.clientHeight - 1) / pitch,
    );
    return {
      start: Math.floor(firstVisibleRow) * cols,
      end: Math.ceil(lastVisibleRow) * cols,
    };
  }

  // ── Render window ─────────────────────────────────────────────────────────
  // In justified mode the buffered endpoints are snapped OUTWARD to row
  // boundaries: renderStart down to the start of its row, renderEnd up to the
  // end of its row. Two invariants follow:
  //   1. every rendered row is complete - flex-wrap line breaks can only match
  //      the packed model when no row is partially rendered, and
  //   2. [visibleStart, visibleEnd) ⊆ [renderStart, renderEnd) even when the
  //      visible range is a uniform-math estimate (as right after a fetch).
  // The parent's updateVisibleThumbnails fetches exactly [renderStart,
  // renderEnd), so the fetch window always equals what is painted
  // (anti-blank-tile).
  const renderStart = computed(() => {
    const raw = Math.max(0, visibleStart.value - renderBuffer.value);
    if (hasJustifiedLayout()) {
      const layout = justifiedLayout.value;
      const n = layout.itemScaledWidths.length;
      if (raw >= n) return raw;
      // Snap down to the start of the row containing raw.
      return layout.rowStarts[rowOfIndex(layout.rowStarts, raw)];
    }
    return raw;
  });

  const renderEnd = computed(() => {
    const raw = Math.min(
      allGridImagesLength.value,
      visibleEnd.value + renderBuffer.value,
    );
    if (hasJustifiedLayout() && raw > 0) {
      const layout = justifiedLayout.value;
      const n = layout.itemScaledWidths.length;
      if (raw >= n) return raw;
      // Snap up to the end of the row containing item raw - 1 (an exclusive
      // endpoint already on a boundary is left unchanged).
      return Math.min(
        n,
        layout.rowStarts[rowOfIndex(layout.rowStarts, raw - 1) + 1],
      );
    }
    return raw;
  });

  // ── Spacers ───────────────────────────────────────────────────────────────
  // Both modes use the same construction: a spacer is the exact offset of the
  // first rendered row (or the exact remaining height below the last one),
  // minus the one row-gap the layout itself inserts next to the spacer.
  // Justified mode: exact pixel sums from rowOffsets (never rows * rowHeight).
  // The grid renders as a flex-wrap container with `gap` between lines, so a
  // rendered spacer line contributes one extra inter-line gap that the spacer
  // height must not double-count.
  // Square mode: the measured pitch and row-gap. The top spacer is always
  // rendered (SQUARE_TOP_SPACER_MIN px when no rows are above), so row r is at
  // squareRowTop(r) whichever rows are rendered.
  const topSpacerHeight = computed(() => {
    if (isJustifiedMode.value) {
      if (!hasJustifiedLayout()) return 0;
      const layout = justifiedLayout.value;
      const start = renderStart.value;
      if (start <= 0) return 0;
      const row = rowOfIndex(
        layout.rowStarts,
        Math.min(start, layout.itemScaledWidths.length - 1),
      );
      // The flex row-gap between the spacer line and the first rendered row
      // supplies JUSTIFIED_ROW_GAP px of the offset.
      return Math.max(0, layout.rowOffsets[row] - JUSTIFIED_ROW_GAP);
    }
    const cols = gridStore.columns;
    const rowsAbove = Math.floor(renderStart.value / cols);
    return SQUARE_TOP_SPACER_MIN + rowsAbove * rowHeight.value;
  });

  const bottomSpacerHeight = computed(() => {
    if (isJustifiedMode.value) {
      if (!hasJustifiedLayout()) return 0;
      const layout = justifiedLayout.value;
      const n = layout.itemScaledWidths.length;
      const end = renderEnd.value;
      if (end >= n) return 0;
      if (end <= 0) return layout.totalHeight;
      const lastRow = rowOfIndex(layout.rowStarts, end - 1);
      // rowOffsets[lastRow + 1] already includes the inter-row gap that the
      // flex layout inserts between the last rendered row and the spacer.
      const nextOffset = layout.rowOffsets[lastRow + 1] ?? layout.totalHeight;
      return Math.max(0, layout.totalHeight - nextOffset);
    }
    const cols = gridStore.columns;
    const lastRenderedRow = Math.floor((renderEnd.value - 1) / cols) + 1;
    const totalRows = Math.ceil(allGridImagesLength.value / cols);
    const rowsBelow = totalRows - lastRenderedRow;
    // The row-gap before the spacer supplies squareRowGap px of it.
    return rowsBelow > 0
      ? Math.max(0, rowsBelow * rowHeight.value - squareRowGap.value)
      : 0;
  });

  // ── Scroll handler ────────────────────────────────────────────────────────
  // At most one range update per animation frame, for as long as the grid
  // scrolls, so a fling keeps rendering the rows it passes. A trailing
  // debounce only updated once the scrolling stopped, so a fast fling ran off
  // the render buffer into blank spacer. The viewport is read in the scroll
  // event, while layout is still clean, and applied in the frame callback, so
  // the frame never forces a layout of a DOM that Vue has just patched (the
  // vue-virtual-scroller scheme). The range only changes at row boundaries,
  // so most frames end at the equality check below. The state is
  // per-instance: each grid tracks its own scroll.
  let scrollFrame = 0;
  let pendingViewport = null;
  function onGridScroll() {
    const el = scrollWrapper.value;
    if (!el) return;
    pendingViewport = {
      scrollTop: el.scrollTop,
      clientHeight: el.clientHeight,
    };
    if (scrollFrame) return;
    scrollFrame = requestAnimationFrame(() => {
      scrollFrame = 0;
      const range = computeVisibleRange(pendingViewport);
      pendingViewport = null;
      if (!range) return;
      if (
        visibleStart.value !== range.start ||
        visibleEnd.value !== range.end
      ) {
        visibleStart.value = range.start;
        visibleEnd.value = range.end;
        onVisibleRangeChange?.();
      }
    });
  }
  // The composable also runs outside a component (unit tests), where there is
  // no scope to tie the pending frame to.
  if (getCurrentScope()) {
    onScopeDispose(() => {
      if (scrollFrame) cancelAnimationFrame(scrollFrame);
    });
  }

  // ── Immediate visible-range recalculation ──────────────────────────────────
  // Used when the layout changes (column count, compact mode, justified
  // repack) without a scroll event - onGridScroll only runs on scroll, and
  // only in the next frame.
  function recalculateVisibleRange() {
    const range = computeVisibleRange();
    if (!range) return;
    visibleStart.value = range.start;
    visibleEnd.value = range.end;
    onVisibleRangeChange?.();
  }

  // ── Keep the range in step with the packed model ──────────────────────────
  // justifiedLayout packs from the aspect ratios of the images that have
  // ARRIVED, so it is null until the first batch lands and then repacks on every
  // change to that list (placeholder 1:1 ratios first, real ratios as batches
  // splice in). Each repack moves every row boundary, while visibleStart/End are
  // plain refs seeded by the fetch from a UNIFORM square-grid estimate
  // (ceil(viewH / cellH) * columns) that does not describe justified geometry.
  // Nothing else recalculates them against the real model: computeVisibleRange
  // returns an empty window while the layout is null, and the only other hook,
  // remeasureJustifiedWidth, early-returns once the width stops changing and so
  // never reaches its own recalculateVisibleRange(). Without this watcher the
  // render window keeps describing a geometry that never existed, so the grid
  // paints its cards outside the viewport (commonly a zero-item window, i.e. a
  // blank grid) until the user scrolls and computeVisibleRange finally reads the
  // packed model.
  watch(justifiedLayout, (layout) => {
    if (!layout || !layout.rowHeights.length) return;
    recalculateVisibleRange();
  });

  // ── Keep the square pitch in step with the rendered grid ──────────────────
  // Runs after Vue has patched the DOM, so it sees the layout the column
  // count, compact mode, thumbnail mode or render window just produced
  // (ImageGrid's own watchers measure before the patch). It is one
  // getBoundingClientRect and one getComputedStyle per render-window change,
  // read after all of Vue's writes for that change; the pitch itself does not
  // depend on the spacers, so applying it converges in one step.
  watch(
    [
      renderStart,
      renderEnd,
      () => gridStore.columns,
      () => gridStore.compactMode,
      () => gridStore.thumbnailMode,
    ],
    (_values, [, , prevCols]) => {
      if (isJustifiedMode.value) return;
      syncSquarePitch(prevCols);
    },
    { flush: "post" },
  );

  // ── Cursor scroll-into-view ───────────────────────────────────────────────
  function scrollCursorIntoView(idx) {
    if (!scrollWrapper.value) return;
    let itemTop;
    let itemBottom;
    if (hasJustifiedLayout()) {
      const layout = justifiedLayout.value;
      const clampedIdx = Math.max(
        0,
        Math.min(idx, layout.itemScaledWidths.length - 1),
      );
      const row = rowOfIndex(layout.rowStarts, clampedIdx);
      // Rows paint one edge gutter below their offset, and a selected card's
      // ring takes one gutter each side: the offset is the ring's top, and its
      // bottom is two gutters past the card's height.
      itemTop = layout.rowOffsets[row];
      itemBottom = itemTop + layout.rowHeights[row] + 2 * JUSTIFIED_ROW_GAP;
    } else {
      const cols = Math.max(1, gridStore.columns || 1);
      const row = Math.floor(idx / cols);
      itemTop = squareRowTop(row);
      itemBottom = itemTop + rowHeight.value - squareRowGap.value;
    }
    const scrollTop = scrollWrapper.value.scrollTop;
    const clientHeight = scrollWrapper.value.clientHeight;
    if (itemTop < scrollTop) {
      scrollWrapper.value.scrollTop = itemTop;
    } else if (itemBottom > scrollTop + clientHeight) {
      scrollWrapper.value.scrollTop = itemBottom - clientHeight;
    }
  }

  return {
    initialRender,
    divisibleViewWindow,
    renderBuffer,
    visibleStart,
    visibleEnd,
    rowHeight,
    renderStart,
    renderEnd,
    topSpacerHeight,
    bottomSpacerHeight,
    getGridColumnWidth,
    updateRowHeightFromGrid,
    // Document offset of square row r (spacer + gap + r * measured pitch).
    squareRowTop,
    recalculateVisibleRange,
    onGridScroll,
    scrollCursorIntoView,
    // Justified-mode additions (null / false when thumbnailMode is 'square').
    isJustifiedMode,
    justifiedLayout,
  };
}
