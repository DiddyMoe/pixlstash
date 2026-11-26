// The order a stack offers to Make cover / Move earlier / Move later. The order
// route refuses a list that misses any member, so a stack whose members are not
// all known to be on screen must offer no order at all.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { ref } from "vue";
import { setActivePinia, createPinia } from "pinia";

vi.mock("../api/stacks", () => ({
  getStack: vi.fn(),
  listStackPictures: vi.fn(),
  createStack: vi.fn(),
  setStackOrder: vi.fn(() => Promise.resolve({})),
  removeStackMembers: vi.fn(),
}));

import { useStackOrdering } from "./useStackOrdering.js";
import { listStackPictures } from "../api/stacks";
import { useSortStore } from "../stores/useSortStore";
import {
  resetShareDragForTests,
  runShareDrag,
} from "../utils/shareDrag.js";

function makeStack(
  images,
  fetched = images,
  { rowFitsMediaFilter, loadFoldsStacks = ref(true) } = {},
) {
  const deps = {
    allGridImages: ref(images),
    lastFetchedGridImages: ref(fetched),
    loadedRanges: ref([]),
    visibleStart: ref(0),
    visibleEnd: ref(images.length),
    renderBuffer: ref(0),
    divisibleViewWindow: ref(4),
    stackReorderDrag: ref(null),
    stackReorderHoverId: ref(null),
    stackReorderHoverSide: ref(null),
    setStackReorderHoverId: vi.fn(),
    setStackReorderHoverSide: vi.fn(),
    selectedImageIds: ref([]),
    preserveScrollOnNextFetch: ref(false),
    loadFoldsStacks,
  };
  const callbacks = new Proxy(
    {},
    {
      get: (_target, name) =>
        name === "rowFitsMediaFilter" && rowFitsMediaFilter
          ? rowFitsMediaFilter
          : vi.fn(() => []),
    },
  );
  const stack = useStackOrdering(deps, {}, vi.fn(), callbacks);
  stack.expandedStackIds.value = new Set(["7"]);
  return Object.assign(stack, {
    grid: deps.allGridImages,
    lastFetchedGridImages: deps.lastFetchedGridImages,
  });
}

const member = (id, position, extra = {}) => ({
  id,
  stack_id: 7,
  stack_position: position,
  ...extra,
});

beforeEach(() => {
  setActivePinia(createPinia());
});

describe("useStackOrdering: the order an open stack offers", () => {
  it("offers the order when every member is on screen", () => {
    const stack = makeStack([
      member(1, 0, { stackCount: 3 }),
      member(2, 1),
      member(3, 2),
    ]);
    expect(stack.getOpenStackPlace({ id: 2, stack_id: 7 })).toEqual({
      stackId: "7",
      position: 1,
      count: 3,
    });
  });

  it("offers none when the badge says more members exist", () => {
    const stack = makeStack([member(1, 0, { stackCount: 4 }), member(2, 1)]);
    expect(stack.getOpenStackPlace({ id: 2, stack_id: 7 })).toBeNull();
    expect(stack.moveStackMember(2, "cover")).toBe(false);
  });

  it("offers none when no row says how big the stack is", () => {
    const stack = makeStack([member(1, 0), member(2, 1)]);
    expect(stack.getOpenStackPlace({ id: 2, stack_id: 7 })).toBeNull();
  });

  it("trusts the members list fetched on expand over a stale badge", () => {
    const stack = makeStack([member(1, 0, { stackCount: 2 }), member(2, 1)]);
    stack.expandedStackMembers.value = new Map([
      ["7", { ids: ["1", "2", "3"], images: [] }],
    ]);
    expect(stack.getOpenStackPlace({ id: 2, stack_id: 7 })).toBeNull();
  });
});

// The lightbox's filmstrip draws an open stack in stack order, cover first,
// whatever the grid is sorted by; the grid has to lay it out the same way.
describe("useStackOrdering: an open stack's layout under a sort", () => {
  const ids = (stack) => stack.grid.value.map((img) => img.id);
  const badges = (stack) =>
    stack.grid.value
      .filter((img) => img.stackCount !== undefined)
      .map((img) => img.id);

  function openOnDateSort(tile, fetched) {
    useSortStore().selectedSort = "DATE";
    const loose = (id) => ({ id, stack_id: null });
    const stack = makeStack([loose(100), tile, loose(101)], fetched);
    stack.expandedStackIds.value = new Set();
    return stack;
  }

  it("lays the members out in stack order, cover first, from a non-cover tile", async () => {
    // Newest first, as a date sort lists them: the tile standing for the
    // stack is not its cover.
    const fetched = [member(2, 1), member(3, 2), member(1, 0)];
    const stack = openOnDateSort(member(2, 1, { stackCount: 3 }), fetched);
    await stack.toggleStackExpand(member(2, 1, { stackCount: 3 }));
    expect(ids(stack)).toEqual([100, 1, 2, 3, 101]);
    expect(badges(stack)).toEqual([1]);
  });

  it("does the same with members fetched from the server", async () => {
    listStackPictures.mockResolvedValueOnce([
      member(3, 2),
      member(1, 0),
      member(2, 1),
    ]);
    const stack = openOnDateSort(member(2, 1, { stackCount: 3 }), []);
    await stack.toggleStackExpand(member(2, 1, { stackCount: 3 }));
    expect(listStackPictures).toHaveBeenLastCalledWith("7");
    expect(ids(stack)).toEqual([100, 1, 2, 3, 101]);
    expect(badges(stack)).toEqual([1]);
  });

  it("collapses back to the cover, still badged", async () => {
    const fetched = [member(2, 1), member(3, 2), member(1, 0)];
    const stack = openOnDateSort(member(2, 1, { stackCount: 3 }), fetched);
    await stack.toggleStackExpand(member(2, 1, { stackCount: 3 }));
    await stack.toggleStackExpand(member(1, 0));
    expect(ids(stack)).toEqual([100, 1, 101]);
    expect(badges(stack)).toEqual([1]);
  });

  it("keeps a tile the members list does not know, at the end of the run", async () => {
    // The lightbox's filmstrip puts such a tile last too.
    listStackPictures.mockResolvedValueOnce([member(3, 1), member(1, 0)]);
    const stack = openOnDateSort(member(2, 1, { stackCount: 3 }), []);
    await stack.toggleStackExpand(member(2, 1, { stackCount: 3 }));
    expect(ids(stack)).toEqual([100, 1, 3, 2, 101]);
    expect(badges(stack)).toEqual([1]);
  });

  it("takes the tile's place in the stack from the members list", async () => {
    // The tile still says it is the cover; the members list knows a newer
    // picture took that place.
    const fetched = [member(2, 1), member(1, 0), member(3, 2)];
    const stack = openOnDateSort(member(2, 0, { stackCount: 3 }), fetched);
    await stack.toggleStackExpand(member(2, 0, { stackCount: 3 }));
    expect(ids(stack)).toEqual([100, 1, 2, 3, 101]);
    const tile = stack.grid.value.find((img) => img.id === 2);
    expect(tile.stack_position).toBe(1);
  });
});

// The members list is read unfiltered, so the Images/Videos filter is applied
// to it as to the stream: the grid renders a window of the filtered list by
// unfiltered index, and a held member the filter drops moves every card after
// it one slot early.
describe("useStackOrdering: an open stack under the Images/Videos filter", () => {
  const ids = (stack) => stack.grid.value.map((img) => img.id);
  const still = (id, position, extra = {}) =>
    member(id, position, { name: `still-${id}.png`, ...extra });
  const clip = (id, position) =>
    member(id, position, { name: `clip-${id}.mp4` });
  const imagesOnly = (img) => !String(img.name).endsWith(".mp4");
  const loose = (id) => ({ id, stack_id: null, name: `loose-${id}.png` });

  function openUnderImages(tile) {
    useSortStore().selectedSort = "DATE";
    const stack = makeStack([loose(100), tile, loose(101)], [], {
      rowFitsMediaFilter: imagesOnly,
    });
    stack.expandedStackIds.value = new Set();
    return stack;
  }

  it("leaves out a member the filter would not show when it opens", async () => {
    listStackPictures.mockResolvedValueOnce([
      still(1, 0),
      still(2, 1),
      clip(3, 2),
    ]);
    const tile = still(2, 1, { stackCount: 3 });
    const stack = openUnderImages(tile);

    await stack.toggleStackExpand(tile);

    expect(ids(stack)).toEqual([100, 1, 2, 101]);
    stack.grid.value.forEach((img, i) => expect(img.idx).toBe(i));
  });

  it("leaves it out when the grid is rebuilt with the stack open", () => {
    useSortStore().selectedSort = "DATE";
    const fetched = [loose(100), still(1, 0, { stackCount: 3 }), loose(101)];
    const stack = makeStack(fetched, fetched, {
      rowFitsMediaFilter: imagesOnly,
    });
    stack.expandedStackMembers.value = new Map([
      [
        "7",
        {
          ids: ["1", "2", "3"],
          images: [still(1, 0), still(2, 1), clip(3, 2)],
        },
      ],
    ]);

    stack.rebuildGridImagesFromLastFetch();

    expect(ids(stack)).toEqual([100, 1, 2, 101]);
  });

  it("offers no drag to reorder a stack it holds only part of", async () => {
    listStackPictures.mockResolvedValueOnce([
      still(1, 0),
      still(2, 1),
      clip(3, 2),
    ]);
    const tile = still(2, 1, { stackCount: 3 });
    const stack = openUnderImages(tile);
    await stack.toggleStackExpand(tile);
    const json = JSON.stringify({ type: "image-ids", imageIds: [1] });
    const event = {
      dataTransfer: {
        types: ["application/json"],
        files: [],
        effectAllowed: "move",
        dropEffect: "",
        getData: (type) => (type === "application/json" ? json : ""),
      },
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
    };

    stack.handleStackReorderDragOver(still(2, 1), event);

    // Not taken as a reorder: the order route would refuse the two it holds.
    expect(event.preventDefault).not.toHaveBeenCalled();
  });
});

// A dropEffect the drag does not allow makes Chromium drop nothing, with no
// drop event; a desktop drag-out (utils/shareDrag.js) allows copy and link only.
describe("useStackOrdering: reorder dragover asks for an allowed effect", () => {
  const COPY = "/home/me/share-copies/0123abcd/1.png";
  const openStack = () =>
    makeStack([member(1, 0, { stackCount: 3 }), member(2, 1), member(3, 2)]);
  const dragover = (dataTransfer) => ({
    dataTransfer,
    preventDefault: vi.fn(),
    stopPropagation: vi.fn(),
  });

  afterEach(() => {
    resetShareDragForTests();
    delete window.pixlstashDesktop;
  });

  it("takes a desktop drag-out with copy", () => {
    window.pixlstashDesktop = { startShareDrag: () => new Promise(() => {}) };
    void runShareDrag([1], [COPY]);
    const stack = openStack();
    const dataTransfer = {
      types: ["Files"],
      files: [],
      effectAllowed: "copyLink",
      dropEffect: "",
      getData: () => "",
    };
    const event = dragover(dataTransfer);

    stack.handleStackReorderDragOver(member(2, 1), event);

    expect(event.preventDefault).toHaveBeenCalled();
    expect(dataTransfer.dropEffect).toBe("copy");
  });

  it("keeps move for an in-page drag that allows only move", () => {
    const stack = openStack();
    const json = JSON.stringify({ type: "image-ids", imageIds: [1] });
    const dataTransfer = {
      types: ["application/json"],
      files: [],
      effectAllowed: "move",
      dropEffect: "",
      getData: (type) => (type === "application/json" ? json : ""),
    };
    const event = dragover(dataTransfer);

    stack.handleStackReorderDragOver(member(2, 1), event);

    expect(event.preventDefault).toHaveBeenCalled();
    expect(dataTransfer.dropEffect).toBe("move");
  });
});

// The server lists every scrapped picture on its own row (the listing's
// `only_deleted` branch), and the heap load keeps them all. A rebuild from
// lastFetchedGridImages must fold as that load did (useGridFetch's
// loadFoldsStacks), whichever caller asks for it: folding the heap on a
// rebuild hid one member of each stack after any removal there.
describe("useStackOrdering: folds a rebuild as the load did", () => {
  // Two two-member stacks and a lone picture, as the heap stream lists them.
  const heapRows = () => [
    { id: 1, stack_id: 7, stack_position: 0, stack_count: 2 },
    { id: 2, stack_id: 7, stack_position: 1, stack_count: 2 },
    { id: 3, stack_id: 8, stack_position: 0, stack_count: 2 },
    { id: 4, stack_id: 8, stack_position: 1, stack_count: 2 },
    { id: 5, stack_id: null },
  ];
  const ids = (stack) => stack.grid.value.map((img) => img.id);

  function heap({ folds = false } = {}) {
    const rows = heapRows();
    const stack = makeStack(rows.slice(), rows, {
      loadFoldsStacks: ref(folds),
    });
    stack.expandedStackIds.value = new Set();
    return stack;
  }

  it("keeps every other row after a removal (removeImagesById)", () => {
    const stack = heap();
    const server = heapRows().map((row) => row.id);
    stack.lastFetchedGridImages.value =
      stack.lastFetchedGridImages.value.filter((row) => row.id !== 2);
    stack.rebuildGridImagesFromLastFetch();
    expect(ids(stack)).toEqual(server.filter((id) => id !== 2));
  });

  it("keeps every row after a stack badge refresh (refreshStackFacets)", () => {
    const stack = heap();
    stack.lastFetchedGridImages.value = stack.lastFetchedGridImages.value.map(
      (row) =>
        row.stack_id === 8 ? { ...row, stack_count: 3, stackCount: 3 } : row,
    );
    stack.rebuildGridImagesFromLastFetch();
    expect(ids(stack)).toEqual([1, 2, 3, 4, 5]);
  });

  it("keeps every row when the stacks toggle or expand / collapse all rebuilds", async () => {
    const stack = heap();
    stack.syncExpandAllStacksFromFetchedImages();
    stack.rebuildGridImagesFromLastFetch();
    expect(ids(stack)).toEqual([1, 2, 3, 4, 5]);
    await stack.expandAllStacks();
    expect(ids(stack)).toEqual([1, 2, 3, 4, 5]);
    await stack.collapseAllStacks();
    expect(ids(stack)).toEqual([1, 2, 3, 4, 5]);
  });

  it("keeps every row after a scoring insert (insertGridImagesById)", () => {
    const stack = heap();
    stack.lastFetchedGridImages.value = [
      ...stack.lastFetchedGridImages.value,
      { id: 6, stack_id: 8, stack_position: 2, stack_count: 3 },
    ];
    stack.rebuildGridImagesFromLastFetch();
    expect(ids(stack)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("still folds each stack to one card in a view the load folded", () => {
    const stack = heap({ folds: true });
    stack.lastFetchedGridImages.value =
      stack.lastFetchedGridImages.value.filter((row) => row.id !== 5);
    stack.rebuildGridImagesFromLastFetch();
    expect(ids(stack)).toEqual([1, 3]);
  });
});
