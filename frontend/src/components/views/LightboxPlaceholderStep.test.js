// Next/Previous onto a slot the grid is still streaming in waits for its row.
//
// The lightbox steps through a snapshot frozen on open, and a streaming grid
// starts as a sized list of placeholders (`{id: null, idx}`) that each batch
// fills in place. A step onto a placeholder used to clear the picture and
// leave the lightbox blank. It now stays on the current picture and moves on
// once the live slot fills, and the snapshot takes the arrived row so the next
// step and the filmstrip agree with the grid.

import { afterEach, describe, it, expect, vi } from "vitest";

// `AppInspector`'s tab band renders a `VIcon` imported from the Vuetify barrel,
// and this suite mounts the overlay without the Vuetify plugin. Stub the
// barrel, per `testing/vuetifyStubs.js`.
vi.mock("vuetify/components", async () => {
  const { vuetifyComponentStubs } = await import("../../testing/vuetifyStubs");
  return vuetifyComponentStubs();
});

import { enableAutoUnmount, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";

// Statically imported for the same reason as LightboxSmartScore.test.js:
// compiling this SFC inside the first test's timeout is what makes it flake.
import Lightbox from "./Lightbox.vue";

const getMock = vi.fn(async (url) => {
  const meta =
    typeof url === "string" && url.match(/pictures\/(\d+)\/metadata/);
  if (meta) {
    return { data: { id: Number(meta[1]), tags: [] } };
  }
  if (typeof url === "string" && url.includes("/workflow")) {
    const notFound = new Error("no workflow");
    notFound.response = { status: 404 };
    throw notFound;
  }
  return { data: [] };
});

vi.mock("../../utils/apiClient", () => ({
  API_BASE_URL: "/api/v1",
  onSessionReset: () => () => {},
  sessionContext: { value: null },
  apiClient: { get: (...a) => getMock(...a), post: vi.fn(), delete: vi.fn() },
  appendShareToken: (u) => u,
  isReadOnly: { value: false },
}));

vi.mock("../../api/stacks", () => ({
  listStackPictures: vi.fn(async () => []),
}));

enableAutoUnmount(afterEach);

const STUBS = {
  RailTags: true,
  LightboxFilmstrip: true,
  LightboxDescriptionPanel: true,
  AddToEntityControl: true,
  CharacterEditor: true,
  StarRatingOverlay: true,
  PluginParametersUI: true,
  ProgressOverlay: true,
  RailPicture: {
    name: "RailPicture",
    props: ["image"],
    template: "<div class='meta'></div>",
  },
};

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));
const settle = async () => {
  for (let i = 0; i < 4; i++) await flush();
};

const shownId = (wrapper) =>
  wrapper.findComponent({ name: "RailPicture" }).props("image")?.id ?? null;

async function openOn(initialImageId, grid) {
  setActivePinia(createPinia());
  const wrapper = mount(Lightbox, {
    props: {
      open: false,
      initialImageId,
      allImages: grid,
      backendUrl: "http://test",
      tagUpdate: { key: 0, pictureIds: [] },
      descriptionUpdate: { key: 0, pictureIds: [] },
      smartScoreUpdate: { key: 0, pictureIds: [] },
    },
    global: { stubs: STUBS },
  });
  await wrapper.setProps({ open: true });
  await settle();
  return wrapper;
}

const streamingGrid = () =>
  reactive([
    { id: 1, idx: 0, tags: [] },
    { id: null, idx: 1 },
    { id: 3, idx: 2, tags: [] },
  ]);

describe("Lightbox step onto a slot still streaming in", () => {
  it("Next waits on the current picture, then shows the row that fills it", async () => {
    const grid = streamingGrid();
    const wrapper = await openOn("1", grid);
    expect(String(shownId(wrapper))).toBe("1");

    await wrapper.find(".lightbox-nav-right").trigger("click");
    await settle();
    expect(String(shownId(wrapper))).toBe("1");

    grid[1] = { id: 2, idx: 1, tags: [] };
    await settle();
    expect(String(shownId(wrapper))).toBe("2");

    // The snapshot took the row: the next step goes on to 3, not back to a
    // placeholder.
    await wrapper.find(".lightbox-nav-right").trigger("click");
    await settle();
    expect(String(shownId(wrapper))).toBe("3");
  });

  it("Previous waits on the current picture, then shows the row that fills it", async () => {
    const grid = streamingGrid();
    const wrapper = await openOn("3", grid);
    expect(String(shownId(wrapper))).toBe("3");

    await wrapper.find(".lightbox-nav-left").trigger("click");
    await settle();
    expect(String(shownId(wrapper))).toBe("3");

    grid[1] = { id: 2, idx: 1, tags: [] };
    await settle();
    expect(String(shownId(wrapper))).toBe("2");

    await wrapper.find(".lightbox-nav-left").trigger("click");
    await settle();
    expect(String(shownId(wrapper))).toBe("1");
  });

  it("steps straight onto a row that already arrived after open", async () => {
    const grid = streamingGrid();
    const wrapper = await openOn("1", grid);
    grid[1] = { id: 2, idx: 1, tags: [] };
    await settle();
    expect(String(shownId(wrapper))).toBe("1");

    await wrapper.find(".lightbox-nav-right").trigger("click");
    await settle();
    expect(String(shownId(wrapper))).toBe("2");
  });

  it("drops the wait when the reader moves elsewhere first", async () => {
    const grid = streamingGrid();
    const wrapper = await openOn("1", grid);
    await wrapper.find(".lightbox-nav-right").trigger("click");
    await settle();
    // Previous wraps from 1 to 3 while the step to slot 1 is still waiting.
    await wrapper.find(".lightbox-nav-left").trigger("click");
    await settle();
    expect(String(shownId(wrapper))).toBe("3");

    grid[1] = { id: 2, idx: 1, tags: [] };
    await settle();
    expect(String(shownId(wrapper))).toBe("3");
  });

  it("skips a trailing slot the stream ended short of", async () => {
    const grid = reactive([
      { id: 1, idx: 0, tags: [] },
      { id: 2, idx: 1, tags: [] },
      { id: null, idx: 2 },
    ]);
    const wrapper = await openOn("2", grid);
    await wrapper.find(".lightbox-nav-right").trigger("click");
    await settle();
    expect(String(shownId(wrapper))).toBe("2");

    // The stream trims the trailing placeholders it never filled, so the
    // slot is gone and the step goes on past it, wrapping to the first.
    grid.splice(2, 1);
    await settle();
    expect(String(shownId(wrapper))).toBe("1");
  });
});

// The other two ways onto a slot of the snapshot: a filmstrip click picks it
// by position, and a delete moves on to the next survivor. Each landed on the
// placeholder itself and blanked the lightbox, as a step used to.
describe("Lightbox pick of a slot still streaming in", () => {
  const pick = (wrapper, idx) =>
    wrapper
      .findComponent({ name: "LightboxFilmstrip" })
      .vm.$emit("select", idx);

  it("a filmstrip pick waits on the current picture, then shows the row", async () => {
    const grid = streamingGrid();
    const wrapper = await openOn("1", grid);

    pick(wrapper, 1);
    await settle();
    expect(String(shownId(wrapper))).toBe("1");

    grid[1] = { id: 2, idx: 1, tags: [] };
    await settle();
    expect(String(shownId(wrapper))).toBe("2");
  });

  it("a filmstrip pick shows a row that already arrived after open", async () => {
    const grid = streamingGrid();
    const wrapper = await openOn("1", grid);
    grid[1] = { id: 2, idx: 1, tags: [] };
    await settle();

    pick(wrapper, 1);
    await settle();
    expect(String(shownId(wrapper))).toBe("2");
  });

  it("a delete waits on the slot it moves on to, then shows its row", async () => {
    const grid = streamingGrid();
    const wrapper = await openOn("1", grid);

    expect(wrapper.vm.removePictures([1])).toBe(true);
    await settle();
    expect(String(shownId(wrapper))).toBe("1");

    grid[1] = { id: 2, idx: 1, tags: [] };
    await settle();
    expect(String(shownId(wrapper))).toBe("2");
  });

  it("a delete moves straight on to a row that already arrived", async () => {
    const grid = streamingGrid();
    const wrapper = await openOn("1", grid);
    grid[1] = { id: 2, idx: 1, tags: [] };
    await settle();

    expect(wrapper.vm.removePictures([1])).toBe(true);
    await settle();
    expect(String(shownId(wrapper))).toBe("2");
    // It moves as onto any next picture: the slide in from the next side.
    expect(wrapper.find(".lightbox-media-inner").classes()).toContain(
      "lightbox-media-inner--arrived-next",
    );
  });

  it("a filmstrip pick of a trailing slot the stream ended short of stays put", async () => {
    const grid = reactive([
      { id: 1, idx: 0, tags: [] },
      { id: 2, idx: 1, tags: [] },
      { id: null, idx: 2 },
    ]);
    const wrapper = await openOn("1", grid);

    pick(wrapper, 2);
    await settle();
    grid.splice(2, 1);
    await settle();
    expect(String(shownId(wrapper))).toBe("1");
  });

  it("a delete onto a trailing slot the stream ended short of shows the one before", async () => {
    const grid = reactive([
      { id: 1, idx: 0, tags: [] },
      { id: 2, idx: 1, tags: [] },
      { id: null, idx: 2 },
    ]);
    const wrapper = await openOn("2", grid);

    expect(wrapper.vm.removePictures([2])).toBe(true);
    await settle();
    grid.splice(2, 1);
    await settle();
    expect(String(shownId(wrapper))).toBe("1");
  });
});
