// The lightbox picks up the open picture's grid row when a streaming grid
// writes it in place.
//
// A lightbox opened while the grid is still loading has the open picture's id
// but not yet its grid row. The stream writes each batch's rows into the
// grid's placeholder array in place, without replacing it. The grid row is the
// only source of the picture's `stack_count` (the metadata read does not carry
// it), and without it the lightbox never loads the stack, so the "N of M in
// this stack" chip stays missing until the reader moves away and back.

import { afterEach, beforeEach, describe, it, expect, vi } from "vitest";

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
import { listStackPictures } from "../../api/stacks";

const getMock = vi.fn(async (url) => {
  if (typeof url === "string" && url.includes("/metadata")) {
    return { data: { id: 7, stack_id: "s1", tags: [] } };
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
  listStackPictures: vi.fn(async () => [
    { id: 7, stack_id: "s1", stack_position: 0, tags: [] },
    { id: 70, stack_id: "s1", stack_position: 1, tags: [] },
    { id: 71, stack_id: "s1", stack_position: 2, tags: [] },
  ]),
}));

enableAutoUnmount(afterEach);

// Each case asserts its own stack read.
beforeEach(() => {
  listStackPictures.mockClear();
});

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

describe("Lightbox opened before the grid streams in", () => {
  it("reads the picture's row when its batch writes it in place", async () => {
    setActivePinia(createPinia());
    const wrapper = mount(Lightbox, {
      props: {
        open: false,
        initialImageId: "7",
        allImages: [],
        backendUrl: "http://test",
        tagUpdate: { key: 0, pictureIds: [] },
        descriptionUpdate: { key: 0, pictureIds: [] },
        smartScoreUpdate: { key: 0, pictureIds: [] },
      },
      global: { stubs: STUBS },
    });
    await wrapper.setProps({ open: true });
    await flush();
    const grid = reactive(
      Array.from({ length: 4 }, (_, i) => ({ id: null, idx: i })),
    );
    await wrapper.setProps({ allImages: grid });
    await flush();
    await flush();
    expect(wrapper.find(".lightbox-stack-chip").exists()).toBe(false);

    grid[1] = {
      id: 7,
      idx: 1,
      stack_id: "s1",
      stack_count: 3,
      stack_position: 0,
      tags: [],
    };
    for (let i = 0; i < 4; i++) await flush();

    expect(listStackPictures).toHaveBeenCalledWith("s1");
    expect(
      wrapper.findComponent({ name: "RailPicture" }).props("image")
        ?.stack_count,
    ).toBe(3);
    expect(wrapper.find(".lightbox-stack-chip").text()).toContain(
      "1 of 3 in this stack",
    );
  });
});

// Opened on a picture while the grid is still streaming (from the Tasks panel,
// or by back/forward), the lightbox freezes a snapshot that already has rows
// but not the open picture's. Its batch then writes the row into the live
// array, which the snapshot never sees.
describe("Lightbox opened while the grid streams in", () => {
  it("reads the picture's row when a later batch writes it in place", async () => {
    setActivePinia(createPinia());
    const grid = reactive([
      { id: 1, idx: 0, tags: [] },
      { id: null, idx: 1 },
      { id: null, idx: 2 },
      { id: 4, idx: 3, tags: [] },
    ]);
    const wrapper = mount(Lightbox, {
      props: {
        open: false,
        initialImageId: "7",
        allImages: grid,
        backendUrl: "http://test",
        tagUpdate: { key: 0, pictureIds: [] },
        descriptionUpdate: { key: 0, pictureIds: [] },
        smartScoreUpdate: { key: 0, pictureIds: [] },
      },
      global: { stubs: STUBS },
    });
    await wrapper.setProps({ open: true });
    await flush();
    await flush();
    expect(listStackPictures).not.toHaveBeenCalled();
    expect(wrapper.find(".lightbox-stack-chip").exists()).toBe(false);

    grid[1] = {
      id: 7,
      idx: 1,
      stack_id: "s1",
      stack_count: 3,
      stack_position: 0,
      tags: [],
    };
    for (let i = 0; i < 4; i++) await flush();

    expect(listStackPictures).toHaveBeenCalledWith("s1");
    expect(
      wrapper.findComponent({ name: "RailPicture" }).props("image")
        ?.stack_count,
    ).toBe(3);
    expect(wrapper.find(".lightbox-stack-chip").text()).toContain(
      "1 of 3 in this stack",
    );
  });
});
