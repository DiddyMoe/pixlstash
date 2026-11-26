// A by-id insert under the Images/Videos filter.
//
// The grid renders a window of the media-filtered list, sliced by the
// unfiltered indices the virtual scroll keeps. A realtime insert (an `added` or
// `restored` picture, or the new-pictures pill) reads rows by id with no
// format narrowing, so a video imported in another tab while Images is on used
// to land in the grid, be dropped by the filter, and move every card after it
// one slot early against its own idx. The insert now skips it.

import { describe, it, expect, beforeEach, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { setActivePinia, createPinia } from "pinia";
import { ref, computed, nextTick } from "vue";
import { useFilterStore } from "../../stores/useFilterStore.js";
import { useSelectionStore } from "../../stores/useSelectionStore.js";
import { useSortStore } from "../../stores/useSortStore.js";

const apiGet = vi.fn();
const apiPost = vi.fn();
const apiPatch = vi.fn();
const apiPut = vi.fn();
const apiDelete = vi.fn();

vi.mock("../../utils/apiClient", () => {
  const isAuthenticated = ref(true);
  const sessionContext = ref({ scope: "ALL" });
  return {
    onSessionReset: () => () => {},
    apiClient: {
      get: (...args) => apiGet(...args),
      post: (...args) => apiPost(...args),
      patch: (...args) => apiPatch(...args),
      put: (...args) => apiPut(...args),
      delete: (...args) => apiDelete(...args),
    },
    activateShareToken: vi.fn(),
    appendShareToken: (url) => url,
    checkLoginStatus: vi.fn(),
    checkSession: vi.fn(),
    isAuthenticated,
    isReadOnly: computed(() => false),
    login: vi.fn(),
    logout: vi.fn(),
    sessionContext,
    setRequestClientId: vi.fn(),
    API_BASE_URL: "/api/v1",
  };
});

vi.mock("vuetify/components", async () => {
  const { vuetifyComponentStubs } = await import("../../testing/vuetifyStubs");
  return vuetifyComponentStubs();
});

vi.mock("vue-router", () => ({
  useRoute: () => ({ query: {}, params: {}, path: "/", name: "grid" }),
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    currentRoute: ref({ query: {} }),
  }),
}));

import ImageGrid from "./ImageGrid.vue";

const COUNT = 30;
const VIDEO_ID = 500;

/** Pictures newest first, one a day; the video sits among them by date. */
function catalog() {
  const rows = new Map();
  for (let i = 0; i < COUNT; i += 1) {
    rows.set(i + 1, {
      id: i + 1,
      name: `still-${i + 1}.png`,
      format: "PNG",
      created_at: new Date(Date.UTC(2026, 0, 1 + i)).toISOString(),
    });
  }
  rows.set(VIDEO_ID, {
    id: VIDEO_ID,
    name: "clip.mp4",
    format: "MP4",
    created_at: new Date(Date.UTC(2026, 0, 15, 12)).toISOString(),
  });
  return rows;
}

let rows;

beforeEach(() => {
  setActivePinia(createPinia());
  rows = catalog();
  apiGet.mockReset();
  apiPost.mockReset();
  apiGet.mockImplementation((url) => {
    const u = String(url ?? "");
    if (u.includes("/pictures/count"))
      return Promise.resolve({ data: { count: 0 } });
    if (u.includes("/pictures/stream")) {
      return Promise.resolve({ data: { pictures: [], done: true } });
    }
    if (u.startsWith("/pictures?")) {
      const ids = new URLSearchParams(u.split("?")[1]).getAll("id");
      return Promise.resolve({
        data: ids.map((id) => rows.get(Number(id))).filter(Boolean),
      });
    }
    return Promise.resolve({ data: [] });
  });
  apiPost.mockResolvedValue({ data: {} });
});

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

async function mountGrid(mode) {
  useSelectionStore().selectedCharacter = "ALL";
  const sortStore = useSortStore();
  sortStore.selectedSort = "DATE";
  sortStore.selectedDescending = true;
  // Set before the mount: a change after it refetches the grid, and that
  // fetch (the count says 0) would empty it under the inserts below.
  useFilterStore().mediaTypeFilter = mode;
  const wrapper = mount(ImageGrid, {
    shallow: true,
    global: {
      config: {
        compilerOptions: { isCustomElement: (tag) => tag.startsWith("v-") },
      },
    },
    props: { backendUrl: "/api/v1" },
  });
  // The mount's own fetch settles first: an insert while it runs is skipped.
  for (let i = 0; i < 4; i += 1) await settle();
  expect(wrapper.vm.imagesLoading).toBe(false);
  // The stills, as the filtered stream would have put them in.
  await wrapper.vm.insertGridImagesById(
    Array.from({ length: COUNT }, (_, i) => i + 1),
  );
  // jsdom lays nothing out, so the window to render is set by hand.
  wrapper.vm.visibleStart = 0;
  wrapper.vm.visibleEnd = COUNT;
  for (let i = 0; i < 4; i += 1) await nextTick();
  return wrapper;
}

/** Every rendered card sits in the slot its own idx names. */
function expectCardsInTheirSlots(vm) {
  const start = vm.renderStart;
  const rendered = vm.gridImagesToRender;
  expect(rendered.length).toBeGreaterThan(0);
  rendered.forEach((img, i) => expect(img.idx).toBe(start + i));
}

function heldIds(vm) {
  return vm.allGridImages.map((img) => img.id);
}

describe("a by-id insert under the Images/Videos filter", () => {
  it("skips a video while Images is on, so no card shifts", async () => {
    const wrapper = await mountGrid("images");

    await wrapper.vm.insertGridImagesById([VIDEO_ID]);
    await nextTick();

    expect(heldIds(wrapper.vm)).not.toContain(VIDEO_ID);
    expect(heldIds(wrapper.vm)).toHaveLength(COUNT);
    expect(wrapper.vm.filteredGridCount).toBe(wrapper.vm.allGridImages.length);
    expectCardsInTheirSlots(wrapper.vm);
    wrapper.unmount();
  });

  it("skips it on a restore too (no highlight), the same way", async () => {
    const wrapper = await mountGrid("images");
    await wrapper.vm.insertGridImagesById([VIDEO_ID], { highlight: false });
    await nextTick();

    expect(heldIds(wrapper.vm)).not.toContain(VIDEO_ID);
    expectCardsInTheirSlots(wrapper.vm);
    wrapper.unmount();
  });

  it("still inserts a still picture under Images, at its sorted place", async () => {
    rows.set(600, {
      id: 600,
      name: "late.png",
      format: "PNG",
      created_at: new Date(Date.UTC(2026, 0, 15, 12)).toISOString(),
    });
    const wrapper = await mountGrid("images");
    await wrapper.vm.insertGridImagesById([600]);
    await nextTick();

    // Newest first: days 30..16, then the 15th at noon, then day 15.
    expect(heldIds(wrapper.vm).indexOf(600)).toBe(COUNT - 15);
    expectCardsInTheirSlots(wrapper.vm);
    wrapper.unmount();
  });

  it("inserts the video in All, where the filter keeps it", async () => {
    const wrapper = await mountGrid("all");
    await wrapper.vm.insertGridImagesById([VIDEO_ID]);
    await nextTick();

    expect(heldIds(wrapper.vm)).toContain(VIDEO_ID);
    expectCardsInTheirSlots(wrapper.vm);
    wrapper.unmount();
  });
});

// An open stack's members are read unfiltered (`listStackPictures`), so the
// grid sifts them as it sifts a by-id insert.
describe("an open stack under the Images/Videos filter", () => {
  const STACK = 7;
  const at = (day) => new Date(Date.UTC(2026, 0, day, 6)).toISOString();
  const members = [
    { id: 700, name: "cover.png", format: "PNG", stack_position: 0 },
    { id: 701, name: "second.png", format: "PNG", stack_position: 1 },
    { id: 702, name: "take.mp4", format: "MP4", stack_position: 2 },
  ].map((m) => ({ ...m, stack_id: STACK, created_at: at(10) }));

  async function openStack(mode) {
    rows.set(700, { ...members[0], stack_count: 3 });
    const listing = apiGet.getMockImplementation();
    apiGet.mockImplementation((url) =>
      String(url ?? "").startsWith(`/stacks/${STACK}/pictures`)
        ? Promise.resolve({ data: members })
        : listing(url),
    );
    const wrapper = await mountGrid(mode);
    await wrapper.vm.insertGridImagesById([700]);
    await nextTick();
    const tile = wrapper.vm.allGridImages.find((img) => img.id === 700);
    await wrapper.vm.toggleStackExpand(tile);
    for (let i = 0; i < 4; i += 1) await nextTick();
    return wrapper;
  }

  it("leaves a video member out while Images is on, so no card shifts", async () => {
    const wrapper = await openStack("images");

    expect(heldIds(wrapper.vm)).toEqual(expect.arrayContaining([700, 701]));
    expect(heldIds(wrapper.vm)).not.toContain(702);
    expect(wrapper.vm.filteredGridCount).toBe(wrapper.vm.allGridImages.length);
    expectCardsInTheirSlots(wrapper.vm);
    wrapper.unmount();
  });

  it("opens every member in All", async () => {
    const wrapper = await openStack("all");

    expect(heldIds(wrapper.vm)).toEqual(
      expect.arrayContaining([700, 701, 702]),
    );
    expectCardsInTheirSlots(wrapper.vm);
    wrapper.unmount();
  });
});
