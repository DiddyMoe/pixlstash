// The grid's first load on mount joins the view's load rather than re-reading
// it (`fetchAllGridImages({ join: true })`, ImageGrid's onMounted timer).
//
// The mount waits 80 ms before its first load and stands down if a load is
// running or has filled the grid. What it cannot see is a load for the same
// view that already failed in those 80 ms: the grid is empty and nothing is
// loading. Joining skips it, so the error stays on screen for "Try again"
// instead of the mount hammering a server that just answered 503. Mounts the
// real ImageGrid, following ImageGridLoadingState.test.js.

import { describe, it, expect, beforeEach, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { setActivePinia, createPinia } from "pinia";
import { useSelectionStore } from "../../stores/useSelectionStore.js";
import { useProjectStore } from "../../stores/useProjectStore.js";
import { useSortStore } from "../../stores/useSortStore.js";
import { ref, nextTick } from "vue";

const apiGet = vi.fn();
const apiPost = vi.fn();
const apiPatch = vi.fn();
const apiPut = vi.fn();
const apiDelete = vi.fn();

// Async factory with a local `await import("vue")`: the store imports above
// pull in apiClient, so this factory runs BEFORE the file's own top-level `vue`
// import has initialised. Closing over that binding throws "Cannot access
// __vi_import__ before initialization".
vi.mock("../../utils/apiClient", async () => {
  const { ref: makeRef, computed: makeComputed } = await import("vue");
  const isAuthenticated = makeRef(true);
  const readOnly = makeRef(false);
  const sessionContext = makeRef({ scope: "ALL" });
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
    isReadOnly: makeComputed(() => readOnly.value),
    login: vi.fn(),
    logout: vi.fn(),
    sessionContext,
    setRequestClientId: vi.fn(),
    API_BASE_URL: "/api/v1",
    // Reached back through the module so a test can flip the session scope.
    __readOnly: readOnly,
  };
});

// The `vuetify/components` barrel pulls component CSS that Vitest cannot load
// from node_modules; hand back a trivial stub for whatever name is imported.
vi.mock("vuetify/components", async () => {
  const { vuetifyComponentStubs } = await import("../../testing/vuetifyStubs");
  return vuetifyComponentStubs();
});

// The grid's own load, captured so a test can start a load the way any other
// trigger does (a store change, a socket event) before the mount's first-load
// timer fires.
const grid = vi.hoisted(() => ({ fetch: null }));
vi.mock("../../composables/useGridFetch.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    useGridFetch: (...args) => {
      const result = actual.useGridFetch(...args);
      grid.fetch = result.fetchAllGridImages;
      return result;
    },
  };
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

function mountGrid({ view = "ALL" } = {}) {
  const selectionStore = useSelectionStore();
  const projectStore = useProjectStore();
  const sortStore = useSortStore();
  selectionStore.selectedCharacter = view;
  selectionStore.selectedSet = null;
  selectionStore.selectedSetIds = [];
  projectStore.projectViewMode = "global";
  projectStore.selectedProjectId = null;
  sortStore.selectedSort = "DATE";
  sortStore.selectedDescending = true;
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

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const COUNT_ZERO = () => Promise.resolve({ data: { count: 0 } });
const DOWN = () =>
  Promise.reject(new Error("Request failed with status code 503"));

function answer({ count }) {
  apiGet.mockImplementation((url) => {
    const u = String(url);
    if (u.includes("/summary")) {
      return Promise.resolve({ data: { image_count: 5 } });
    }
    if (u.includes("/pictures/count")) return count();
    return Promise.resolve({ data: [] });
  });
}

beforeEach(() => {
  setActivePinia(createPinia());
  apiGet.mockReset();
  apiPost.mockReset();
  apiPost.mockResolvedValue({ data: {} });
  grid.fetch = null;
});

const countCalls = () =>
  apiGet.mock.calls.filter(([url]) => String(url).includes("/pictures/count"))
    .length;

describe("the grid's first load on mount", () => {
  it("loads the view when nothing has", async () => {
    answer({ count: COUNT_ZERO });
    mountGrid();
    await wait(300);

    expect(countCalls()).toBe(1);
  });

  it("does not re-read a view whose load just failed", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    answer({ count: DOWN });
    const wrapper = mountGrid();
    // Another trigger loads the view inside the mount's 80 ms wait, and the
    // load fails.
    await grid.fetch();
    expect(countCalls()).toBe(1);

    // Past the mount's wait and the empty card's delay, inside the 2.5 s in
    // which a failed load is joined.
    await wait(800);
    await nextTick();

    expect(countCalls()).toBe(1);
    expect(wrapper.find(".empty-state").text()).toContain(
      "Couldn't load pictures",
    );
  });
});
