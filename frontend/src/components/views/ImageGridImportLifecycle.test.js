// The grid's import announcements, against App's upload flag.
//
// App sets wsStore.isUploadInProgress on `import-started` and clears it on
// `import-ended`; while set, the socket takes every `picture_imported` as this
// tab's own upload. An import the grid announced must therefore always be
// ended, including when the grid goes away mid-upload (a view change). Vue
// drops an emit from a component already unmounted, so the grid ends it just
// before.
//
// Mounts the real ImageGrid, as ImageGridSquareGeometry.test.js does.

import { describe, it, expect, beforeEach, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { setActivePinia, createPinia } from "pinia";
import { useSelectionStore } from "../../stores/useSelectionStore.js";
import { useProjectStore } from "../../stores/useProjectStore.js";
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
});

// Test Utils clears a wrapper's emitted history as unmount() begins, so read
// after an unmount this holds only what the unmount itself emitted.
const ended = (wrapper) => wrapper.emitted("import-ended") ?? [];

describe("the grid's import announcements", () => {
  it("ends an import it announced when it is unmounted mid-upload", async () => {
    const wrapper = mountGrid();
    await nextTick();
    wrapper.vm.handleImportStarted();
    expect(wrapper.emitted("import-started")).toHaveLength(1);

    wrapper.unmount();

    expect(ended(wrapper)).toHaveLength(1);
  });

  it("announces no end on unmount when no import was going", async () => {
    const wrapper = mountGrid();
    await nextTick();

    wrapper.unmount();

    expect(ended(wrapper)).toHaveLength(0);
  });

  it("ends a finished import once, not again on unmount", async () => {
    const wrapper = mountGrid();
    await nextTick();
    wrapper.vm.handleImportStarted();
    wrapper.vm.handleImportCancelled();
    expect(ended(wrapper)).toHaveLength(1);

    wrapper.unmount();

    expect(ended(wrapper)).toHaveLength(0);
  });
});
