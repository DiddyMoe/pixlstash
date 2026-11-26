// A Scrapheap run that has been asked to stop is still a run going.
//
// The server runs one Scrapheap run at a time and refuses a second start until
// the first has ended. A run asked to stop goes on to the end of its slice
// first, so until the poll says it has ended the grid must keep refusing every
// gesture that would start another (Delete forever, Restore), as it does for
// any run going: a confirm opened then would be typed for a start the server
// refuses. ImageGridScrapheapRun.test.js pins the gate for a run going and a
// start still out; this file pins it for a stopping run.
//
// Mounts the real ImageGrid, as ImageGridScrapheapRun.test.js does.

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { setActivePinia, createPinia } from "pinia";
import { nextTick, ref } from "vue";
import { useScrapheapRunStore } from "../../stores/useScrapheapRunStore.js";
import { useNoticeStore } from "../../stores/useNoticeStore.js";
import { useSelectionStore } from "../../stores/useSelectionStore.js";

const apiGet = vi.fn();
const apiPost = vi.fn();
const apiDelete = vi.fn();

vi.mock("../../utils/apiClient", async () => {
  const { ref: makeRef, computed: makeComputed } = await import("vue");
  return {
    onSessionReset: () => () => {},
    apiClient: {
      get: (...args) => apiGet(...args),
      post: (...args) => apiPost(...args),
      patch: vi.fn(),
      put: vi.fn(),
      delete: (...args) => apiDelete(...args),
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
  useSelectionStore().selectedCharacter = "SCRAPHEAP";
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

/** A purge run as `serialize_run` shapes it, asked to stop and not yet ended. */
function stoppingRun() {
  return {
    id: "example-run",
    kind: "purge",
    status: "running",
    active: true,
    include_protected: false,
    cancel_requested: true,
    total: 50000,
    processed: 1500,
    deleted_count: 1500,
    skipped_count: 0,
    skipped_locked_count: 0,
    restored_count: 0,
    skipped_restored: 0,
    skipped_locked: [],
    snapshots_with_deleted: [],
    segment_batch_ids: [],
    ghosts_kept: 0,
    ghosts_cascaded: 0,
    error_code: null,
    error: null,
  };
}

let consoleWarn;

beforeEach(() => {
  setActivePinia(createPinia());
  sessionStorage.clear();
  apiGet.mockReset();
  apiPost.mockReset();
  apiDelete.mockReset();
  apiGet.mockResolvedValue({ data: { pictures: [], count: 0, total: 0 } });
  apiPost.mockResolvedValue({ data: {} });
  consoleWarn = vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  useScrapheapRunStore().forget();
  consoleWarn.mockRestore();
});

describe("a gesture while a Scrapheap run is stopping", () => {
  it.each([
    ["Empty scrapheap", (vm) => vm.confirmEmptyScrapheap()],
    ["Delete forever on a selection", (vm) => vm.deleteSelected()],
    ["Restore all", (vm) => vm.confirmRestoreScrapheap()],
    ["Restore on a selection", (vm) => vm.removeFromGroup()],
  ])("%s is refused, and says so, until the run ends", async (_gesture, act) => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const wrapper = mountGrid();
    await nextTick();
    wrapper.vm.allGridImages = [{ id: 1, name: "test-1.png" }];
    wrapper.vm.selectedImageIds = [1];
    const store = useScrapheapRunStore();
    store.run = stoppingRun();
    await nextTick();
    expect(store.action).toBe(null);
    apiPost.mockClear();
    apiDelete.mockClear();

    await act(wrapper.vm);
    await nextTick();

    expect(apiPost).not.toHaveBeenCalled();
    expect(apiDelete).not.toHaveBeenCalled();
    expect(confirmSpy).not.toHaveBeenCalled();
    expect(wrapper.vm.deleteForeverOpen).toBe(false);
    expect(
      useNoticeStore().notices.find((n) => n.key === "scrapheap-run-busy"),
    ).toMatchObject({
      level: "info",
      text:
        "Another Scrapheap run is going, and its progress shows above the " +
        "grid. This can start once it ends.",
    });
    expect(wrapper.vm.selectedImageIds).toEqual([1]);

    confirmSpy.mockRestore();
    wrapper.unmount();
  });
});
