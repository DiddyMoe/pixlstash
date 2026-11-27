// An export of a selection names its pictures in the request body: as `?id=`
// parameters a large selection is a URL the server or a proxy refuses. An
// export of the whole view still sends the view's query. Mounts the real grid
// the way ImageGridExportPolling.test.js does.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { ref } from "vue";
import { useSearchStore } from "../../stores/useSearchStore";
import { useSelectionStore } from "../../stores/useSelectionStore";
import { useSortStore } from "../../stores/useSortStore";
import { SEARCH_CHANCE_CEILING } from "../../api/pictures";

const apiGet = vi.fn();
const apiPost = vi.fn();

vi.mock("../../utils/apiClient", async () => {
  const { ref, computed } = await import("vue");
  return {
    apiClient: {
      get: (...args) => apiGet(...args),
      post: (...args) => apiPost(...args),
      patch: vi.fn(), put: vi.fn(), delete: vi.fn(),
    },
    onSessionReset: () => () => {},
    activateShareToken: vi.fn(),
    appendShareToken: (url) => url,
    checkLoginStatus: vi.fn(), checkSession: vi.fn(),
    isAuthenticated: ref(true), isReadOnly: computed(() => false),
    login: vi.fn(), logout: vi.fn(),
    sessionContext: ref({ scope: "ALL" }),
    setRequestClientId: vi.fn(),
    newOperationBatchId: vi.fn(() => `cli-${crypto.randomUUID()}`),
    operationBatchHeaders: (batchId) => ({ headers: { "X-Operation-Batch-Id": batchId } }),
    API_BASE_URL: "/api/v1",
  };
});
vi.mock("vuetify/components", async () => {
  const { vuetifyComponentStubs } = await import("../../testing/vuetifyStubs");
  return vuetifyComponentStubs();
});
vi.mock("vue-router", () => ({
  useRoute: () => ({ query: {}, params: {}, path: "/", name: "grid" }),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), currentRoute: ref({ query: {} }) }),
}));

import ImageGrid from "./ImageGrid.vue";

const DESTINATION = "/home/me/exports";
// Never completes, so each test sees only how the export started.
const RUNNING = { status: "in_progress", processed: 0, total: 3, progress: 0 };

let wrapper;

beforeEach(() => {
  setActivePinia(createPinia());
  useSelectionStore().selectedCharacter = "ALL";
  useSortStore().selectedSort = "DATE";
  apiGet.mockReset().mockImplementation(async (url) => {
    const path = String(url);
    if (path === "/pictures/export/status") return { data: RUNNING };
    if (path === "/pictures/export" || path.startsWith("/pictures/export?")) {
      return { data: { task_id: "test-task" } };
    }
    return { data: path.includes("/count") ? { count: 0 } : [] };
  });
  apiPost.mockReset().mockImplementation(async (url) => ({
    data: String(url).startsWith("/pictures/export") ? { task_id: "test-task" } : {},
  }));
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
});

async function mountGrid() {
  wrapper = mount(ImageGrid, {
    shallow: true,
    props: { backendUrl: "/api/v1" },
    global: {
      stubs: { "v-icon": true, "v-snackbar": true },
      config: { compilerOptions: { isCustomElement: (tag) => tag.startsWith("v-") } },
    },
  });
  await flushPromises();
}

function exportCalls(mock) {
  return mock.mock.calls.filter(
    ([url]) => String(url).startsWith("/pictures/export") && !String(url).includes("/status"),
  );
}

describe("ImageGrid export of a selection", () => {
  it("a ZIP export posts the selected ids in the body, none in the URL", async () => {
    await mountGrid();
    wrapper.vm.selectedImageIds = [3, 1, 2];
    const started = wrapper.vm.exportCurrentViewToZip({});
    await flushPromises();

    expect(exportCalls(apiGet)).toEqual([]);
    const calls = exportCalls(apiPost);
    expect(calls).toHaveLength(1);
    const [url, body] = calls[0];
    expect(url).toBe("/pictures/export");
    expect(body.picture_ids).toEqual([3, 1, 2]);
    expect(body.caption_mode).toBe("description");
    wrapper.unmount();
    wrapper = null;
    await started;
  });

  it("a folder export posts the selected ids and the destination in the body", async () => {
    await mountGrid();
    wrapper.vm.selectedImageIds = [3, 1, 2];
    const started = wrapper.vm.exportCurrentViewToFolder({ destination: DESTINATION });
    await flushPromises();

    const calls = exportCalls(apiPost);
    expect(calls).toHaveLength(1);
    const [url, body] = calls[0];
    expect(url).toBe("/pictures/export/folder/selection");
    expect(body.destination).toBe(DESTINATION);
    expect(body.picture_ids).toEqual([3, 1, 2]);
    wrapper.unmount();
    wrapper = null;
    await started;
  });

  it("with nothing selected, a ZIP export sends the view's query", async () => {
    await mountGrid();
    const started = wrapper.vm.exportCurrentViewToZip({});
    await flushPromises();

    expect(exportCalls(apiPost)).toEqual([]);
    const calls = exportCalls(apiGet);
    expect(calls).toHaveLength(1);
    expect(String(calls[0][0])).toMatch(/^\/pictures\/export\?/);
    expect(String(calls[0][0])).not.toMatch(/[?&]id=/);
    wrapper.unmount();
    wrapper = null;
    await started;
  });

  it("a search view's export sends the threshold the grid's search uses", async () => {
    useSearchStore().searchQuery = "zebra";
    await mountGrid();
    const started = wrapper.vm.exportCurrentViewToZip({});
    await flushPromises();

    const calls = exportCalls(apiGet);
    expect(calls).toHaveLength(1);
    const params = new URLSearchParams(String(calls[0][0]).split("?")[1]);
    expect(params.get("query")).toBe("zebra");
    expect(params.get("threshold")).toBe(String(SEARCH_CHANCE_CEILING));
    wrapper.unmount();
    wrapper = null;
    await started;
  });
});
