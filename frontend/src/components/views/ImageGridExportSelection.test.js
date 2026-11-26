// What a ZIP or folder export asks the server for. A selection is named in the
// body of POST /pictures/export (ZIP) or POST /pictures/export/folder/selection
// (folder), in its order and with the Scrapheap flag of the view it was made
// in: as `?id=` parameters a large selection is a URL a server or a proxy
// refuses. An export of the whole view still sends the view's own query, and a
// search view's query asks for the threshold the grid's search uses. Mounts the
// real grid the way ImageGridExportPolling.test.js does.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { ref } from "vue";
import { SEARCH_CHANCE_CEILING } from "../../api/pictures";
import { useSearchStore } from "../../stores/useSearchStore";
import { useSelectionStore } from "../../stores/useSelectionStore";
import { useSortStore } from "../../stores/useSortStore";
import { SCRAPHEAP_PICTURES_ID } from "../../stores/useViewStore";

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

const TASK_ID = "test-task";

let wrapper;

beforeEach(() => {
  setActivePinia(createPinia());
  useSelectionStore().selectedCharacter = "ALL";
  useSortStore().selectedSort = "DATE";
  apiGet.mockReset().mockImplementation(async (url) => {
    const path = String(url);
    if (path === "/pictures/export/status") {
      // Ends the follow at once: what was asked for is all these test.
      return { data: { status: "failed", processed: 0, total: 0 } };
    }
    if (path === "/pictures/export" || path.startsWith("/pictures/export?")) {
      return { data: { task_id: TASK_ID } };
    }
    return { data: path.includes("/count") ? { count: 0 } : [] };
  });
  apiPost.mockReset().mockImplementation(async (url) => ({
    data:
      url === "/pictures/export" || String(url).startsWith("/pictures/export/folder")
        ? { task_id: TASK_ID }
        : {},
  }));
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  vi.restoreAllMocks();
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

async function exportZip(options = {}) {
  apiGet.mockClear();
  apiPost.mockClear();
  await wrapper.vm.exportCurrentViewToZip(options);
  await flushPromises();
}

function exportPosts() {
  return apiPost.mock.calls.filter(([url]) => url === "/pictures/export");
}

function exportGets() {
  return apiGet.mock.calls
    .map(([url]) => String(url))
    .filter((url) => url === "/pictures/export" || url.startsWith("/pictures/export?"));
}

describe("a ZIP export of a selection", () => {
  it("names the selected pictures in the POST body, in selection order", async () => {
    await mountGrid();
    wrapper.vm.selectedImageIds = [12, 11, 13];

    await exportZip({ captionMode: "tags", tagFormat: "underscores", resolution: "half" });

    expect(exportGets()).toEqual([]);
    expect(exportPosts()).toEqual([
      [
        "/pictures/export",
        {
          picture_ids: [12, 11, 13],
          only_deleted: false,
          export_type: "full",
          caption_mode: "tags",
          tag_format: "underscores",
          include_character_name: true,
          use_original_file_names: false,
          resolution: "half",
          keep_generation_data: false,
        },
      ],
    ]);
  });

  it("is a Scrapheap export when the selection was made in the Scrapheap", async () => {
    useSelectionStore().selectedCharacter = SCRAPHEAP_PICTURES_ID;
    await mountGrid();
    wrapper.vm.selectedImageIds = [21, 22];

    await exportZip();

    expect(exportGets()).toEqual([]);
    expect(exportPosts()).toHaveLength(1);
    expect(exportPosts()[0][1]).toMatchObject({
      picture_ids: [21, 22],
      only_deleted: true,
    });
  });
});

describe("a ZIP export of the whole view", () => {
  it("sends the view's query to the GET, with no ids and no POST", async () => {
    await mountGrid();

    await exportZip({ resolution: "quarter" });

    expect(exportPosts()).toEqual([]);
    const gets = exportGets();
    expect(gets).toHaveLength(1);
    const query = new URLSearchParams(gets[0].split("?")[1]);
    expect(query.getAll("id")).toEqual([]);
    expect(query.get("sort")).toBe("DATE");
    expect(query.get("resolution")).toBe("quarter");
    expect(query.get("export_type")).toBe("full");
    expect(query.has("threshold")).toBe(false);
  });

  it("of a search asks for the threshold the grid's search uses", async () => {
    // The grid's search sends searchPictures' default threshold
    // (useGridFetch passes none; pictures.test.js pins the default).
    useSearchStore().searchQuery = "zebra";
    await mountGrid();

    await exportZip();

    const gets = exportGets();
    expect(gets).toHaveLength(1);
    const query = new URLSearchParams(gets[0].split("?")[1]);
    expect(query.get("query")).toBe("zebra");
    expect(query.get("threshold")).toBe(String(SEARCH_CHANCE_CEILING));
    expect(query.has("text_match_only")).toBe(false);
  });

  it("of a search narrowed to text matches asks the search for the same", async () => {
    // The grid's search sends text_match_only with the switch on
    // (useGridFetch.test.js); the export sends it too, so the server narrows
    // both by one rule and the ZIP holds the pictures the grid shows.
    useSearchStore().searchQuery = "zebra";
    await mountGrid();
    // A new query resets the switch (useSearchStore), so it is turned on after.
    useSearchStore().textMatchesOnly = true;
    await flushPromises();

    await exportZip();

    const gets = exportGets();
    expect(gets).toHaveLength(1);
    const query = new URLSearchParams(gets[0].split("?")[1]);
    expect(query.get("query")).toBe("zebra");
    expect(query.get("text_match_only")).toBe("true");
  });
});

const DESTINATION = "/test-export-destination";
const FOLDER_SELECTION = "/pictures/export/folder/selection";

async function exportFolder(options = {}) {
  apiGet.mockClear();
  apiPost.mockClear();
  await wrapper.vm.exportCurrentViewToFolder({ destination: DESTINATION, ...options });
  await flushPromises();
}

function folderSelectionPosts() {
  return apiPost.mock.calls.filter(([url]) => url === FOLDER_SELECTION);
}

function folderQueryPosts() {
  return apiPost.mock.calls
    .map(([url]) => String(url))
    .filter((url) => url.startsWith("/pictures/export/folder?"));
}

describe("a folder export of a selection", () => {
  it("names the selected pictures and the folder in the POST body, in selection order", async () => {
    await mountGrid();
    wrapper.vm.selectedImageIds = [12, 11, 13];

    await exportFolder({ captionMode: "tags", tagFormat: "underscores", resolution: "half" });

    expect(folderQueryPosts()).toEqual([]);
    expect(folderSelectionPosts()).toEqual([
      [
        FOLDER_SELECTION,
        {
          destination: DESTINATION,
          picture_ids: [12, 11, 13],
          only_deleted: false,
          export_type: "full",
          caption_mode: "tags",
          tag_format: "underscores",
          include_character_name: true,
          use_original_file_names: false,
          resolution: "half",
          keep_generation_data: false,
        },
      ],
    ]);
  });

  it("keeps a large selection out of every URL", async () => {
    await mountGrid();
    const selection = Array.from({ length: 5000 }, (_, index) => 100000 + index);
    wrapper.vm.selectedImageIds = selection;

    await exportFolder();

    expect(folderSelectionPosts()).toHaveLength(1);
    expect(folderSelectionPosts()[0][1].picture_ids).toEqual(selection);
    const urls = [...apiPost.mock.calls, ...apiGet.mock.calls].map(([url]) => String(url));
    expect(urls.filter((url) => url.includes("id=")).length).toBe(0);
    expect(Math.max(...urls.map((url) => url.length))).toBeLessThan(200);
  });

  it("is a Scrapheap export when the selection was made in the Scrapheap", async () => {
    useSelectionStore().selectedCharacter = SCRAPHEAP_PICTURES_ID;
    await mountGrid();
    wrapper.vm.selectedImageIds = [21, 22];

    await exportFolder();

    expect(folderQueryPosts()).toEqual([]);
    expect(folderSelectionPosts()).toHaveLength(1);
    expect(folderSelectionPosts()[0][1]).toMatchObject({
      destination: DESTINATION,
      picture_ids: [21, 22],
      only_deleted: true,
    });
  });
});

describe("a folder export of the whole view", () => {
  it("sends the view's query and the folder, with no ids and no body", async () => {
    await mountGrid();

    await exportFolder({ resolution: "quarter" });

    expect(folderSelectionPosts()).toEqual([]);
    const posts = folderQueryPosts();
    expect(posts).toHaveLength(1);
    const query = new URLSearchParams(posts[0].split("?")[1]);
    expect(query.getAll("id")).toEqual([]);
    expect(query.get("destination")).toBe(DESTINATION);
    expect(query.get("sort")).toBe("DATE");
    expect(query.get("resolution")).toBe("quarter");
    expect(query.has("threshold")).toBe(false);
  });

  it("of a search asks for the threshold the grid's search uses", async () => {
    useSearchStore().searchQuery = "zebra";
    await mountGrid();

    await exportFolder();

    const posts = folderQueryPosts();
    expect(posts).toHaveLength(1);
    const query = new URLSearchParams(posts[0].split("?")[1]);
    expect(query.get("query")).toBe("zebra");
    expect(query.get("threshold")).toBe(String(SEARCH_CHANCE_CEILING));
  });

  it("of a search narrowed to text matches asks the search for the same", async () => {
    useSearchStore().searchQuery = "zebra";
    await mountGrid();
    // A new query resets the switch (useSearchStore), so it is turned on after.
    useSearchStore().textMatchesOnly = true;
    await flushPromises();

    await exportFolder();

    const posts = folderQueryPosts();
    expect(posts).toHaveLength(1);
    const query = new URLSearchParams(posts[0].split("?")[1]);
    expect(query.get("text_match_only")).toBe("true");
  });
});
