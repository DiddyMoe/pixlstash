// What the grid does with the Scrapheap's background runs.
//
// Emptying a 50,000-picture heap in one request outlasted the request's
// timeout ("Couldn't empty the scrapheap. timeout of 60000ms exceeded"), so
// the Scrapheap's gestures start a run on the server and the run store follows
// it (`stores/useScrapheapRunStore`). The grid starts the run and, once the
// run ends, says how it ended, re-reads the view and settles the selection. A
// start answered by a run that has already ended (as a run over a handful of
// pictures can be) ends at once, which is how these tests reach the outcome
// without timers.
//
// Mounts the real ImageGrid, as ImageGridEmptyScrapheapFromSidebar.test.js
// does.

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { setActivePinia, createPinia } from "pinia";
import { useSelectionStore } from "../../stores/useSelectionStore.js";
import { useProjectStore } from "../../stores/useProjectStore.js";
import { useSortStore } from "../../stores/useSortStore.js";
import { useScrapheapRunStore } from "../../stores/useScrapheapRunStore.js";
import { useNoticeStore } from "../../stores/useNoticeStore.js";
import { ref, nextTick } from "vue";

const apiGet = vi.fn();
const apiPost = vi.fn();
const apiPatch = vi.fn();
const apiPut = vi.fn();
const apiDelete = vi.fn();

// Async factory with a local `await import("vue")`: the store imports above
// pull in apiClient, so this factory runs BEFORE the file's own top-level `vue`
// import has initialised.
vi.mock("../../utils/apiClient", async () => {
  const { ref: makeRef, computed: makeComputed } = await import("vue");
  const isAuthenticated = makeRef(true);
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
    isReadOnly: makeComputed(() => false),
    login: vi.fn(),
    logout: vi.fn(),
    sessionContext,
    setRequestClientId: vi.fn(),
    API_BASE_URL: "/api/v1",
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
    push: vi.fn(),
    replace: vi.fn(),
    currentRoute: ref({ query: {} }),
  }),
}));

import ImageGrid from "./ImageGrid.vue";

const RUNS_URL = "/pictures/scrapheap/runs";

function mountGrid(view = "SCRAPHEAP") {
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

/** A run as `serialize_run` shapes it: a purge that has completed, by default. */
function endedRun(overrides = {}) {
  return {
    id: "example-run",
    kind: "purge",
    status: "complete",
    active: false,
    include_protected: false,
    cancel_requested: false,
    total: 0,
    processed: 0,
    deleted_count: 0,
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
    ...overrides,
  };
}

/** Answer a run's start with `run`; every other POST with an empty body. */
function startAnswers(run) {
  apiPost.mockImplementation((url) =>
    Promise.resolve({ data: String(url) === RUNS_URL ? run : {} }),
  );
}

function runStarts() {
  return apiPost.mock.calls.filter(([url]) => String(url) === RUNS_URL);
}

function noticeWithKey(key) {
  return useNoticeStore().notices.find((notice) => notice.key === key);
}

/** A served failure, as axios rejects with one; `detail` is the body's. */
function served(status, detail) {
  return Object.assign(
    new Error(`Request failed with status code ${status}`),
    { response: { status, data: { detail } } },
  );
}

let consoleWarn;

beforeEach(() => {
  setActivePinia(createPinia());
  apiGet.mockReset();
  apiPost.mockReset();
  apiPatch.mockReset();
  apiPut.mockReset();
  apiDelete.mockReset();
  apiGet.mockResolvedValue({ data: { pictures: [], count: 0, total: 0 } });
  apiPost.mockResolvedValue({ data: {} });
  consoleWarn = vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  consoleWarn.mockRestore();
});

describe("a Scrapheap run this tab started", () => {
  it("reports an emptied heap, warns of the snapshots it touched, and lets the outcome go", async () => {
    const snapshot = { id: 9, name: "example-snapshot", picture_count: 3 };
    startAnswers(
      endedRun({
        total: 3,
        processed: 3,
        deleted_count: 2,
        skipped_count: 1,
        snapshots_with_deleted: [snapshot],
      }),
    );
    const wrapper = mountGrid();
    await nextTick();
    wrapper.vm.deleteForeverMode = "all";
    wrapper.vm.deleteForeverConfirmToken = "test-confirmation";

    await wrapper.vm.confirmDeleteForever({ includeProtected: false });

    expect(runStarts()).toEqual([
      [
        RUNS_URL,
        {
          kind: "purge",
          include_protected: false,
          confirm_token: "test-confirmation",
        },
      ],
    ]);
    await vi.waitFor(() =>
      expect(noticeWithKey("scrapheap-run")).toMatchObject({
        level: "success",
        text:
          "Emptied the Scrapheap: 2 pictures deleted forever. " +
          "1 reference-folder original was kept.",
      }),
    );
    expect(wrapper.vm.snapshotsWithDeletedOpen).toBe(true);
    expect(wrapper.vm.snapshotsWithDeleted).toEqual([snapshot]);
    expect(wrapper.emitted("refresh-sidebar")).toHaveLength(1);
    // Acted on once: the store holds no outcome for a second watcher run.
    expect(useScrapheapRunStore().outcome).toBeNull();

    wrapper.unmount();
  });

  it("keeps the locked pictures selected after Delete forever, and says why", async () => {
    startAnswers(
      endedRun({
        total: 2,
        processed: 2,
        deleted_count: 1,
        skipped_locked_count: 1,
        skipped_locked: [2],
      }),
    );
    const wrapper = mountGrid();
    await nextTick();
    wrapper.vm.selectedImageIds = [1, 2, 3];
    wrapper.vm.lastSelectedImageId = 1;
    wrapper.vm.deleteForeverMode = "selection";
    wrapper.vm.deleteForeverIds = [1, 2];
    wrapper.vm.deleteForeverConfirmToken = "test-confirmation";

    await wrapper.vm.confirmDeleteForever({ includeProtected: false });

    expect(runStarts()[0][1]).toEqual({
      kind: "purge",
      picture_ids: [1, 2],
      include_protected: false,
      confirm_token: "test-confirmation",
    });
    await vi.waitFor(() =>
      expect(noticeWithKey("delete-skipped-locked")).toMatchObject({
        level: "warning",
        text: "1 picture deleted forever; 1 is in a locked set and was kept.",
      }),
    );
    // The locked-set notice says it all; no second card repeats it.
    expect(noticeWithKey("scrapheap-run")).toBeUndefined();
    // The destroyed picture leaves the selection; the kept one, and the one
    // never sent, stay.
    expect(wrapper.vm.selectedImageIds).toEqual([2, 3]);
    expect(wrapper.vm.lastSelectedImageId).toBeNull();

    wrapper.unmount();
  });

  it("starts Restore all as a run and reports it", async () => {
    startAnswers(
      endedRun({
        kind: "restore",
        total: 12000,
        processed: 12000,
        restored_count: 12000,
        segment_batch_ids: ["b1", "b2", "b3"],
      }),
    );
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const wrapper = mountGrid();
    await nextTick();
    // Restore all waits for something to restore.
    wrapper.vm.allGridImages = [{ id: 1, name: "test-1.png" }];
    await nextTick();

    await wrapper.vm.confirmRestoreScrapheap();

    expect(runStarts()).toEqual([[RUNS_URL, { kind: "restore" }]]);
    await vi.waitFor(() =>
      expect(noticeWithKey("scrapheap-run")).toMatchObject({
        level: "success",
        text:
          `Restored ${(12000).toLocaleString()} pictures from the Scrapheap. ` +
          "History records it as 3 steps, each undone on its own.",
      }),
    );

    confirmSpy.mockRestore();
    wrapper.unmount();
  });

  it("warns, not errs, when the start is refused for a run already going", async () => {
    const going = endedRun({ status: "running", active: true, total: 10 });
    apiPost.mockImplementation((url) =>
      String(url) === RUNS_URL
        ? Promise.reject(
            served(409, {
              code: "scrapheap_run_active",
              message: "A Scrapheap run is already going.",
              run: going,
            }),
          )
        : Promise.resolve({ data: {} }),
    );
    const wrapper = mountGrid();
    await nextTick();
    wrapper.vm.deleteForeverMode = "all";
    wrapper.vm.deleteForeverConfirmToken = "test-confirmation";

    await wrapper.vm.confirmDeleteForever({ includeProtected: false });

    expect(noticeWithKey("scrapheap-empty")).toMatchObject({
      level: "warning",
      text:
        "Another Scrapheap run is going, and its progress shows above the " +
        "grid. This can start once it ends.",
    });
    // The bar now shows the run the refusal named.
    expect(useScrapheapRunStore().run).toMatchObject({ id: going.id, active: true });

    useScrapheapRunStore().forget();
    wrapper.unmount();
  });
});

describe("a gesture while a Scrapheap run is busy", () => {
  // The server runs one at a time, so every gesture that would start another
  // is refused before any confirm opens or any request leaves: a confirmation
  // typed for a start the server would refuse is typed for nothing. Each
  // gesture has its own entry point and its own guard, so each one is driven.
  const GESTURES = [
    ["Empty scrapheap", (vm) => vm.confirmEmptyScrapheap()],
    ["Delete forever on a selection", (vm) => vm.deleteSelected()],
    ["Restore all", (vm) => vm.confirmRestoreScrapheap()],
    ["Restore on a selection", (vm) => vm.removeFromGroup()],
  ];
  const BUSY = [
    [
      "a run is going",
      (store) => {
        store.run = endedRun({ status: "running", active: true, total: 10 });
      },
      "Another Scrapheap run is going, and its progress shows above the " +
        "grid. This can start once it ends.",
    ],
    [
      "a start has no answer yet",
      (store) => {
        store.action = "start";
      },
      "The last Scrapheap request is still waiting for an answer.",
    ],
  ];
  const CASES = GESTURES.flatMap(([gesture, act]) =>
    BUSY.map(([state, makeBusy, text]) => [gesture, state, act, makeBusy, text]),
  );

  it.each(CASES)(
    "%s is refused, and says so, while %s",
    async (_gesture, _state, act, makeBusy, text) => {
      const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
      const wrapper = mountGrid();
      await nextTick();
      // Something to act on, so no gesture stops short for want of it.
      wrapper.vm.allGridImages = [{ id: 1, name: "test-1.png" }];
      wrapper.vm.selectedImageIds = [1];
      makeBusy(useScrapheapRunStore());
      await nextTick();
      apiPost.mockClear();
      apiDelete.mockClear();

      await act(wrapper.vm);
      await nextTick();

      expect(apiPost).not.toHaveBeenCalled();
      expect(apiDelete).not.toHaveBeenCalled();
      expect(confirmSpy).not.toHaveBeenCalled();
      expect(wrapper.vm.deleteForeverOpen).toBe(false);
      // Refused aloud, never dropped: a gesture that silently does nothing is
      // how the sidebar's Empty once looked like it only navigated.
      expect(noticeWithKey("scrapheap-run-busy")).toMatchObject({
        level: "info",
        text,
      });
      // The selection is left for once the run ends.
      expect(wrapper.vm.selectedImageIds).toEqual([1]);

      confirmSpy.mockRestore();
      useScrapheapRunStore().forget();
      wrapper.unmount();
    },
  );
});

describe("a Scrapheap run another tab or an earlier visit started", () => {
  const elsewhere = () => ({
    run: endedRun({ total: 4, processed: 4, deleted_count: 4 }),
    context: null,
    startedHere: false,
  });

  it("is reported in the Scrapheap, without claiming this tab's gesture", async () => {
    const wrapper = mountGrid();
    await nextTick();

    useScrapheapRunStore().outcome = elsewhere();

    await vi.waitFor(() =>
      expect(noticeWithKey("scrapheap-run")).toMatchObject({
        level: "success",
        text: "Deleted 4 pictures forever.",
      }),
    );
    expect(wrapper.vm.snapshotsWithDeletedOpen).toBe(false);
    expect(useScrapheapRunStore().outcome).toBeNull();

    wrapper.unmount();
  });

  it("is let go quietly anywhere else", async () => {
    const wrapper = mountGrid("ALL");
    await nextTick();

    useScrapheapRunStore().outcome = elsewhere();

    await vi.waitFor(() =>
      expect(useScrapheapRunStore().outcome).toBeNull(),
    );
    expect(noticeWithKey("scrapheap-run")).toBeUndefined();
    expect(wrapper.emitted("refresh-sidebar")).toHaveLength(1);

    wrapper.unmount();
  });
});

describe("Stop", () => {
  function following() {
    useScrapheapRunStore().run = endedRun({
      status: "running",
      active: true,
      total: 50000,
      processed: 1000,
    });
  }

  it("asks the server to stop the run the bar shows", async () => {
    const wrapper = mountGrid();
    await nextTick();
    following();
    apiPost.mockClear();
    apiPost.mockResolvedValue({
      data: endedRun({ status: "running", active: true, cancel_requested: true }),
    });

    await wrapper.vm.stopScrapheapRun();

    expect(apiPost.mock.calls).toEqual([[`${RUNS_URL}/example-run/cancel`]]);
    expect(noticeWithKey("scrapheap-run-stop")).toBeUndefined();

    useScrapheapRunStore().forget();
    wrapper.unmount();
  });

  it("says why a stop was refused", async () => {
    const wrapper = mountGrid();
    await nextTick();
    following();
    apiPost.mockRejectedValue(served(500, "Internal Server Error"));

    await wrapper.vm.stopScrapheapRun();

    expect(noticeWithKey("scrapheap-run-stop")).toMatchObject({
      level: "error",
      text: "Couldn't stop the run. Internal Server Error.",
    });

    useScrapheapRunStore().forget();
    wrapper.unmount();
  });
});
