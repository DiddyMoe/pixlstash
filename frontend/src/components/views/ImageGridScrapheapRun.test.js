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
import { useUserPrefsStore } from "../../stores/useUserPrefsStore.js";
import {
  activeConfirm,
  registerConfirmHost,
  resolveConfirm,
  unregisterConfirmHost,
} from "../../composables/useConfirm";
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
  // Navigation resolves, as the router's does: closing the lightbox waits on it.
  useRouter: () => ({
    push: vi.fn(() => Promise.resolve()),
    replace: vi.fn(() => Promise.resolve()),
    currentRoute: ref({ query: {} }),
  }),
}));

import ImageGrid from "./ImageGrid.vue";

const RUNS_URL = "/pictures/scrapheap/runs";
/** A start asks the server to answer a short run at once. */
const START_OPTIONS = { headers: { Prefer: "wait=1" } };

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
  // The runs a tab started outlive its store, as they do a reload.
  sessionStorage.clear();
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
  unregisterConfirmHost();
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
          // The gesture's own id: an unanswered start is resent as the same
          // request, and the server answers it with the run it started.
          request_id: "cli-test-gesture",
        },
        START_OPTIONS,
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
    // The store offers it (App.vue mounts the dialog), so it survives a
    // change of view.
    expect(useScrapheapRunStore().snapshotsWithDeletedOpen).toBe(true);
    expect(useScrapheapRunStore().snapshotsWithDeleted).toEqual([snapshot]);
    expect(wrapper.emitted("refresh-sidebar")).toHaveLength(1);
    // Acted on once: the store holds no outcome for a second watcher run.
    expect(useScrapheapRunStore().outcome).toBeNull();

    wrapper.unmount();
  });

  it("keeps the heap's cards up until the re-read after an Empty answers", async () => {
    // A run can leave pictures behind (locked, protected, a Stop). Clearing
    // the grid before the re-read blanked the whole heap for its round trip.
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
    // The view's first read has run, so the re-read is of the view on screen.
    await vi.waitFor(() =>
      expect(
        apiGet.mock.calls.some(([url]) =>
          String(url).startsWith("/pictures/count?"),
        ),
      ).toBe(true),
    );
    await vi.waitFor(() => expect(wrapper.vm.imagesLoading).toBe(false));
    wrapper.vm.allGridImages = [
      { id: 1, idx: 0, thumbnail: "thumb-1" },
      { id: 2, idx: 1, thumbnail: "thumb-2" },
    ];
    wrapper.vm.selectedImageIds = [1, 2];
    let answerCount;
    const heldCount = new Promise((resolve) => {
      answerCount = resolve;
    });
    apiGet.mockImplementation((url) => {
      const path = String(url);
      if (path.startsWith("/pictures/count?")) return heldCount;
      if (path.startsWith("/pictures/stream?")) {
        return Promise.resolve({ data: { pictures: [{ id: 2 }] } });
      }
      return Promise.resolve({ data: { pictures: [], count: 0, total: 0 } });
    });
    wrapper.vm.deleteForeverMode = "all";
    wrapper.vm.deleteForeverConfirmToken = "test-confirmation";

    await wrapper.vm.confirmDeleteForever({ includeProtected: false });
    await vi.waitFor(() =>
      expect(noticeWithKey("scrapheap-run")).toBeDefined(),
    );

    expect(wrapper.vm.selectedImageIds).toEqual([]);
    expect(wrapper.vm.allGridImages.map((row) => row.id)).toEqual([1, 2]);

    answerCount({ data: { count: 1 } });
    await vi.waitFor(() =>
      expect(wrapper.vm.allGridImages.map((row) => row.id)).toEqual([2]),
    );
    // The card that stayed kept its thumbnail through the swap.
    expect(wrapper.vm.allGridImages[0].thumbnail).toBe("thumb-2");

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
      request_id: "cli-test-gesture",
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
    // App.vue mounts the app's confirm host; the browser's box never opens.
    registerConfirmHost();
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const wrapper = mountGrid();
    await nextTick();
    // Restore all waits for something to restore.
    wrapper.vm.allGridImages = [{ id: 1, name: "test-1.png" }];
    await nextTick();

    const asked = wrapper.vm.confirmRestoreScrapheap();
    await nextTick();
    expect(activeConfirm.value?.options).toMatchObject({
      title: "Restore all pictures?",
      confirmLabel: "Restore all",
      danger: false,
    });
    expect(runStarts()).toEqual([]);
    resolveConfirm(true);
    await asked;

    expect(confirmSpy).not.toHaveBeenCalled();
    expect(runStarts()).toEqual([
      [
        RUNS_URL,
        { kind: "restore", request_id: "cli-test-gesture" },
        START_OPTIONS,
      ],
    ]);
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

  it("asks Restore all with the heap's count only when the count is the whole heap's", async () => {
    registerConfirmHost();
    const wrapper = mountGrid();
    await nextTick();
    wrapper.vm.allGridImages = [{ id: 1, name: "test-1.png" }];
    // The Scrapheap summary; the one loaded card is no count of the heap.
    wrapper.vm.totalCurrentCategoryCount = 12000;
    await nextTick();

    const counted = wrapper.vm.confirmRestoreScrapheap();
    await nextTick();
    expect(activeConfirm.value?.options.message).toBe(
      `Restore all ${(12000).toLocaleString()} pictures in the Scrapheap? ` +
        "They return to the library.",
    );
    resolveConfirm(false);
    await counted;

    // With hidden tags applied the summary leaves those pictures out, and
    // Restore all restores them too: no number rather than a wrong one.
    useUserPrefsStore().applyTagFilter = true;
    await nextTick();
    wrapper.vm.totalCurrentCategoryCount = 12000;
    const uncounted = wrapper.vm.confirmRestoreScrapheap();
    await nextTick();
    expect(activeConfirm.value?.options.message).toBe(
      "Restore every picture in the Scrapheap? They return to the library.",
    );
    resolveConfirm(false);
    await uncounted;

    expect(runStarts()).toEqual([]);
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

  // The store raises the notice when it takes the run, in any view
  // (useScrapheapRunStore.test.js, "announcing an end, in any view"). The
  // outcome is set by hand here, so what is left is the grid's own part.
  it("is let go in the Scrapheap, leaving the notice to the store", async () => {
    const wrapper = mountGrid();
    await nextTick();

    useScrapheapRunStore().outcome = elsewhere();

    await vi.waitFor(() => expect(useScrapheapRunStore().outcome).toBeNull());
    expect(noticeWithKey("scrapheap-run")).toBeUndefined();
    expect(useScrapheapRunStore().snapshotsWithDeletedOpen).toBe(false);
    expect(wrapper.emitted("refresh-sidebar")).toHaveLength(1);

    wrapper.unmount();
  });

  it("is let go anywhere else", async () => {
    const wrapper = mountGrid("ALL");
    await nextTick();

    useScrapheapRunStore().outcome = elsewhere();

    await vi.waitFor(() => expect(useScrapheapRunStore().outcome).toBeNull());
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

describe("a restore from the lightbox", () => {
  // A restore run's slices are minted by the server (`srv-`); the lightbox's
  // own id is what keeps its pill from being taken for one of them.
  it("sends its own gesture id", async () => {
    const wrapper = mountGrid();
    await nextTick();
    wrapper.vm.lightboxCtxImage = { id: 7, name: "test-7.png" };

    await wrapper.vm.handleLightboxScrapheapRestore();

    expect(apiPost).toHaveBeenCalledWith(
      "/pictures/scrapheap/restore",
      { picture_ids: [7] },
      { headers: { "X-Operation-Batch-Id": "cli-test-gesture" } },
    );
    wrapper.unmount();
  });

  it("keeps the picture and the lightbox when the restore fails", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const wrapper = mountGrid();
    await nextTick();
    wrapper.vm.allGridImages = [{ id: 7, name: "test-7.png" }];
    wrapper.vm.lightboxOpen = true;
    wrapper.vm.lightboxCtxImage = { id: 7, name: "test-7.png" };
    apiPost.mockRejectedValueOnce(served(500, "disk full"));

    await wrapper.vm.handleLightboxScrapheapRestore();
    await nextTick();

    expect(noticeWithKey("scrapheap-restore-lightbox")).toMatchObject({
      level: "error",
    });
    expect(wrapper.vm.allGridImages.map((row) => row.id)).toEqual([7]);
    expect(wrapper.vm.lightboxOpen).toBe(true);
    expect(wrapper.emitted("refresh-sidebar")).toBeUndefined();
    errorSpy.mockRestore();
    wrapper.unmount();
  });
});

describe("a key pressed in a confirm over the Scrapheap", () => {
  // The confirm dialogs keep only Escape and Enter for themselves, so every
  // other key reaches the window, where the grid's shortcuts listen. Delete on
  // the confirm's own "Delete forever" button used to start a second delete
  // preview, which blanks the confirm token first: the confirm then sent no
  // token, the server refused it, and nothing was deleted. The DOM the dialog
  // puts up is what Vuetify renders for an open dialog (an active overlay with a
  // scrim); AppDialog.keys.test.js pins that against the real components.
  const PREVIEW_URL = "/pictures/scrapheap/delete-preview";

  function previews() {
    return apiPost.mock.calls.filter(([url]) => String(url) === PREVIEW_URL);
  }

  function openModal() {
    const overlay = document.createElement("div");
    overlay.className = "v-overlay v-overlay--active v-dialog";
    overlay.innerHTML =
      '<div class="v-overlay__scrim"></div>' +
      '<div class="v-overlay__content"><button type="button">Delete forever</button></div>';
    document.body.appendChild(overlay);
    return overlay;
  }

  function pressOn(target, key) {
    target.dispatchEvent(
      new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }),
    );
  }

  it("leaves Delete forever's token alone, and the confirm still deletes", async () => {
    let minted = 0;
    apiPost.mockImplementation((url) => {
      if (String(url) === PREVIEW_URL) {
        minted += 1;
        return Promise.resolve({
          data: {
            confirm_token: `test-preview-${minted}`,
            total_count: 2,
            protected_count: 0,
            unprotected_count: 2,
            protected: [],
            locked_count: 0,
          },
        });
      }
      if (String(url) === RUNS_URL) {
        return Promise.resolve({
          data: endedRun({ total: 2, processed: 2, deleted_count: 2 }),
        });
      }
      return Promise.resolve({ data: {} });
    });
    const wrapper = mountGrid();
    await nextTick();
    wrapper.vm.allGridImages = [
      { id: 1, idx: 0, name: "test-1.png" },
      { id: 2, idx: 1, name: "test-2.png" },
    ];
    wrapper.vm.selectedImageIds = [1, 2];
    await nextTick();

    // Delete on the grid asks what Delete forever would destroy.
    pressOn(document.body, "Delete");
    await vi.waitFor(() => expect(wrapper.vm.deleteForeverOpen).toBe(true));
    expect(previews()).toHaveLength(1);
    expect(wrapper.vm.deleteForeverConfirmToken).toBe("test-preview-1");

    // The confirm is up, focus on its button: Delete and Backspace are its.
    const overlay = openModal();
    const button = overlay.querySelector("button");
    button.focus();
    pressOn(button, "Delete");
    pressOn(button, "Backspace");
    await nextTick();

    expect(previews()).toHaveLength(1);
    expect(wrapper.vm.deleteForeverConfirmToken).toBe("test-preview-1");

    await wrapper.vm.confirmDeleteForever({ includeProtected: false });
    overlay.remove();

    expect(runStarts()).toHaveLength(1);
    expect(runStarts()[0][1]).toMatchObject({
      kind: "purge",
      picture_ids: [1, 2],
      confirm_token: "test-preview-1",
    });

    wrapper.unmount();
  });

  it("does not open Delete forever behind another confirm", async () => {
    registerConfirmHost();
    const wrapper = mountGrid();
    await nextTick();
    wrapper.vm.allGridImages = [{ id: 1, idx: 0, name: "test-1.png" }];
    wrapper.vm.selectedImageIds = [1];
    wrapper.vm.totalCurrentCategoryCount = 1;
    await nextTick();

    // Restore all asks first; its dialog is the one on screen.
    const asked = wrapper.vm.confirmRestoreScrapheap();
    await nextTick();
    expect(activeConfirm.value).not.toBeNull();
    const overlay = openModal();
    pressOn(overlay.querySelector("button"), "Delete");
    await nextTick();
    await nextTick();

    expect(previews()).toEqual([]);
    expect(wrapper.vm.deleteForeverOpen).toBe(false);

    resolveConfirm(false);
    await asked;
    overlay.remove();
    wrapper.unmount();
  });
});
