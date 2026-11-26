// Keep cover only over a selection larger than one request, through the REAL
// grid.
//
// Reported bug: a Ctrl+A over a library of stacked duplicates opened the dialog
// on "The preview could not be read, so these counts are unknown. Close this
// and try again." The helpers in `utils/keepCoverOnlyRequests.js` are tested on
// their own; this file checks that the grid actually routes through them, mints
// one gesture id, and settles the grid correctly on each way a run can end.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { ref } from "vue";
import { useSelectionStore } from "../../stores/useSelectionStore";
import { useSortStore } from "../../stores/useSortStore";
import { useOperationStore } from "../../stores/useOperationStore";
import { useNoticeStore } from "../../stores/useNoticeStore";

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

const PREVIEW_URL = "/stacks/keep-cover-only/preview";
const RUN_URL = "/stacks/keep-cover-only";
const OP_TYPE = "stack.keep_cover_only";

/** Stacks 1..1201: three requests of 500, 500 and 201. */
const STACK_IDS = Array.from({ length: 1201 }, (_, i) => i + 1);
/** Each stack is shown by its cover, a two-deep stack. */
const coverOf = (stackId) => stackId * 10;
/** And holds one copy, the picture a collapse moves. */
const copyOf = (stackId) => stackId * 10 + 1;

function previewFor(stackIds) {
  return {
    stacks_selected: stackIds.length,
    stacks_eligible: stackIds.length,
    pictures_moving: stackIds.length,
    picture_ids_moving: stackIds.map(copyOf),
    bytes_held_by_copies: 0,
    scrapheap_retention_days: null,
    keep_recipes: false,
  };
}

function resultFor(stackIds) {
  return {
    status: "success",
    stacks_collapsed: stackIds.length,
    stack_ids_collapsed: stackIds,
    pictures_moved: stackIds.length,
    picture_ids_moved: stackIds.map(copyOf),
    cover_picture_ids: stackIds.map(coverOf),
    stacks_skipped_locked: [],
  };
}

/** A request the server answered with a refusal. */
function refusal(status, detail) {
  return Object.assign(new Error(`Request failed with status code ${status}`), {
    response: { status, data: { detail } },
  });
}

/** A request whose answer never arrived. */
function lostAnswer() {
  return Object.assign(new Error("Network Error"), { code: "ERR_NETWORK" });
}

function bodiesTo(url) {
  return apiPost.mock.calls.filter(([called]) => called === url).map(([, body]) => body);
}

function gridQueryCount() {
  return apiGet.mock.calls.filter(([url]) => String(url).includes("/pictures/count"))
    .length;
}

function noticeTexts() {
  return useNoticeStore().notices.map((notice) => notice.text);
}

let wrapper;
let setup;

beforeEach(() => {
  setActivePinia(createPinia());
  useSelectionStore().selectedCharacter = "ALL";
  useSortStore().selectedSort = "DATE";
  apiGet.mockReset().mockImplementation((url) => Promise.resolve({
    data: String(url).includes("/count") ? { count: 0 } : [],
  }));
  apiPost.mockReset().mockImplementation(async (url, body) => ({
    data: url === PREVIEW_URL ? previewFor(body.stack_ids) : resultFor(body.stack_ids),
  }));
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  vi.restoreAllMocks();
});

/** Mount the grid holding every stack's cover, all of them selected (Ctrl+A). */
async function selectEveryStack() {
  wrapper = mount(ImageGrid, {
    shallow: true,
    props: { backendUrl: "/api/v1" },
    global: {
      stubs: { "v-icon": true, "v-snackbar": true },
      config: { compilerOptions: { isCustomElement: (tag) => tag.startsWith("v-") } },
    },
  });
  await flushPromises();
  setup = wrapper.vm.$.setupState;
  const covers = STACK_IDS.map((stackId) => ({
    id: coverOf(stackId),
    stack_id: stackId,
    stack_position: 0,
    stack_count: 2,
    tags: [],
  }));
  wrapper.vm.allGridImages = covers;
  setup.lastFetchedGridImages = covers;
  wrapper.vm.selectedImageIds = covers.map((cover) => cover.id);
  await flushPromises();
  apiGet.mockClear();
}

/** Open the dialog from the Edit stack menu and wait for its preview. */
async function openDialog() {
  await setup.openKeepCoverOnly();
  await flushPromises();
}

describe("Keep cover only over a selection of 1,201 stacks", () => {
  it("previews in requests the server accepts and adds the figures up", async () => {
    await selectEveryStack();
    await openDialog();

    expect(bodiesTo(PREVIEW_URL).map((body) => body.stack_ids.length)).toEqual([
      500, 500, 201,
    ]);
    expect(setup.keepCoverOnlyPreviewError).toBeNull();
    expect(setup.keepCoverOnlyPreview.pictures_moving).toBe(1201);
    expect(setup.keepCoverOnlyLoading).toBe(false);
  });

  it("collapses under one gesture id and one receipt", async () => {
    await selectEveryStack();
    await openDialog();
    const operationStore = useOperationStore();
    const noteNextReceipt = vi.spyOn(operationStore, "noteNextReceipt");

    await setup.runKeepCoverOnly();
    await flushPromises();

    const runs = bodiesTo(RUN_URL);
    expect(runs.map((body) => body.stack_ids.length)).toEqual([500, 500, 201]);
    expect(runs[0].batch_id).toMatch(/^cli-/);
    expect(new Set(runs.map((body) => body.batch_id)).size).toBe(1);
    // Each request promises exactly what its own preview listed.
    expect(runs[2].expected_picture_ids).toEqual(STACK_IDS.slice(1000).map(copyOf));

    // The receipt names the newest step; the note gives the gesture's total.
    expect(noteNextReceipt).toHaveBeenCalledWith(
      OP_TYPE,
      expect.stringContaining("In all, 1,201 pictures from 1,201 stacks"),
    );
    expect(operationStore.ghostPictureIds).toHaveLength(1201);
    expect(wrapper.vm.selectedImageIds).toEqual([]);
    expect(setup.keepCoverOnlyOpen).toBe(false);
    expect(setup.keepCoverOnlyProgress).toBeNull();
    // No grid refetch: it would take the ghosted tiles, and their undo, away.
    expect(gridQueryCount()).toBe(0);
  });

  it("mints a new gesture id for the next collapse", async () => {
    await selectEveryStack();
    await openDialog();
    await setup.runKeepCoverOnly();
    const first = bodiesTo(RUN_URL)[0].batch_id;

    wrapper.vm.selectedImageIds = [coverOf(1)];
    await flushPromises();
    await openDialog();
    await setup.runKeepCoverOnly();
    expect(bodiesTo(RUN_URL).at(-1).batch_id).not.toBe(first);
  });

  it("settles what finished when a later request is refused", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    await selectEveryStack();
    await openDialog();
    const operationStore = useOperationStore();
    const noteNextReceipt = vi.spyOn(operationStore, "noteNextReceipt");
    let runs = 0;
    apiPost.mockImplementation(async (url, body) => {
      runs += 1;
      if (runs === 2) throw refusal(423, "Stack 600 is in a locked set");
      return { data: resultFor(body.stack_ids) };
    });

    await setup.runKeepCoverOnly();
    await flushPromises();

    // The first 500 really moved: ghosted, out of the selection, one undo away.
    expect(operationStore.ghostPictureIds).toEqual(STACK_IDS.slice(0, 500).map(copyOf));
    expect(wrapper.vm.selectedImageIds).toEqual(STACK_IDS.slice(500).map(coverOf));
    expect(noteNextReceipt).toHaveBeenCalledWith(OP_TYPE, expect.any(String));
    // The rest are the user's to finish, and the notice says how.
    expect(noticeTexts()).toEqual([
      "Collapsed 500 of 1,201 stacks, then stopped. Stack 600 is in a locked set. "
        + "The other 701 are still selected: run Keep cover only again to finish, "
        + "or press Ctrl+Z to take back what was done.",
    ]);
    expect(setup.keepCoverOnlyOpen).toBe(false);
    expect(gridQueryCount()).toBe(0);
  });

  it("re-reads the grid when an answer was lost", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    await selectEveryStack();
    await openDialog();
    const noteNextReceipt = vi.spyOn(useOperationStore(), "noteNextReceipt");
    let runs = 0;
    apiPost.mockImplementation(async (url, body) => {
      runs += 1;
      if (runs === 2) throw lostAnswer();
      return { data: resultFor(body.stack_ids) };
    });

    await setup.runKeepCoverOnly();
    await flushPromises();

    expect(noticeTexts()[0]).toMatch(
      /^Collapsed 500 of 1,201 stacks, then lost the server's answer/,
    );
    // The second request may have committed, so totals are not claimed...
    expect(noteNextReceipt).not.toHaveBeenCalled();
    // ...and the grid is read again instead of trusted, before the notice
    // says it was.
    expect(gridQueryCount()).toBeGreaterThan(0);
    expect(noticeTexts()[0]).toContain(
      "The grid has been reloaded to show where things stand.",
    );
  });

  it("says the grid could not be reloaded when that re-read fails too", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    await selectEveryStack();
    await openDialog();
    let runs = 0;
    apiPost.mockImplementation(async (url, body) => {
      runs += 1;
      if (runs === 2) {
        // The connection is gone, so the grid's re-read fails with it.
        apiGet.mockImplementation(() => Promise.reject(lostAnswer()));
        throw lostAnswer();
      }
      return { data: resultFor(body.stack_ids) };
    });

    await setup.runKeepCoverOnly();
    await flushPromises();

    expect(gridQueryCount()).toBeGreaterThan(0);
    expect(noticeTexts()).toHaveLength(1);
    expect(noticeTexts()[0]).toContain(
      "The grid could not be reloaded either, so reload the page to see where things stand.",
    );
    expect(noticeTexts()[0]).not.toContain("has been reloaded");
  });

  it("asks again when the first request finds the plan grew", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    await selectEveryStack();
    await openDialog();
    apiPost.mockImplementation(async (url, body) => {
      if (url === RUN_URL) throw refusal(409, "The plan changed since the preview");
      return { data: previewFor(body.stack_ids) };
    });
    apiPost.mockClear();

    await setup.runKeepCoverOnly();
    await flushPromises();

    // Nothing moved, so the dialog stays open on a fresh preview.
    expect(bodiesTo(RUN_URL)).toHaveLength(1);
    expect(bodiesTo(PREVIEW_URL)).toHaveLength(3);
    expect(setup.keepCoverOnlyOpen).toBe(true);
    expect(setup.keepCoverOnlyPreview.pictures_moving).toBe(1201);
    expect(noticeTexts()).toEqual([
      "Couldn't collapse those stacks. The plan changed since the preview.",
    ]);
  });
});

describe("a Keep cover only preview that fails", () => {
  it("keeps the server's classification and asks again on Try again", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    await selectEveryStack();
    apiPost.mockRejectedValueOnce(refusal(500, "database is locked"));
    await openDialog();

    expect(setup.keepCoverOnlyPreviewError).toMatchObject({
      kind: "server",
      status: 500,
      detail: "database is locked",
    });
    // Nothing to confirm with: a failed read is never a screen of zeroes.
    expect(setup.keepCoverOnlyPreview).toBeNull();

    apiPost.mockClear();
    await setup.loadKeepCoverOnlyPreview();
    await flushPromises();
    // Try again asks about the same stacks the dialog opened on.
    expect(bodiesTo(PREVIEW_URL).flatMap((body) => body.stack_ids)).toEqual(STACK_IDS);
    expect(setup.keepCoverOnlyPreviewError).toBeNull();
    expect(setup.keepCoverOnlyPreview.pictures_moving).toBe(1201);
  });

  it("stops asking once the dialog is closed", async () => {
    await selectEveryStack();
    let signal;
    let answer;
    apiPost.mockImplementation((url, body, config) => {
      signal = config?.signal;
      return new Promise((resolve) => {
        answer = () => resolve({ data: previewFor(body.stack_ids) });
      });
    });
    const opening = setup.openKeepCoverOnly();
    await flushPromises();
    expect(signal?.aborted).toBe(false);

    setup.closeKeepCoverOnly();
    expect(signal.aborted).toBe(true);
    answer();
    await opening;
    await flushPromises();

    // The first request's late answer is discarded, and no second is sent.
    expect(bodiesTo(PREVIEW_URL)).toHaveLength(1);
    expect(setup.keepCoverOnlyPreview).toBeNull();
    expect(setup.keepCoverOnlyOpen).toBe(false);
  });

  // One request, answered anyway: the abort cannot stop a walk that has
  // nothing left to send, so only the dialog's own token keeps the figures of a
  // dismissed dialog out of the next one.
  it("discards a last answer that arrives after the dialog closed", async () => {
    await selectEveryStack();
    wrapper.vm.selectedImageIds = [coverOf(1)];
    await flushPromises();
    let answer;
    apiPost.mockImplementation((url, body) => new Promise((resolve) => {
      answer = () => resolve({ data: previewFor(body.stack_ids) });
    }));
    const opening = setup.openKeepCoverOnly();
    await flushPromises();

    setup.closeKeepCoverOnly();
    answer();
    await opening;
    await flushPromises();

    expect(setup.keepCoverOnlyPreview).toBeNull();
    expect(setup.keepCoverOnlyLoading).toBe(false);
  });
});
