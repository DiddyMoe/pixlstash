// An export is followed until the server says it is over: completed, failed,
// or no longer known to it. No count of polls ends it, so a big export that is
// still running is never reported as timed out. The Abort button and leaving
// the grid stop the follow at once, and a server that cannot be reached for a
// while is asked again. Mounts the real grid the way
// ImageGridLibraryMoveChunks.test.js does and drives the two export entry
// points the selection pill calls (ZIP download and folder export).

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { ref } from "vue";
import { useNoticeStore } from "../../stores/useNoticeStore";
import { useSelectionStore } from "../../stores/useSelectionStore";
import { useSortStore } from "../../stores/useSortStore";

const apiGet = vi.fn();
const apiPost = vi.fn();
// What `GET /pictures/export/status` answers, per test.
const exportStatus = vi.fn();

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
import ProgressOverlay from "../widgets/ProgressOverlay.vue";

const TASK_ID = "test-task";
const ZIP_URL = `/pictures/export/download/${TASK_ID}`;
const DESTINATION = "/home/me/exports";
const TOTAL = 5000;

// What axios rejects with: `isAxiosError`, and a `response` only when the
// server answered.
function networkError() {
  return Object.assign(new Error("Network Error"), {
    isAxiosError: true,
    code: "ERR_NETWORK",
  });
}
function httpError(status, detail) {
  return Object.assign(new Error(`Request failed with status code ${status}`), {
    isAxiosError: true,
    response: { status, data: detail === undefined ? {} : { detail } },
  });
}

function running(processed) {
  return { status: "in_progress", processed, total: TOTAL, progress: processed / TOTAL };
}

const kinds = [
  {
    name: "ZIP export",
    key: "export-zip",
    start: (vm) => vm.exportCurrentViewToZip({}),
    completed: (processed) => ({
      status: "completed",
      processed,
      total: processed,
      progress: 1,
      download_url: ZIP_URL,
    }),
    expectFinished: () => {
      expect(apiGet).toHaveBeenCalledWith(ZIP_URL, { responseType: "blob" });
      expect(HTMLAnchorElement.prototype.click).toHaveBeenCalledTimes(1);
    },
    gone:
      "Lost track of this export: the server no longer has it (it restarted, " +
      "or the library changed). Export again to download the ZIP.",
    refused: "Lost track of this export.",
    left:
      "Left the grid before the ZIP export finished, so it won't download. " +
      "Export again to get it.",
  },
  {
    name: "folder export",
    key: "export-folder",
    start: (vm) => vm.exportCurrentViewToFolder({ destination: DESTINATION }),
    completed: (processed) => ({
      status: "completed",
      processed,
      total: processed,
      progress: 1,
      opened: true,
    }),
    expectFinished: (processed) => {
      expect(notices()).toEqual([
        {
          level: "success",
          key: "export-folder",
          // Grouped as a person reads it: TOTAL is 5000, so "5,000".
          text: `Exported ${processed.toLocaleString()} pictures to ${DESTINATION}.`,
        },
      ]);
    },
    gone:
      `Lost track of the export to ${DESTINATION}: the server no longer has it ` +
      "(it restarted, or the library changed). Check that folder before " +
      "exporting again.",
    refused: `Lost track of the export to ${DESTINATION}.`,
    left:
      `Left the grid before the export to ${DESTINATION} finished. The server ` +
      "keeps writing it there.",
  },
];

let wrapper;
let noticeStore;
let posted;

beforeEach(() => {
  setActivePinia(createPinia());
  useSelectionStore().selectedCharacter = "ALL";
  useSortStore().selectedSort = "DATE";
  noticeStore = useNoticeStore();
  posted = [];
  noticeStore.$onAction(({ name, args: [text, options] }) => {
    if (["info", "success", "warning", "error"].includes(name)) {
      posted.push({ level: name, key: options?.key ?? null, text });
    }
  });
  exportStatus.mockReset();
  apiGet.mockReset().mockImplementation(async (url, config) => {
    const path = String(url);
    if (path === "/pictures/export/status") {
      return { data: await exportStatus(config?.params?.task_id) };
    }
    if (path === "/pictures/export" || path.startsWith("/pictures/export?")) {
      return { data: { task_id: TASK_ID } };
    }
    if (path === ZIP_URL) {
      return {
        data: new Blob(["test-zip"]),
        headers: { "content-disposition": 'attachment; filename="test-export.zip"' },
      };
    }
    return { data: path.includes("/count") ? { count: 0 } : [] };
  });
  apiPost.mockReset().mockImplementation(async (url) => ({
    data: String(url).startsWith("/pictures/export/folder?")
      ? { task_id: TASK_ID }
      : {},
  }));
  // jsdom has no object URLs and would try to navigate on an anchor click.
  URL.createObjectURL = vi.fn(() => "blob:test-export");
  URL.revokeObjectURL = vi.fn();
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
  // The middle of each retry's jitter range, so the waits are repeatable.
  vi.spyOn(Math, "random").mockReturnValue(0.5);
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  noticeStore.clear();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

// The grid settles on real timers; only what the export does after this runs
// on fake ones, so the timer count below is the export's own.
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
  vi.useFakeTimers();
}

function card() {
  return wrapper.vm.$.setupState.exportProgress;
}

// Every notice the export posted, including ones whose countdown has run out.
function notices() {
  return posted;
}

function statusCalls() {
  return exportStatus.mock.calls.length;
}

async function advanceSeconds(seconds) {
  for (let i = 0; i < seconds; i++) await vi.advanceTimersByTimeAsync(1000);
}

// Microtasks only: a wait that ignored the abort would still be pending.
async function settleMicrotasks() {
  for (let i = 0; i < 20; i++) await Promise.resolve();
}

// A request that answers only when the test resolves or rejects it.
function hold() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

// Holds the start request of `kind` until the returned handle settles it.
function holdStart(kind) {
  const held = hold();
  if (kind.key === "export-zip") {
    const answer = apiGet.getMockImplementation();
    apiGet.mockImplementation((url, ...args) =>
      String(url).startsWith("/pictures/export?") ? held.promise : answer(url, ...args),
    );
  } else {
    apiPost.mockImplementation(() => held.promise);
  }
  return held;
}

function pressAbort() {
  wrapper.findComponent(ProgressOverlay).vm.$emit("abort");
}

describe.each(kinds)("$name", (kind) => {
  it("follows an export that keeps moving past 600 s, then finishes it", async () => {
    await mountGrid();
    const answers = 620;
    exportStatus.mockImplementation(async () =>
      statusCalls() <= answers ? running(statusCalls()) : kind.completed(answers),
    );

    const done = kind.start(wrapper.vm);
    await advanceSeconds(610);

    // Ten minutes in and still running: followed, asked about once a second
    // because every answer moved, and nothing says it failed or timed out.
    expect(statusCalls()).toBe(611);
    expect(card().visible).toBe(true);
    expect(card().status).toBe("in_progress");
    expect(card().processed).toBe(611);
    expect(card().total).toBe(TOTAL);
    expect(card().message).toBe("Exporting images...");
    expect(notices()).toEqual([]);

    await advanceSeconds(15);
    await done;

    expect(statusCalls()).toBe(answers + 1);
    kind.expectFinished(answers);
    expect(notices().filter((n) => n.level === "error")).toEqual([]);
  });

  it("keeps following an export that has not moved for over 10 min", async () => {
    await mountGrid();
    let stuck = true;
    exportStatus.mockImplementation(async () =>
      stuck ? running(40) : kind.completed(TOTAL),
    );

    const done = kind.start(wrapper.vm);
    await advanceSeconds(11 * 60);

    // Asked less often as nothing moves, never more than 10 s apart, and still
    // running as far as anyone can tell.
    expect(statusCalls()).toBeGreaterThan(60);
    expect(statusCalls()).toBeLessThan(80);
    expect(card().visible).toBe(true);
    expect(card().status).toBe("in_progress");
    expect(card().message).toBe(
      "Still working on the server; no progress for 10 min",
    );
    expect(notices()).toEqual([]);

    stuck = false;
    await advanceSeconds(11);
    await done;

    kind.expectFinished(TOTAL);
  });

  it("stops at once on Abort: no further status request, no timer left", async () => {
    await mountGrid();
    exportStatus.mockImplementation(async () => running(statusCalls()));

    const done = kind.start(wrapper.vm);
    await advanceSeconds(3);
    expect(statusCalls()).toBe(4);

    wrapper.findComponent(ProgressOverlay).vm.$emit("abort");
    await settleMicrotasks();

    expect(card().cancelRequested).toBe(true);
    expect(card().status).toBe("cancelled");
    expect(card().message).toBe("Export cancelled.");
    expect(card().visible).toBe(false);
    expect(vi.getTimerCount()).toBe(0);
    await done;

    await vi.advanceTimersByTimeAsync(10 * 60 * 1000);
    expect(statusCalls()).toBe(4);
    expect(notices()).toEqual([]);
  });

  it("stops when the grid closes: no further status request, no timer left", async () => {
    await mountGrid();
    exportStatus.mockImplementation(async () => running(statusCalls()));

    const done = kind.start(wrapper.vm);
    await advanceSeconds(3);
    expect(statusCalls()).toBe(4);

    wrapper.unmount();
    wrapper = null;
    await settleMicrotasks();

    const told = notices();
    // The notice's own countdown is the only timer left; it is not the export's.
    noticeStore.clear();
    expect(vi.getTimerCount()).toBe(0);
    await vi.advanceTimersByTimeAsync(10 * 60 * 1000);
    expect(statusCalls()).toBe(4);
    await done;
    expect(told).toEqual([{ level: "warning", key: kind.key, text: kind.left }]);
  });

  it("rides out a server that cannot be reached for a while", async () => {
    await mountGrid();
    const replies = [
      running(10),
      networkError(),
      httpError(503),
      Object.assign(new Error("timeout of 60000ms exceeded"), {
        isAxiosError: true,
        code: "ECONNABORTED",
      }),
      running(20),
      kind.completed(TOTAL),
    ];
    exportStatus.mockImplementation(async () => {
      const reply = replies.shift();
      if (reply instanceof Error) throw reply;
      return reply;
    });

    const done = kind.start(wrapper.vm);
    await advanceSeconds(1);
    expect(statusCalls()).toBe(2);

    // Still running for all anyone knows: the card says so and stays up.
    expect(card().visible).toBe(true);
    expect(card().status).toBe("in_progress");
    expect(card().message).toBe("Reconnecting to the server...");
    expect(notices()).toEqual([]);

    await advanceSeconds(10);
    await done;

    expect(statusCalls()).toBe(6);
    kind.expectFinished(TOTAL);
    expect(notices().filter((n) => n.level === "error")).toEqual([]);
  });

  it("ends on the server's refusal and says its reason", async () => {
    await mountGrid();
    const reason = "This share link can no longer see that export.";
    exportStatus
      .mockResolvedValueOnce(running(5))
      .mockRejectedValueOnce(httpError(403, reason));

    const done = kind.start(wrapper.vm);
    await advanceSeconds(1);

    expect(statusCalls()).toBe(2);
    expect(notices()).toEqual([
      { level: "warning", key: kind.key, text: `${kind.refused} ${reason}` },
    ]);
    // Not a failure card: nothing says the export failed.
    expect(card().visible).toBe(false);
    expect(card().status).toBe("idle");
    noticeStore.clear();
    expect(vi.getTimerCount()).toBe(0);
    await vi.advanceTimersByTimeAsync(60 * 1000);
    expect(statusCalls()).toBe(2);
    await done;
  });

  it("says it lost track when the server no longer has the export", async () => {
    await mountGrid();
    exportStatus
      .mockResolvedValueOnce(running(5))
      .mockRejectedValueOnce(httpError(404, "Task not found"));

    const done = kind.start(wrapper.vm);
    await advanceSeconds(1);

    expect(statusCalls()).toBe(2);
    expect(notices()).toEqual([{ level: "warning", key: kind.key, text: kind.gone }]);
    expect(card().visible).toBe(false);
    expect(card().status).toBe("idle");
    await vi.advanceTimersByTimeAsync(60 * 1000);
    expect(statusCalls()).toBe(2);
    await done;
  });

  it("reports a failure only when the server says the export failed", async () => {
    await mountGrid();
    exportStatus
      .mockResolvedValueOnce(running(5))
      .mockResolvedValueOnce({ status: "failed", processed: 5, total: TOTAL, progress: 0 });

    const done = kind.start(wrapper.vm);
    await advanceSeconds(1);

    expect(statusCalls()).toBe(2);
    expect(card().status).toBe("failed");
    expect(card().message).toBe("Export failed");
    expect(notices()).toEqual([
      { level: "error", key: kind.key, text: "Export failed. Export failed on server." },
    ]);
    await done;
  });

  // A request can hang for the client's 60 s timeout, which is when Abort or
  // leaving the grid is wanted most: neither waits for its answer.
  it("Abort with a status request pending: card cancelled at once, late answer dropped, nothing posted", async () => {
    await mountGrid();
    const held = hold();
    exportStatus.mockImplementation(() => held.promise);
    const done = kind.start(wrapper.vm);
    await settleMicrotasks();
    expect(statusCalls()).toBe(1);

    pressAbort();
    await settleMicrotasks();
    const atAbort = { visible: card().visible, status: card().status };

    held.resolve(kind.completed(TOTAL));
    await done;
    await vi.advanceTimersByTimeAsync(5000);
    expect(atAbort).toEqual({ visible: false, status: "cancelled" });
    expect(card().status).toBe("cancelled");
    expect(HTMLAnchorElement.prototype.click).toHaveBeenCalledTimes(0);
    expect(notices()).toEqual([]);
    expect(statusCalls()).toBe(1);
  });

  it("Abort with the start request pending: card closed at once, no status request after", async () => {
    await mountGrid();
    const held = holdStart(kind);
    const done = kind.start(wrapper.vm);
    await settleMicrotasks();

    pressAbort();
    await settleMicrotasks();
    const atAbort = { visible: card().visible, status: card().status };

    held.resolve({ data: { task_id: TASK_ID } });
    await done;
    await vi.advanceTimersByTimeAsync(5000);
    expect(atAbort).toEqual({ visible: false, status: "cancelled" });
    expect(statusCalls()).toBe(0);
    expect(notices()).toEqual([]);
  });

  it("Abort with the start request pending, which then fails: no failure posted", async () => {
    await mountGrid();
    const held = holdStart(kind);
    const done = kind.start(wrapper.vm);
    await settleMicrotasks();

    pressAbort();
    await settleMicrotasks();
    held.reject(httpError(500, "test-server-error"));
    await done;
    await vi.advanceTimersByTimeAsync(5000);
    expect(notices()).toEqual([]);
    expect(card().status).toBe("cancelled");
  });

  it("unmount with a status request pending: one warning, posted at once", async () => {
    await mountGrid();
    const held = hold();
    exportStatus.mockImplementation(() => held.promise);
    const done = kind.start(wrapper.vm);
    await settleMicrotasks();

    wrapper.unmount();
    wrapper = null;
    await settleMicrotasks();
    const atUnmount = [...notices()];

    held.resolve(kind.completed(TOTAL));
    await done;
    await vi.advanceTimersByTimeAsync(5000);
    const left = [{ level: "warning", key: kind.key, text: kind.left }];
    expect(atUnmount).toEqual(left);
    expect(notices()).toEqual(left);
    expect(HTMLAnchorElement.prototype.click).toHaveBeenCalledTimes(0);
  });

  it("unmount with the start request pending: one warning, posted at once, no status request", async () => {
    await mountGrid();
    const held = holdStart(kind);
    const done = kind.start(wrapper.vm);
    await settleMicrotasks();

    wrapper.unmount();
    wrapper = null;
    await settleMicrotasks();
    const atUnmount = [...notices()];

    held.resolve({ data: { task_id: TASK_ID } });
    await done;
    const left = [{ level: "warning", key: kind.key, text: kind.left }];
    expect(atUnmount).toEqual(left);
    expect(notices()).toEqual(left);
    expect(statusCalls()).toBe(0);
  });

  it("Abort during one start, then a second export: the first does not claim the grid was left", async () => {
    await mountGrid();
    const held = holdStart(kind);
    const first = kind.start(wrapper.vm);
    await settleMicrotasks();
    pressAbort();
    await settleMicrotasks();

    // The second export's start answers at once and it keeps running.
    if (kind.key === "export-zip") {
      const holding = apiGet.getMockImplementation();
      apiGet.mockImplementation((url, ...args) =>
        String(url).startsWith("/pictures/export?")
          ? Promise.resolve({ data: { task_id: TASK_ID } })
          : holding(url, ...args),
      );
    } else {
      apiPost.mockImplementation(async () => ({ data: { task_id: TASK_ID } }));
    }
    exportStatus.mockResolvedValue(running(10));
    const second = kind.start(wrapper.vm);
    await settleMicrotasks();

    held.resolve({ data: { task_id: TASK_ID } });
    await first;
    await settleMicrotasks();
    expect(notices()).toEqual([]);
    expect(card().visible).toBe(true);

    pressAbort();
    await settleMicrotasks();
    await vi.advanceTimersByTimeAsync(2000);
    await second;
  });
});

// Once the server has built the ZIP, the transfer is no longer a follow: the
// file is saved even if the grid closes meanwhile.
it("ZIP export: leaving the grid during the download still saves the file and says nothing contrary", async () => {
  await mountGrid();
  exportStatus.mockResolvedValue(kinds[0].completed(TOTAL));
  const held = hold();
  const answer = apiGet.getMockImplementation();
  apiGet.mockImplementation((url, ...args) =>
    url === ZIP_URL ? held.promise : answer(url, ...args),
  );
  const done = kinds[0].start(wrapper.vm);
  await settleMicrotasks();
  expect(apiGet).toHaveBeenCalledWith(ZIP_URL, { responseType: "blob" });

  wrapper.unmount();
  wrapper = null;
  await settleMicrotasks();
  held.resolve({ data: new Blob(["test-zip"]), headers: {} });
  await done;
  await vi.advanceTimersByTimeAsync(3000);
  expect(HTMLAnchorElement.prototype.click).toHaveBeenCalledTimes(1);
  expect(notices()).toEqual([]);
});
