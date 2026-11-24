import { describe, it, expect, beforeEach, vi } from "vitest";

// The transport, not `api/stacks`: these tests read the request bodies the real
// wrappers build, so a field the wrapper drops fails here too.
vi.mock("./apiClient", () => ({
  API_BASE_URL: "/api/v1",
  apiClient: { get: vi.fn(), post: vi.fn(), patch: vi.fn(), delete: vi.fn() },
}));

import { apiClient } from "./apiClient";
import {
  KEEP_COVER_ONLY_STACKS_PER_REQUEST,
  KeepCoverOnlyIncompleteError,
  keepCoverOnlySelection,
  mergeKeepCoverOnlyPreviews,
  mergeKeepCoverOnlyResults,
  partitionKeepCoverOnlyStacks,
  previewKeepCoverOnlySelection,
} from "./keepCoverOnlyRequests";

const PREVIEW_URL = "/stacks/keep-cover-only/preview";
const RUN_URL = "/stacks/keep-cover-only";

/** Stack ids 1..n. */
function stacks(n) {
  return Array.from({ length: n }, (_, i) => i + 1);
}

/** The bodies posted to one URL, in order. */
function bodiesTo(url) {
  return apiClient.post.mock.calls
    .filter(([called]) => called === url)
    .map(([, body]) => body);
}

/** A preview for a request of stack ids: one copy per stack, id = stack + 10000. */
function previewFor(stackIds) {
  return {
    stacks_selected: stackIds.length,
    stacks_eligible: stackIds.length,
    pictures_moving: stackIds.length,
    picture_ids_moving: stackIds.map((id) => id + 10000),
    bytes_held_by_copies: stackIds.length * 1000,
    scrapheap_retention_days: 30,
    keep_recipes: false,
  };
}

/** A collapse result for a request of stack ids, shaped like `previewFor`. */
function resultFor(stackIds) {
  return {
    status: "success",
    stacks_collapsed: stackIds.length,
    stack_ids_collapsed: stackIds,
    pictures_moved: stackIds.length,
    picture_ids_moved: stackIds.map((id) => id + 10000),
    cover_picture_ids: stackIds,
    stacks_skipped_locked: [],
    batch_id: "cli-gesture",
  };
}

beforeEach(() => {
  apiClient.post.mockReset();
  apiClient.post.mockImplementation(async (url, body) => ({
    data:
      url === PREVIEW_URL ? previewFor(body.stack_ids) : resultFor(body.stack_ids),
  }));
});

describe("partitionKeepCoverOnlyStacks", () => {
  it("splits at the request size and keeps the selection's order", () => {
    const requests = partitionKeepCoverOnlyStacks(stacks(1201));
    expect(requests.map((r) => r.length)).toEqual([500, 500, 201]);
    expect(requests.flat()).toEqual(stacks(1201));
  });

  // A repeated id would be sent twice and counted twice in the merged figure.
  it("drops repeats and anything that is not a stack id", () => {
    expect(
      partitionKeepCoverOnlyStacks([3, "3", 4, 0, -1, 2.5, NaN, null, "x", 5], 2),
    ).toEqual([[3, 4], [5]]);
  });

  it("is empty for an empty selection rather than one empty request", () => {
    expect(partitionKeepCoverOnlyStacks([])).toEqual([]);
    expect(partitionKeepCoverOnlyStacks(undefined)).toEqual([]);
  });

  it("refuses a request size that would loop forever", () => {
    expect(() => partitionKeepCoverOnlyStacks([1], 0)).toThrow(RangeError);
  });

  // The server refuses more than 2000 ids in one call.
  it("never asks the server for more than its cap", () => {
    expect(KEEP_COVER_ONLY_STACKS_PER_REQUEST).toBeLessThanOrEqual(2000);
  });
});

describe("merging reports", () => {
  it("sums counts, concatenates lists and keeps the last setting", () => {
    const merged = mergeKeepCoverOnlyPreviews([
      {
        pictures_moving: 3,
        picture_ids_moving: [1, 2, 3],
        scrapheap_retention_days: 30,
      },
      {
        pictures_moving: 2,
        picture_ids_moving: [4, 5],
        scrapheap_retention_days: 30,
      },
    ]);
    expect(merged).toEqual({
      pictures_moving: 5,
      picture_ids_moving: [1, 2, 3, 4, 5],
      // A window, not a quantity: two requests under 30 days are still 30.
      scrapheap_retention_days: 30,
    });
  });

  // `null` is the default "never empties on its own", which is a value, not an
  // absence: a later request's null must not be skipped for an earlier 30.
  it("keeps a setting's null rather than skipping it", () => {
    const merged = mergeKeepCoverOnlyPreviews([
      { scrapheap_retention_days: 30 },
      { scrapheap_retention_days: null },
    ]);
    expect(merged).toEqual({ scrapheap_retention_days: null });
  });

  it("never invents a field no report carried", () => {
    expect(mergeKeepCoverOnlyPreviews([{ pictures_moving: 1 }])).toEqual({
      pictures_moving: 1,
    });
  });

  it("merges skip rows from every request into one result", () => {
    const merged = mergeKeepCoverOnlyResults([
      { stacks_skipped_locked: [{ stack_id: 1 }], pictures_staying: 1 },
      { stacks_skipped_locked: [{ stack_id: 9 }], pictures_staying: 2 },
    ]);
    expect(merged.stacks_skipped_locked).toEqual([
      { stack_id: 1 },
      { stack_id: 9 },
    ]);
    expect(merged.pictures_staying).toBe(3);
  });
});

describe("previewKeepCoverOnlySelection", () => {
  // The bug: a Ctrl+A over a stacked library sent every stack in one preview
  // and the dialog showed "the preview could not be read".
  it("reads one preview per request, none of them past the cap", async () => {
    const { report } = await previewKeepCoverOnlySelection(stacks(1201));
    const bodies = bodiesTo(PREVIEW_URL);
    expect(bodies.map((b) => b.stack_ids.length)).toEqual([500, 500, 201]);
    expect(bodies.flatMap((b) => b.stack_ids)).toEqual(stacks(1201));
    expect(report.pictures_moving).toBe(1201);
    expect(report.stacks_eligible).toBe(1201);
    expect(report.bytes_held_by_copies).toBe(1201000);
  });

  it("hands each request the moving ids of its own preview", async () => {
    const { requests } = await previewKeepCoverOnlySelection(stacks(501));
    expect(requests).toEqual([
      {
        stackIds: stacks(500),
        expectedPictureIds: stacks(500).map((id) => id + 10000),
      },
      { stackIds: [501], expectedPictureIds: [10501] },
    ]);
  });

  it("stops between requests once aborted, and says so", async () => {
    const controller = new AbortController();
    apiClient.post.mockImplementation(async (url, body) => {
      controller.abort();
      return { data: previewFor(body.stack_ids) };
    });
    await expect(
      previewKeepCoverOnlySelection(stacks(1001), {
        signal: controller.signal,
      }),
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(bodiesTo(PREVIEW_URL)).toHaveLength(1);
  });

  // An abort must reach the request already on the wire, not only the next.
  it("passes the signal down to the transport", async () => {
    const controller = new AbortController();
    await previewKeepCoverOnlySelection([1], { signal: controller.signal });
    expect(apiClient.post).toHaveBeenCalledWith(
      PREVIEW_URL,
      { stack_ids: [1] },
      { signal: controller.signal },
    );
  });
});

describe("keepCoverOnlySelection", () => {
  it("sends every request under the one batch id, with its preview's ids", async () => {
    const { requests } = await previewKeepCoverOnlySelection(stacks(1001));
    apiClient.post.mockClear();
    const { result, operations } = await keepCoverOnlySelection(requests, {
      batchId: "cli-gesture",
    });
    const bodies = bodiesTo(RUN_URL);
    expect(bodies.map((b) => b.stack_ids.length)).toEqual([500, 500, 1]);
    expect(new Set(bodies.map((b) => b.batch_id))).toEqual(
      new Set(["cli-gesture"]),
    );
    expect(bodies[2].expected_picture_ids).toEqual([11001]);
    expect(result.pictures_moved).toBe(1001);
    expect(result.picture_ids_moved).toHaveLength(1001);
    expect(operations).toBe(3);
  });

  // A request whose stacks were all skipped records no history step, so the
  // receipt must not be told the gesture took more steps than it did.
  it("counts the history steps, not the requests", async () => {
    const { requests } = await previewKeepCoverOnlySelection(stacks(1001));
    apiClient.post.mockImplementation(async (url, body) => ({
      data: body.stack_ids.length === 1
        ? { ...resultFor([]), stacks_skipped_locked: [{ stack_id: 1001 }] }
        : resultFor(body.stack_ids),
    }));
    const { operations } = await keepCoverOnlySelection(requests, {
      batchId: "cli-gesture",
    });
    expect(operations).toBe(2);
  });

  // Without a batch id every request would mint its own, and one Ctrl+A would
  // take several Ctrl+Z to undo.
  it("refuses to run without the gesture's batch id", async () => {
    await expect(
      keepCoverOnlySelection([{ stackIds: [1], expectedPictureIds: [] }]),
    ).rejects.toThrow(TypeError);
    expect(apiClient.post).not.toHaveBeenCalled();
  });

  it("reports progress before the walk and after every request", async () => {
    const onProgress = vi.fn();
    const { requests } = await previewKeepCoverOnlySelection(stacks(1001));
    await keepCoverOnlySelection(requests, {
      batchId: "cli-gesture",
      onProgress,
    });
    expect(onProgress.mock.calls.map(([p]) => p)).toEqual([
      { stacksDone: 0, stacksTotal: 1001 },
      { stacksDone: 500, stacksTotal: 1001 },
      { stacksDone: 1000, stacksTotal: 1001 },
      { stacksDone: 1001, stacksTotal: 1001 },
    ]);
  });

  it("stops at the first failure and carries what already happened", async () => {
    const { requests } = await previewKeepCoverOnlySelection(stacks(1001));
    const refusal = Object.assign(new Error("Request failed"), {
      response: { status: 409, data: { detail: "preview again" } },
    });
    let calls = 0;
    apiClient.post.mockImplementation(async (url, body) => {
      calls += 1;
      if (calls === 2) throw refusal;
      return { data: resultFor(body.stack_ids) };
    });
    const error = await keepCoverOnlySelection(requests, {
      batchId: "cli-gesture",
    }).catch((e) => e);

    expect(error).toBeInstanceOf(KeepCoverOnlyIncompleteError);
    expect(error.cause).toBe(refusal);
    expect(error.stacksDone).toBe(500);
    expect(error.stacksTotal).toBe(1001);
    expect(error.operations).toBe(1);
    expect(error.result.picture_ids_moved).toEqual(
      stacks(500).map((id) => id + 10000),
    );
    // The server answered, so the failed request wrote nothing.
    expect(error.outcomeUnknown).toBe(false);
    // And the third request was never sent.
    expect(calls).toBe(2);
  });

  // A timeout or a dropped connection: the request may have committed.
  it("marks the outcome unknown when the failure has no response", async () => {
    apiClient.post.mockRejectedValue(new Error("timeout of 60000ms exceeded"));
    const error = await keepCoverOnlySelection(
      [{ stackIds: [1], expectedPictureIds: [10001] }],
      { batchId: "cli-gesture" },
    ).catch((e) => e);
    expect(error.outcomeUnknown).toBe(true);
    expect(error.stacksDone).toBe(0);
    expect(error.result).toEqual({});
  });
});
