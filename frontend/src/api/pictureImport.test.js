import { describe, it, expect, beforeEach, vi } from "vitest";

// Pattern for API-module tests: mock the singleton apiClient, assert the module
// builds the right URL and returns the mapped body (never the axios envelope).
vi.mock("../utils/apiClient", () => ({
  API_BASE_URL: "/api/v1",
  apiClient: { get: vi.fn(), post: vi.fn(), delete: vi.fn() },
}));

import { apiClient } from "../utils/apiClient";
import {
  commitStaging,
  fetchStagingStatus,
  IMPORT_ENDPOINTS,
  openStagingSession,
  stageFiles,
} from "./pictureImport";

beforeEach(() => {
  apiClient.get.mockReset();
  apiClient.post.mockReset();
});

describe("api/pictureImport: opening a staging session", () => {
  it("names the library the server opened the session in", async () => {
    apiClient.post.mockResolvedValue({
      data: { staging_id: "s1", safe_threshold: 3, library_uuid: "lib-9" },
    });
    const session = await openStagingSession({ backendUrl: "/be" });
    expect(session).toMatchObject({
      stagingId: "s1",
      safeThreshold: 3,
      libraryUuid: "lib-9",
    });
  });

  it("names no library when an older server does not say", async () => {
    apiClient.post.mockResolvedValue({ data: { staging_id: "s1" } });
    const session = await openStagingSession({ backendUrl: "/be" });
    expect(session.libraryUuid).toBe("");
  });
});

describe("api/pictureImport: resending a batch", () => {
  it("sends the batch's key as a quoted Idempotency-Key header", async () => {
    apiClient.post.mockResolvedValue({ data: { staged: 1 } });
    await stageFiles({
      backendUrl: "/be",
      stagingId: "s1",
      files: [new File(["x"], "a.png")],
      idempotencyKey: "batch-key-1",
    });
    const config = apiClient.post.mock.calls[0][2];
    // The server reads a structured-field string: the quotes are part of it.
    expect(config.headers["Idempotency-Key"]).toBe('"batch-key-1"');
    expect(config.headers["Content-Type"]).toBe("multipart/form-data");
  });

  it("sends no Idempotency-Key when the caller has none", async () => {
    apiClient.post.mockResolvedValue({ data: { staged: 1 } });
    await stageFiles({
      backendUrl: "/be",
      stagingId: "s1",
      files: [new File(["x"], "a.png")],
    });
    const config = apiClient.post.mock.calls[0][2];
    expect(config.headers).not.toHaveProperty("Idempotency-Key");
  });

  it("says when the server answered a resend from its first reply", async () => {
    apiClient.post.mockResolvedValue({
      data: { staged: 2, received: ["a.png"], replayed: true },
    });
    const batch = await stageFiles({
      backendUrl: "/be",
      stagingId: "s1",
      files: [new File(["x"], "a.png")],
      idempotencyKey: "batch-key-1",
    });
    expect(batch.replayed).toBe(true);
  });

  it("says when a commit was answered from an earlier commit", async () => {
    apiClient.post.mockResolvedValue({
      data: { task_id: "t-1", staged_count: 2, replayed: true },
    });
    const commit = await commitStaging({ backendUrl: "/be", stagingId: "s1" });
    expect(commit.taskId).toBe("t-1");
    expect(commit.replayed).toBe(true);
  });
});

describe("api/pictureImport: where each file sat in the drop", () => {
  it("sends one relpath per file, in file order", async () => {
    apiClient.post.mockResolvedValue({ data: { staged: 3 } });
    await stageFiles({
      backendUrl: "/be",
      stagingId: "s1",
      files: [
        new File(["x"], "x.png"),
        new File(["x"], "x.png"),
        new File(["x"], "y.png"),
      ],
      relpaths: ["/a/x.png", "/b/x.png", ""],
    });
    const form = apiClient.post.mock.calls[0][1];
    expect(form.getAll("relpath")).toEqual(["/a/x.png", "/b/x.png", ""]);
    expect(form.getAll("file")).toHaveLength(3);
  });

  it("sends an empty relpath for each file the caller has no path for", async () => {
    apiClient.post.mockResolvedValue({ data: { staged: 2 } });
    await stageFiles({
      backendUrl: "/be",
      stagingId: "s1",
      files: [new File(["x"], "a.png"), new File(["x"], "b.png")],
    });
    const form = apiClient.post.mock.calls[0][1];
    expect(form.getAll("relpath")).toEqual(["", ""]);
  });
});

describe("api/pictureImport: what the server is doing", () => {
  it("passes the importer's phase through, null once it is over", async () => {
    apiClient.get.mockResolvedValueOnce({
      data: { stage: "importing", phase: "persisting" },
    });
    apiClient.get.mockResolvedValueOnce({ data: { stage: "completed" } });
    const running = await fetchStagingStatus({
      backendUrl: "",
      stagingId: "s",
    });
    const done = await fetchStagingStatus({ backendUrl: "", stagingId: "s" });
    expect(running.phase).toBe("persisting");
    expect(done.phase).toBeNull();
  });

  it("hands an abort signal to the request when given one", async () => {
    apiClient.get.mockResolvedValue({ data: { stage: "importing" } });
    const controller = new AbortController();
    await fetchStagingStatus({
      backendUrl: "",
      stagingId: "s",
      outcomesFrom: 3,
      signal: controller.signal,
    });
    expect(apiClient.get).toHaveBeenCalledWith(
      IMPORT_ENDPOINTS.status("", "s"),
      { params: { outcomes_from: 3 }, signal: controller.signal },
    );
  });
});

describe("api/pictureImport: staging status buckets", () => {
  it("maps every completion bucket, Scrapheap matches included", async () => {
    apiClient.get.mockResolvedValue({
      data: {
        stage: "completed",
        staged: 4,
        total: 4,
        processed: 4,
        task_id: "t-1",
        imported_count: 1,
        duplicate_count: 1,
        scrapheaped_count: 2,
        scrapheaped_picture_ids: [7],
        failed_count: 0,
        cancelled_count: 0,
      },
    });

    const status = await fetchStagingStatus({
      backendUrl: "/be",
      stagingId: "s1",
    });

    expect(apiClient.get).toHaveBeenCalledWith(
      IMPORT_ENDPOINTS.status("/be", "s1"),
      { params: { outcomes_from: 0 } },
    );
    expect(status.importedCount).toBe(1);
    expect(status.duplicateCount).toBe(1);
    expect(status.scrapheapedCount).toBe(2);
    // Per-PICTURE, while the count is per-FILE: two staged copies of one
    // scrapheaped picture name its id once.
    expect(status.scrapheapedPictureIds).toEqual([7]);
    // The buckets are disjoint and sum to the total, so no summary line can
    // overstate what happened.
    expect(
      status.importedCount +
        status.duplicateCount +
        status.scrapheapedCount +
        status.failedCount +
        status.cancelledCount,
    ).toBe(status.total);
  });

  it("defaults the Scrapheap fields when an older backend omits them", async () => {
    apiClient.get.mockResolvedValue({
      data: { stage: "completed", imported_count: 3 },
    });
    const status = await fetchStagingStatus({
      backendUrl: "",
      stagingId: "s2",
    });
    expect(status.scrapheapedCount).toBeNull();
    // An array, never null: the caller iterates it to build the restore offer.
    expect(status.scrapheapedPictureIds).toEqual([]);
  });

  it("ignores a non-array scrapheaped_picture_ids rather than passing it on", async () => {
    apiClient.get.mockResolvedValue({
      data: { stage: "completed", scrapheaped_picture_ids: "7" },
    });
    const status = await fetchStagingStatus({
      backendUrl: "",
      stagingId: "s3",
    });
    expect(status.scrapheapedPictureIds).toEqual([]);
  });

  it("asks only for the outcomes it has not seen, and passes them on (#2512)", async () => {
    const outcomes = [{ key: "k2", outcome: "duplicate", picture_id: 9 }];
    apiClient.get.mockResolvedValue({
      data: { stage: "importing", outcomes },
    });
    const status = await fetchStagingStatus({
      backendUrl: "",
      stagingId: "s4",
      outcomesFrom: 1,
    });
    expect(apiClient.get).toHaveBeenCalledWith(
      IMPORT_ENDPOINTS.status("", "s4"),
      { params: { outcomes_from: 1 } },
    );
    expect(status.outcomes).toEqual(outcomes);
  });
});
