import { describe, it, expect, beforeEach, vi } from "vitest";

vi.mock("../utils/apiClient", () => ({
  API_BASE_URL: "/api/v1",
  apiClient: { get: vi.fn(), post: vi.fn(), delete: vi.fn() },
  operationBatchHeaders: (batchId) => batchId
    ? { headers: { "X-Operation-Batch-Id": batchId } }
    : undefined,
}));

import { apiClient } from "../utils/apiClient";
import {
  getAnomalyRegion,
  listPicturesByIds,
  PICTURES_BY_IDS_URL_BUDGET,
  picturesByIdsTargets,
  streamPicturesAmongIds,
  getPictureCount,
  streamPictures,
  getLikenessGroups,
  faceSearch,
  likenessSearch,
  setLikenessSearch,
  searchPictures,
  getPictureStats,
  clearGuestScoreSession,
  getPictureMetadata,
  getThumbnails,
  deletePictures,
  movePicturesToLibrary,
  setPicturesProject,
  restoreScrapheap,
  startScrapheapRun,
  getScrapheapRun,
  getLatestScrapheapRun,
  cancelScrapheapRun,
  undoScrapheapRun,
  startExport,
  startExportByIds,
  getExportStatus,
  downloadExport,
  downloadPicture,
} from "./pictures";

beforeEach(() => {
  apiClient.get.mockReset();
  apiClient.post.mockReset();
  apiClient.delete.mockReset();
});

describe("api/pictures", () => {
  it("carries the shared move gesture id without imposing a copy timeout", async () => {
    apiClient.post.mockResolvedValue({ data: { moved: [4] } });
    expect(await movePicturesToLibrary([4], "archive", "cli-move-test"))
      .toEqual({ moved: [4] });
    expect(apiClient.post).toHaveBeenCalledWith(
      "/pictures/move-to-library",
      { picture_ids: [4], library_uuid: "archive" },
      { headers: { "X-Operation-Batch-Id": "cli-move-test" }, timeout: 0 },
    );
  });
  it("listPicturesByIds repeats the id param once per picture", async () => {
    apiClient.get.mockResolvedValue({ data: [{ id: 4 }, { id: 5 }] });
    const result = await listPicturesByIds([4, 5]);
    expect(apiClient.get).toHaveBeenCalledWith("/pictures?id=4&id=5");
    expect(result).toEqual([{ id: 4 }, { id: 5 }]);
  });

  it("listPicturesByIds requests the by-ids route", async () => {
    apiClient.get.mockResolvedValue({ data: [] });
    await listPicturesByIds([4]);
    expect(apiClient.get).toHaveBeenCalledWith("/pictures?id=4");
  });

  it("listPicturesByIds appends the projection when asked", async () => {
    apiClient.get.mockResolvedValue({ data: [] });
    await listPicturesByIds([4, 5], { fields: "grid" });
    expect(apiClient.get).toHaveBeenCalledWith("/pictures?id=4&id=5&fields=grid");
  });

  it("listPicturesByIds splits a long list into requests of at most 1000 ids, in order", async () => {
    // One-digit ids, so the count cap binds before the byte budget does.
    const ids = Array.from({ length: 2001 }, (_, i) => (i % 9) + 1);
    apiClient.get.mockImplementation(async (url) => ({
      data: new URLSearchParams(url.split("?")[1])
        .getAll("id")
        .map((id) => ({ id: Number(id) })),
    }));
    const rows = await listPicturesByIds(ids, { fields: "grid" });
    const sent = apiClient.get.mock.calls.map(
      ([url]) => new URLSearchParams(url.split("?")[1]),
    );
    expect(sent.map((p) => p.getAll("id").length)).toEqual([1000, 1000, 1]);
    for (const p of sent) expect(p.get("fields")).toBe("grid");
    expect(rows.map((r) => r.id)).toEqual(ids);
  });

  it("keeps the worst-case by-ids request line under 8,000 bytes", async () => {
    // Ten-digit ids, the longest a picture id gets in practice, behind the
    // real prefix and a share link's token (secrets.token_urlsafe(32)).
    const ids = Array.from({ length: 3000 }, (_, i) => 9_000_000_000 + i);
    const token = "x".repeat(43);
    apiClient.get.mockImplementation(async (url) => ({
      data: new URLSearchParams(url.split("?")[1])
        .getAll("id")
        .map((id) => ({ id: Number(id) })),
    }));
    const rows = await listPicturesByIds(ids, { fields: "grid" });

    const targets = apiClient.get.mock.calls.map(([url]) => url);
    expect(targets.length).toBeGreaterThan(1);
    for (const target of targets) {
      expect(target.length).toBeLessThanOrEqual(PICTURES_BY_IDS_URL_BUDGET);
      const requestLine = `GET /api/v1${target}&token=${token} HTTP/1.1`;
      expect(new TextEncoder().encode(requestLine).length).toBeLessThan(8000);
    }
    // Full requests: each but the last would pass the budget with one more id.
    for (const target of targets.slice(0, -1)) {
      expect(target.length + "&id=9000000000".length).toBeGreaterThan(
        PICTURES_BY_IDS_URL_BUDGET,
      );
    }
    expect(rows.map((r) => r.id)).toEqual(ids);
  });

  it("picturesByIdsTargets keeps the encoding and the empty list as one request", () => {
    expect(picturesByIdsTargets([4, 5], "grid")).toEqual([
      "/pictures?id=4&id=5&fields=grid",
    ]);
    expect(picturesByIdsTargets([])).toEqual(["/pictures?"]);
  });

  it("streamPicturesAmongIds narrows the stream query by id, 1000 at a time, from the first row", async () => {
    // One-digit ids, so the count cap binds before the byte budget does.
    const ids = Array.from({ length: 1500 }, (_, i) => (i % 9) + 1);
    const signal = new AbortController().signal;
    apiClient.get.mockImplementation(async (url) => ({
      data: {
        pictures: new URLSearchParams(url.split("?")[1])
          .getAll("id")
          .map((id) => ({ id: Number(id) })),
      },
    }));
    const rows = await streamPicturesAmongIds("fields=id&sort=DATE", ids, { signal });

    const calls = apiClient.get.mock.calls;
    for (const [, options] of calls) expect(options).toEqual({ signal });
    const sent = calls.map(([url]) => {
      expect(url.startsWith("/pictures/stream?fields=id&sort=DATE&id=")).toBe(true);
      return new URLSearchParams(url.split("?")[1]);
    });
    expect(sent.map((p) => p.getAll("id").length)).toEqual([1000, 500]);
    for (const p of sent) {
      expect(p.get("offset")).toBe("0");
      expect(p.get("batch_limit")).toBe("1000");
    }
    expect(rows.map((r) => r.id)).toEqual(ids);
  });

  it("streamPicturesAmongIds keeps each request within the by-ids URL budget", async () => {
    const ids = Array.from({ length: 1200 }, (_, i) => 9_000_000_000 + i);
    apiClient.get.mockResolvedValue({ data: { pictures: [] } });
    await streamPicturesAmongIds("fields=id", ids);
    const targets = apiClient.get.mock.calls.map(([url]) => url);
    expect(targets.length).toBeGreaterThan(2);
    for (const target of targets) {
      expect(target.length).toBeLessThanOrEqual(PICTURES_BY_IDS_URL_BUDGET);
    }
    expect(
      targets.flatMap((t) => new URLSearchParams(t.split("?")[1]).getAll("id")),
    ).toEqual(ids.map(String));
  });

  it("streamPicturesAmongIds sends nothing for no ids", async () => {
    expect(await streamPicturesAmongIds("fields=id", [])).toEqual([]);
    expect(apiClient.get).not.toHaveBeenCalled();
  });

  it("streamPicturesAmongIds rejects an answer with no pictures list", async () => {
    apiClient.get.mockResolvedValue({ data: [] });
    await expect(streamPicturesAmongIds("fields=id", [4])).rejects.toThrow(
      "streamPicturesAmongIds: a stream read returned no pictures list",
    );
  });

  it("listPicturesByIds rejects when any chunk fails, rather than return part", async () => {
    const ids = Array.from({ length: 1500 }, (_, i) => i + 1);
    apiClient.get
      .mockResolvedValueOnce({ data: [{ id: 1 }] })
      .mockRejectedValueOnce(new Error("chunk 2 failed"));
    await expect(listPicturesByIds(ids)).rejects.toThrow("chunk 2 failed");
  });

  it("getPictureCount appends a filter query when there is one", async () => {
    apiClient.get.mockResolvedValue({ data: { count: 12 } });
    const result = await getPictureCount("stack_leaders_only=true");
    expect(apiClient.get).toHaveBeenCalledWith(
      "/pictures/count?stack_leaders_only=true",
    );
    expect(result.count).toBe(12);
  });

  it("getPictureCount omits the query separator when unfiltered", async () => {
    apiClient.get.mockResolvedValue({ data: { count: 0 } });
    await getPictureCount();
    expect(apiClient.get).toHaveBeenCalledWith("/pictures/count");
  });

  // The grid runs several batches concurrently, so offset/limit must land on
  // the wire exactly as given rather than being tracked inside the module.
  it("streamPictures appends the caller's offset and batch limit", async () => {
    apiClient.get.mockResolvedValue({ data: { pictures: [] } });
    await streamPictures("fields=grid", { offset: 200, batchLimit: 50 });
    expect(apiClient.get).toHaveBeenCalledWith(
      "/pictures/stream?fields=grid&offset=200&batch_limit=50",
    );
  });

  it("getLikenessGroups encodes the threshold and appends the filter", async () => {
    apiClient.get.mockResolvedValue({ data: [] });
    await getLikenessGroups(0.4, "character_id=2");
    expect(apiClient.get).toHaveBeenCalledWith(
      "/pictures/likeness-groups?threshold=0.4&character_id=2",
    );
  });

  it("getLikenessGroups omits an empty filter", async () => {
    apiClient.get.mockResolvedValue({ data: [] });
    await getLikenessGroups(0.4);
    expect(apiClient.get).toHaveBeenCalledWith(
      "/pictures/likeness-groups?threshold=0.4",
    );
  });

  it("faceSearch POSTs the source face and a top-n cap", async () => {
    apiClient.post.mockResolvedValue({ data: [] });
    await faceSearch(7);
    expect(apiClient.post).toHaveBeenCalledWith(
      "/pictures/face-search?source_face_id=7&top_n=500",
    );
  });

  // Several sources are combined by MINIMUM similarity, so each one has to
  // reach the server as its own repeated param.
  it("likenessSearch repeats one source_picture_ids param per source", async () => {
    apiClient.post.mockResolvedValue({ data: [] });
    await likenessSearch([1, 2]);
    expect(apiClient.post).toHaveBeenCalledWith(
      "/pictures/likeness-search?source_picture_ids=1&source_picture_ids=2&top_n=500&threshold=0.05",
    );
  });

  // The set suggestion fetches wide (floor 0) and excludes the set's own
  // members, so its length is the count the "Add" button can promise.
  it("setLikenessSearch queries by set, excludes its members, asks for tags", async () => {
    apiClient.post.mockResolvedValue({ data: [] });
    await setLikenessSearch(4);
    expect(apiClient.post).toHaveBeenCalledWith(
      "/pictures/likeness-search?source_set_id=4&top_n=500&threshold=0&exclude_set_id=4&include_tag_counts=true",
    );
  });

  // The threshold is the top of what the server's score gives a picture by
  // chance; at 0.1 a nonsense search kept every picture.
  it("searchPictures encodes the text, keeps scores above chance and appends the scope filter", async () => {
    apiClient.get.mockResolvedValue({ data: [] });
    await searchPictures("red car", { query: "character_id=2" });
    expect(apiClient.get).toHaveBeenCalledWith(
      "/pictures/search?query=red%20car&threshold=0.3&top_n=10000&character_id=2",
    );
  });

  it("getPictureStats merges the filter query with the section params", async () => {
    apiClient.get.mockResolvedValue({ data: {} });
    await getPictureStats("character_id=2", { include: "cooc" });
    expect(apiClient.get).toHaveBeenCalledWith(
      "/pictures/stats?character_id=2",
      { params: { include: "cooc" } },
    );
  });

  it("getPictureStats drops the separator when there is no filter", async () => {
    apiClient.get.mockResolvedValue({ data: {} });
    await getPictureStats();
    expect(apiClient.get).toHaveBeenCalledWith("/pictures/stats", undefined);
  });

  it("clearGuestScoreSession DELETEs the session scores", async () => {
    apiClient.delete.mockResolvedValue({ data: {} });
    await clearGuestScoreSession();
    expect(apiClient.delete).toHaveBeenCalledWith(
      "/pictures/guest-scores/session",
    );
  });

  it("getPictureMetadata omits the query when nothing is asked for", async () => {
    apiClient.get.mockResolvedValue({ data: {} });
    await getPictureMetadata(42);
    expect(apiClient.get).toHaveBeenCalledWith("/pictures/42/metadata");
  });

  it("getPictureMetadata adds the smart-score and cache-buster params", async () => {
    apiClient.get.mockResolvedValue({ data: {} });
    await getPictureMetadata(42, { smartScore: true, cacheBuster: 99 });
    expect(apiClient.get).toHaveBeenCalledWith(
      "/pictures/42/metadata?smart_score=true&cb=99",
    );
  });

  it("getThumbnails POSTs the id batch", async () => {
    apiClient.post.mockResolvedValue({ data: {} });
    await getThumbnails(["1", "2"]);
    expect(apiClient.post).toHaveBeenCalledWith("/pictures/thumbnails", {
      ids: ["1", "2"],
    });
  });

  it("deletePictures sends the ids as the DELETE body", async () => {
    apiClient.delete.mockResolvedValue({ data: { skipped_locked: [] } });
    await deletePictures([1, 2]);
    expect(apiClient.delete).toHaveBeenCalledWith("/pictures", {
      data: { picture_ids: [1, 2] },
    });
  });

  // No mode means SET the project; sending mode:undefined would be a different
  // request shape, so the key must be absent entirely.
  it("setPicturesProject omits the mode when setting", async () => {
    apiClient.patch = vi.fn().mockResolvedValue({ data: {} });
    await setPicturesProject([1], 5);
    expect(apiClient.patch).toHaveBeenCalledWith("/pictures/project", {
      picture_ids: [1],
      project_id: 5,
    });
  });

  it("setPicturesProject includes the mode when adding or removing", async () => {
    apiClient.patch = vi.fn().mockResolvedValue({ data: {} });
    await setPicturesProject([1], 5, { mode: "remove" });
    expect(apiClient.patch).toHaveBeenCalledWith("/pictures/project", {
      picture_ids: [1],
      project_id: 5,
      mode: "remove",
    });
  });

  // The one-shot restore is for named pictures only; the whole heap goes
  // through a restore run, and the server refuses an id-less body.
  it("restoreScrapheap always names the pictures it restores", async () => {
    apiClient.post.mockResolvedValue({ data: {} });
    await restoreScrapheap([4], { batchId: "cli-test-b5" });
    expect(apiClient.post.mock.calls[0][0]).toBe("/pictures/scrapheap/restore");
    expect(apiClient.post.mock.calls[0][1]).toEqual({ picture_ids: [4] });
  });

  it("restoreScrapheap scopes to the given ids", async () => {
    apiClient.post.mockResolvedValue({ data: {} });
    await restoreScrapheap([3]);
    expect(apiClient.post).toHaveBeenCalledWith(
      "/pictures/scrapheap/restore",
      { picture_ids: [3] },
      undefined,
    );
  });

  // A gesture's own id keeps its History step apart from a restore run's.
  it("restoreScrapheap sends the gesture's batch id", async () => {
    apiClient.post.mockResolvedValue({ data: {} });
    await restoreScrapheap([3], { batchId: "cli-lightbox1" });
    expect(apiClient.post).toHaveBeenCalledWith(
      "/pictures/scrapheap/restore",
      { picture_ids: [3] },
      { headers: { "X-Operation-Batch-Id": "cli-lightbox1" } },
    );
  });

  const START_HEADERS = { headers: { Prefer: "wait=1" } };

  // The run request forbids unknown fields, so each kind sends only its own:
  // a restore carrying include_protected or confirm_token would be a 422.
  it("startScrapheapRun sends a whole-heap purge with its token and no ids", async () => {
    const run = { id: "run-a", active: true };
    apiClient.post.mockResolvedValue({
      data: run,
      headers: { "retry-after": "1" },
    });
    const reply = await startScrapheapRun({
      kind: "purge",
      includeProtected: true,
      confirmToken: "test-token-a",
    });
    expect(apiClient.post).toHaveBeenCalledWith(
      "/pictures/scrapheap/runs",
      {
        kind: "purge",
        include_protected: true,
        confirm_token: "test-token-a",
      },
      START_HEADERS,
    );
    expect(reply.run).toBe(run);
    expect(reply.retryAfterMs).toBe(1000);
  });

  it("startScrapheapRun scopes a purge to the selection", async () => {
    apiClient.post.mockResolvedValue({ data: { id: "run-b" }, headers: {} });
    await startScrapheapRun({
      kind: "purge",
      pictureIds: [3, 4],
      confirmToken: "test-token-b",
    });
    expect(apiClient.post).toHaveBeenCalledWith(
      "/pictures/scrapheap/runs",
      {
        kind: "purge",
        picture_ids: [3, 4],
        include_protected: false,
        confirm_token: "test-token-b",
      },
      START_HEADERS,
    );
  });

  // The server refuses a purge without the preview's single-use
  // confirm_token, so the field must always be on the wire - never silently
  // dropped when a caller forgets it.
  it("startScrapheapRun always sends confirm_token on a purge", async () => {
    apiClient.post.mockResolvedValue({ data: { id: "run-d" }, headers: {} });
    await startScrapheapRun({ kind: "purge" });
    expect(apiClient.post).toHaveBeenCalledWith(
      "/pictures/scrapheap/runs",
      {
        kind: "purge",
        include_protected: false,
        confirm_token: undefined,
      },
      START_HEADERS,
    );
  });

  it("startScrapheapRun sends a restore with no purge-only fields", async () => {
    apiClient.post.mockResolvedValue({ data: { id: "run-c" }, headers: {} });
    await startScrapheapRun({
      kind: "restore",
      pictureIds: [7],
      includeProtected: true,
      confirmToken: "test-token-c",
    });
    expect(apiClient.post).toHaveBeenCalledWith(
      "/pictures/scrapheap/runs",
      {
        kind: "restore",
        picture_ids: [7],
      },
      START_HEADERS,
    );
  });

  // A short run answers at once (RFC 7240 wait): 200 with the ended run and
  // no Retry-After, so the caller follows nothing and polls nothing.
  it("startScrapheapRun asks to wait a second and returns a run that ended in it", async () => {
    const run = { id: "run-e", active: false, status: "complete" };
    apiClient.post.mockResolvedValue({
      status: 200,
      data: run,
      headers: { "preference-applied": "wait=1", vary: "Prefer" },
    });
    const reply = await startScrapheapRun({ kind: "restore" });
    expect(apiClient.post.mock.calls[0][2].headers.Prefer).toBe("wait=1");
    expect(reply).toEqual({ run, retryAfterMs: null });
  });

  // The gesture's request id rides in the body so a start whose answer
  // was lost can be sent again and get the same run back.
  it("startScrapheapRun sends the gesture's request id", async () => {
    apiClient.post.mockResolvedValue({ data: { id: "run-r" }, headers: {} });
    await startScrapheapRun({
      kind: "restore",
      pictureIds: [7],
      requestId: "cli-test-b3",
    });
    expect(apiClient.post.mock.calls[0][1]).toEqual({
      kind: "restore",
      picture_ids: [7],
      request_id: "cli-test-b3",
    });
  });

  it("startScrapheapRun sends the request id on a purge too", async () => {
    apiClient.post.mockResolvedValue({ data: { id: "run-p" }, headers: {} });
    await startScrapheapRun({
      kind: "purge",
      confirmToken: "test-token-p",
      requestId: "cli-test-b3p",
    });
    expect(apiClient.post.mock.calls[0][1].request_id).toBe("cli-test-b3p");
  });

  // Undo is a run of its own, started like one (wait, request id).
  it("undoScrapheapRun starts the undo of a restore run", async () => {
    const run = { id: "run-u", kind: "undo", active: true };
    apiClient.post.mockResolvedValue({
      status: 202,
      data: run,
      headers: { "retry-after": "1" },
    });
    const reply = await undoScrapheapRun("a/b", { requestId: "cli-test-u" });
    expect(apiClient.post).toHaveBeenCalledWith(
      "/pictures/scrapheap/runs/a%2Fb/undo",
      { request_id: "cli-test-u" },
      START_HEADERS,
    );
    expect(reply).toEqual({ run, retryAfterMs: 1000 });
  });

  it("getScrapheapRun and cancelScrapheapRun address the run by its id", async () => {
    apiClient.get.mockResolvedValue({ data: { id: "a/b" }, headers: {} });
    apiClient.post.mockResolvedValue({
      data: { id: "a/b", active: true },
      headers: { "retry-after": "1" },
    });
    expect((await getScrapheapRun("a/b")).retryAfterMs).toBeNull();
    expect(apiClient.get).toHaveBeenCalledWith(
      "/pictures/scrapheap/runs/a%2Fb",
    );
    expect((await cancelScrapheapRun("a/b")).retryAfterMs).toBe(1000);
    expect(apiClient.post).toHaveBeenCalledWith(
      "/pictures/scrapheap/runs/a%2Fb/cancel",
    );
  });

  it("getLatestScrapheapRun unwraps the run, and null before the first", async () => {
    const run = { id: "run-d" };
    apiClient.get.mockResolvedValueOnce({ data: { run } });
    expect(await getLatestScrapheapRun()).toBe(run);
    expect(apiClient.get).toHaveBeenCalledWith(
      "/pictures/scrapheap/runs/latest",
    );
    apiClient.get.mockResolvedValueOnce({ data: { run: null } });
    expect(await getLatestScrapheapRun()).toBeNull();
  });

  it("startExport appends the selection query", async () => {
    apiClient.get.mockResolvedValue({ data: { task_id: "t1" } });
    const result = await startExport("id=1&export_type=full");
    expect(apiClient.get).toHaveBeenCalledWith(
      "/pictures/export?id=1&export_type=full",
    );
    expect(result.task_id).toBe("t1");
  });

  it("startExportByIds posts the selection, in its order, in the body", async () => {
    apiClient.post.mockResolvedValue({ data: { task_id: "t2" } });
    const result = await startExportByIds(["7", 3, 5], {
      only_deleted: true,
      resolution: "half",
    });
    expect(apiClient.post).toHaveBeenCalledWith("/pictures/export", {
      picture_ids: [7, 3, 5],
      only_deleted: true,
      resolution: "half",
    });
    expect(apiClient.get).not.toHaveBeenCalled();
    expect(result.task_id).toBe("t2");
  });

  it("getExportStatus polls by task id", async () => {
    apiClient.get.mockResolvedValue({ data: { status: "in_progress" } });
    await getExportStatus("t1");
    expect(apiClient.get).toHaveBeenCalledWith("/pictures/export/status", {
      params: { task_id: "t1" },
    });
  });

  // The ZIP name lives in a response HEADER: a body-only return would rename
  // every download to the fallback, so the module parses it here.
  it("downloadExport returns the blob and the server's filename", async () => {
    const blob = new Blob(["zip"]);
    apiClient.get.mockResolvedValue({
      data: blob,
      headers: { "content-disposition": 'attachment; filename="trip.zip"' },
    });
    const result = await downloadExport("/pictures/export/download/t1");
    expect(apiClient.get).toHaveBeenCalledWith(
      "/pictures/export/download/t1",
      { responseType: "blob" },
    );
    expect(result).toEqual({ blob, filename: "trip.zip" });
  });

  it("downloadExport falls back to a default name without the header", async () => {
    const blob = new Blob(["zip"]);
    apiClient.get.mockResolvedValue({ data: blob, headers: {} });
    const result = await downloadExport("/dl");
    expect(result.filename).toBe("pixlstash_export.zip");
  });

  it("downloadPicture fetches the original media as a blob", async () => {
    const blob = new Blob(["picture"]);
    apiClient.get.mockResolvedValue({ data: blob });

    const result = await downloadPicture(42, ".JPG", {
      version: "pixel-hash",
    });

    expect(apiClient.get).toHaveBeenCalledWith(
      "/pictures/42.jpg?v=pixel-hash",
      { responseType: "blob" },
    );
    expect(result).toBe(blob);
  });

  it("getAnomalyRegion passes the tag as a query param", async () => {
    apiClient.get.mockResolvedValue({ data: { bbox: [0, 0, 1, 1] } });
    const result = await getAnomalyRegion(42, "hat");
    expect(apiClient.get).toHaveBeenCalledWith("/pictures/42/anomaly_region", {
      params: { tag: "hat" },
    });
    expect(result).toEqual({ bbox: [0, 0, 1, 1] });
  });

  // The caller caches a miss on rejection, so an unknown tag must reject
  // rather than resolve to null.
  it("getAnomalyRegion propagates an unknown-tag failure", async () => {
    apiClient.get.mockRejectedValue(new Error("404"));
    await expect(getAnomalyRegion(42, "nope")).rejects.toThrow("404");
  });
});
