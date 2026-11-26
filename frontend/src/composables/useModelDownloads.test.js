// What a pre-flight answer offers for download, and starting those once each.

import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

import {
  downloadableInPreflight,
  downloadModels,
} from "./useModelDownloads";
import { startModelDownload } from "../api/modelDownloads";
import { useNoticeStore } from "../stores/useNoticeStore";
import { useTasksStore } from "../stores/useTasksStore";

vi.mock("../api/modelDownloads", () => ({
  startModelDownload: vi.fn(),
  cancelModelDownload: vi.fn(),
}));
vi.mock("../api/workers", () => ({ getWorkerProgress: vi.fn() }));

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
});

describe("downloadableInPreflight", () => {
  it("takes linked missing models and left-out LoRAs, not the owner's skips", () => {
    const answer = {
      groups: [
        {
          reasons: [
            {
              code: "missing_models",
              models: [
                { file: "vae/a.safetensors", downloadable: true },
                { file: "b.safetensors" },
                { downloadable: true },
              ],
            },
            { code: "missing_nodes", models: [{ file: "c.safetensors", downloadable: true }] },
          ],
          bypassed_loras: [
            { file: "lora-d.safetensors", downloadable: true },
            { file: "lora-e.safetensors", downloadable: true, requested: true },
            { file: "lora-f.safetensors", downloadable: true, recipe: true },
            { file: "lora-g.safetensors" },
          ],
        },
      ],
    };
    expect(downloadableInPreflight(answer)).toEqual([
      "vae/a.safetensors",
      "lora-d.safetensors",
    ]);
    expect(downloadableInPreflight(null)).toEqual([]);
  });
});

describe("downloadModels", () => {
  it("starts each name once, skips one already running, and says so once", async () => {
    startModelDownload.mockImplementation(async (file) => ({
      id: file,
      file: file.split("/").pop(),
      status: "running",
    }));
    useTasksStore().downloadStarted({
      id: "x",
      file: "running.safetensors",
      status: "running",
    });
    const started = await downloadModels([
      "vae/a.safetensors",
      "a.safetensors",
      "b.safetensors",
      "running.safetensors",
    ]);
    expect(started).toBe(2);
    expect(startModelDownload.mock.calls.map(([file]) => file)).toEqual([
      "vae/a.safetensors",
      "b.safetensors",
    ]);
    expect(useNoticeStore().notices.map((n) => n.text)).toEqual([
      "Downloading 2 models.",
    ]);
  });

  it("says so when everything asked for is already downloading", async () => {
    useTasksStore().downloadStarted({
      id: "x",
      file: "running.safetensors",
      status: "running",
    });
    expect(await downloadModels(["loras/running.safetensors"])).toBe(0);
    expect(startModelDownload).not.toHaveBeenCalled();
    expect(useNoticeStore().notices.map((n) => n.text)).toEqual([
      "Already downloading: the Tasks list in the title bar shows how far.",
    ]);
  });

  // The server queues downloads past the few it fetches at once, and refuses
  // a second start of a queued file as of a running one.
  it("skips a file the last poll listed as queued, and says it is queued", async () => {
    useTasksStore().modelDownloads = [
      { id: "x", file: "queued.safetensors", status: "queued" },
    ];
    expect(await downloadModels(["loras/queued.safetensors"])).toBe(0);
    expect(startModelDownload).not.toHaveBeenCalled();
    expect(useNoticeStore().notices.map((n) => n.text)).toEqual([
      "Already queued: the Tasks list in the title bar shows them waiting.",
    ]);
  });

  // Several on their way read queued only when every one of them waits.
  it("says downloading when one of those on their way is running", async () => {
    useTasksStore().modelDownloads = [
      { id: "x", file: "queued.safetensors", status: "queued" },
      { id: "y", file: "running.safetensors", status: "running" },
    ];
    expect(
      await downloadModels(["queued.safetensors", "running.safetensors"]),
    ).toBe(0);
    expect(useNoticeStore().notices.map((n) => n.text)).toEqual([
      "Already downloading: the Tasks list in the title bar shows how far.",
    ]);
  });

  // The server's duplicate-start rule folds case, so a file on its way
  // under another spelling is not offered again, nor is one name started twice.
  it("matches names with their case folded, as the server does", async () => {
    startModelDownload.mockImplementation(async (file) => ({
      id: file,
      file: file.split("/").pop(),
      status: "running",
    }));
    useTasksStore().modelDownloads = [
      { id: "x", file: "Coming_LoRA.safetensors", status: "queued" },
    ];
    const started = await downloadModels([
      "loras/coming_lora.SAFETENSORS",
      "Fresh_VAE.safetensors",
      "vae/fresh_vae.safetensors",
    ]);
    expect(started).toBe(1);
    expect(startModelDownload.mock.calls.map(([file]) => file)).toEqual([
      "Fresh_VAE.safetensors",
    ]);
  });

  // Where `toLowerCase()` and `casefold()` differ the server still refuses,
  // and its words are what the owner reads.
  it("shows the server's words when it refuses a start as a duplicate", async () => {
    useTasksStore().modelDownloads = [
      { id: "x", file: "STRASSE.safetensors", status: "running" },
    ];
    startModelDownload.mockRejectedValueOnce({
      response: {
        status: 409,
        data: { detail: "test-detail: that file is already downloading." },
      },
    });
    expect(await downloadModels(["straße.safetensors"])).toBe(0);
    expect(startModelDownload).toHaveBeenCalledWith("straße.safetensors");
    expect(useNoticeStore().notices.map((n) => n.text)).toEqual([
      "test-detail: that file is already downloading.",
    ]);
  });

  it("reports a refusal and still starts the rest", async () => {
    startModelDownload
      .mockRejectedValueOnce({ response: { data: { detail: "No folder." } } })
      .mockResolvedValueOnce({ id: "b", file: "b.safetensors", status: "running" });
    expect(await downloadModels(["a.safetensors", "b.safetensors"])).toBe(1);
    expect(useNoticeStore().notices.map((n) => n.text).sort()).toEqual([
      "Downloading b.safetensors.",
      "No folder.",
    ]);
  });
});
