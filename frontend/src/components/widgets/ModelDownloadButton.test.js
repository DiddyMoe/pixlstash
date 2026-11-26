// The download beside a missing model: it sends only the name, says so with a
// passing notice that opens the task list, and is busy while that file is.

import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

import ModelDownloadButton from "./ModelDownloadButton.vue";
import { startModelDownload } from "../../api/modelDownloads";
import { useNoticeStore } from "../../stores/useNoticeStore";
import { useSidebarStore } from "../../stores/useSidebarStore";
import { useTasksStore } from "../../stores/useTasksStore";

vi.mock("../../api/modelDownloads", () => ({
  startModelDownload: vi.fn(),
  cancelModelDownload: vi.fn(),
}));
vi.mock("../../api/workers", () => ({ getWorkerProgress: vi.fn() }));

const stubs = { "v-icon": true, Tooltip: true };

function mountButton(file = "SDXL/flux1-dev.safetensors") {
  return mount(ModelDownloadButton, { props: { file }, global: { stubs } });
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
});

describe("ModelDownloadButton", () => {
  it("starts the download by name and says so in passing", async () => {
    const download = {
      id: "d-1",
      file: "flux1-dev.safetensors",
      status: "running",
    };
    startModelDownload.mockResolvedValue(download);
    const wrapper = mountButton();
    expect(wrapper.attributes("aria-label")).toBe(
      "Download flux1-dev.safetensors",
    );
    await wrapper.trigger("click");
    await flushPromises();
    expect(startModelDownload).toHaveBeenCalledWith(
      "SDXL/flux1-dev.safetensors",
    );
    expect(useTasksStore().modelDownloads).toEqual([download]);
    // Busy now: the file is downloading, so a second press sends nothing.
    expect(wrapper.attributes("aria-label")).toBe(
      "Downloading flux1-dev.safetensors",
    );
    await wrapper.trigger("click");
    expect(startModelDownload).toHaveBeenCalledTimes(1);

    const [notice] = useNoticeStore().notices;
    expect(notice.text).toBe("Downloading flux1-dev.safetensors.");
    expect(notice.timeout).toBeGreaterThan(0);
    const sidebar = useSidebarStore();
    const show = vi.spyOn(sidebar, "showTasksTab");
    notice.action.handler();
    expect(show).toHaveBeenCalled();
  });

  // The server fetches a few at once and queues the rest; it refuses a second
  // start of a queued file as of a running one.
  // The server refuses a second start of a file in any case, so the
  // button does not offer one for a download spelled otherwise.
  it("is busy while its file is on its way under another case", async () => {
    useTasksStore().modelDownloads = [
      { id: "d-1", file: "FLUX1-Dev.safetensors", status: "running" },
    ];
    const wrapper = mountButton();
    expect(wrapper.attributes("aria-label")).toBe(
      "Downloading flux1-dev.safetensors",
    );
    await wrapper.trigger("click");
    await flushPromises();
    expect(startModelDownload).not.toHaveBeenCalled();
  });

  it("is busy while its file is queued, and says it is queued", async () => {
    startModelDownload.mockResolvedValue({
      id: "d-1",
      file: "flux1-dev.safetensors",
      status: "queued",
    });
    const wrapper = mountButton();
    await wrapper.trigger("click");
    await flushPromises();
    expect(wrapper.attributes("aria-label")).toBe(
      "Queued to download flux1-dev.safetensors",
    );
    await wrapper.trigger("click");
    expect(startModelDownload).toHaveBeenCalledTimes(1);

    // Its slot comes free: it reads as downloading.
    useTasksStore().modelDownloads = [
      { id: "d-1", file: "flux1-dev.safetensors", status: "running" },
    ];
    await flushPromises();
    expect(wrapper.attributes("aria-label")).toBe(
      "Downloading flux1-dev.safetensors",
    );
  });

  it("says why when the server refuses", async () => {
    startModelDownload.mockRejectedValue({
      response: { data: { detail: "flux1-dev.safetensors is already there." } },
    });
    const wrapper = mountButton();
    await wrapper.trigger("click");
    await flushPromises();
    expect(useNoticeStore().notices.map((n) => n.text)).toEqual([
      "flux1-dev.safetensors is already there.",
    ]);
    expect(wrapper.attributes("aria-label")).toBe(
      "Download flux1-dev.safetensors",
    );
  });
});
