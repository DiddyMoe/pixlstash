// Project Files: a file dropped on the panel is an attachment, not a picture
// to import. The window's capture handler (useWindowFileImport) sees the drop
// first; a drop elsewhere still imports (useWindowFileImport.test.js).

import { afterEach, describe, expect, it, vi } from "vitest";
import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { defineComponent, h, ref } from "vue";

vi.mock("vuetify/components", () => ({
  VProgressCircular: { template: "<i />" },
}));

const api = vi.hoisted(() => ({
  listProjectAttachments: vi.fn(),
  uploadProjectAttachment: vi.fn(),
  addProjectAttachmentUrl: vi.fn(),
  deleteProjectAttachment: vi.fn(),
}));
vi.mock("../../api/projects", () => api);

import ProjectFiles from "./ProjectFiles.vue";
import { useWindowFileImport } from "../../composables/useWindowFileImport";

enableAutoUnmount(afterEach);

const startLocalImport = vi.fn();

const Host = defineComponent({
  setup() {
    useWindowFileImport({
      sidebarRef: ref({ startLocalImport, currentProjectId: 3 }),
    });
    return () => h(ProjectFiles, { projectId: 3, backendUrl: "/api/v1" });
  },
});

function pictureDrop() {
  const file = new File(["x"], "reference.jpg", { type: "image/jpeg" });
  const entry = {
    isFile: true,
    isDirectory: false,
    name: file.name,
    fullPath: `/${file.name}`,
    file: (ok) => ok(file),
  };
  const event = new Event("drop", { bubbles: true, cancelable: true });
  event.dataTransfer = {
    types: ["Files"],
    files: [file],
    items: [
      { kind: "file", webkitGetAsEntry: () => entry, getAsFile: () => file },
    ],
    getData: () => "",
  };
  return { event, file };
}

async function dropOn(selector, { open }) {
  setActivePinia(createPinia());
  startLocalImport.mockClear();
  api.listProjectAttachments.mockReset().mockResolvedValue([]);
  api.uploadProjectAttachment.mockReset().mockResolvedValue({});
  const wrapper = mount(Host, {
    attachTo: document.body,
    global: {
      stubs: {
        Tooltip: true,
        "v-icon": { template: "<i />" },
        AppButton: { template: '<button type="button"><slot /></button>' },
      },
    },
  });
  if (open) {
    await wrapper.find(".pf-header").trigger("click");
    await flushPromises();
  }
  const { event, file } = pictureDrop();
  wrapper.find(selector).element.dispatchEvent(event);
  // The window handler awaits the DataTransfer walk before importing.
  await new Promise((resolve) => setTimeout(resolve, 0));
  await flushPromises();
  return file;
}

describe("a picture dropped on Project Files", () => {
  it.each([
    ["the collapsed header", ".pf-header", false],
    ["the open panel", ".pf-panel", true],
  ])("on %s is attached and not imported", async (_, selector, open) => {
    const file = await dropOn(selector, { open });

    expect(api.uploadProjectAttachment).toHaveBeenCalledWith(3, file);
    expect(startLocalImport).not.toHaveBeenCalled();
  });
});
