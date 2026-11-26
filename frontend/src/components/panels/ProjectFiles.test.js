// Project Files: removing an attachment asks first, in the app's dialog.
//
// An uploaded file is deleted from disk with its row, so that question is a
// danger one; a saved link is only a row, so removing it is not. Either way the
// delete call waits for the answer, and a no deletes nothing.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { defineComponent, h, ref } from "vue";

vi.mock("vuetify/components", () => ({
  VProgressCircular: { template: "<i />" },
  VIcon: { template: "<i />" },
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
import {
  activeConfirm,
  registerConfirmHost,
  resolveConfirm,
  unregisterConfirmHost,
} from "../../composables/useConfirm";

enableAutoUnmount(afterEach);

const UPLOADED = {
  id: 7,
  original_filename: "brief.pdf",
  mime_type: "application/pdf",
  file_size: 4096,
  created_at: "2026-11-20T10:00:00Z",
  url: null,
};
const LINK = {
  id: 8,
  original_filename: "Reference board",
  url: "https://example.com/board",
  created_at: "2026-11-20T10:00:00Z",
};

beforeEach(() => {
  api.listProjectAttachments.mockReset().mockResolvedValue([UPLOADED, LINK]);
  api.deleteProjectAttachment.mockReset().mockResolvedValue({});
  registerConfirmHost();
});

afterEach(() => {
  unregisterConfirmHost();
  vi.restoreAllMocks();
});

/** Mounts the panel, opens it, and returns the Remove button of `file`'s card. */
async function removeButtonFor(file) {
  const wrapper = mount(ProjectFiles, {
    props: { projectId: 3, backendUrl: "/api/v1" },
    global: {
      stubs: {
        Tooltip: true,
        "v-icon": { template: "<i />" },
        AppButton: { template: '<button type="button"><slot /></button>' },
      },
    },
  });
  await wrapper.find(".pf-header").trigger("click");
  await flushPromises();
  const card = wrapper
    .findAll(".pf-file-card")
    .find((c) => c.find(".pf-file-name").text().includes(file.original_filename));
  expect(card, `no card for ${file.original_filename}`).toBeTruthy();
  return { wrapper, remove: card.find('button.pf-file-delete[aria-label="Remove"]') };
}

/** Clicks Remove twice: answers no, then yes, checking the delete waits. */
async function answerNoThenYes(remove, fileId) {
  await remove.trigger("click");
  await flushPromises();
  const options = activeConfirm.value.options;
  resolveConfirm(false);
  await flushPromises();
  expect(api.deleteProjectAttachment).not.toHaveBeenCalled();

  await remove.trigger("click");
  await flushPromises();
  expect(api.deleteProjectAttachment).not.toHaveBeenCalled();
  resolveConfirm(true);
  await flushPromises();
  expect(api.deleteProjectAttachment).toHaveBeenCalledWith(3, fileId);
  return options;
}

describe("removing a project attachment", () => {
  it("asks a danger question for an uploaded file, and deletes it only on yes", async () => {
    const nativeConfirm = vi.spyOn(window, "confirm");
    const { wrapper, remove } = await removeButtonFor(UPLOADED);

    const options = await answerNoThenYes(remove, UPLOADED.id);
    expect(options).toMatchObject({
      title: 'Delete "brief.pdf"?',
      message: "The file is removed from this project and deleted from disk.",
      warning: "There is no undo for this.",
      confirmLabel: "Delete file",
      danger: true,
    });
    expect(nativeConfirm).not.toHaveBeenCalled();
    expect(wrapper.text()).not.toContain("brief.pdf");
  });

  it("asks a plain question for a link, since the page it points to is untouched", async () => {
    const nativeConfirm = vi.spyOn(window, "confirm");
    const { remove } = await removeButtonFor(LINK);

    const options = await answerNoThenYes(remove, LINK.id);
    expect(options).toMatchObject({
      title: 'Remove the link to "Reference board"?',
      message: "The page it points to is not affected.",
      confirmLabel: "Remove link",
      danger: false,
    });
    expect(nativeConfirm).not.toHaveBeenCalled();
  });
});

// A file dropped on Project Files is an attachment, not a picture to import:
// the window's capture handler (useWindowFileImport) leaves a zone marked
// `data-owns-file-drop` to its own handler, or the picture would be both
// attached and imported into the library.
describe("a picture dropped on Project Files", () => {
  const windowImport = vi.fn();
  const Host = defineComponent({
    setup() {
      useWindowFileImport({
        sidebarRef: ref({
          startLocalImport: windowImport,
          currentProjectId: 3,
        }),
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
    windowImport.mockClear();
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
    await flushPromises();
    if (open) {
      await wrapper.find(".pf-header").trigger("click");
      await flushPromises();
    }
    const { event, file } = pictureDrop();
    wrapper.find(selector).element.dispatchEvent(event);
    for (let i = 0; i < 4; i += 1) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    return file;
  }

  it.each([
    ["the collapsed header", ".pf-header", false],
    ["the open panel", ".pf-panel", true],
  ])("on %s attaches it and imports nothing", async (_, selector, open) => {
    const file = await dropOn(selector, { open });

    expect(api.uploadProjectAttachment).toHaveBeenCalledWith(3, file);
    expect(windowImport).not.toHaveBeenCalled();
  });
});
