import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

const api = vi.hoisted(() => ({
  listImportFolders: vi.fn(),
  listReferenceFolders: vi.fn(),
  createFolder: vi.fn(),
  patchFolder: vi.fn(),
  deleteFolder: vi.fn(),
}));
vi.mock("../../api/folders", () => api);
vi.mock("../editors/FolderBrowser.vue", () => ({
  default: { name: "FolderBrowser", template: "<div />" },
}));
vi.mock("../widgets/AppDialog.vue", () => ({
  default: {
    name: "AppDialog",
    props: ["open", "title"],
    template:
      "<div v-if='open' class='dialog-stub'><slot /><slot name='footer' /></div>",
  },
}));

import ImportFoldersSection from "./ImportFoldersSection.vue";
import {
  activeConfirm,
  registerConfirmHost,
  resolveConfirm,
  unregisterConfirmHost,
} from "../../composables/useConfirm";

const RENDERS = {
  id: 3,
  folder: "/home/me/renders",
  label: "ComfyUI output",
  delete_after_import: false,
  picture_count: 12,
  last_imported_at: null,
  missing: true,
};

function mountSection() {
  return mount(ImportFoldersSection, {
    props: { open: true },
    global: {
      stubs: {
        VMenu: { template: "<div><slot name='activator' :props='{}' /><slot /></div>" },
        VIcon: true,
      },
    },
  });
}

function button(wrapper, text) {
  return wrapper.findAll("button").find((b) => b.text().trim() === text);
}

beforeEach(() => {
  setActivePinia(createPinia());
  api.listImportFolders.mockResolvedValue({ folders: [RENDERS] });
  api.listReferenceFolders.mockResolvedValue({ folders: [], in_docker: false });
  api.createFolder.mockResolvedValue({});
  api.patchFolder.mockResolvedValue({});
  api.deleteFolder.mockReset().mockResolvedValue({});
});

afterEach(() => {
  unregisterConfirmHost();
  vi.restoreAllMocks();
});

describe("ImportFoldersSection", () => {
  it("lists each folder with what happens to the original and its state", async () => {
    const wrapper = mountSection();
    await flushPromises();
    const row = wrapper.find(".ifs-list .ifs-row");
    expect(row.text()).toContain("ComfyUI output");
    expect(row.text()).toContain("Keep original");
    expect(row.find("[role='status']").text()).toBe("Not found");
    expect(row.text()).toContain("12");
  });

  it("says so when a folder is on disk but the server cannot read it", async () => {
    api.listImportFolders.mockResolvedValue({
      folders: [
        {
          ...RENDERS,
          missing: false,
          unreadable: true,
          last_imported_at: "2026-01-02T03:04:05Z",
        },
      ],
    });
    const wrapper = mountSection();
    await flushPromises();
    const status = wrapper.find(".ifs-list .ifs-row [role='status']");
    expect(status.exists()).toBe(true);
    expect(status.text()).toBe("Can't be read");
  });

  it("names a refused folder's rule under its path, ahead of any other state", async () => {
    const rule = "Path is in a restricted system directory: /etc";
    api.listImportFolders.mockResolvedValue({
      folders: [
        // Refused and gone from disk: the refusal is what the user must fix.
        { ...RENDERS, id: 3, missing: true, refused: rule },
        // Refused while readable on disk.
        { ...RENDERS, id: 4, label: "Linked", missing: false, refused: rule },
        { ...RENDERS, id: 5, label: "Plain", missing: false },
      ],
    });
    const wrapper = mountSection();
    await flushPromises();
    const rows = wrapper.findAll(".ifs-list .ifs-row");
    for (const row of rows.slice(0, 2)) {
      expect(row.find("[role='status']").text()).toBe("Refused");
      expect(row.find(".ifs-why").text()).toBe(rule);
    }
    // A folder the rules accept carries no reason line and no status.
    expect(rows[2].find(".ifs-why").exists()).toBe(false);
    expect(rows[2].find("[role='status']").exists()).toBe(false);
    expect(rows[2].text()).toContain("Nothing yet");
  });

  it("adds a folder named after its last component when no name is given", async () => {
    const wrapper = mountSection();
    await flushPromises();
    await button(wrapper, "Add import folder…").trigger("click");
    await wrapper.find(".dialog-stub input").setValue("/home/me/Scans/");
    await wrapper
      .findAll(".dialog-stub [role='radio']")
      .find((r) => r.text() === "Delete original")
      .trigger("click");
    await button(wrapper, "Add").trigger("click");
    await flushPromises();
    expect(api.createFolder).toHaveBeenCalledWith("import", {
      folder: "/home/me/Scans/",
      label: "Scans",
      delete_after_import: true,
    });
    expect(wrapper.find(".dialog-stub").exists()).toBe(false);
  });

  it("edits the name and keeps the folder, and shows why a save failed", async () => {
    api.patchFolder.mockRejectedValueOnce({
      response: { data: { detail: "That folder overlaps a library." } },
    });
    const wrapper = mountSection();
    await flushPromises();
    await button(wrapper, "Edit…").trigger("click");
    const name = wrapper
      .findAll(".dialog-stub input")
      .find((i) => i.element.value === "ComfyUI output");
    await name.setValue("Renders");
    await button(wrapper, "Save").trigger("click");
    await flushPromises();
    expect(api.patchFolder).toHaveBeenCalledWith("import", 3, {
      label: "Renders",
      delete_after_import: false,
    });
    expect(wrapper.find(".ifs-form__error").text()).toContain("overlaps");
  });

  it("asks in the app's dialog before it stops watching, and removes nothing on no", async () => {
    registerConfirmHost();
    const nativeConfirm = vi.spyOn(window, "confirm");
    const wrapper = mountSection();
    await flushPromises();

    await button(wrapper, "Remove").trigger("click");
    await flushPromises();
    expect(activeConfirm.value.options).toMatchObject({
      title: 'Stop watching "ComfyUI output"?',
      message:
        'PixlStash will stop watching "ComfyUI output". What it already imported stays in the library.',
      confirmLabel: "Stop watching",
      danger: false,
    });
    resolveConfirm(false);
    await flushPromises();
    expect(api.deleteFolder).not.toHaveBeenCalled();

    await button(wrapper, "Remove").trigger("click");
    await flushPromises();
    expect(api.deleteFolder).not.toHaveBeenCalled();
    resolveConfirm(true);
    await flushPromises();
    expect(api.deleteFolder).toHaveBeenCalledWith("import", 3);
    expect(nativeConfirm).not.toHaveBeenCalled();
  });

  it("names the folder in the browser's confirm when the app's dialog is not mounted", async () => {
    unregisterConfirmHost();
    const nativeConfirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    const wrapper = mountSection();
    await flushPromises();

    await button(wrapper, "Remove").trigger("click");
    await flushPromises();
    expect(nativeConfirm).toHaveBeenCalledTimes(1);
    expect(nativeConfirm.mock.calls[0][0]).toContain('"ComfyUI output"');
    expect(api.deleteFolder).not.toHaveBeenCalled();
  });
});
