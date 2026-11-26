// PhotosImportDialog - a folder dropped on its drop zone, with an item the
// browser cannot read.
//
// Two handlers walk that drop: the window's capture handler
// (useWindowFileImport) and the dialog's own zone. The person hears about the
// unreadable item once, not once per walk, and the readable picture is handed
// on to the importer once, by the zone: the window's handler leaves the
// pictures of a zone marked `data-owns-file-drop` to it.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent, h, ref } from "vue";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

import PhotosImportDialog from "./PhotosImportDialog.vue";
import { useWindowFileImport } from "../../composables/useWindowFileImport";
import { useNoticeStore } from "../../stores/useNoticeStore";

vi.mock("../../api/projects.js", () => ({
  listProjects: vi.fn(async () => []),
}));

const PassThrough = { template: "<div><slot /></div>" };
const STUBS = {
  AppDialog: PassThrough,
  ProjectEditor: true,
  AppButton: true,
  "v-select": true,
  "v-tabs": PassThrough,
  "v-tab": true,
  "v-window": PassThrough,
  "v-window-item": PassThrough,
};

const windowImport = vi.fn();
const dialogImport = vi.fn();

const Host = defineComponent({
  setup() {
    useWindowFileImport({
      sidebarRef: ref({
        startLocalImport: windowImport,
        currentProjectId: null,
      }),
    });
    return () =>
      h(PhotosImportDialog, { open: true, onLocalImport: dialogImport });
  },
});

/**
 * A folder drop holding one readable picture and one the browser refuses;
 * with `{unsupported: true}`, one file the importer does not take; with
 * `{onlyUnreadable: true}`, only the item the browser refuses.
 */
function folderDrop({ unsupported = false, onlyUnreadable = false } = {}) {
  const fileEntry = (name, file) => ({
    isFile: true,
    isDirectory: false,
    name,
    fullPath: `/shoot/${name}`,
    file: (ok, fail) =>
      file ? ok(file) : fail?.(new DOMException("gone", "NotFoundError")),
  });
  const children = unsupported
    ? [fileEntry("notes.docx", new File(["x"], "notes.docx"))]
    : onlyUnreadable
      ? [fileEntry("gone.jpg", null)]
      : [
          fileEntry("keep.jpg", new File(["x"], "keep.jpg")),
          fileEntry("gone.jpg", null),
        ];
  const entry = {
    isFile: false,
    isDirectory: true,
    name: "shoot",
    fullPath: "/shoot",
    createReader: () => {
      let done = false;
      return {
        readEntries(cb) {
          if (done) return cb([]);
          done = true;
          cb(children);
        },
      };
    },
  };
  const event = new Event("drop", { bubbles: true, cancelable: true });
  event.dataTransfer = {
    types: ["Files"],
    files: [new File([], "shoot")],
    items: [{ kind: "file", webkitGetAsEntry: () => entry }],
    getData: () => "",
  };
  return event;
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

let wrapper;

beforeEach(() => {
  setActivePinia(createPinia());
  windowImport.mockClear();
  dialogImport.mockClear();
  vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  wrapper?.unmount();
  vi.restoreAllMocks();
});

describe("a folder dropped on the import dialog with an unreadable item", () => {
  it("warns once and imports the readable picture", async () => {
    wrapper = mount(Host, {
      attachTo: document.body,
      global: { stubs: STUBS },
    });
    const zone = wrapper.find(".local-import-dropzone");
    expect(zone.exists()).toBe(true);

    zone.element.dispatchEvent(folderDrop());
    for (let i = 0; i < 4; i++) await flush();

    const unreadable = useNoticeStore().notices.filter(
      (n) => n.key === "import-unreadable-items",
    );
    expect(unreadable.map((n) => [n.text, n.count])).toEqual([
      ["1 item could not be read and was skipped", 1],
    ]);
    expect(
      useNoticeStore().notices.filter(
        (n) => n.key === "import-unsupported-files",
      ),
    ).toEqual([]);
    expect(dialogImport).toHaveBeenCalledTimes(1);
    expect(dialogImport.mock.calls[0][0].files.map((f) => f.name)).toEqual([
      "keep.jpg",
    ]);
    expect(windowImport).not.toHaveBeenCalled();
  });

  it("says once that nothing in the drop is supported", async () => {
    wrapper = mount(Host, {
      attachTo: document.body,
      global: { stubs: STUBS },
    });

    wrapper
      .find(".local-import-dropzone")
      .element.dispatchEvent(folderDrop({ unsupported: true }));
    for (let i = 0; i < 4; i++) await flush();

    expect(
      useNoticeStore()
        .notices.filter((n) => n.key === "import-unsupported-files")
        .map((n) => [n.text, n.count]),
    ).toEqual([
      ["None of those files are a supported image, video or archive.", 1],
    ]);
    expect(dialogImport).not.toHaveBeenCalled();
    expect(windowImport).not.toHaveBeenCalled();
  });

  it("says only that the item could not be read when nothing else came", async () => {
    wrapper = mount(Host, {
      attachTo: document.body,
      global: { stubs: STUBS },
    });

    wrapper
      .find(".local-import-dropzone")
      .element.dispatchEvent(folderDrop({ onlyUnreadable: true }));
    for (let i = 0; i < 4; i++) await flush();

    const notices = useNoticeStore().notices;
    expect(
      notices
        .filter((n) => n.key === "import-unreadable-items")
        .map((n) => [n.text, n.count]),
    ).toEqual([["1 item could not be read and was skipped", 1]]);
    // The drop did carry a file; "nothing supported" would be wrong.
    expect(notices.filter((n) => n.key === "import-unsupported-files")).toEqual(
      [],
    );
    expect(dialogImport).not.toHaveBeenCalled();
    expect(windowImport).not.toHaveBeenCalled();
  });
});
