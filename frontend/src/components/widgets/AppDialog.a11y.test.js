// AppDialog's `alert` prop, against Vuetify's real VDialog.
//
// A confirmation that stops the work is an alertdialog, named by its title and
// described by its message (WAI-ARIA APG,
// https://www.w3.org/WAI/ARIA/apg/patterns/alertdialog/). The other dialog
// tests mock `vuetify/components` away, which hides the whole question: which
// element Vuetify puts its own role="dialog" on, and whether ours replaces it
// there rather than nesting a second dialog. `vi.mock` is file-scoped, so this
// needs its own file with the real components (as
// settings/LibrarySwitchOverlay.a11y.test.js does).

import { afterEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { createVuetify } from "vuetify";
import * as vuetifyComponents from "vuetify/components";
import * as vuetifyDirectives from "vuetify/directives";
import { nextTick } from "vue";
import { createPinia, setActivePinia } from "pinia";

import AppDialog from "./AppDialog.vue";
import ConfirmDialog from "./ConfirmDialog.vue";
import DeleteForeverDialog from "./DeleteForeverDialog.vue";
import RetentionReductionDialog from "./RetentionReductionDialog.vue";
import ShelfRemoveFolderDialog from "../panels/ShelfRemoveFolderDialog.vue";
import {
  resolveConfirm,
  unregisterConfirmHost,
  useConfirm,
} from "../../composables/useConfirm";

const vuetify = createVuetify({
  components: vuetifyComponents,
  directives: vuetifyDirectives,
});

let wrapper = null;

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  unregisterConfirmHost();
});

async function render(component, props = {}) {
  wrapper = mount(component, {
    attachTo: document.body,
    props,
    global: { plugins: [vuetify] },
  });
  await nextTick();
  await nextTick();
}

/** The text an AT joins from an IDREF list, in order. */
function fromIdRefs(element, attribute) {
  const refs = element.getAttribute(attribute);
  expect(refs, `${attribute} is missing`).toBeTruthy();
  return refs
    .split(/\s+/)
    .map((id) => {
      const target = document.getElementById(id);
      expect(target, `${attribute} points at missing id "${id}"`).toBeTruthy();
      return target.textContent.trim().replace(/\s+/g, " ");
    })
    .join(" ");
}

/** The one alertdialog on the page; no plain dialog nested around or in it. */
function theAlertDialog() {
  const dialogs = document.querySelectorAll('[role="alertdialog"]');
  expect(dialogs).toHaveLength(1);
  expect(document.querySelectorAll('[role="dialog"]')).toHaveLength(0);
  return dialogs[0];
}

describe("AppDialog alert", () => {
  it("keeps a plain dialog a dialog, described by nothing", async () => {
    await render(AppDialog, { open: true, title: "Settings" });
    const dialog = document.querySelector('[role="dialog"]');
    expect(dialog).not.toBeNull();
    expect(document.querySelectorAll('[role="alertdialog"]')).toHaveLength(0);
    expect(dialog.hasAttribute("aria-describedby")).toBe(false);
  });

  it("makes every useConfirm question an alertdialog, described by its message", async () => {
    await render(ConfirmDialog);
    const pending = useConfirm().confirm({
      title: "Restore all pictures?",
      message: "They return to the library.",
      warning: "2 share links stop working.",
    });
    await nextTick();
    await nextTick();

    const dialog = theAlertDialog();
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    expect(fromIdRefs(dialog, "aria-labelledby")).toBe("Restore all pictures?");
    // textContent runs the two paragraphs together; an AT reads them apart.
    const description = fromIdRefs(dialog, "aria-describedby");
    expect(description).toContain("They return to the library.");
    expect(description).toContain("2 share links stop working.");

    resolveConfirm(false);
    await expect(pending).resolves.toBe(false);
  });

  it("describes Delete forever by what it spares and destroys, not by the path list", async () => {
    await render(DeleteForeverDialog, {
      open: true,
      totalCount: 4,
      protectedCount: 1,
      unprotectedCount: 2,
      lockedCount: 1,
      protectedPaths: ["/ref/a.png"],
    });

    const dialog = theAlertDialog();
    expect(fromIdRefs(dialog, "aria-labelledby")).toBe("Delete forever?");
    const description = fromIdRefs(dialog, "aria-describedby");
    expect(description).toContain(
      document.querySelector(".lock-note").textContent.trim(),
    );
    expect(description).toContain(
      document.querySelector(".confirm-body").textContent.trim(),
    );
    expect(description).not.toContain("/ref/a.png");
  });

  it("makes shortening the auto-empty window an alertdialog, described by its body", async () => {
    await render(RetentionReductionDialog, {
      open: true,
      title: "Shorten the auto-empty window?",
      body: "1,200 pictures will be deleted forever from 3 May.",
      warning: "This cannot be undone.",
    });

    const dialog = theAlertDialog();
    expect(fromIdRefs(dialog, "aria-labelledby")).toBe(
      "Shorten the auto-empty window?",
    );
    expect(fromIdRefs(dialog, "aria-describedby")).toContain(
      "1,200 pictures will be deleted forever from 3 May.",
    );
  });

  it("describes removing a model folder by its lead sentence, not its file columns", async () => {
    setActivePinia(createPinia());
    await render(ShelfRemoveFolderDialog, {
      open: true,
      source: { folderId: 1, path: "Lora/old", name: "old" },
      plan: {
        dry_run: true,
        folder_id: 1,
        path: "Lora/old",
        name: "old",
        files: [
          { relpath: "notes.txt", size: 20, model_id: null, link: false },
        ],
        models: [],
        breaks: { people: [], sets: [], workflows: [] },
        pictures: 0,
        copies: [],
        fingerprint: "fp-1",
      },
    });

    const dialog = theAlertDialog();
    expect(fromIdRefs(dialog, "aria-labelledby")).toBe("Remove folder");
    const description = fromIdRefs(dialog, "aria-describedby");
    expect(description).toBe(
      document
        .querySelector('[data-testid="remove-folder-lead"]')
        .textContent.trim()
        .replace(/\s+/g, " "),
    );
    expect(description).not.toContain("notes.txt");
  });
});
