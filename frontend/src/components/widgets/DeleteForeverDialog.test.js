// Delete forever is the last consent before files leave the disk, so every
// figure it shows must read the same way the progress beside it does: "3,000",
// never "3000". These tests pin the grouped figure in each sentence and label
// that carries a count, and that the singular and plural wording still follows
// the number itself.

import { describe, it, expect } from "vitest";
import { mount } from "@vue/test-utils";

import DeleteForeverDialog from "./DeleteForeverDialog.vue";

// The real AppDialog renders through Vuetify's overlay teleport, which puts the
// footer out of the wrapper's reach. This stub keeps both slots in place.
const AppDialogStub = {
  name: "AppDialog",
  props: { open: Boolean, title: String, size: String },
  emits: ["close"],
  template: `<div class="dlg">
      <slot />
      <div class="dlg-footer"><slot name="footer" /></div>
    </div>`,
};

function mountDialog(props) {
  return mount(DeleteForeverDialog, {
    props: { open: true, ...props },
    global: { stubs: { "v-icon": true, AppDialog: AppDialogStub } },
  });
}

describe("DeleteForeverDialog: counts read as grouped figures", () => {
  it("groups the figure in the standard body", () => {
    const wrapper = mountDialog({ totalCount: 3000, unprotectedCount: 3000 });
    expect(wrapper.find(".confirm-body").text()).toBe(
      "This permanently deletes 3,000 pictures, including their files on disk. This cannot be undone.",
    );
  });

  it("keeps the singular wording for one picture", () => {
    const wrapper = mountDialog({ totalCount: 1, unprotectedCount: 1 });
    expect(wrapper.find(".confirm-body").text()).toBe(
      "This permanently deletes 1 picture, including its file on disk. This cannot be undone.",
    );
  });

  it("groups every figure in the protected flow", () => {
    const wrapper = mountDialog({
      totalCount: 4500,
      protectedCount: 1500,
      unprotectedCount: 3000,
      protectedPaths: ["/ref/a.png", "/ref/b.png"],
    });
    expect(wrapper.find(".confirm-body").text()).toBe(
      "This permanently deletes 4,500 pictures. 1,500 of them are reference-folder originals - deleting them also destroys the original files on disk. This cannot be undone.",
    );
    expect(wrapper.find(".ref-warn-more").text()).toBe(
      "+ 1,498 more protected original files",
    );
    expect(wrapper.find(".type-confirm-hint").text()).toBe(
      "Type DELETE to permanently destroy the 1,500 protected originals.",
    );
    const labels = wrapper.findAll(".footer-wrap button").map((b) => b.text());
    expect(labels).toContain("Delete unprotected only (3,000)");
    expect(labels).toContain("Delete all - incl. 1,500 protected");
  });
});

describe("DeleteForeverDialog: a capped protected list", () => {
  // The delete preview sends at most 200 protected paths and the
  // exact count beside them; the dialog lists what it got and counts the rest.
  it("lists the 200 paths it was sent and counts the other 100", () => {
    const protectedPaths = Array.from(
      { length: 200 },
      (_, i) => `/ref/${String(i).padStart(3, "0")}.png`,
    );
    const wrapper = mountDialog({
      totalCount: 300,
      protectedCount: 300,
      unprotectedCount: 0,
      protectedPaths,
    });
    const items = wrapper.findAll(".ref-warn-paths li");
    expect(items).toHaveLength(201);
    expect(items[0].text()).toBe("/ref/000.png");
    expect(items[199].text()).toBe("/ref/199.png");
    expect(wrapper.find(".ref-warn-more").text()).toBe(
      "+ 100 more protected original files",
    );
  });
});
