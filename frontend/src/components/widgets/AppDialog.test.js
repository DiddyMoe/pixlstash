// AppDialog - the dialog keyboard contract (owner decision, 2026-07-29).
//
// Escape dismisses and plain Enter accepts, handled on the dialog's own
// subtree so no page-level Escape owner is consulted first. Enter is inert
// wherever the key already has a meaning: multiline fields, buttons and links
// (native activation must win, so Enter on a focused Cancel cancels), selects,
// summaries, anything that already handled the event, and the Enter that ends
// an input method's composition.

import { describe, it, expect, vi } from "vitest";
import { mount } from "@vue/test-utils";

vi.mock("vuetify/components", () => ({
  VIcon: { name: "v-icon", template: "<i><slot /></i>" },
  VDialog: { name: "v-dialog", template: "<div><slot /></div>" },
  // Tooltip.vue wraps VTooltip: render the activator only, as a closed tip does.
  VTooltip: {
    name: "VTooltip",
    setup:
      (_p, { slots }) =>
      () =>
        slots.activator?.({ props: {} }),
  },
}));

import AppDialog from "./AppDialog.vue";

const BodyStub = {
  template: `
    <div>
      <input class="txt" type="text" />
      <textarea class="area"></textarea>
      <button class="btn" type="button">Cancel</button>
      <div class="row" role="radio" tabindex="0"></div>
    </div>
  `,
};

function mountDialog(props = {}, options = {}) {
  return mount(AppDialog, {
    props: { open: true, title: "T", ...props },
    slots: { default: BodyStub },
    ...options,
  });
}

describe("AppDialog keyboard contract", () => {
  it("emits close on Escape from anywhere inside", async () => {
    const w = mountDialog();
    await w.find(".txt").trigger("keydown", { key: "Escape" });
    expect(w.emitted("close")).toBeTruthy();
  });

  it("does not close on Escape while persistent", async () => {
    const w = mountDialog({ persistent: true });
    await w.find(".txt").trigger("keydown", { key: "Escape" });
    expect(w.emitted("close")).toBeFalsy();
  });

  it("does not accept on Enter while persistent - a destructive accept needs its button", async () => {
    const w = mountDialog({ persistent: true });
    await w.find(".txt").trigger("keydown", { key: "Enter" });
    expect(w.emitted("accept")).toBeFalsy();
  });

  it("emits accept on plain Enter from a single-line input", async () => {
    const w = mountDialog();
    await w.find(".txt").trigger("keydown", { key: "Enter" });
    expect(w.emitted("accept")).toBeTruthy();
  });

  it("leaves Enter alone in a textarea - newlines beat accept", async () => {
    const w = mountDialog();
    await w.find(".area").trigger("keydown", { key: "Enter" });
    expect(w.emitted("accept")).toBeFalsy();
  });

  it("leaves Enter alone on a button - native activation wins", async () => {
    const w = mountDialog();
    await w.find(".btn").trigger("keydown", { key: "Enter" });
    expect(w.emitted("accept")).toBeFalsy();
  });

  it("respects a descendant that already handled Enter", async () => {
    const w = mountDialog();
    const row = w.find(".row").element;
    row.addEventListener("keydown", (e) => e.preventDefault());
    await w.find(".row").trigger("keydown", { key: "Enter" });
    expect(w.emitted("accept")).toBeFalsy();
  });

  it("leaves the Enter that ends an input method's composition to the input method", async () => {
    const w = mountDialog();
    await w
      .find(".txt")
      .trigger("keydown", { key: "Enter", isComposing: true });
    expect(w.emitted("accept")).toBeFalsy();
  });

  it("leaves Safari's composition-ending Enter (keyCode 229) to the input method", async () => {
    const w = mountDialog();
    await w.find(".txt").trigger("keydown", { key: "Enter", keyCode: 229 });
    expect(w.emitted("accept")).toBeFalsy();
  });

  it.each([
    ["isComposing", { isComposing: true }],
    ["Safari's keyCode 229", { keyCode: 229 }],
  ])(
    "leaves the Escape that cancels a composition (%s) to the input method",
    async (_name, init) => {
      // In the document, so an event that is not stopped reaches the window.
      const w = mountDialog({}, { attachTo: document.body });
      // Vuetify's overlay closes on any Escape that reaches the window.
      const onWindow = vi.fn();
      window.addEventListener("keydown", onWindow);
      try {
        await w.find(".txt").trigger("keydown", { key: "Escape", ...init });
        // Before unmount: unmounting clears the wrapper's emitted history.
        expect(w.emitted("close")).toBeFalsy();
        expect(onWindow).not.toHaveBeenCalled();
      } finally {
        window.removeEventListener("keydown", onWindow);
        w.unmount();
      }
    },
  );

  it("ignores modified Enter - Ctrl+Enter stays a dialog-local shortcut", async () => {
    const w = mountDialog();
    await w.find(".txt").trigger("keydown", { key: "Enter", ctrlKey: true });
    expect(w.emitted("accept")).toBeFalsy();
  });

  it("closes from update:model-value only, never from click:outside", async () => {
    // Vuetify emits click:outside even on a persistent dialog, which let a
    // stray click dismiss the folder-mapping wizard mid-answer.
    const w = mountDialog({ persistent: true });
    const dialog = w.findComponent({ name: "v-dialog" });

    dialog.vm.$emit("click:outside");
    expect(w.emitted("close")).toBeFalsy();

    dialog.vm.$emit("update:modelValue", false);
    expect(w.emitted("close")).toHaveLength(1);
  });

  it("the header close button is named and emits close", async () => {
    const w = mountDialog();
    const close = w.find('button[aria-label="Close"]');
    expect(close.exists()).toBe(true);
    await close.trigger("click");
    expect(w.emitted("close")).toHaveLength(1);
  });

  it("names the dialog by its heading", () => {
    const w = mountDialog();
    const id = w.find("h2").attributes("id");
    expect(id).toBeTruthy();
    expect(w.find("[aria-labelledby]").attributes("aria-labelledby")).toBe(id);
  });

  it("sizes by step, md by default", () => {
    expect(mountDialog().find(".app-dialog").classes()).toContain(
      "app-dialog--md",
    );
    expect(mountDialog({ size: "lg" }).find(".app-dialog").classes()).toContain(
      "app-dialog--lg",
    );
  });
});
