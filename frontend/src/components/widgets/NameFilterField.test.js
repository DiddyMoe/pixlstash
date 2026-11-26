// NameFilterField - Escape clears the text first, and only a second press
// reaches the menu or view behind it. The Escape that cancels an input
// method's composition does neither.

import { afterEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";

vi.mock("vuetify/components", () => ({
  VIcon: { name: "v-icon", template: "<i><slot /></i>" },
}));

import NameFilterField from "./NameFilterField.vue";

const onWindow = vi.fn();
let wrapper = null;

function mountField(modelValue) {
  wrapper = mount(NameFilterField, {
    props: { modelValue },
    attachTo: document.body,
  });
  window.addEventListener("keydown", onWindow);
  return wrapper;
}

function escape(init = {}) {
  const event = new KeyboardEvent("keydown", {
    key: "Escape",
    bubbles: true,
    cancelable: true,
    ...init,
  });
  wrapper.find("input").element.dispatchEvent(event);
  return event;
}

afterEach(() => {
  window.removeEventListener("keydown", onWindow);
  onWindow.mockReset();
  wrapper?.unmount();
  wrapper = null;
});

describe("NameFilterField Escape", () => {
  it("clears the text and keeps the key from the menu behind", () => {
    mountField("flux");
    const event = escape();
    expect(wrapper.emitted("update:modelValue")).toEqual([[""]]);
    expect(event.defaultPrevented).toBe(true);
    expect(onWindow).not.toHaveBeenCalled();
  });

  it("lets the key through once the field is empty", () => {
    mountField("");
    escape();
    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
    expect(onWindow).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["isComposing", { isComposing: true }],
    ["Safari's keyCode 229", { keyCode: 229 }],
  ])(
    "leaves the Escape that cancels a composition (%s) to the input method",
    (_name, init) => {
      // Safari's arrives after the composition has ended, when the field may
      // already be empty: it must not close the menu either.
      for (const value of ["flux", ""]) {
        mountField(value);
        const event = escape(init);
        expect(wrapper.emitted("update:modelValue")).toBeUndefined();
        expect(event.defaultPrevented).toBe(false);
        expect(onWindow).not.toHaveBeenCalled();
        window.removeEventListener("keydown", onWindow);
        wrapper.unmount();
        wrapper = null;
      }
    },
  );
});
