// ModelFolderNameField - Enter applies, Escape cancels, and every key stops
// at the field: it sits inside a tree whose Enter activates the focused row
// and a dialog whose Escape closes it. The Enter and Escape of an input
// method's composition are the IME's: stopped all the same, acting on nothing.

import { afterEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";

import ModelFolderNameField from "./ModelFolderNameField.vue";

const onWindow = vi.fn();
let wrapper = null;

function mountField() {
  wrapper = mount(ModelFolderNameField, {
    props: { modelValue: "Portraits", label: "Folder name" },
    attachTo: document.body,
  });
  window.addEventListener("keydown", onWindow);
  return wrapper;
}

function press(key, init = {}) {
  const event = new KeyboardEvent("keydown", {
    key,
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

describe("ModelFolderNameField keys", () => {
  it("applies on Enter and cancels on Escape, keeping both from the tree", () => {
    mountField();
    expect(press("Enter").defaultPrevented).toBe(true);
    expect(press("Escape").defaultPrevented).toBe(true);
    expect(wrapper.emitted("submit")).toHaveLength(1);
    expect(wrapper.emitted("cancel")).toHaveLength(1);
    expect(onWindow).not.toHaveBeenCalled();
  });

  it.each([
    ["isComposing", { isComposing: true }],
    ["Safari's keyCode 229", { keyCode: 229 }],
  ])(
    "leaves the Enter and Escape of a composition (%s) to the input method",
    (_name, init) => {
      mountField();
      for (const key of ["Enter", "Escape"]) {
        expect(press(key, init).defaultPrevented).toBe(false);
      }
      expect(wrapper.emitted("submit")).toBeUndefined();
      expect(wrapper.emitted("cancel")).toBeUndefined();
      expect(onWindow).not.toHaveBeenCalled();
    },
  );
});
