// useGlobalKeydown - where the undo chord is allowed to act.
//
// The handler's own invariant is that every undo raises a receipt, which is why
// it already declines behind a modal scrim. The model shelf is the same rule
// reached from the other side: it mounts no `ActionReceipt` and no
// `UndoControl`, so an undo fired there would revert a library action taken on
// a screen the reader has left, with nothing on this one to say so. These tests
// pin the positive control beside the negative - over-declining would be its
// own regression.

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { ref } from "vue";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

import { SHELF_NO_UNDO_KEY, useGlobalKeydown } from "./useGlobalKeydown";
import { useNoticeStore } from "../stores/useNoticeStore";
import { useOperationStore } from "../stores/useOperationStore";
import { useSearchStore } from "../stores/useSearchStore";

/** A host component whose only job is to install the listener. */
const Host = {
  setup() {
    useGlobalKeydown({
      gridContainer: ref(null),
      sidebarRef: ref(null),
      shortcutsDialogOpen: ref(false),
    });
    return () => null;
  },
};

let wrapper;
let store;
let notices;

function press(key, init = {}) {
  window.dispatchEvent(
    new KeyboardEvent("keydown", {
      key,
      ctrlKey: true,
      bubbles: true,
      ...init,
    }),
  );
}

/** Mount the shelf's root signal - the class `ModelShelf.vue` renders. */
function mountShelfMarker(className = "shelf") {
  const el = document.createElement("div");
  el.className = className;
  document.body.appendChild(el);
  return el;
}

beforeEach(() => {
  setActivePinia(createPinia());
  store = useOperationStore();
  notices = useNoticeStore();
  vi.spyOn(store, "undo").mockResolvedValue(undefined);
  vi.spyOn(store, "redo").mockResolvedValue(undefined);
  wrapper = mount(Host, { attachTo: document.body });
});

afterEach(() => {
  wrapper?.unmount();
  document.body.innerHTML = "";
  vi.restoreAllMocks();
});

describe("the undo chord", () => {
  it("undoes and redoes from a narrating view, silently", () => {
    press("z");
    expect(store.undo).toHaveBeenCalledTimes(1);
    press("y");
    expect(store.redo).toHaveBeenCalledTimes(1);
    press("z", { shiftKey: true });
    expect(store.redo).toHaveBeenCalledTimes(2);
    // The receipt narrates a real undo; a notice here would be a second voice.
    expect(notices.notices).toHaveLength(0);
  });

  it("takes a live local receipt's action on the shelf ahead of the log", async () => {
    mountShelfMarker();
    const undo = vi.fn().mockResolvedValue(true);
    const redo = vi.fn().mockResolvedValue({});
    store.setLocalReceiptHost(true);
    store.showLocalReceipt({ summary: "Added", icon: "mdi-plus", undo, redo });
    press("y");
    expect(undo).not.toHaveBeenCalled();
    // The wrong direction says which way works, never "not undoable".
    expect(notices.notices.at(-1).text).toContain(
      "Undo takes that change back",
    );
    expect(notices.notices.at(-1).key).toBe(SHELF_NO_UNDO_KEY);
    press("z");
    await vi.waitFor(() => expect(store.receipt?.mode).toBe("undone"));
    expect(undo).toHaveBeenCalledTimes(1);
    press("y");
    await vi.waitFor(() => expect(redo).toHaveBeenCalledTimes(1));
    expect(store.undo).not.toHaveBeenCalled();
    expect(store.redo).not.toHaveBeenCalled();
  });

  // Every grid-replacing destination now carries UndoControl (in
  // TbGlobalActions) and an ActionReceipt, so the chord acts there as it does
  // on the grid. They used to decline (#1415) because they had neither.
  it.each([
    ["the model shelf", "shelf"],
    ["the Workflows grid", "wfv"],
    ["Moves", "mv"],
    ["Insights", "ins"],
  ])("undoes and redoes on %s, silently", (_name, className) => {
    mountShelfMarker(className);
    press("z");
    expect(store.undo).toHaveBeenCalledTimes(1);
    press("z", { shiftKey: true });
    expect(store.redo).toHaveBeenCalledTimes(1);
    expect(notices.notices).toHaveLength(0);
  });
});

describe("Escape and the sidebar's Find in library (#2448)", () => {
  let sidebar;
  let host;

  beforeEach(() => {
    sidebar = { clearFind: vi.fn(() => true) };
    const sidebarRef = ref(sidebar);
    host = mount(
      {
        setup() {
          useGlobalKeydown({
            gridContainer: ref(null),
            sidebarRef,
            shortcutsDialogOpen: ref(false),
          });
          return () => null;
        },
      },
      { attachTo: document.body },
    );
  });

  afterEach(() => host.unmount());

  it("clears Find from anywhere outside a text field", () => {
    press("Escape", { ctrlKey: false });
    expect(sidebar.clearFind).toHaveBeenCalled();
  });

  it("leaves it alone while typing elsewhere or under a dialog", () => {
    const input = document.createElement("input");
    document.body.appendChild(input);
    input.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    expect(sidebar.clearFind).not.toHaveBeenCalled();

    const overlay = document.createElement("div");
    overlay.className = "v-overlay--active";
    overlay.innerHTML = '<div class="v-overlay__scrim"></div>';
    document.body.appendChild(overlay);
    press("Escape", { ctrlKey: false });
    expect(sidebar.clearFind).not.toHaveBeenCalled();
  });
});

// A key pressed in a modal dialog is the dialog's: AppDialog keeps only Escape
// and Enter for itself, so the rest reach the window. Behind a dialog, Home or
// PageDown used to scroll the grid (and swallow the dialog's own scrolling), F
// moved focus to the search field, and ? opened the shortcuts sheet on top.
describe("the app's shortcuts behind a modal dialog", () => {
  let grid;
  let shortcutsDialogOpen;
  let host;
  let searchFocus;

  beforeEach(() => {
    grid = { onGlobalKeyPress: vi.fn() };
    shortcutsDialogOpen = ref(false);
    host = mount(
      {
        setup() {
          useGlobalKeydown({
            gridContainer: ref(grid),
            sidebarRef: ref(null),
            shortcutsDialogOpen,
          });
          return () => null;
        },
      },
      { attachTo: document.body },
    );
    searchFocus = vi.spyOn(useSearchStore(), "requestSearchFocus");
  });

  afterEach(() => host.unmount());

  /** What Vuetify renders for an open dialog: an active overlay with a scrim. */
  function openModal() {
    const overlay = document.createElement("div");
    overlay.className = "v-overlay v-overlay--active v-dialog";
    overlay.innerHTML = '<div class="v-overlay__scrim"></div>';
    document.body.appendChild(overlay);
    return overlay;
  }

  function pressPlain(key) {
    const event = new KeyboardEvent("keydown", {
      key,
      bubbles: true,
      cancelable: true,
    });
    window.dispatchEvent(event);
    return event;
  }

  it("leaves Home, End, PageUp, PageDown, F and ? to the dialog", () => {
    openModal();
    const presses = ["Home", "End", "PageUp", "PageDown", "f", "?", "F1"].map(
      pressPlain,
    );

    expect(grid.onGlobalKeyPress).not.toHaveBeenCalled();
    expect(searchFocus).not.toHaveBeenCalled();
    expect(shortcutsDialogOpen.value).toBe(false);
    // The dialog keeps the browser's own handling of its keys too.
    expect(presses.some((event) => event.defaultPrevented)).toBe(false);
  });

  it("still closes the shortcuts sheet with its own key", () => {
    shortcutsDialogOpen.value = true;
    openModal();
    pressPlain("?");
    expect(shortcutsDialogOpen.value).toBe(false);
  });

  it("acts on the same keys with no dialog open", () => {
    pressPlain("PageDown");
    pressPlain("f");
    pressPlain("?");
    expect(grid.onGlobalKeyPress).toHaveBeenCalledWith(
      "PageDown",
      expect.any(KeyboardEvent),
    );
    // (The file's own host is mounted too, so the count is not this one's.)
    expect(searchFocus).toHaveBeenCalled();
    expect(shortcutsDialogOpen.value).toBe(true);
  });
});

// A key pressed on an item of an open menu is the menu's (WAI-ARIA APG menu
// pattern: an open menu owns the keyboard). A menu has no scrim, so the dialog
// checks above miss it, and Vuetify lets its keys bubble to the window: Home
// on a menu item scrolled the grid, F moved focus to the search field, Ctrl+Z
// undid a library change, Escape cleared Find as well as closing the menu.
describe("the app's shortcuts inside an open menu", () => {
  let grid;
  let sidebar;
  let shortcutsDialogOpen;
  let host;
  let searchFocus;

  beforeEach(() => {
    // Only this host listens: the file's own one has no sidebar, and its F2
    // would claim the key before this one could open the editor.
    wrapper.unmount();
    wrapper = null;
    grid = { onGlobalKeyPress: vi.fn() };
    sidebar = {
      clearFind: vi.fn(() => true),
      openCurrentSelectionEditor: vi.fn(),
    };
    shortcutsDialogOpen = ref(false);
    host = mount(
      {
        setup() {
          useGlobalKeydown({
            gridContainer: ref(grid),
            sidebarRef: ref(sidebar),
            shortcutsDialogOpen,
          });
          return () => null;
        },
      },
      { attachTo: document.body },
    );
    searchFocus = vi.spyOn(useSearchStore(), "requestSearchFocus");
  });

  afterEach(() => host.unmount());

  /** An open Vuetify overlay holding one item; a menu has no scrim. */
  function openOverlay(className, { scrim = false } = {}) {
    const overlay = document.createElement("div");
    overlay.className = `v-overlay v-overlay--active ${className}`;
    overlay.innerHTML =
      (scrim ? '<div class="v-overlay__scrim"></div>' : "") +
      '<div class="v-overlay__content"><button type="button" role="menuitem">Item</button></div>';
    document.body.appendChild(overlay);
    return overlay.querySelector("button");
  }

  function pressOn(target, key, init = {}) {
    const event = new KeyboardEvent("keydown", {
      key,
      bubbles: true,
      cancelable: true,
      ...init,
    });
    target.dispatchEvent(event);
    return event;
  }

  const MENU_KEYS = [
    ["Home", { key: "Home" }],
    ["PageDown", { key: "PageDown" }],
    ["F", { key: "f" }],
    ["?", { key: "?" }],
    ["F2", { key: "F2" }],
    ["Ctrl+Z", { key: "z", ctrlKey: true }],
    ["Escape", { key: "Escape" }],
  ];

  it.each(MENU_KEYS)("leaves %s on a menu item to the menu", (_, { key, ...init }) => {
    const item = openOverlay("v-menu");
    const event = pressOn(item, key, init);

    expect(grid.onGlobalKeyPress).not.toHaveBeenCalled();
    expect(searchFocus).not.toHaveBeenCalled();
    expect(shortcutsDialogOpen.value).toBe(false);
    expect(sidebar.openCurrentSelectionEditor).not.toHaveBeenCalled();
    expect(store.undo).not.toHaveBeenCalled();
    expect(sidebar.clearFind).not.toHaveBeenCalled();
    // The menu keeps the browser's own handling of its keys too.
    expect(event.defaultPrevented).toBe(false);
  });

  // The control: the same menu stays open, and the same keys pressed outside
  // it still act, so the guard reads where the key was pressed, not whether a
  // menu exists.
  const APP_EFFECTS = {
    Home: () =>
      expect(grid.onGlobalKeyPress).toHaveBeenCalledWith(
        "Home",
        expect.any(KeyboardEvent),
      ),
    PageDown: () =>
      expect(grid.onGlobalKeyPress).toHaveBeenCalledWith(
        "PageDown",
        expect.any(KeyboardEvent),
      ),
    F: () => expect(searchFocus).toHaveBeenCalled(),
    "?": () => expect(shortcutsDialogOpen.value).toBe(true),
    F2: () => expect(sidebar.openCurrentSelectionEditor).toHaveBeenCalled(),
    "Ctrl+Z": () => expect(store.undo).toHaveBeenCalled(),
  };

  it.each(MENU_KEYS.filter(([name]) => name in APP_EFFECTS))(
    "still acts on %s pressed outside an open menu",
    (name, { key, ...init }) => {
      openOverlay("v-menu");
      const button = document.createElement("button");
      document.body.appendChild(button);
      pressOn(button, key, init);

      APP_EFFECTS[name]();
    },
  );

  it("still closes the keyboard-help sheet with ? pressed inside it", () => {
    shortcutsDialogOpen.value = true;
    const button = openOverlay("v-dialog", { scrim: true });
    pressOn(button, "?");

    expect(shortcutsDialogOpen.value).toBe(false);
  });
});
