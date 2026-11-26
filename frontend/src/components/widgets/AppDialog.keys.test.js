// A key pressed in an open dialog is the dialog's, against Vuetify's real
// VDialog.
//
// AppDialog keeps Escape and plain Enter for itself and lets every other key
// bubble, so the window's shortcut listeners (the grid's, useGridKeyboardNav;
// the app's, useGlobalKeydown) see Delete, PageDown and the rest. They decline
// while a modal overlay is up, and they find one by the scrim Vuetify renders
// for it (`.v-overlay--active .v-overlay__scrim`). The other dialog tests mock
// `vuetify/components` away, which hides whether the real dialog renders that
// scrim at all; `vi.mock` is file-scoped, so this needs its own file with the
// real components (as AppDialog.a11y.test.js does).

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createVuetify } from "vuetify";
import * as vuetifyComponents from "vuetify/components";
import * as vuetifyDirectives from "vuetify/directives";
import { nextTick, ref } from "vue";
import { createPinia, setActivePinia } from "pinia";

vi.mock("../../utils/apiClient", () => ({
  API_BASE_URL: "/api/v1",
  isReadOnly: { value: false },
  newOperationBatchId: () => "cli-test",
  onSessionReset: () => () => {},
}));

import AppDialog from "./AppDialog.vue";
import ConfirmDialog from "./ConfirmDialog.vue";
import DeleteForeverDialog from "./DeleteForeverDialog.vue";
import { useGridKeyboardNav } from "../../composables/useGridKeyboardNav.js";
import { useGlobalKeydown } from "../../composables/useGlobalKeydown.js";
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
let shortcuts = null;
let deleteSelected;
let grid;

/**
 * The grid's and the app's window listeners, as ImageGrid and App install
 * them, with a selection for Delete to act on.
 */
const Shortcuts = {
  setup() {
    const nav = useGridKeyboardNav(
      {
        scrollWrapper: ref(null),
        allGridImages: ref([{ id: 1 }, { id: 2 }]),
        rowHeight: ref(128),
        visibleStart: ref(0),
        lightboxOpen: ref(false),
        reviewOverlayOpen: ref(false),
        showSelectionBar: ref(true),
        searchResultsActive: ref(false),
        selectedImageIds: ref([1, 2]),
        lastSelectedImageId: null,
        cursorIdx: ref(0),
        isMultiCharacterView: ref(false),
        isSetOverlapView: ref(false),
        hoveredImageIdx: ref(null),
        toolbarSelectionMenuOpen: ref(false),
        isJustifiedMode: ref(false),
        justifiedLayout: ref(null),
      },
      {},
      vi.fn(),
      {
        clearFaceSelection: vi.fn(),
        clearSearchQuery: vi.fn(),
        scrollCursorIntoView: vi.fn(),
        openLightbox: vi.fn(),
        deleteSelected,
        selectionBarRef: ref(null),
        applyScoresForSelection: vi.fn(),
        setScore: vi.fn(),
      },
    );
    useGlobalKeydown({
      gridContainer: ref(grid),
      sidebarRef: ref(null),
      shortcutsDialogOpen: ref(false),
    });
    window.addEventListener("keydown", nav.handleKeyDown);
    return { handleKeyDown: nav.handleKeyDown };
  },
  unmounted() {
    window.removeEventListener("keydown", this.handleKeyDown);
  },
  render: () => null,
};

beforeEach(() => {
  setActivePinia(createPinia());
  deleteSelected = vi.fn();
  grid = { onGlobalKeyPress: vi.fn() };
  shortcuts = mount(Shortcuts, { attachTo: document.body });
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  shortcuts?.unmount();
  shortcuts = null;
  unregisterConfirmHost();
});

async function render(component, props = {}, slots = {}) {
  wrapper = mount(component, {
    attachTo: document.body,
    props,
    slots,
    global: { plugins: [vuetify] },
  });
  await nextTick();
  await nextTick();
}

function press(target, key) {
  target.dispatchEvent(
    new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }),
  );
}

/**
 * A dialog button by its label (a key hint may follow it): focus on it is not
 * a typing surface.
 */
function dialogButton(text) {
  const button = [...document.querySelectorAll(".v-overlay--active button")]
    .find((el) => el.textContent.trim().startsWith(text));
  expect(button, `no "${text}" button in the open dialog`).toBeTruthy();
  return button;
}

function expectShortcutsInert(button) {
  button.focus();
  for (const key of ["Delete", "Backspace", "PageDown", "Home"]) {
    press(button, key);
  }
  expect(deleteSelected).not.toHaveBeenCalled();
  expect(grid.onGlobalKeyPress).not.toHaveBeenCalled();
}

describe("keys pressed in an open dialog stay there", () => {
  it("reach the grid with no dialog open (the control)", () => {
    press(document.body, "Delete");
    press(document.body, "PageDown");
    expect(deleteSelected).toHaveBeenCalledTimes(1);
    expect(grid.onGlobalKeyPress).toHaveBeenCalledWith(
      "PageDown",
      expect.any(KeyboardEvent),
    );
  });

  it("from Delete forever's own button", async () => {
    await render(DeleteForeverDialog, {
      open: true,
      totalCount: 2,
      protectedCount: 0,
      unprotectedCount: 2,
    });
    expectShortcutsInert(dialogButton("Delete forever"));
  });

  it("from a confirm that is not a delete (Restore all's)", async () => {
    await render(ConfirmDialog);
    const pending = useConfirm().confirm({
      title: "Restore all pictures?",
      message: "They return to the library.",
      confirmLabel: "Restore",
    });
    await nextTick();
    await nextTick();
    expectShortcutsInert(dialogButton("Cancel"));
    resolveConfirm(false);
    await pending;
  });

  it("from any AppDialog", async () => {
    await render(
      AppDialog,
      { open: true, title: "Settings" },
      { default: '<button type="button">Pick</button>' },
    );
    expectShortcutsInert(dialogButton("Pick"));
  });
});
