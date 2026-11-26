import { afterEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { nextTick } from "vue";

vi.mock("vuetify/components", () => ({
  VIcon: { template: "<i><slot /></i>" },
}));
vi.mock("../widgets/AppDialog.vue", () => ({
  default: {
    props: ["open", "title"],
    template: `
      <div class="app-dialog-stub">
        <slot name="header-right" />
        <slot />
      </div>
    `,
  },
}));
vi.mock("../widgets/AppButton.vue", () => ({
  default: { template: "<button><slot /></button>" },
}));

const sectionStub = vi.hoisted(() => ({
  template: '<div class="section-stub" />',
}));
// The panes that are told both whether the dialog is open and whether their own
// tab is the one showing. The stub writes both props out so a test can read them.
const paneStub = vi.hoisted(() => (cls) => ({
  props: { open: Boolean, active: Boolean },
  template: `<div class="${cls}" :data-open="String(open)" :data-active="String(active)" />`,
}));
vi.mock("./AccountSection.vue", () => ({ default: sectionStub }));
vi.mock("./AppearanceSection.vue", () => ({ default: sectionStub }));
vi.mock("./BehaviourSection.vue", () => ({
  default: paneStub("behaviour-stub"),
}));
vi.mock("./ComputeSection.vue", () => ({ default: paneStub("compute-stub") }));
vi.mock("./LibrariesSection.vue", () => ({ default: sectionStub }));
vi.mock("./ImportFoldersSection.vue", () => ({ default: sectionStub }));
vi.mock("./ModelFoldersSection.vue", () => ({ default: sectionStub }));
vi.mock("./PrivacySection.vue", () => ({ default: sectionStub }));
vi.mock("./SharingSection.vue", () => ({ default: sectionStub }));
vi.mock("./ScrapheapSection.vue", () => ({ default: sectionStub }));
vi.mock("./SnapshotsSection.vue", () => ({
  default: paneStub("snapshots-stub"),
}));
vi.mock("./GhostsSection.vue", () => ({ default: sectionStub }));
vi.mock("./AiToolkitSetup.vue", () => ({ default: sectionStub }));
vi.mock("./SmartScoreSection.vue", () => ({ default: sectionStub }));
vi.mock("./ComfyuiHostSection.vue", () => ({
  default: {
    props: { first: Boolean },
    template: '<div class="comfyui-host-stub" :data-first="String(first)" />',
  },
}));

import UserSettingsDialog from "./UserSettingsDialog.vue";
import { sessionContext } from "../../utils/apiClient";

function mountDialog() {
  return mount(UserSettingsDialog, {
    props: { open: true, initialTab: "libraries" },
    global: {
      stubs: {
        VIcon: true,
      },
    },
  });
}

afterEach(() => {
  sessionContext.value = null;
});

describe("UserSettingsDialog library navigation", () => {
  it("deep-links an owner to the semantic Libraries region", async () => {
    sessionContext.value = { scope: "ALL" };
    const wrapper = mountDialog();
    await nextTick();

    const librariesNav = wrapper.find("#settings-nav-libraries");
    expect(librariesNav.attributes("aria-current")).toBe("page");
    expect(librariesNav.attributes("aria-controls")).toBe(
      "settings-pane-libraries",
    );
    expect(wrapper.find("#settings-pane-libraries").attributes("role")).toBe(
      "region",
    );
  });

  it("groups the rail under five headings, and has no layout tab", async () => {
    sessionContext.value = { scope: "ALL" };
    const wrapper = mountDialog();
    await nextTick();

    // The whole order, headings included, not an adjacency: an index
    // comparison between two entries also holds when one of them is missing.
    // Server is desktop-only and absent under jsdom; GPU & compute and
    // ComfyUI stay, because a browser has a GPU picker and connects ComfyUI.
    // No Library layout item: the layout belongs to whichever library is
    // open, so it is a dialog off the active library's overflow menu.
    expect(
      wrapper
        .findAll(".settings-nav-heading, .settings-nav-item")
        .map((n) => n.attributes("id") || n.text()),
    ).toEqual([
      "General",
      "settings-nav-appearance",
      "settings-nav-privacy",
      "Library",
      "settings-nav-libraries",
      "settings-nav-recovery",
      "Tags",
      "settings-nav-behaviour",
      "settings-nav-smart-score",
      "Models & AI",
      "settings-nav-models",
      "settings-nav-training",
      "settings-nav-comfyui",
      "settings-nav-compute",
      "Access",
      "settings-nav-account",
    ]);
  });

  it.each([
    ["import-folders", "libraries"],
    ["scrapheap", "recovery"],
    ["snapshots", "recovery"],
    ["sharing", "privacy"],
  ])(
    "lands a link to the old %s pane on %s, at its section",
    async (oldId, pane) => {
      sessionContext.value = { scope: "ALL" };
      const scrolled = vi.fn();
      Element.prototype.scrollIntoView = scrolled;
      const wrapper = mount(UserSettingsDialog, {
        props: { open: true, initialTab: oldId },
        global: { stubs: { VIcon: true } },
        attachTo: document.body,
      });
      try {
        await nextTick();
        await nextTick();

        expect(
          wrapper.find(`#settings-nav-${pane}`).attributes("aria-current"),
        ).toBe("page");
        const anchor = wrapper.find(`#settings-anchor-${oldId}`);
        expect(anchor.exists()).toBe(true);
        expect(
          wrapper.find(`#settings-pane-${pane}`).element.contains(anchor.element),
        ).toBe(true);
        expect(scrolled.mock.contexts).toEqual([anchor.element]);
      } finally {
        wrapper.unmount();
        delete Element.prototype.scrollIntoView;
      }
    },
  );

  it("gives every rail item a pane matching its aria-controls", async () => {
    sessionContext.value = { scope: "ALL" };
    const wrapper = mountDialog();
    await nextTick();

    for (const item of wrapper.findAll(".settings-nav-item")) {
      const pane = wrapper.find(`#${item.attributes("aria-controls")}`);
      expect(pane.exists()).toBe(true);
      expect(pane.attributes("role")).toBe("region");
      expect(pane.attributes("aria-labelledby")).toBe(item.attributes("id"));
    }
  });

  it("gives the ComfyUI host its own pane in a browser", async () => {
    sessionContext.value = { scope: "ALL" };
    const wrapper = mount(UserSettingsDialog, {
      props: { open: true, initialTab: "comfyui" },
      global: { stubs: { VIcon: true } },
    });
    await nextTick();

    expect(wrapper.find("#settings-nav-comfyui").attributes("aria-current")).toBe(
      "page",
    );
    const pane = wrapper.find("#settings-pane-comfyui");
    // Alone in the pane, so it carries no divider above nothing.
    expect(pane.find(".comfyui-host-stub").attributes("data-first")).toBe(
      "true",
    );
    // The Compute pane is on a browser as well (which GPU, what it may take).
    expect(wrapper.find(".compute-stub").exists()).toBe(true);
  });

  it("moves an open dialog to the pane a notice asks for", async () => {
    // Notices sit above the dialog, so the saved-address notice's "Open
    // Settings" can ask for the ComfyUI pane while Settings is already open on
    // another one.
    sessionContext.value = { scope: "ALL" };
    const wrapper = mountDialog();
    await nextTick();
    await wrapper.find("#settings-nav-appearance").trigger("click");
    expect(
      wrapper.find("#settings-nav-appearance").attributes("aria-current"),
    ).toBe("page");

    await wrapper.setProps({ initialTab: "comfyui" });

    expect(wrapper.find("#settings-nav-comfyui").attributes("aria-current")).toBe(
      "page",
    );
    expect(
      wrapper.find("#settings-nav-appearance").attributes("aria-current"),
    ).toBeUndefined();
  });

  it("moves an open dialog back to a pane asked for again", async () => {
    // Opened on ComfyUI from a notice, clicked away to Appearance, and the
    // notice asks for ComfyUI again: the pane named is the same, so only the
    // request count changes.
    sessionContext.value = { scope: "ALL" };
    const wrapper = mount(UserSettingsDialog, {
      props: { open: true, initialTab: "comfyui", tabRequest: 1 },
      global: { stubs: { VIcon: true } },
    });
    await nextTick();
    await wrapper.find("#settings-nav-appearance").trigger("click");
    expect(
      wrapper.find("#settings-nav-appearance").attributes("aria-current"),
    ).toBe("page");

    await wrapper.setProps({ tabRequest: 2 });

    expect(wrapper.find("#settings-nav-comfyui").attributes("aria-current")).toBe(
      "page",
    );
    expect(
      wrapper.find("#settings-nav-appearance").attributes("aria-current"),
    ).toBeUndefined();
  });

  it("splits ComfyUI and Compute into two desktop panes", async () => {
    // `isDesktop` is read when the module is evaluated, so the bridge has to
    // be in place before the import.
    sessionContext.value = { scope: "ALL" };
    window.pixlstashDesktop = {};
    try {
      vi.resetModules();
      const { default: Desktop } = await import("./UserSettingsDialog.vue");
      const wrapper = mount(Desktop, {
        props: { open: true, initialTab: "compute" },
        global: { stubs: { VIcon: true } },
      });
      await nextTick();

      expect(
        wrapper.findAll(".settings-nav-item").map((n) => n.attributes("id")),
      ).toEqual([
        "settings-nav-appearance",
        "settings-nav-privacy",
        "settings-nav-libraries",
        "settings-nav-recovery",
        "settings-nav-behaviour",
        "settings-nav-smart-score",
        "settings-nav-models",
        "settings-nav-training",
        "settings-nav-comfyui",
        "settings-nav-compute",
        "settings-nav-backend",
        "settings-nav-account",
      ]);
      expect(wrapper.find("#settings-nav-compute").attributes("aria-current")).toBe(
        "page",
      );
      const compute = wrapper.find("#settings-pane-compute");
      expect(compute.find(".compute-stub").exists()).toBe(true);
      expect(compute.find(".comfyui-host-stub").exists()).toBe(false);
      const comfyui = wrapper.find("#settings-pane-comfyui");
      expect(comfyui.find(".comfyui-host-stub").attributes("data-first")).toBe(
        "true",
      );
      expect(comfyui.find(".compute-stub").exists()).toBe(false);
    } finally {
      delete window.pixlstashDesktop;
      vi.resetModules();
    }
  });

  it("does not disclose the nav entry or pane in read-only/share mode", async () => {
    sessionContext.value = { scope: "READ", resource_type: "picture_set" };
    const wrapper = mountDialog();
    await nextTick();

    expect(wrapper.find("#settings-nav-libraries").exists()).toBe(false);
    expect(wrapper.find("#settings-pane-libraries").exists()).toBe(false);
    expect(wrapper.find("#settings-nav-appearance").attributes("aria-current")).toBe(
      "page",
    );
    // One item left: a heading over it would be noise.
    expect(wrapper.findAll(".settings-nav-item")).toHaveLength(1);
    expect(wrapper.find(".settings-nav-heading").exists()).toBe(false);
  });
});

// Every pane is mounted while the dialog is open; v-show only hides the ones
// whose tab is not showing. A pane told `active` is the one the user can see,
// so it may poll or make its first read; the others must be told they are not.
describe("UserSettingsDialog tells each pane whether its tab is showing", () => {
  const PANES = {
    behaviour: "#settings-pane-behaviour .behaviour-stub",
    snapshots: "#settings-pane-recovery .snapshots-stub",
    compute: "#settings-pane-compute .compute-stub",
  };

  /** The panes whose given prop is true, by name. */
  function panesWith(wrapper, prop, panes = PANES) {
    return Object.entries(panes)
      .filter(([, selector]) => {
        const pane = wrapper.find(selector);
        expect(pane.exists()).toBe(true);
        return pane.attributes(`data-${prop}`) === "true";
      })
      .map(([name]) => name);
  }

  it("marks only the pane of the tab that is showing", async () => {
    sessionContext.value = { scope: "ALL" };
    const wrapper = mountDialog();
    await nextTick();

    expect(panesWith(wrapper, "open")).toEqual([
      "behaviour",
      "snapshots",
      "compute",
    ]);
    expect(panesWith(wrapper, "active")).toEqual([]);

    await wrapper.find("#settings-nav-behaviour").trigger("click");
    expect(panesWith(wrapper, "active")).toEqual(["behaviour"]);

    await wrapper.find("#settings-nav-recovery").trigger("click");
    expect(panesWith(wrapper, "active")).toEqual(["snapshots"]);

    await wrapper.find("#settings-nav-compute").trigger("click");
    expect(panesWith(wrapper, "active")).toEqual(["compute"]);
  });

  it.each([
    ["behaviour", "behaviour"],
    ["recovery", "snapshots"],
    ["compute", "compute"],
  ])(
    "takes the %s tab's pane out of view when the dialog closes on it",
    async (tab, pane) => {
      sessionContext.value = { scope: "ALL" };
      const wrapper = mount(UserSettingsDialog, {
        props: { open: true, initialTab: tab },
        global: { stubs: { VIcon: true } },
      });
      await nextTick();
      expect(panesWith(wrapper, "active")).toEqual([pane]);

      await wrapper.setProps({ open: false });
      expect(panesWith(wrapper, "open")).toEqual([]);
      expect(panesWith(wrapper, "active")).toEqual([]);
    },
  );

  it("tells the desktop Server pane apart from the Compute pane", async () => {
    // `isDesktop` is read when the module is evaluated, so the bridge has to
    // be in place before the import.
    sessionContext.value = { scope: "ALL" };
    window.pixlstashDesktop = {};
    try {
      vi.resetModules();
      const { default: Desktop } = await import("./UserSettingsDialog.vue");
      const wrapper = mount(Desktop, {
        props: { open: true, initialTab: "backend" },
        global: { stubs: { VIcon: true } },
      });
      await nextTick();
      const panes = {
        compute: PANES.compute,
        backend: "#settings-pane-backend .compute-stub",
      };

      expect(panesWith(wrapper, "active", panes)).toEqual(["backend"]);

      await wrapper.setProps({ open: false });
      expect(panesWith(wrapper, "active", panes)).toEqual([]);
    } finally {
      delete window.pixlstashDesktop;
      vi.resetModules();
    }
  });
});
