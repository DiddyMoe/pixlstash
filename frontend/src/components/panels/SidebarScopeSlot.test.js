// The project dropdown's counts: locale-formatted, and marked when their last
// refresh failed, as the sidebar marks its own rows (SideBar `isCountStale`).
// Before, a failed refresh left the dropdown showing the old number as if it
// were current, while the row under it was already marked.

import { describe, it, expect, vi } from "vitest";
import { mount } from "@vue/test-utils";

vi.mock("vuetify/components", async () => {
  const { vuetifyComponentStubs } = await import("../../testing/vuetifyStubs");
  return vuetifyComponentStubs();
});

import SidebarScopeSlot from "./SidebarScopeSlot.vue";

const STALE_TIP = "Couldn't refresh this count";
const PROJECTS = [{ id: 7, name: "Example project" }];

function mountSlot(counts) {
  return mount(SidebarScopeSlot, {
    props: { projects: PROJECTS, counts },
    global: { stubs: { Tooltip: true } },
  });
}

function metaFor(wrapper, name) {
  const row = wrapper
    .findAll(".ctx-item")
    .find((el) => el.find(".ctx-label-text").text() === name);
  return row.find(".ctx-meta");
}

function isMarkedStale(meta) {
  const tip = meta.find("tooltip-stub");
  return (
    meta.classes().includes("ctx-meta--stale") &&
    tip.exists() &&
    tip.attributes("text") === STALE_TIP
  );
}

describe("SidebarScopeSlot counts", () => {
  it("marks the counts that could not refresh, and only those", () => {
    const wrapper = mountSlot({
      any: 1234567,
      none: 12,
      byId: { 7: 3 },
      stale: { any: true, none: false },
    });
    expect(metaFor(wrapper, "Any project").text()).toBe(
      (1234567).toLocaleString(),
    );
    expect(isMarkedStale(metaFor(wrapper, "Any project"))).toBe(true);
    expect(isMarkedStale(metaFor(wrapper, "Not in a project"))).toBe(false);
    expect(isMarkedStale(metaFor(wrapper, "Example project"))).toBe(false);
  });

  it("marks Not in a project's count on its own", () => {
    const wrapper = mountSlot({
      any: 5,
      none: 4,
      stale: { any: false, none: true },
    });
    expect(metaFor(wrapper, "Not in a project").text()).toBe("4");
    expect(isMarkedStale(metaFor(wrapper, "Not in a project"))).toBe(true);
    expect(isMarkedStale(metaFor(wrapper, "Any project"))).toBe(false);
  });

  it("shows a missing count blank and a real 0 as 0, unmarked", () => {
    const wrapper = mountSlot({ any: 0 });
    expect(metaFor(wrapper, "Any project").text()).toBe("0");
    expect(metaFor(wrapper, "Not in a project").text()).toBe("");
    expect(metaFor(wrapper, "Example project").text()).toBe("");
    expect(isMarkedStale(metaFor(wrapper, "Any project"))).toBe(false);
  });
});
