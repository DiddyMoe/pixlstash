// The active-filter strip's tail, "N of M": M is the view unfiltered, N is
// what the grid has counted under the filters. The reported fault was this
// tail reading "0 of 207,680" while the grid was still loading the filtered
// view, so the tests pin that an uncounted view says so, and that a real 0
// still reads as 0.

import { describe, it, expect, beforeEach, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { setActivePinia, createPinia } from "pinia";

import FilterStrip from "./FilterStrip.vue";
import { useFilterStore } from "../../stores/useFilterStore.js";
import { useGridStore } from "../../stores/useGridStore.js";
import { resetFilterCounts } from "../../composables/useFilterCounts";

vi.mock("../../api/pictures", () => ({
  getPictureCount: vi.fn().mockResolvedValue({ count: 207680 }),
}));

const big = (207680).toLocaleString();

async function mountStrip(props = {}) {
  // A filter on, so the strip is there at all.
  useFilterStore().stackStateFilter = "stacked";
  const wrapper = mount(FilterStrip, {
    props: { countBaseQuery: "set_id=4", ...props },
    global: { stubs: { "v-icon": true, Tooltip: true } },
  });
  await flushPromises();
  return wrapper;
}

const tail = (wrapper) => wrapper.find(".filter-strip-count").text();

beforeEach(() => {
  setActivePinia(createPinia());
  resetFilterCounts();
});

describe("FilterStrip", () => {
  it("says the view is uncounted until the grid has counted it", async () => {
    const wrapper = await mountStrip();
    expect(wrapper.text()).toContain("Stacks");
    expect(tail(wrapper)).toBe(`– of ${big}`);

    useGridStore().matchCount = 1234;
    await flushPromises();
    expect(tail(wrapper)).toBe(`${(1234).toLocaleString()} of ${big}`);
  });

  it("states a view that matched nothing as 0", async () => {
    const wrapper = await mountStrip();
    useGridStore().matchCount = 0;
    await flushPromises();
    expect(tail(wrapper)).toBe(`0 of ${big}`);
  });

  it("prints an owner's own label instead of the grid's count", async () => {
    // The Workflows screen hands its own tail; the grid's count is not its.
    const wrapper = await mountStrip({ ofLabel: "12 workflows" });
    expect(tail(wrapper)).toBe("12 workflows");
  });
});

// The grid counts a collapsed stack once and the Details panel counts every
// picture in it, so while a stack is collapsed the tail says which it is.
describe("FilterStrip's count with stacks collapsed", () => {
  const tip = (wrapper) =>
    wrapper.find(".filter-strip-count tooltip-stub");

  it("says a collapsed stack counts once", async () => {
    const grid = useGridStore();
    grid.totalStackCount = 3;
    grid.expandedStackCount = 1;
    const wrapper = await mountStrip();
    expect(tip(wrapper).attributes("text")).toBe(
      "A collapsed stack counts once here. The Details panel counts every picture in it.",
    );
  });

  it("says nothing when every stack is open", async () => {
    const grid = useGridStore();
    grid.totalStackCount = 2;
    grid.expandedStackCount = 2;
    const wrapper = await mountStrip();
    expect(tip(wrapper).exists()).toBe(false);
  });

  it("says nothing over an owner's own label", async () => {
    const grid = useGridStore();
    grid.totalStackCount = 3;
    grid.expandedStackCount = 0;
    const wrapper = await mountStrip({ ofLabel: "12 workflows" });
    expect(tip(wrapper).exists()).toBe(false);
  });
});
