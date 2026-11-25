// The bar above the Scrapheap's grid: Restore all, Empty scrapheap and the
// auto-empty policy, or, while a background run empties or restores the heap,
// that run's progress and its Stop.
//
// A 50,000-picture heap takes minutes to empty, and the run goes on on the
// server whatever the page does, so this bar is where the user learns how far
// it has got and that they may keep working. The tests pin the sentence each
// state says, which controls each state offers, that a request on the wire
// holds its own control rather than allowing a second click, and that only a
// change of state reaches the live region.

import { describe, it, expect } from "vitest";
import { mount } from "@vue/test-utils";

import EmptyScrapHeap from "./EmptyScrapHeap.vue";

const globalOpts = { global: { stubs: { "v-icon": true } } };

/** A run as `serialize_run` shapes it: a purge under way, by default. */
function run(overrides = {}) {
  return {
    id: "example-run",
    kind: "purge",
    status: "running",
    active: true,
    include_protected: false,
    cancel_requested: false,
    total: 50123,
    processed: 12530,
    deleted_count: 12000,
    skipped_count: 30,
    skipped_locked_count: 500,
    restored_count: 0,
    skipped_restored: 0,
    skipped_locked: [],
    snapshots_with_deleted: null,
    segment_batch_ids: [],
    error_code: null,
    error: null,
    ...overrides,
  };
}

function mountBar(props = {}) {
  return mount(EmptyScrapHeap, {
    ...globalOpts,
    props: { visible: true, retentionLabel: "30 days", ...props },
  });
}

/** Text with runs of whitespace collapsed, so template wrapping is not asserted. */
function textOf(wrapper) {
  return wrapper.text().replace(/\s+/g, " ").trim();
}

const n = (value) => value.toLocaleString();

const byTestId = (w, id) => w.find(`[data-testid="${id}"]`);
const stopButton = (w) => byTestId(w, "scrapheap-run-stop");
const progressLine = (w) => byTestId(w, "scrapheap-run-progress");
const announcement = (w) => byTestId(w, "scrapheap-run-announcement");
const progressbar = (w) => w.find('[role="progressbar"]');
const buttonNamed = (w, label) =>
  w.findAll("button").find((b) => b.text().trim() === label);

describe("EmptyScrapHeap: no run", () => {
  it("offers the heap's own actions and its policy", () => {
    const w = mountBar();
    expect(buttonNamed(w, "Restore all")).toBeDefined();
    expect(buttonNamed(w, "Empty scrapheap")).toBeDefined();
    expect(textOf(w)).toContain("Auto-empty: 30 days");
    expect(stopButton(w).exists()).toBe(false);
    expect(progressLine(w).exists()).toBe(false);
    expect(progressbar(w).exists()).toBe(false);
  });

  it("keeps the heap's actions for a run that has ended", () => {
    // The store keeps the last run for its outcome; an ended one is history,
    // and the bar goes back to what the heap can do now.
    const w = mountBar({
      run: run({ status: "complete", active: false, processed: 50123 }),
    });
    expect(buttonNamed(w, "Empty scrapheap")).toBeDefined();
    expect(stopButton(w).exists()).toBe(false);
    expect(progressbar(w).exists()).toBe(false);
  });

  it("sends each action up, and holds the ones the grid disables", async () => {
    const w = mountBar();
    await buttonNamed(w, "Restore all").trigger("click");
    await buttonNamed(w, "Empty scrapheap").trigger("click");
    await w.find(".retention-note__change").trigger("click");
    expect(w.emitted("restore-scrapheap")).toHaveLength(1);
    expect(w.emitted("empty-scrapheap")).toHaveLength(1);
    expect(w.emitted("open-retention-settings")).toHaveLength(1);

    const held = mountBar({ disabled: true, restoreDisabled: true });
    expect(buttonNamed(held, "Restore all").attributes("disabled")).toBeDefined();
    expect(
      buttonNamed(held, "Empty scrapheap").attributes("disabled"),
    ).toBeDefined();
  });

  it("renders nothing when the grid is not the Scrapheap", () => {
    const w = mount(EmptyScrapHeap, {
      ...globalOpts,
      props: { visible: false, run: run() },
    });
    expect(w.find(".selection-bar-overlay").exists()).toBe(false);
  });
});

describe("EmptyScrapHeap: a run under way", () => {
  it("says how far a purge has got and that work can go on", () => {
    const w = mountBar({ run: run() });
    expect(textOf(progressLine(w))).toBe(
      `Deleting forever: ${n(12530)} of ${n(50123)} pictures. You can keep working.`,
    );
    // 12530 / 50123 = 25%: the share processed, not the share deleted (24%).
    expect(w.find(".run-meta").text()).toBe("25%");
  });

  it("says how far a restore has got", () => {
    const w = mountBar({
      run: run({ kind: "restore", processed: 3, total: 4, restored_count: 3 }),
    });
    expect(textOf(progressLine(w))).toBe(
      "Restoring: 3 of 4 pictures. You can keep working.",
    );
    expect(w.find(".run-meta").text()).toBe("75%");
  });

  it("takes the bar from the heap's actions, so none can start a second run", () => {
    const w = mountBar({ run: run() });
    expect(buttonNamed(w, "Restore all")).toBeUndefined();
    expect(buttonNamed(w, "Empty scrapheap")).toBeUndefined();
  });

  it("is a progress bar with the figures, not a live region", () => {
    // Progress moves every poll; a region that re-read itself would talk over
    // everything else on the page.
    const w = mountBar({ run: run() });
    const bar = progressbar(w);
    expect(bar.attributes("aria-label")).toBe("Delete forever progress");
    expect(bar.attributes("aria-valuenow")).toBe("25");
    expect(bar.attributes("aria-valuetext")).toBe(
      `${n(12530)} of ${n(50123)} pictures done`,
    );
    expect(bar.attributes("aria-live")).toBeUndefined();
    expect(progressLine(w).attributes("aria-live")).toBeUndefined();
    expect(bar.find(".run-track__fill").attributes("style")).toContain(
      "width: 25%",
    );

    const restore = mountBar({ run: run({ kind: "restore" }) });
    expect(progressbar(restore).attributes("aria-label")).toBe(
      "Restore progress",
    );
  });

  it("does not claim a share before the run has counted its pictures", () => {
    const queued = mountBar({
      run: run({ status: "pending", total: 0, processed: 0 }),
    });
    expect(textOf(progressLine(queued))).toBe(
      "Delete forever is queued and starts when the server gets to it. You can keep working.",
    );
    expect(queued.find(".run-meta").exists()).toBe(false);
    expect(progressbar(queued).attributes("aria-valuenow")).toBeUndefined();
    expect(progressbar(queued).attributes("aria-valuetext")).toBeUndefined();

    const restoreQueued = mountBar({
      run: run({ kind: "restore", status: "pending" }),
    });
    expect(textOf(progressLine(restoreQueued))).toContain(
      "Restore is queued and starts when the server gets to it.",
    );

    // Running but not yet counted: no share either, never NaN%.
    const uncounted = mountBar({ run: run({ total: 0, processed: 0 }) });
    expect(uncounted.find(".run-meta").exists()).toBe(false);
    expect(progressbar(uncounted).attributes("aria-valuenow")).toBeUndefined();
  });

  it("never quotes more than all of it", () => {
    const w = mountBar({ run: run({ total: 10, processed: 12 }) });
    expect(w.find(".run-meta").text()).toBe("100%");
    expect(progressbar(w).attributes("aria-valuenow")).toBe("100");
  });

  it("says the figures are stale when the server stops answering", () => {
    const w = mountBar({ run: run(), runUnreachable: true });
    expect(textOf(byTestId(w, "scrapheap-run-unreachable"))).toBe(
      "No answer from the server; these are the last figures read.",
    );
    expect(textOf(progressLine(w))).not.toContain("You can keep working.");
  });
});

describe("EmptyScrapHeap: stopping a run", () => {
  it("sends Stop up", async () => {
    const w = mountBar({ run: run() });
    await stopButton(w).trigger("click");
    expect(w.emitted("stop-run")).toHaveLength(1);
  });

  it("holds Stop while its own request is on the wire", async () => {
    const w = mountBar({ run: run(), runAction: "cancel" });
    const stop = stopButton(w);
    expect(stop.attributes("aria-busy")).toBe("true");
    expect(stop.attributes("disabled")).toBeDefined();
    await stop.trigger("click");
    expect(w.emitted("stop-run")).toBeUndefined();
  });

  it("holds Stop, without a spinner, while a start is on the wire", () => {
    const stop = stopButton(mountBar({ run: run(), runAction: "start" }));
    expect(stop.attributes("disabled")).toBeDefined();
    expect(stop.attributes("aria-busy")).toBeUndefined();
  });

  it("offers Stop when nothing is on the wire", () => {
    const stop = stopButton(mountBar({ run: run() }));
    expect(stop.attributes("disabled")).toBeUndefined();
    expect(stop.attributes("aria-busy")).toBeUndefined();
  });

  it("says what a stopping run has done so far, and offers no second Stop", () => {
    const purge = mountBar({ run: run({ cancel_requested: true }) });
    expect(textOf(progressLine(purge))).toBe(
      `Stopping after the current step: ${n(12000)} deleted so far. You can keep working.`,
    );
    expect(stopButton(purge).exists()).toBe(false);
    expect(purge.find(".run-meta").exists()).toBe(false);

    const restore = mountBar({
      run: run({ kind: "restore", cancel_requested: true, restored_count: 7 }),
    });
    expect(textOf(progressLine(restore))).toBe(
      "Stopping after the current step: 7 restored so far. You can keep working.",
    );
  });
});

describe("EmptyScrapHeap: the live region", () => {
  it("is mounted with no run, so the start of one is announced", () => {
    const w = mountBar();
    const region = announcement(w);
    expect(region.attributes("role")).toBe("status");
    expect(region.attributes("aria-live")).toBe("polite");
    expect(region.text()).toBe("");
  });

  it("announces a change of state and never the figures", async () => {
    const w = mountBar();
    await w.setProps({ run: run() });
    expect(announcement(w).text()).toBe(
      "Deleting pictures forever in the background.",
    );
    await w.setProps({ run: run({ processed: 40000 }) });
    expect(announcement(w).text()).toBe(
      "Deleting pictures forever in the background.",
    );
    await w.setProps({ run: run({ cancel_requested: true }) });
    expect(announcement(w).text()).toBe("Stopping after the current step.");
    await w.setProps({ run: run({ active: false, status: "cancelled" }) });
    expect(announcement(w).text()).toBe("");

    await w.setProps({ run: run({ kind: "restore" }) });
    expect(announcement(w).text()).toBe(
      "Restoring pictures from the Scrapheap in the background.",
    );
  });
});
