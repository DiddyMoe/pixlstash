// The banner that follows a background Auto-stack run above the duplicate
// queue.
//
// The run outlives the request that started it, so this banner is the only
// place the user learns how far it got and what it left behind. The tests pin
// the figures each state quotes, which controls each state offers, that a
// pending request holds its own control rather than allowing a second click,
// and that the banner stays out of the live region the destination owns.

import { describe, it, expect } from "vitest";
import { mount } from "@vue/test-utils";

import DedupAutoStackBanner from "./DedupAutoStackBanner.vue";

const globalOpts = { global: { stubs: { "v-icon": true } } };

/** A run as `serialize_run` shapes it, running an apply by default. */
function run(overrides = {}) {
  return {
    id: 7,
    request_id: "example-request",
    kind: "apply",
    status: "running",
    active: true,
    target_run_id: null,
    total: 123456,
    processed: 37036,
    applied: 30864,
    superseded: 0,
    blocked: 0,
    failed: 0,
    pictures: 96000,
    cancel_requested: false,
    can_revert: false,
    target_can_revert: false,
    error_code: null,
    error: null,
    ...overrides,
  };
}

function mountBanner(runOverrides = {}, props = {}) {
  return mount(DedupAutoStackBanner, {
    ...globalOpts,
    props: { run: run(runOverrides), ...props },
  });
}

/** Text with runs of whitespace collapsed, so template wrapping is not asserted. */
function textOf(wrapper) {
  return wrapper.text().replace(/\s+/g, " ");
}

const n = (value) => value.toLocaleString();

const stopButton = (w) => w.find('[data-testid="auto-stack-stop"]');
const revertButton = (w) => w.find('[data-testid="auto-stack-revert"]');
const dismissButton = (w) => w.find('[data-testid="auto-stack-dismiss"]');
const progressbar = (w) => w.find('[role="progressbar"]');

describe("DedupAutoStackBanner: an apply run", () => {
  it("says how far the run has got and that work can go on", () => {
    const w = mountBanner();
    const text = textOf(w);
    expect(text).toContain(
      `Auto-stacking exact matches. ${n(37036)} of ${n(123456)} groups done, ${n(30864)} stacked. You can keep working.`,
    );
    // 37036 / 123456 = 30%: the share processed, not the share stacked (25%).
    expect(w.find(".as-banner__meta").text()).toBe("30%");
  });

  it("is a progress bar with the figures, not a live region", () => {
    // Progress moves every poll; the destination's one live region announces
    // changes of state, so a banner that re-read itself would talk over it.
    const w = mountBanner();
    expect(w.attributes("aria-live")).toBeUndefined();
    expect(w.attributes("role")).toBe("group");
    const bar = progressbar(w);
    expect(bar.attributes("aria-valuenow")).toBe("30");
    expect(bar.attributes("aria-valuetext")).toBe(
      `${n(37036)} of ${n(123456)} groups done`,
    );
    expect(w.find(".as-banner__fill").attributes("style")).toContain(
      "width: 30%",
    );
  });

  it("offers Stop while running, and nothing that needs the run to end", () => {
    const w = mountBanner();
    expect(stopButton(w).exists()).toBe(true);
    expect(revertButton(w).exists()).toBe(false);
    expect(dismissButton(w).exists()).toBe(false);
  });

  it("says a queued run has not started, without inventing progress", () => {
    const w = mountBanner({ status: "pending", processed: 0, applied: 0 });
    expect(textOf(w)).toContain(
      "Auto-stack is queued and starts when the server gets to it.",
    );
    expect(w.find(".as-banner__meta").text()).toBe("Queued");
    expect(progressbar(w).attributes("aria-valuenow")).toBeUndefined();
  });

  it("drops Stop once a stop has been asked for, and says it is stopping", () => {
    const w = mountBanner({ cancel_requested: true });
    expect(textOf(w)).toContain(
      `Stopping Auto-stack after the current step. ${n(30864)} groups stacked so far.`,
    );
    expect(w.find(".as-banner__meta").text()).toBe("Stopping");
    expect(stopButton(w).exists()).toBe(false);
  });

  it("reports a finished run's outcome and drops the progress bar", () => {
    const w = mountBanner(
      {
        status: "complete",
        active: false,
        processed: 123456,
        applied: 123400,
        pictures: 380000,
      },
      { canRevert: true },
    );
    const text = textOf(w);
    expect(text).toContain(
      `Auto-stack finished. ${n(123400)} groups stacked (${n(380000)} pictures).`,
    );
    expect(text).not.toContain("already left the queue");
    expect(text).not.toContain("could not be stacked");
    expect(progressbar(w).exists()).toBe(false);
    expect(w.find(".as-banner__meta").exists()).toBe(false);
    expect(w.classes()).not.toContain("as-banner--warning");
    expect(revertButton(w).text()).toBe("Undo run");
    expect(dismissButton(w).exists()).toBe(true);
    expect(stopButton(w).exists()).toBe(false);
  });

  it("names the groups that had moved on and the ones it could not stack", () => {
    const text = textOf(
      mountBanner({
        status: "complete",
        active: false,
        applied: 123000,
        superseded: 500,
        blocked: 300,
        failed: 84,
      }),
    );
    // Blocked and failed are one figure to the user: still in the queue. The
    // whole run of sentences is asserted, so a lost space between them fails.
    expect(text).toContain(
      `(${n(96000)} pictures). ${n(500)} had already left the queue. ${n(384)} could not be stacked and are still in the queue.`,
    );
  });

  it("offers no Undo run when the server says there is nothing to undo", () => {
    const w = mountBanner({ status: "complete", active: false });
    expect(revertButton(w).exists()).toBe(false);
    expect(dismissButton(w).exists()).toBe(true);
  });

  it("says a stopped run left the rest in the queue", () => {
    const w = mountBanner({ status: "cancelled", active: false });
    expect(textOf(w)).toContain(
      `Auto-stack stopped. ${n(30864)} of ${n(123456)} groups stacked; the rest are still in the queue.`,
    );
    expect(w.classes()).not.toContain("as-banner--warning");
  });

  it("warns that a restart interrupted the run", () => {
    const w = mountBanner({ status: "interrupted", active: false });
    expect(textOf(w)).toContain(
      "Auto-stack was interrupted when the server restarted.",
    );
    expect(w.classes()).toContain("as-banner--warning");
  });

  it("warns that the run failed and quotes the server's reason", () => {
    const w = mountBanner({
      status: "failed",
      active: false,
      error: "The library is read-only.",
    });
    const text = textOf(w);
    expect(text).toContain(
      `Auto-stack stopped on an error. ${n(30864)} of ${n(123456)} groups stacked; the rest are still in the queue. The library is read-only.`,
    );
    expect(w.classes()).toContain("as-banner--warning");
  });
});

describe("DedupAutoStackBanner: a revert run", () => {
  const revert = (overrides = {}) => ({
    kind: "revert",
    target_run_id: 7,
    total: 676,
    processed: 100,
    applied: 25000,
    ...overrides,
  });

  it("counts History steps undone and groups back in the queue", () => {
    const w = mountBanner(revert());
    expect(textOf(w)).toContain(
      `Undoing Auto-stack. 100 of 676 steps undone, ${n(25000)} groups back in the queue.`,
    );
    expect(progressbar(w).attributes("aria-valuetext")).toBe(
      "100 of 676 steps undone",
    );
    expect(stopButton(w).exists()).toBe(true);
  });

  it("reports a finished undo with nothing left to offer but Dismiss", () => {
    const w = mountBanner(
      revert({ status: "complete", active: false, applied: 123400 }),
    );
    expect(textOf(w)).toContain(
      `Auto-stack undone. ${n(123400)} groups are back in the queue.`,
    );
    expect(revertButton(w).exists()).toBe(false);
    expect(dismissButton(w).exists()).toBe(true);
  });

  it("warns an undo blocked by later work, and offers to finish it", () => {
    const w = mountBanner(revert({ status: "blocked", active: false }), {
      canRevert: true,
    });
    expect(textOf(w)).toContain(
      "Undo the later work from History first, then choose Finish undo.",
    );
    expect(w.classes()).toContain("as-banner--warning");
    expect(revertButton(w).text()).toBe("Finish undo");
  });

  it("warns that the undo failed and quotes the server's reason", () => {
    const w = mountBanner(
      revert({ status: "failed", active: false, error: "Disk full." }),
      { canRevert: true },
    );
    expect(textOf(w)).toContain(
      `Undo stopped on an error. ${n(25000)} groups are back in the queue; the rest are still stacked. Disk full.`,
    );
    expect(w.classes()).toContain("as-banner--warning");
    expect(revertButton(w).text()).toBe("Finish undo");
  });

  it("says a stopped undo left the rest stacked", () => {
    const w = mountBanner(revert({ status: "cancelled", active: false }), {
      canRevert: true,
    });
    expect(textOf(w)).toContain(
      `Undo stopped. ${n(25000)} groups are back in the queue; the rest are still stacked.`,
    );
    expect(w.classes()).not.toContain("as-banner--warning");
  });
});

describe("DedupAutoStackBanner: controls", () => {
  it("asks the parent to stop, undo and dismiss", async () => {
    const running = mountBanner();
    await stopButton(running).trigger("click");
    expect(running.emitted("stop")).toHaveLength(1);

    const ended = mountBanner(
      { status: "complete", active: false },
      { canRevert: true },
    );
    await revertButton(ended).trigger("click");
    await dismissButton(ended).trigger("click");
    expect(ended.emitted("revert")).toHaveLength(1);
    expect(ended.emitted("dismiss")).toHaveLength(1);
  });

  it("holds Stop as working, not clickable, while its request is out", async () => {
    const w = mountBanner({}, { action: "cancel" });
    const stop = stopButton(w);
    expect(stop.attributes("aria-busy")).toBe("true");
    expect(stop.attributes("disabled")).toBeDefined();
    await stop.trigger("click");
    expect(w.emitted("stop")).toBeUndefined();
  });

  // The store sends a Stop for the run shown here even while a start or an
  // undo is still out, so Stop does not wait for them.
  it.each(["start", "revert"])(
    "keeps Stop clickable while a %s request is out",
    async (action) => {
      const w = mountBanner({}, { action });
      const stop = stopButton(w);
      expect(stop.attributes("disabled")).toBeUndefined();
      expect(stop.attributes("aria-busy")).toBeUndefined();
      await stop.trigger("click");
      expect(w.emitted("stop")).toHaveLength(1);
    },
  );

  it("holds Undo run as working and the other controls still", () => {
    const w = mountBanner(
      { status: "complete", active: false },
      { canRevert: true, action: "revert", revertOut: true },
    );
    expect(revertButton(w).attributes("aria-busy")).toBe("true");
    expect(revertButton(w).attributes("disabled")).toBeDefined();
    expect(dismissButton(w).attributes("disabled")).toBeDefined();
    expect(dismissButton(w).attributes("aria-busy")).toBeUndefined();
  });

  // A Stop can go out beside an undo still waiting for its answer, and the run
  // can end before either is answered. `action` then reads "cancel": the
  // undo's own flag keeps Finish undo working.
  it("holds Undo run as working while a Stop is out beside the undo", async () => {
    const w = mountBanner(
      { kind: "revert", target_run_id: 7, status: "cancelled", active: false },
      { canRevert: true, action: "cancel", revertOut: true },
    );
    const undo = revertButton(w);
    expect(undo.text()).toBe("Finish undo");
    expect(undo.attributes("aria-busy")).toBe("true");
    expect(undo.attributes("disabled")).toBeDefined();
    await undo.trigger("click");
    expect(w.emitted("revert")).toBeUndefined();
  });

  it("holds Undo run while a new run's start request is out", () => {
    // The store refuses a second request while one is pending; the control
    // says so rather than inviting a click that would be refused.
    const w = mountBanner(
      { status: "complete", active: false },
      { canRevert: true, action: "start" },
    );
    expect(revertButton(w).attributes("disabled")).toBeDefined();
    expect(revertButton(w).attributes("aria-busy")).toBeUndefined();
  });

  it("says when the figures are the last ones read", () => {
    expect(
      mountBanner({}, { unreachable: true })
        .find('[data-testid="auto-stack-banner-unreachable"]')
        .text(),
    ).toContain(
      "No answer from the server, so these are the last figures read.",
    );
    expect(
      mountBanner()
        .find('[data-testid="auto-stack-banner-unreachable"]')
        .exists(),
    ).toBe(false);
  });
});
