// The auto-stack dialog is the single consent for a bulk change, so these tests
// pin what the user is promised before they give it: the full row list
// including the zero deletions, a stable layout while the dry run lands, and a
// confirm button that cannot fire on numbers nobody has seen.

import { describe, it, expect } from "vitest";
import { mount } from "@vue/test-utils";

import DedupAutoStackDialog from "./DedupAutoStackDialog.vue";
import AppButton from "./AppButton.vue";

const AppDialogStub = {
  name: "AppDialog",
  template:
    "<div><slot name='header-right'/><slot/><slot name='footer'/></div>",
};

const globalOpts = {
  global: { stubs: { "v-icon": true, AppDialog: AppDialogStub } },
};

// The backend's dry-run report. Every figure about the run rides in
// `dry_run_summary`, derived from one read of one group list so the rows cannot
// disagree with each other. `groups_by_tier` counts only what the run would act
// on (exact-only), so the "left in the queue" figure is deliberately a separate
// prop fed from the counts endpoint's per-tier split.
const SUMMARY = {
  groups: 1240,
  groups_by_tier: { exact: 1240, near: 0, embedding: 0 },
  pictures: 2600,
  covers_gaining_tags: 300,
  covers_gaining_score: 120,
  covers_gaining_metadata: 318,
};

const PREVIEW = {
  dry_run: true,
  groups: 1240,
  pictures: 2600,
  dry_run_summary: SUMMARY,
  failures: [],
};

function mountDialog(props = {}) {
  return mount(DedupAutoStackDialog, {
    ...globalOpts,
    props: { open: true, preview: PREVIEW, queueRemaining: 88, ...props },
  });
}

/** The footer buttons, in render order: Cancel, Create (N) stacks. */
function footerButtons(wrapper) {
  return wrapper.findAllComponents(AppButton);
}

describe("DedupAutoStackDialog: the dry run", () => {
  it("states the zero deletions out loud alongside the other counts", () => {
    // "Nothing is deleted" is the invariant the whole feature rests on; hiding
    // the row because it is zero is how the user stops believing it.
    const rows = mountDialog().findAll(".as-row");
    expect(rows).toHaveLength(5);
    expect(rows[4].find(".as-term").text()).toContain("Files deleted");
    expect(rows[4].find(".as-value").text()).toBe("0");
    expect(rows[0].find(".as-value").text()).toBe((1240).toLocaleString());
    expect(rows[1].find(".as-value").text()).toBe((2600).toLocaleString());
  });

  // The design promises this row, and it is the answer to the only real
  // question a bulk stack raises: whether collapsing copies loses anything.
  it("reports how many covers gain metadata from the copies", () => {
    const rows = mountDialog().findAll(".as-row");
    expect(rows[2].find(".as-term").text()).toContain(
      "Covers gaining metadata",
    );
    expect(rows[2].find(".as-value").text()).toBe((318).toLocaleString());
  });

  // One read, one set of numbers. Taking the run's counts off the top level
  // while the metadata row came from the summary is how two rows of the same
  // dialog end up describing two different moments.
  it("reads the run's counts from the dry-run summary, not the envelope", () => {
    const rows = mountDialog({
      preview: {
        ...PREVIEW,
        groups: 7,
        pictures: 9,
        dry_run_summary: SUMMARY,
      },
    }).findAll(".as-row");
    expect(rows[0].find(".as-value").text()).toBe((1240).toLocaleString());
    expect(rows[1].find(".as-value").text()).toBe((2600).toLocaleString());
  });

  // A server that predates the summary still gets a working dialog rather than
  // a column of zeroes over a live Create button.
  it("falls back to the envelope when no summary is served", () => {
    const rows = mountDialog({
      preview: { dry_run: true, groups: 12, pictures: 30, failures: [] },
    }).findAll(".as-row");
    expect(rows[0].find(".as-value").text()).toBe("12");
    expect(rows[1].find(".as-value").text()).toBe("30");
    expect(rows[2].find(".as-value").text()).toBe("0");
  });

  it("keeps every row in place while the dry run is in flight", () => {
    // Swapping the rows for a spinner would resize the dialog and move the
    // confirm button out from under the pointer when the counts land.
    const rows = mountDialog({ preview: null, loading: true }).findAll(
      ".as-row",
    );
    expect(rows).toHaveLength(5);
    // The three rows the dry run reports go to placeholders; the queue count
    // and the zero deletions are known without it and stay readable.
    expect(rows.map((r) => r.find(".as-value").text())).toEqual([
      "–",
      "–",
      "–",
      "88",
      "0",
    ]);
  });

  // The run goes on in the background and is undone from its banner, as a
  // run of its own: one Ctrl+Z would reverse only the newest of its History
  // steps, so the consent must not promise that.
  it("says the run goes on in the background and is undone as a whole", () => {
    const note = mountDialog().find(".as-reversible");
    expect(note.text()).toContain("Runs in the background");
    expect(note.text()).toContain("Undo run reverses every stack it made");
    expect(note.find("kbd").exists()).toBe(false);
  });
});

describe("DedupAutoStackDialog: the confirmation", () => {
  it("refuses to confirm before the counts land, or when there is nothing to stack", () => {
    // Confirming during the dry run would act on numbers the user never saw.
    const loading = footerButtons(
      mountDialog({ preview: null, loading: true }),
    )[1];
    expect(loading.find("button").attributes("disabled")).toBeDefined();

    const empty = footerButtons(
      mountDialog({
        preview: {
          ...PREVIEW,
          groups: 0,
          dry_run_summary: {
            ...SUMMARY,
            groups: 0,
            groups_by_tier: { exact: 0, near: 0, embedding: 0 },
          },
        },
      }),
    )[1];
    expect(empty.find("button").attributes("disabled")).toBeDefined();
  });

  // A failed dry run and a genuinely empty one must not be the same screen: a
  // column of zeroes reads as "there is nothing to stack" when the truth is
  // "nobody was able to ask".
  it("says so when the dry run could not be read", () => {
    const wrapper = mountDialog({
      preview: null,
      previewError: { kind: "server", status: 500, detail: "" },
    });
    expect(wrapper.find(".as-failed").exists()).toBe(true);
    expect(
      wrapper.findAll(".as-row").map((r) => r.find(".as-value").text()),
    ).toEqual(["–", "–", "–", "88", "0"]);
    expect(
      footerButtons(wrapper)[1].find("button").attributes("disabled"),
    ).toBeDefined();
  });

  it("stays quiet about failure on a dry run that worked", () => {
    expect(mountDialog().find(".as-failed").exists()).toBe(false);
  });

  // "Create 0 stacks" under a failure notice is a figure nobody reported: the
  // label drops its number whenever the rows above it are dashes.
  it.each([
    ["in flight", { preview: null, loading: true }],
    [
      "failed",
      {
        preview: null,
        previewError: { kind: "timeout", status: null, detail: "" },
      },
    ],
    [
      "failed after an earlier answer",
      { previewError: { kind: "server", status: 502, detail: "" } },
    ],
  ])("names no count while the dry run is %s", (_state, props) => {
    const wrapper = mountDialog(props);
    const button = footerButtons(wrapper)[1];
    expect(button.text()).toBe("Create stacks");
    expect(button.find("button").attributes("disabled")).toBeDefined();
    expect(
      wrapper.findAll(".as-row").map((r) => r.find(".as-value").text()),
    ).toEqual(["–", "–", "–", "88", "0"]);
  });

  // Each failure says what happened and, where retrying can help, how; a
  // refused request names the server's own reason.
  it.each([
    [
      "a refused body",
      {
        kind: "validation",
        status: 422,
        detail: "Input should be a valid boolean",
      },
      "The server refused the preview request: Input should be a valid boolean.",
    ],
    [
      "a refused request whose reason ends its own sentence",
      { kind: "validation", status: 400, detail: "Unknown scope." },
      "The server refused the preview request: Unknown scope.",
    ],
    [
      "a refused request with no reason",
      { kind: "validation", status: 422, detail: "" },
      "The server refused the preview request.",
    ],
    [
      "a missing permission",
      { kind: "forbidden", status: 403, detail: "Not allowed" },
      "You don't have permission to stack duplicates here.",
    ],
    [
      "a timeout",
      { kind: "timeout", status: null, detail: "" },
      "The preview took too long to answer, so these counts are unknown. Close this and try again.",
    ],
    [
      "an unreachable server",
      { kind: "network", status: null, detail: "" },
      "The server could not be reached, so these counts are unknown. Close this and try again.",
    ],
    [
      "a server fault",
      { kind: "server", status: 500, detail: "" },
      "The preview failed on the server, so these counts are unknown. Close this and try again.",
    ],
  ])("words %s as what it was", (_label, previewError, sentence) => {
    const wrapper = mountDialog({ preview: null, previewError });
    expect(wrapper.find(".as-failed").text()).toBe(sentence);
  });

  // A cancelled request was cancelled because the dialog moved on; it is not a
  // failure to report, and its counts are still unknown.
  it("says nothing about a cancelled dry run and still names no count", () => {
    const wrapper = mountDialog({
      preview: null,
      previewError: { kind: "cancelled", status: null, detail: "" },
    });
    expect(wrapper.find(".as-failed").exists()).toBe(false);
    expect(footerButtons(wrapper)[1].text()).toBe("Create stacks");
  });

  // The consent has to describe what the run actually does to the copies it
  // collapses, or the user is agreeing to a word ("stack") rather than an
  // outcome.
  it("promises that the other copies and their metadata survive", () => {
    const lede = mountDialog().find(".as-lede").text();
    expect(lede).toContain("stays in your library");
    expect(lede).toContain("moves onto the stack");
  });

  it("names the count it is about to create and emits the confirmation", () => {
    const wrapper = mountDialog();
    const button = footerButtons(wrapper)[1];
    expect(button.text()).toContain(`Create ${(1240).toLocaleString()} stacks`);

    button.find("button").trigger("click");
    expect(wrapper.emitted("confirm")).toHaveLength(1);
  });
});

describe("DedupAutoStackDialog: starting the run", () => {
  // Pending, not refused: the button holds still with a spinner while the
  // start is on the wire, so a second click cannot send a second start.
  it("holds the confirm button while the start is on the wire", () => {
    const button = footerButtons(mountDialog({ busy: true }))[1];
    expect(button.props("loading")).toBe(true);
    expect(button.find("button").attributes("disabled")).toBeDefined();
    expect(button.find("button").attributes("aria-busy")).toBe("true");
  });

  it("stays quiet about the start until one fails", () => {
    expect(mountDialog().find(".as-start-failed").exists()).toBe(false);
  });

  // The incident this guards: a start with no answer said only that the
  // outcome was unknown. It now says what confirming again does, because the
  // second confirm sends the same request and cannot start a second run.
  it.each([
    [
      "no answer",
      { kind: "timeout", status: null, detail: "", code: null },
      "No answer came from the server, so it is not known whether the run started. Confirm again to retry: it sends the same request, which cannot start a second run.",
    ],
    [
      "an unreachable server",
      { kind: "network", status: null, detail: "", code: null },
      "No answer came from the server, so it is not known whether the run started. Confirm again to retry: it sends the same request, which cannot start a second run.",
    ],
    [
      "a server fault",
      { kind: "server", status: 500, detail: "", code: null },
      "The run could not start on the server. Confirm again to retry: it sends the same request, which cannot start a second run.",
    ],
    [
      "another run already going",
      {
        kind: "conflict",
        status: 409,
        detail: "An auto-stack run is already active.",
        code: "auto_stack_run_active",
      },
      "Another Auto-stack run is already going, and its progress shows above the queue. Start this one once it has finished.",
    ],
    [
      "a refused request",
      {
        kind: "validation",
        status: 422,
        detail: "That request id already started a different run",
        code: "request_id_reused",
      },
      "The server refused to start the run: That request id already started a different run.",
    ],
    [
      "a missing permission",
      { kind: "forbidden", status: 403, detail: "", code: null },
      "You don't have permission to stack duplicates here.",
    ],
  ])("words %s as what it was", (_label, startError, sentence) => {
    const wrapper = mountDialog({ startError });
    expect(wrapper.find(".as-start-failed").text()).toBe(sentence);
  });

  it.each([["cancelled"], ["busy"]])(
    "says nothing about a %s start",
    (kind) => {
      const wrapper = mountDialog({
        startError: { kind, status: null, detail: "", code: null },
      });
      expect(wrapper.find(".as-start-failed").exists()).toBe(false);
    },
  );

  // Confirming again is the advice after no answer, so the button must allow
  // it; against a run already going it would only be refused again.
  it("allows a second confirm after no answer, but not against a running run", () => {
    const retry = footerButtons(
      mountDialog({
        startError: { kind: "timeout", status: null, detail: "", code: null },
      }),
    )[1];
    expect(retry.find("button").attributes("disabled")).toBeUndefined();

    const blocked = footerButtons(
      mountDialog({
        startError: {
          kind: "conflict",
          status: 409,
          detail: "",
          code: "auto_stack_run_active",
        },
      }),
    )[1];
    expect(blocked.find("button").attributes("disabled")).toBeDefined();
  });
});
