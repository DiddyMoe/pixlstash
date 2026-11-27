// The duplicate triage queue.
//
// These tests cover the parts of the destination that are invisible in a
// screenshot and expensive to get wrong: the live region has to outlive the row
// that emptied the queue, the tier popover has to be dismissible without a
// mouse, a failed verdict has to be reported rather than swallowed, and a
// read-only session must not be shown a verdict it can never give.

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { setActivePinia, createPinia } from "pinia";
import { reactive } from "vue";
import {
  DEFAULT_THUMBNAIL_SIZE_LEVEL,
  stripHeightForSizeLevel,
} from "../../utils/thumbnailSizes";

/**
 * The row pitch the view ESTIMATES at the default size - the strip height (or
 * the info column's floor, whichever is taller) plus the row's own chrome.
 *
 * Derived from the ladder rather than written out, because happy-dom lays
 * nothing out: `measureRowPitch` never refines the estimate here, so every
 * scroll assertion below is in units of this number, and re-scaling the ladder
 * must not turn them into arithmetic puzzles.
 */
const MIN_ROW_CONTENT_PX = 89;
const ROW_CHROME_PX = 28;
const PITCH =
  Math.max(
    stripHeightForSizeLevel(DEFAULT_THUMBNAIL_SIZE_LEVEL),
    MIN_ROW_CONTENT_PX,
  ) + ROW_CHROME_PX;

vi.mock("../../api/dedup", () => ({
  getPolicy: vi.fn(),
  listGroups: vi.fn(),
  getCounts: vi.fn(),
  startScan: vi.fn(),
  stackGroup: vi.fn(),
  keepGroupSeparate: vi.fn(),
  applyVerdictBatch: vi.fn(),
  reopenGroup: vi.fn(),
  // Auto-stack: a dry run that only counts, and the background run it starts,
  // which the queue follows by polling until it ends.
  previewAutoStackExact: vi.fn(),
  startAutoStackRun: vi.fn(),
  getAutoStackRun: vi.fn(),
  getLatestAutoStackRun: vi.fn(),
  cancelAutoStackRun: vi.fn(),
  revertAutoStackRun: vi.fn(),
  // The lazy half of the stack contract: a row's expansion reads its members
  // only when the user opens one.
  listStackMembers: vi.fn(),
  // The third page (design D5). Its potentially expensive list stays lazy
  // until the user opens the page.
  listMixedStacks: vi.fn(),
  splitMixedStack: vi.fn(),
  unstackMixedStack: vi.fn(),
  keepMixedStack: vi.fn(),
  clearMixedStackKeep: vi.fn(),
  MAX_STACK_MEMBER_PAGE: 200,
  MAX_MIXED_STACK_PAGE: 200,
  GLOBAL_SCOPE: "global",
}));

// The queue opens itself from the URL's scope, so it needs a route. Reactive,
// as the real one is: the view watches parts of the query, and a plain object
// would keep those watchers silent - hiding exactly the class of bug where a
// mirror write into the URL feeds back into the component's own route sync.
const routeMock = reactive({ name: "duplicates", query: {} });
const routerReplace = vi.fn();
const routerPush = vi.fn();
vi.mock("vue-router", () => ({
  useRoute: () => routeMock,
  useRouter: () => ({
    replace: (...a) => routerReplace(...a),
    // Only the queue-clear route to the stacks pushes; the URL mirror replaces.
    push: (...a) => routerPush(...a),
  }),
}));

vi.mock("../../api/pictures", () => ({
  pictureThumbnailUrl: (id) => `/pictures/thumbnails/${id}.webp`,
}));

// The read-only flag is a module-level computed over the session; the tests
// drive it directly rather than faking a whole session. The factories are
// hoisted above every top-level binding, so the shared refs and spies are built
// inside them and read back through the mocked modules.
let batchCounter = 0;
vi.mock("../../utils/apiClient", async () => {
  const { ref: makeRef } = await import("vue");
  return {
    isReadOnly: makeRef(false),
    onSessionReset: () => () => {},
    API_BASE_URL: "http://backend.test/api/v1",
    newOperationBatchId: () => `cli-test-${(batchCounter += 1)}`,
  };
});

vi.mock("../../stores/useOperationStore", () => {
  const undo = vi.fn();
  // The queue subscribes to the shared store's actions to reload after an
  // undo/redo; the mock records the listeners so a test can drive one.
  const actionListeners = [];
  const operationStoreMock = {
    undo,
    refresh: vi.fn(),
    nextUndo: null,
    nextRedo: null,
    past: [],
    operations: [],
    // The dedup store keeps a followed run's History steps off the pill.
    setNarrationFilter: vi.fn(),
    $onAction: (cb) => {
      actionListeners.push(cb);
      return () => {};
    },
  };
  return {
    useOperationStore: () => operationStoreMock,
    // Named helpers UndoControl imports from the same module.
    summarizeOperation: () => "",
    formatOperationTime: () => "",
    iconForOpType: () => "mdi-history",
    __operationStoreMock: operationStoreMock,
    __actionListeners: actionListeners,
  };
});

vi.mock("../../stores/useNoticeStore", () => {
  const error = vi.fn();
  const info = vi.fn();
  const warning = vi.fn();
  return { useNoticeStore: () => ({ error, info, warning }) };
});

import {
  getPolicy,
  listGroups,
  getCounts,
  startScan,
  stackGroup,
  keepGroupSeparate,
  applyVerdictBatch,
  reopenGroup,
  previewAutoStackExact,
  startAutoStackRun,
  getAutoStackRun,
  getLatestAutoStackRun,
  cancelAutoStackRun,
  revertAutoStackRun,
  listStackMembers,
  listMixedStacks,
  splitMixedStack,
  unstackMixedStack,
  keepMixedStack,
  clearMixedStackKeep,
} from "../../api/dedup";
import { isReadOnly as readOnlyRef } from "../../utils/apiClient";
import { useNoticeStore } from "../../stores/useNoticeStore";
import {
  __operationStoreMock,
  __actionListeners,
} from "../../stores/useOperationStore";
import DuplicateQueue from "./DuplicateQueue.vue";
import { useDedupStore } from "../../stores/useDedupStore";

/** A queue group in the backend's shape, with `n` candidates. */
function group(signature, n = 2) {
  const base = Number(signature.replace(/\D/g, "")) * 100;
  return {
    signature,
    tier: "near",
    confidence: 0.93,
    member_count: n,
    cover_picture_id: null,
    why: [],
    candidates: Array.from({ length: n }, (_, i) => ({
      picture_id: base + i,
      width: 4000,
      height: 3000,
      megapixels: 12,
    })),
  };
}

/** One `MixedStackModel` row from `GET /dedup/mixed-stacks`. */
function mixedStack(over = {}) {
  return {
    stack_id: 42,
    threshold: 0.9,
    member_count: 5,
    member_ids: [7, 8, 9, 10, 11],
    membership_fingerprint: "fp-42",
    component_count: 2,
    component_sizes: [4, 1],
    components: [[7, 8, 9, 10], [11]],
    largest_component_size: 4,
    stranded_picture_ids: [11],
    weakest_edge: 0.91,
    unhashed_picture_ids: [],
    suggested_action: "split",
    kept: false,
    leader_picture_id: 7,
    leader_thumbnail_version: null,
    ...over,
  };
}

/** One page of that list. */
function mixedPage(stacks, over = {}) {
  return {
    threshold: 0.9,
    total: stacks.length,
    kept_total: 0,
    live_stack_count: 209,
    offset: 0,
    limit: 100,
    next_offset: null,
    stacks,
    ...over,
  };
}

/** An auto-stack run in the backend's shape: apply run 7, a third of the way. */
function runFixture(over = {}) {
  return {
    id: 7,
    request_id: "cli-fake-request",
    kind: "apply",
    status: "running",
    active: true,
    target_run_id: null,
    total: 120,
    processed: 40,
    applied: 40,
    superseded: 0,
    blocked: 0,
    failed: 0,
    pictures: 80,
    failures: [],
    segment_batch_ids: [],
    cancel_requested: false,
    can_revert: false,
    error_code: null,
    error: null,
    created_at: "2026-05-12T14:22:00Z",
    ...over,
  };
}

/** Run 8, undoing apply run 7, a third of the way. */
function revertRun(over = {}) {
  return runFixture({ id: 8, kind: "revert", target_run_id: 7, ...over });
}

/** An axios rejection carrying a served status and `detail`. */
function refusal(status, detail) {
  return Object.assign(new Error(`Request failed with status code ${status}`), {
    response: { status, data: detail == null ? {} : { detail } },
  });
}

/** A promise the test settles by hand, to order replies at will. */
function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const autoStackDialog = (wrapper) =>
  wrapper.findComponent({ name: "DedupAutoStackDialog" });
const autoStackTrigger = (wrapper) =>
  wrapper.find('[data-testid="auto-stack-trigger"]');
/** What the queue's one live region last said. */
const announced = (wrapper) =>
  wrapper.find('[data-testid="dedup-announcement"]').text();

/** The bounds `GET /dedup/policy` publishes. */
const BOUNDS = {
  min_threshold: 0.65,
  max_threshold: 0.99999,
  tiers: ["exact", "near", "embedding"],
  always_on_tiers: ["exact"],
  tier_requires: { exact: null, near: "exact", embedding: "near" },
  scope_types: ["global", "project", "set", "character", "folder"],
  verdicts: ["stacked", "keep_separate"],
  max_page_size: 200,
};

/** The text of the one-surface tooltip a control carries. */
const tipText = (el) => el.findComponent({ name: "Tooltip" }).props("text");

const globalOpts = {
  global: {
    stubs: {
      "v-icon": true,
      "v-progress-circular": true,
      DedupCompareDialog: true,
      DedupAutoStackDialog: true,
      ActionReceipt: true,
      "v-slider": true,
      // Renders the activator, never the tip: a control attached through the
      // slot must still be there to press.
      VTooltip: {
        template: '<slot name="activator" :props="{}" :is-active="false" />',
      },
      // Its History popover needs Vuetify's v-menu; the queue only needs to
      // know the control is mounted.
      UndoControl: true,
    },
  },
};

/**
 * Mount the queue over one served page and let the first load settle.
 *
 * `total` is the server's count of the WHOLE queue, which is larger than the
 * page whenever there is more to page in.
 */
async function mountQueue(groups, { byTier = {}, total = null } = {}) {
  getPolicy.mockResolvedValue({
    defaults: { near_enabled: false, embedding_enabled: false, threshold: 0.9 },
    bounds: BOUNDS,
  });
  listGroups.mockResolvedValue({
    groups,
    total: total ?? groups.length,
    offset: 0,
    limit: 20,
    scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
  });
  getCounts.mockResolvedValue({
    unresolved_groups: total ?? groups.length,
    by_tier: byTier,
    scopes: [],
    scan: { status: "complete" },
  });
  const store = useDedupStore();
  await store.loadPolicy();
  await store.refreshCounts();
  await store.loadFirstPage();
  const wrapper = mount(DuplicateQueue, {
    ...globalOpts,
    attachTo: document.body,
  });
  await wrapper.vm.$nextTick();
  return { wrapper, store };
}

let errorSpy;

beforeEach(() => {
  // The queue's thumbnail size is remembered in localStorage, and the row pitch
  // every spacer is sized from follows it. A case that changes the size would
  // otherwise resize the rows of every case after it.
  window.localStorage.clear();
  setActivePinia(createPinia());
  readOnlyRef.value = false;
  routeMock.name = "duplicates";
  routeMock.query = {};
  routerReplace.mockReset();
  routerPush.mockReset();
  __actionListeners.length = 0;
  __operationStoreMock.refresh.mockReset();
  __operationStoreMock.nextUndo = null;
  __operationStoreMock.nextRedo = null;
  __operationStoreMock.past = [];
  __operationStoreMock.operations = [];
  vi.spyOn(console, "warn").mockImplementation(() => {});
  const notices = useNoticeStore();
  errorSpy = notices.error;
  for (const fn of [
    getPolicy,
    listGroups,
    getCounts,
    stackGroup,
    keepGroupSeparate,
    applyVerdictBatch,
    reopenGroup,
    previewAutoStackExact,
    startAutoStackRun,
    getAutoStackRun,
    getLatestAutoStackRun,
    cancelAutoStackRun,
    revertAutoStackRun,
    listStackMembers,
    listMixedStacks,
    splitMixedStack,
    unstackMixedStack,
    keepMixedStack,
    clearMixedStackKeep,
    errorSpy,
    notices.info,
    notices.warning,
  ]) {
    fn.mockReset();
  }
  // Most libraries have no mixed stack at all, so an empty list is the default
  // answer here too. Cases that need rows override it.
  listMixedStacks.mockResolvedValue(mixedPage([]));
  // The queue looks for the newest Auto-stack run when it opens; most cases
  // run in a library that never had one.
  getLatestAutoStackRun.mockResolvedValue(null);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("DuplicateQueue - what a screen reader hears", () => {
  // A live region that unmounts with the last row takes the verdict that
  // emptied the queue down with it, so the one announcement a user most needs
  // is the one they would never hear.
  it("keeps the live region alive once the queue empties", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    stackGroup.mockResolvedValue({ ok: true });

    expect(wrapper.find('[role="status"][aria-live="polite"]').exists()).toBe(
      true,
    );
    await wrapper.vm.$nextTick();

    await store.stack(store.groups[0]);
    await wrapper.vm.$nextTick();

    expect(store.hasGroups).toBe(false);
    expect(wrapper.find('[role="status"][aria-live="polite"]').exists()).toBe(
      true,
    );
    wrapper.unmount();
  });

  // The visible hint strip is a row of glyphs hidden from assistive tech, so
  // the full model has to be stated somewhere it can be read, including the two
  // keys the strip has no room for.
  it("describes the whole keyboard model, including X and the digits", async () => {
    const { wrapper } = await mountQueue([group("g1")]);
    const help = wrapper.find("#dq-key-help");
    expect(wrapper.attributes("aria-describedby")).toBe("dq-key-help");
    const text = help.text();
    for (const phrase of [
      "1 to 9",
      "X leaves",
      "Escape",
      "ever deleted",
      // Amendment #3's scheme, stated where a screen reader can find it.
      "Enter or S",
      "K keeps it separate",
      "Down moves on without deciding",
    ]) {
      expect(text).toContain(phrase);
    }
    wrapper.unmount();
  });
});

describe("DuplicateQueue - the tier popover", () => {
  const TIERS = [{ key: "near", label: "Near duplicates", count: 4 }];

  it("closes on Escape and gives the focus back to its button", async () => {
    const { wrapper } = await mountQueue([group("g1")], { tiers: TIERS });
    const button = wrapper.find(
      '.dq-tier-wrap [data-testid="dedup-tier-trigger"]',
    );

    await button.trigger("click");
    expect(wrapper.findComponent({ name: "DedupTierMenu" }).exists()).toBe(
      true,
    );
    expect(button.attributes("aria-expanded")).toBe("true");

    await wrapper.find(".dq").trigger("keydown", { key: "Escape" });
    expect(wrapper.findComponent({ name: "DedupTierMenu" }).exists()).toBe(
      false,
    );
    expect(document.activeElement).toBe(button.element);
    wrapper.unmount();
  });

  it("closes on a pointer press outside itself", async () => {
    const { wrapper } = await mountQueue([group("g1")], { tiers: TIERS });
    await wrapper
      .find('.dq-tier-wrap [data-testid="dedup-tier-trigger"]')
      .trigger("click");
    expect(wrapper.findComponent({ name: "DedupTierMenu" }).exists()).toBe(
      true,
    );

    document.dispatchEvent(
      new window.MouseEvent("mousedown", { bubbles: true }),
    );
    await wrapper.vm.$nextTick();
    expect(wrapper.findComponent({ name: "DedupTierMenu" }).exists()).toBe(
      false,
    );
    wrapper.unmount();
  });

  it("leaves the popover alone for a press inside it", async () => {
    const { wrapper } = await mountQueue([group("g1")], { tiers: TIERS });
    const wrap = wrapper.find(".dq-tier-wrap");
    await wrap.find('[data-testid="dedup-tier-trigger"]').trigger("click");

    wrap.element.dispatchEvent(
      new window.MouseEvent("mousedown", { bubbles: true }),
    );
    await wrapper.vm.$nextTick();
    expect(wrapper.findComponent({ name: "DedupTierMenu" }).exists()).toBe(
      true,
    );
    wrapper.unmount();
  });
});

describe("DuplicateQueue - the filter on the Decided page", () => {
  /** Flip to Decided and open the filter popover over it. */
  async function decidedQueue(over = {}) {
    const mounted = await mountQueue([group("g1")], { total: 1 });
    // Opening the queue from the route is what marks the filter selection as
    // adopted, and until it is the URL mirror deliberately stays silent.
    // mountQueue pre-loads the rows, so the view's own mount takes the
    // already-showing fast path; the open is done here instead.
    await mounted.store.openQueue();
    await flushPromises();
    listGroups.mockResolvedValue({
      groups: [group("d1", 2, { verdict: "stacked" })],
      total: 5,
      offset: 0,
      limit: 20,
      by_verdict: { stacked: 3, keep_separate: 2 },
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
      ...over,
    });
    await mounted.wrapper
      .find('[data-testid="decided-toggle"]')
      .trigger("click");
    await flushPromises();
    await mounted.wrapper.vm.$nextTick();
    return mounted;
  }

  // The tier gate says nothing about a decision already made - the server
  // ignores it on the decided page - so the button behind the filter icon has
  // to open the control that is actually in force.
  it("swaps the tier menu for the verdict menu", async () => {
    const { wrapper } = await decidedQueue();
    await wrapper
      .find('.dq-tier-wrap [data-testid="dedup-tier-trigger"]')
      .trigger("click");
    expect(wrapper.findComponent({ name: "DedupVerdictMenu" }).exists()).toBe(
      true,
    );
    expect(wrapper.findComponent({ name: "DedupTierMenu" }).exists()).toBe(
      false,
    );
    wrapper.unmount();
  });

  // A trigger reading "Exact only" over a page showing every decision would be
  // a plain lie: the label has to name the filter that is in force.
  it("names the verdict filter on its own button", async () => {
    const { wrapper, store } = await decidedQueue();
    const button = wrapper.find(
      '.dq-tier-wrap [data-testid="dedup-tier-trigger"]',
    );
    expect(button.attributes("aria-label")).toBe("All decisions");

    await store.setVerdictEnabled("keep_separate", false);
    await wrapper.vm.$nextTick();
    expect(button.attributes("aria-label")).toBe("Stacked");
    wrapper.unmount();
  });

  it("shows decided candidates individually and removes stack expansion", async () => {
    const decidedGroup = {
      ...deckGroup("decided-deck", 12),
      verdict: "stacked",
      decided_at: "2026-08-02T12:00:00",
    };
    const { wrapper } = await decidedQueue({
      groups: [decidedGroup],
      total: 1,
    });

    const row = wrapper.findComponent({ name: "DedupGroupRow" });
    expect(row.props("collapseStacks")).toBe(false);
    expect(row.findAll(".gthumb")).toHaveLength(2);
    expect(row.findComponent({ name: "StackBadge" }).exists()).toBe(false);

    listStackMembers.mockClear();
    await wrapper.find(".dq").trigger("keydown", { key: "e" });
    await flushPromises();
    expect(listStackMembers).not.toHaveBeenCalled();
    expect(wrapper.find('[data-testid="dedup-announcement"]').text()).toContain(
      "already shown",
    );

    const compare = wrapper.findComponent({ name: "DedupCompareDialog" });
    expect(compare.props("collapseStacks")).toBe(false);
    expect(compare.props("readOnly")).toBe(true);
    wrapper.unmount();
  });

  it("narrows the page from the menu and says so", async () => {
    const { wrapper } = await decidedQueue();
    await wrapper
      .find('.dq-tier-wrap [data-testid="dedup-tier-trigger"]')
      .trigger("click");
    listGroups.mockClear();
    wrapper
      .findComponent({ name: "DedupVerdictMenu" })
      .vm.$emit("toggle", "keep_separate", false);
    await flushPromises();
    await wrapper.vm.$nextTick();

    expect(listGroups).toHaveBeenLastCalledWith(
      expect.objectContaining({ decided: true, verdicts: ["stacked"] }),
    );
    expect(wrapper.find('[data-testid="dedup-announcement"]').text()).toContain(
      "Hiding kept separate groups",
    );
    // The popover stays up: with two verdicts, hiding one is usually followed
    // by hiding or restoring the other.
    expect(wrapper.findComponent({ name: "DedupVerdictMenu" }).exists()).toBe(
      true,
    );
    wrapper.unmount();
  });

  // The selection is part of the ADDRESS, exactly as the tier gate is, so a
  // refresh or a shared link reopens the page the same way.
  it("mirrors a narrowed selection into the URL and drops it when widened", async () => {
    const { wrapper, store } = await decidedQueue();
    routerReplace.mockClear();
    await store.setVerdictEnabled("stacked", false);
    await wrapper.vm.$nextTick();
    expect(routerReplace).toHaveBeenLastCalledWith({
      query: expect.objectContaining({
        view: "decided",
        verdict: "keep_separate",
      }),
    });

    routerReplace.mockClear();
    await store.setVerdictEnabled("stacked", true);
    await wrapper.vm.$nextTick();
    expect(routerReplace.mock.calls.at(-1)[0].query.verdict).toBeUndefined();
    wrapper.unmount();
  });
});

describe("DuplicateQueue - when a verdict does not land", () => {
  it.each(["partial", "failed"])(
    "never renders Queue clear when the latest scan is %s",
    async (status) => {
      const { wrapper, store } = await mountQueue([]);
      store.scan = { ...store.scan, status, error: "comparison work omitted" };
      await wrapper.vm.$nextTick();

      expect(wrapper.text()).toContain(
        status === "failed" ? "Scan failed" : "Scan incomplete",
      );
      expect(wrapper.text()).not.toContain("Queue clear");
      wrapper.unmount();
    },
  );

  it("never renders Queue clear when the queue state failed to load", async () => {
    const { wrapper, store } = await mountQueue([]);
    store.error = new Error("network down");
    await wrapper.vm.$nextTick();

    expect(wrapper.text()).toContain("Could not confirm the duplicate queue");
    expect(wrapper.text()).not.toContain("Queue clear");
    wrapper.unmount();
  });

  // A failed verdict leaves the row where it was, which on a queue whose whole
  // promise is auto-advance reads as a dead keypress.
  it("raises a notice rather than swallowing the failure", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    stackGroup.mockRejectedValue(new Error("nope"));

    wrapper.findComponent({ name: "DedupGroupRow" }).vm.$emit("stack");
    await flushPromises();

    expect(errorSpy).toHaveBeenCalledTimes(1);
    expect(errorSpy.mock.calls[0][0]).toContain("still in the queue");
    expect(store.hasGroups).toBe(true);
    wrapper.unmount();
  });
});

describe("DuplicateQueue - keeping a group separate", () => {
  // The backend deliberately records NO operation for this verdict, so no
  // receipt will come. The narration is transient and points at the Decided
  // page, which is the standing way back (it replaced the sticky notice).
  it("narrates transiently and points at Decided, with no action attached", async () => {
    const { wrapper } = await mountQueue([group("g1", 3)]);
    keepGroupSeparate.mockResolvedValue({ verdict: "keep_separate" });
    const info = useNoticeStore().info;

    wrapper.findComponent({ name: "DedupGroupRow" }).vm.$emit("keep-separate");
    await flushPromises();

    expect(info).toHaveBeenCalledTimes(1);
    const [text, opts] = info.mock.calls[0];
    expect(text).toContain("under Decided");
    expect(opts?.action).toBeUndefined();
    wrapper.unmount();
  });

  // A backend that HAS made keep-separate undoable mirrors the stack
  // response (batch_id present): the standard undo receipt narrates it, so
  // the info toast would say the same thing twice and stands down.
  it("hands narration to the receipt when the backend recorded the verdict", async () => {
    const { wrapper } = await mountQueue([group("g1", 3)]);
    keepGroupSeparate.mockResolvedValue({
      verdict: "keep_separate",
      batch_id: "srv-9",
    });
    const info = useNoticeStore().info;

    wrapper.findComponent({ name: "DedupGroupRow" }).vm.$emit("keep-separate");
    await flushPromises();

    expect(info).not.toHaveBeenCalled();
    expect(__operationStoreMock.refresh).toHaveBeenCalledWith({
      narrate: true,
    });
    wrapper.unmount();
  });
});

describe("DuplicateQueue - one toolbar", () => {
  // The queue used to carry a second bar whose right half was a row of key
  // hints. Every one of those keys is already stated on the row it acts on, in
  // Compare's footer, or in the description a screen reader reads, so the bar
  // was explanation the user had to look past on every visit.
  it("carries the count and the Decided toggle, and no key hints", async () => {
    const { wrapper } = await mountQueue([group("g1"), group("g2")]);
    const toolbar = wrapper.find(".dq-toolbar");
    expect(toolbar.find(".qtitle").text()).toContain("2 groups to review");
    expect(toolbar.find('[data-testid="decided-toggle"]').exists()).toBe(true);
    expect(wrapper.find(".qhead").exists()).toBe(false);
    expect(wrapper.find(".khint").exists()).toBe(false);
    expect(toolbar.findAll("kbd")).toHaveLength(0);
    // The keys themselves are not hidden, they are stated where they act: the
    // focused row still wears its Enter/S/C chips.
    expect(wrapper.find(".grow--focus").findAll("kbd").length).toBeGreaterThan(
      0,
    );
    wrapper.unmount();
  });

  // The one thing that stayed on a second row, and it is state rather than
  // explanation: it appears with the selection and leaves with it.
  it("raises the bulk bar only while a selection is live", async () => {
    const { wrapper, store } = await mountQueue([group("g1"), group("g2")]);
    expect(wrapper.find(".qselbar").exists()).toBe(false);

    store.toggleSelected(0);
    store.toggleSelected(1);
    await wrapper.vm.$nextTick();
    expect(wrapper.find(".qselbar").text()).toContain("2 groups selected");

    store.clearSelection();
    await wrapper.vm.$nextTick();
    expect(wrapper.find(".qselbar").exists()).toBe(false);
    wrapper.unmount();
  });

  // The slider drives the rows, so the height it publishes has to reach them.
  it("hands the size level's height to every row", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    const heightOf = () =>
      wrapper.findComponent({ name: "DedupGroupRow" }).props("thumbHeight");
    expect(heightOf()).toBe(196);

    store.setSizeLevel(6);
    await wrapper.vm.$nextTick();
    expect(heightOf()).toBe(406);
    wrapper.unmount();
  });
});

describe("DuplicateQueue - who owns the keyboard", () => {
  // The bug: the handler was bound on the queue root, so the shortcuts only
  // worked while the DOM focus was inside it. One click on a sidebar row and
  // every key went dead, with nothing on screen to say why.
  it("still answers keys after the focus has left the queue", async () => {
    const { wrapper, store } = await mountQueue([group("g1"), group("g2")]);
    const elsewhere = document.createElement("button");
    document.body.appendChild(elsewhere);
    elsewhere.focus();
    expect(document.activeElement).toBe(elsewhere);

    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }),
    );
    expect(store.focusIndex).toBe(1);

    elsewhere.remove();
    wrapper.unmount();
  });

  // The other half of the same coin: a document-bound handler must not answer
  // keys meant for a dialog raised over the queue.
  it("hands the keyboard to a dialog the queue did not open", async () => {
    const { wrapper, store } = await mountQueue([group("g1"), group("g2")]);
    const scrim = document.createElement("div");
    scrim.className = "v-overlay--active";
    scrim.innerHTML = '<div class="v-overlay__scrim"></div>';
    document.body.appendChild(scrim);

    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }),
    );
    expect(store.focusIndex).toBe(0);

    scrim.remove();
    wrapper.unmount();
  });

  // A menu raised over the queue (the title bar's, the sidebar's) has no
  // scrim, so the dialog signal misses it; a key pressed in it is its own.
  it("hands the keyboard to a menu the queue did not open", async () => {
    const { wrapper, store } = await mountQueue([group("g1"), group("g2")]);
    const menu = document.createElement("div");
    menu.className = "v-overlay v-overlay--active v-menu";
    menu.innerHTML =
      '<div class="v-overlay__content"><button type="button">Item</button></div>';
    document.body.appendChild(menu);

    const item = menu.querySelector("button");
    item.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }),
    );
    expect(store.focusIndex).toBe(0);
    // The same key outside the menu still moves the queue.
    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }),
    );
    expect(store.focusIndex).toBe(1);

    menu.remove();
    wrapper.unmount();
  });

  // And it must stop listening when the destination is left, or a key pressed
  // in the grid would move a cursor in a queue that is no longer on screen.
  it("stops listening once the view is gone", async () => {
    const { wrapper, store } = await mountQueue([group("g1"), group("g2")]);
    wrapper.unmount();
    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }),
    );
    expect(store.focusIndex).toBe(0);
  });

  // Ctrl+A pages the queue in, so it is not instant and it can stop short.
  // Both facts are narrated: a bulk verdict on a set whose size the user never
  // saw is exactly what the announcement is for.
  it("narrates what Ctrl+A actually selected", async () => {
    const page1 = Array.from({ length: 20 }, (_, i) => group(`g${i + 1}`));
    const { wrapper, store } = await mountQueue(page1, { total: 30 });
    listGroups.mockResolvedValue({
      groups: Array.from({ length: 10 }, (_, i) => group(`g${i + 21}`)),
      total: 30,
      offset: 30,
      limit: 200,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });

    document.dispatchEvent(
      new window.KeyboardEvent("keydown", {
        key: "a",
        ctrlKey: true,
        bubbles: true,
      }),
    );
    await flushPromises();
    await wrapper.vm.$nextTick();

    expect(store.selectionCount).toBe(30);
    expect(wrapper.find('[data-testid="dedup-announcement"]').text()).toContain(
      "Selected all 30 groups",
    );
    wrapper.unmount();
  });
});

describe("DuplicateQueue - the toolbar hands the keyboard back", () => {
  function key(name, target = document) {
    const event = new window.KeyboardEvent("keydown", {
      key: name,
      bubbles: true,
      cancelable: true,
    });
    target.dispatchEvent(event);
    return event;
  }

  // The user's report: after changing something in the toolbar, Enter/S and
  // the arrows went to the focused control (or a still-open popover) instead
  // of the queue, until a click in the grid.
  it("a tier toggle returns focus to the queue and the keys act again", async () => {
    const { wrapper, store } = await mountQueue([group("g1"), group("g2")]);
    await wrapper
      .find('.dq-tier-wrap [data-testid="dedup-tier-trigger"]')
      .trigger("click");
    wrapper
      .findComponent({ name: "DedupTierMenu" })
      .vm.$emit("toggle", "near", true);
    await flushPromises();

    expect(document.activeElement).toBe(wrapper.find(".dq").element);
    key("ArrowDown");
    expect(store.focusIndex).toBe(1);
    stackGroup.mockResolvedValue({});
    key("Enter");
    await flushPromises();
    expect(stackGroup).toHaveBeenCalled();
    wrapper.unmount();
  });

  it("a pointer-committed threshold change hands the keyboard back with the popover open", async () => {
    const { wrapper, store } = await mountQueue([group("g1"), group("g2")]);
    await wrapper
      .find('.dq-tier-wrap [data-testid="dedup-tier-trigger"]')
      .trigger("click");
    const menu = wrapper.findComponent({ name: "DedupTierMenu" });
    // The drag begins with a pointer press inside the popover.
    menu.element.dispatchEvent(
      new window.MouseEvent("mousedown", { bubbles: true }),
    );
    menu.vm.$emit("threshold", 0.8);
    await flushPromises();

    // The popover stays open (the count is what the user tunes against)...
    expect(wrapper.findComponent({ name: "DedupTierMenu" }).exists()).toBe(
      true,
    );
    // ...but the keyboard is the queue's again.
    expect(document.activeElement).toBe(wrapper.find(".dq").element);
    key("ArrowDown");
    expect(store.focusIndex).toBe(1);
    wrapper.unmount();
  });

  // Every keyboard arrow fires its own change; yanking focus after the first
  // would turn the rest of the tuning into row moves.
  it("keyboard threshold tuning keeps focus on the slider", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    // The slider is disabled until a looser tier is on.
    store.nearEnabled = true;
    await wrapper
      .find('.dq-tier-wrap [data-testid="dedup-tier-trigger"]')
      .trigger("click");
    await wrapper.vm.$nextTick();
    const input = wrapper.find(".dth-input");
    input.element.focus();
    wrapper
      .findComponent({ name: "DedupTierMenu" })
      .vm.$emit("threshold", 0.85);
    await flushPromises();
    expect(document.activeElement).toBe(input.element);
    wrapper.unmount();
  });

  it("keys pressed on the threshold slider never fire verdicts", async () => {
    const { wrapper } = await mountQueue([group("g1")]);
    await wrapper
      .find('.dq-tier-wrap [data-testid="dedup-tier-trigger"]')
      .trigger("click");
    const input = wrapper.find(".dth-input").element;
    input.focus();
    stackGroup.mockResolvedValue({});
    key("s", input);
    key("Enter", input);
    await flushPromises();
    expect(stackGroup).not.toHaveBeenCalled();
    expect(keepGroupSeparate).not.toHaveBeenCalled();
    wrapper.unmount();
  });

  // The ⋯ panel takes the same deal as the tier popover: the keys pressed
  // inside it are the menu's, not the queue's. Without it, S on an open menu
  // would stack the group behind it.
  it("keys pressed inside the ⋯ panel never fire verdicts", async () => {
    const { wrapper } = await mountQueue([group("g1")]);
    await wrapper.find(".dq-overflow .tbo-trigger").trigger("click");
    const row = wrapper.find('[data-testid="mixed-row"]').element;
    row.focus();
    stackGroup.mockResolvedValue({});
    key("s", row);
    await flushPromises();
    expect(stackGroup).not.toHaveBeenCalled();
    expect(keepGroupSeparate).not.toHaveBeenCalled();
    wrapper.unmount();
  });

  // ...and only while it is OPEN. A closed ⋯ is an ordinary toolbar button,
  // and the keys go on working on it exactly as they do on the tier trigger
  // beside it - the alternative is two adjacent triggers with opposite
  // keyboard behaviour.
  it("keys pressed on the closed ⋯ trigger still reach the queue", async () => {
    const { wrapper } = await mountQueue([group("g1")]);
    const trigger = wrapper.find(".dq-overflow .tbo-trigger").element;
    trigger.focus();
    stackGroup.mockResolvedValue({});
    key("s", trigger);
    await flushPromises();
    expect(stackGroup).toHaveBeenCalled();
    wrapper.unmount();
  });

  it("Escape on the threshold slider still dismisses the popover to its trigger", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    store.nearEnabled = true;
    const trigger = wrapper.find(
      '.dq-tier-wrap [data-testid="dedup-tier-trigger"]',
    );
    await trigger.trigger("click");
    await wrapper.vm.$nextTick();
    const input = wrapper.find(".dth-input");
    input.element.focus();
    await input.trigger("keydown", { key: "Escape" });
    expect(wrapper.findComponent({ name: "DedupTierMenu" }).exists()).toBe(
      false,
    );
    expect(document.activeElement).toBe(trigger.element);
    wrapper.unmount();
  });

  it("a pointer-committed size change hands the keyboard back", async () => {
    const { wrapper } = await mountQueue([group("g1")]);
    wrapper.findComponent(".dq-size-slider").vm.$emit("end", 4);
    await wrapper.vm.$nextTick();
    expect(document.activeElement).toBe(wrapper.find(".dq").element);
    wrapper.unmount();
  });

  it("flipping to Decided hands the keyboard to the list", async () => {
    const { wrapper } = await mountQueue([group("g1")]);
    await wrapper.find('[data-testid="decided-toggle"]').trigger("click");
    await flushPromises();
    expect(document.activeElement).toBe(wrapper.find(".dq").element);
    wrapper.unmount();
  });

  // Moving BETWEEN controls must never get focus yanked: only a committed
  // change hands it back.
  it("Tab through the toolbar is never claimed by the queue", async () => {
    const { wrapper } = await mountQueue([group("g1")]);
    const decided = wrapper.find('[data-testid="decided-toggle"]').element;
    decided.focus();
    const event = key("Tab", decided);
    expect(event.defaultPrevented).toBe(false);
    expect(document.activeElement).toBe(decided);
    wrapper.unmount();
  });
});

describe("DuplicateQueue - the shell chrome", () => {
  // Duplicates replaces the grid, and with it the grid's toolbar; the
  // app-wide chrome (the stats rail toggle, undo/redo) is not the grid's and
  // must not vanish with it. Settings is the title bar's (#2158), so no
  // toolbar carries a second one.
  it("carries the stats toggle and undo/redo like every other view", async () => {
    const { wrapper } = await mountQueue([group("g1")]);
    expect(wrapper.find(".tb-stats-btn").exists()).toBe(true);
    expect(wrapper.findComponent({ name: "UndoControl" }).exists()).toBe(true);
    expect(wrapper.find('button[aria-label="Settings"]').exists()).toBe(false);
    wrapper.unmount();
  });

  // The decision record's canonical tail, identical in every view:
  // [separator] [UndoControl] [rail toggle], the last two from TbGlobalActions. The ⋯ is NOT part of the
  // tail - it belongs to the left group, whose controls it collapses, and the
  // app-wide cluster never folds into it (amendment #4).
  it("orders the tail separator → UndoControl → TbGlobalActions, ⋯ elsewhere", async () => {
    const { wrapper } = await mountQueue([group("g1")]);
    const undo = wrapper.findComponent({ name: "UndoControl" }).element;
    // TbGlobalActions is multi-root; its rail toggle is a stable anchor.
    const globalActions = wrapper.find(".tb-stats-btn").element;
    const follows = (a, b) =>
      Boolean(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);

    expect(undo.previousElementSibling.classList.contains("dq-tb-sep")).toBe(
      true,
    );
    expect(follows(undo, globalActions)).toBe(true);
    expect(wrapper.find(".dq-tb-left .dq-overflow").exists()).toBe(true);
    expect(wrapper.find(".dq-tb-right .dq-overflow").exists()).toBe(false);
    wrapper.unmount();
  });

  // The ⋯ stands where the controls it collapses stood: at the end of the
  // toggle run, before D-S1.
  it("puts the ⋯ after the page toggles and before the separator", async () => {
    const { wrapper } = await mountQueue([group("g1")]);
    const burger = wrapper.find(".dq-tb-left .dq-overflow").element;
    expect(
      burger.previousElementSibling.getAttribute("data-testid") ===
        "mixed-toggle",
    ).toBe(true);
    expect(burger.nextElementSibling.classList.contains("dq-tb-sep")).toBe(
      true,
    );
    wrapper.unmount();
  });

  // Fold = CSS both ways: every control the ⋯ collapses exists as a bar
  // button AND as a row, and the container query at ≤882 flips which of the
  // pair is visible. Nothing else may be in there - the tier gate, the scope
  // pill, the count and the app-wide tail all stay on the bar at every width.
  it("carries a row for each folded page toggle and nothing else", async () => {
    const { wrapper } = await mountQueue([group("g1")]);
    await wrapper.find(".dq-overflow .tbo-trigger").trigger("click");
    const rows = wrapper.findAll(".dq-overflow .ctx-item");
    expect(rows).toHaveLength(2);
    expect(rows[0].text()).toContain("Decided");
    expect(rows[1].text()).toContain("Mixed stacks");
    // A folded control keeps the name its bar button carried. Name-from-content
    // would otherwise cut Mixed stacks down to its label and count, losing the
    // sentence that says what a mixed stack is.
    expect(rows[0].attributes("aria-label")).toBe(
      wrapper.find('[data-testid="decided-toggle"]').attributes("aria-label"),
    );
    expect(rows[1].attributes("aria-label")).toBe(
      wrapper.find('[data-testid="mixed-toggle"]').attributes("aria-label"),
    );
    // Each row's bar twin carries the fold class that hides it at the same
    // width, so exactly one of the pair is ever on screen.
    for (const toggle of [
      wrapper.find('[data-testid="decided-toggle"]'),
      wrapper.find('[data-testid="mixed-toggle"]'),
    ]) {
      expect(toggle.classes()).toContain("dq-fold-906");
    }
    wrapper.unmount();
  });

  // A fold class is a name shared between the template and one @container
  // rung, and nothing else connects them: rename it in one place and that rung
  // silently stops firing, which no mounted test can see (jsdom evaluates no
  // container query) and no reviewer reads off a diff. Both re-placements of
  // this ladder renamed these classes, so the halves are checked against each
  // other rather than against a literal.
  it("every fold class the template sets is hidden by a rung", async () => {
    const { readFileSync } = await import("node:fs");
    const source = readFileSync(
      `${process.cwd()}/src/components/views/DuplicateQueue.vue`,
      "utf8",
    );
    const template = source.slice(0, source.indexOf("<script setup>"));
    const style = source.slice(source.indexOf("<style scoped>"));
    const set = (text, pattern) =>
      [...new Set([...text.matchAll(pattern)].map((m) => m[1]))].sort();
    const used = set(template, /['"\s](dq-fold-\d+)['":\s]/g);
    const hidden = set(style, /\.(dq-fold-\d+)\s*\{/g);
    expect(used.length).toBeGreaterThan(0);
    expect(used).toEqual(hidden);
  });

  // A row does the same thing its button does.
  it("navigates from the folded row, and closes the panel behind it", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    await wrapper.find(".dq-overflow .tbo-trigger").trigger("click");
    listGroups.mockResolvedValue({
      groups: [],
      total: 0,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    await wrapper.find('[data-testid="mixed-row"]').trigger("click");
    await wrapper.vm.$nextTick();
    expect(store.showingMixed).toBe(true);
    expect(wrapper.find(".dq-overflow .ctx-item").exists()).toBe(false);
    wrapper.unmount();
  });

  // The way OUT of a sub-page is not a foldable: on the Decided and Mixed
  // pages the surviving toggle reads "Back to review" and is the visible exit,
  // so it keeps its place on the bar and the ⋯ (which would then hold nothing)
  // does not mount at all.
  it("never folds a Back to review toggle, and drops the ⋯ with it", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    listGroups.mockResolvedValue({
      groups: [],
      total: 0,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    await store.toggleDecided();
    await wrapper.vm.$nextTick();

    const back = wrapper.find('[data-testid="decided-toggle"]');
    expect(back.attributes("aria-label")).toBe("Back to review");
    expect(back.classes()).not.toContain("dq-fold-906");
    expect(wrapper.find(".dq-overflow").exists()).toBe(false);
    wrapper.unmount();
  });

  // The separator amendments: D-S1's left flank is populated at every width -
  // by the toggles above 906 and by the ⋯ that replaces them below it - so
  // both rules render at all widths and neither carries a fold class.
  it("both separators render at all widths, neither carrying a fold class", async () => {
    const { wrapper } = await mountQueue([group("g1")]);
    const separators = wrapper.findAll(".dq-toolbar .dq-tb-sep");
    expect(separators).toHaveLength(2);
    for (const separator of separators) {
      expect(separator.classes().some((c) => c.startsWith("dq-fold-"))).toBe(
        false,
      );
    }
    wrapper.unmount();
  });

  // The Decided toggle folds at ≤882 and compresses to an arrow while it is
  // the way back, so it must carry its own accessible name and keep its
  // pressed state at every width.
  it("the Decided toggle exposes its label and keeps aria-pressed", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    const toggle = wrapper.find('[data-testid="decided-toggle"]');
    expect(tipText(toggle)).toBe("Decided");
    expect(toggle.attributes("aria-label")).toBe("Decided");
    expect(toggle.attributes("aria-pressed")).toBe("false");
    expect(toggle.find(".dq-toggle-label").text()).toBe("Decided");

    listGroups.mockResolvedValue({
      groups: [],
      total: 0,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    await store.toggleDecided();
    await wrapper.vm.$nextTick();
    const flipped = wrapper.find('[data-testid="decided-toggle"]');
    expect(tipText(flipped)).toBe("Back to review");
    expect(flipped.attributes("aria-label")).toBe("Back to review");
    expect(flipped.attributes("aria-pressed")).toBe("true");
    wrapper.unmount();
  });

  // The tier trigger's label ellipsizes under pressure and hides entirely at
  // ≤1087, so the button must carry its own accessible name at every width
  // (WCAG 4.1.2 - a hidden span would leave it empty).
  it("the tier button exposes its label as tooltip and aria-label", async () => {
    const { wrapper } = await mountQueue([group("g1")]);
    const button = wrapper.find(
      '.dq-tier-wrap [data-testid="dedup-tier-trigger"]',
    );
    const label = wrapper.vm.tierLabel;
    expect(label).toBeTruthy();
    expect(tipText(button)).toBe(label);
    expect(button.attributes("aria-label")).toBe(label);
    expect(button.find(".dq-tier-label").text()).toBe(label);
    wrapper.unmount();
  });

  // Undo is owner-only on the server, but the control stays mounted and inert
  // in a read-only session so the feature is still visible; UndoControl owns
  // the disabled state.
  it("keeps undo/redo in a read-only session", async () => {
    readOnlyRef.value = true;
    const { wrapper } = await mountQueue([group("g1")]);
    expect(wrapper.findComponent({ name: "UndoControl" }).exists()).toBe(true);
    wrapper.unmount();
  });
});

describe("DuplicateQueue - undo puts the queue back", () => {
  /** Drive the queue's operation-store subscription as Pinia would. */
  async function runUndoAction(name, args = [], result = {}) {
    const afters = [];
    for (const listener of __actionListeners) {
      listener({ name, args, after: (cb) => afters.push(cb) });
    }
    for (const cb of afters) await cb(result);
  }

  // The regression this pins: undoing a stack verdict reopened the group
  // server-side and corrected the badge, but the visible list kept showing
  // one row fewer until a remount - the count said N+1 over N rows.
  it("reloads the queue after an undo that reverted a dedup operation", async () => {
    const { wrapper, store } = await mountQueue([group("g1"), group("g2")]);
    // The undo toast is the shared receipt, mounted here like every view.
    expect(wrapper.findComponent({ name: "ActionReceipt" }).exists()).toBe(
      true,
    );

    __operationStoreMock.nextUndo = {
      id: 9,
      op_type: "dedup.stack",
      batch_id: "b1",
    };
    listGroups.mockClear();
    listGroups.mockResolvedValue({
      groups: [group("g1"), group("g2"), group("g9")],
      total: 3,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });

    await runUndoAction("undo");
    await flushPromises();
    expect(listGroups).toHaveBeenCalledTimes(1);
    expect(listGroups.mock.calls[0][0].offset).toBe(0);
    expect(store.groups.map((g) => g.signature)).toContain("g9");
    wrapper.unmount();
  });

  it("restores a stacked group inside the virtual window without losing the viewport or focus", async () => {
    const first = Array.from({ length: 200 }, (_, i) => group(`g${i + 1}`));
    const { wrapper, store } = await mountQueue(first, { total: 240 });
    const list = wrapper.find(".qlist");
    Object.defineProperty(list.element, "clientHeight", {
      configurable: true,
      value: 6 * PITCH,
    });

    // Work far enough down the queue that a first-page reload cannot preserve
    // context. The focused row is the one that slid into the stacked row's old
    // place; two still-open rows are multi-selected around it.
    store.setFocus(174);
    store.toggleSelected(173);
    store.setFocus(174);
    await flushPromises();
    expect(store.focusedGroup.signature).toBe("g175");
    list.element.scrollTop = 170 * PITCH + 7;
    list.element.dispatchEvent(new window.Event("scroll"));

    __operationStoreMock.nextUndo = {
      id: 9,
      op_type: "dedup.stack",
      batch_id: "b1",
      target_ids: [9900, 9901],
    };
    const restored = group("g99-restored");
    // Requested at offset 150: the old focus (g175) shifts down one when the
    // restored group returns immediately before it.
    const refreshed = [
      ...Array.from({ length: 24 }, (_, i) => group(`g${151 + i}`)),
      restored,
      ...Array.from({ length: 66 }, (_, i) => group(`g${175 + i}`)),
    ];
    listGroups.mockClear();
    listGroups.mockResolvedValue({
      groups: refreshed,
      total: 241,
      offset: 150,
      limit: 200,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });

    await runUndoAction("undo");
    await flushPromises();
    await wrapper.vm.$nextTick();

    expect(listGroups).toHaveBeenCalledTimes(1);
    expect(listGroups.mock.calls[0][0]).toMatchObject({
      offset: 150,
      limit: 200,
    });
    expect(store.windowStart).toBe(150);
    expect(store.focusedGroup.signature).toBe("g175");
    expect(store.isSelected("g174")).toBe(true);
    expect(store.isSelected("g175")).toBe(true);
    expect(list.element.scrollTop).toBe(170 * PITCH + 7);
    expect(
      wrapper.find('[data-testid="dedup-group-g99-restored"]').exists(),
    ).toBe(true);
    wrapper.unmount();
  });

  it("does not reconcile the queue when a stack undo fails", async () => {
    const { wrapper } = await mountQueue([group("g1"), group("g2")]);
    __operationStoreMock.nextUndo = { id: 9, op_type: "dedup.stack" };
    listGroups.mockClear();

    await runUndoAction("undo", [], null);
    await flushPromises();

    expect(listGroups).not.toHaveBeenCalled();
    wrapper.unmount();
  });

  it("redo of a dedup operation reloads the same way", async () => {
    const { wrapper } = await mountQueue([group("g1")]);
    __operationStoreMock.nextRedo = { id: 4, op_type: "dedup.stack" };
    listGroups.mockClear();
    await runUndoAction("redo");
    await flushPromises();
    expect(listGroups).toHaveBeenCalledTimes(1);
    wrapper.unmount();
  });

  // The Decided screen participates in the same lifecycle: the post-undo
  // reload targets whichever flip is showing, because loadFirstPage carries
  // `decided: showingDecided`. Undoing a verdict from the flip removes the
  // group from Decided (it is back in the queue)...
  it("undo of a keep-separate while ON the Decided flip reloads the Decided list", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    listGroups.mockResolvedValue({
      groups: [
        { ...group("g9"), verdict: "keep_separate", decided_at: "2026-07-30" },
      ],
      total: 1,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    await store.toggleDecided();
    await wrapper.vm.$nextTick();
    expect(store.groups.map((g) => g.signature)).toEqual(["g9"]);

    __operationStoreMock.nextUndo = { id: 5, op_type: "dedup.keep_separate" };
    listGroups.mockClear();
    getCounts.mockClear();
    listGroups.mockResolvedValue({
      groups: [],
      total: 0,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    await runUndoAction("undo");
    await flushPromises();

    expect(listGroups).toHaveBeenCalledTimes(1);
    expect(listGroups.mock.calls[0][0].decided).toBe(true);
    expect(store.showingDecided).toBe(true);
    expect(store.groups).toHaveLength(0);
    // The badge reconciles from the server on the same pass (item 5).
    expect(getCounts).toHaveBeenCalled();
    wrapper.unmount();
  });

  // ...and redo puts it back on Decided.
  it("redo while ON the Decided flip returns the group to Decided", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    listGroups.mockResolvedValue({
      groups: [],
      total: 0,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    await store.toggleDecided();
    await wrapper.vm.$nextTick();
    expect(store.groups).toHaveLength(0);

    __operationStoreMock.nextRedo = { id: 6, op_type: "dedup.keep_separate" };
    listGroups.mockClear();
    listGroups.mockResolvedValue({
      groups: [
        { ...group("g9"), verdict: "keep_separate", decided_at: "2026-07-30" },
      ],
      total: 1,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    await runUndoAction("redo");
    await flushPromises();

    expect(listGroups.mock.calls[0][0].decided).toBe(true);
    expect(store.groups.map((g) => g.signature)).toEqual(["g9"]);
    expect(store.showingDecided).toBe(true);
    wrapper.unmount();
  });

  // The undo controls and the receipt surface are toolbar/root chrome, not
  // the queue list's: the flip must not unmount them.
  it("keeps the undo controls and receipt surface on the Decided flip", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    listGroups.mockResolvedValue({
      groups: [
        { ...group("g9"), verdict: "stacked", decided_at: "2026-07-30" },
      ],
      total: 1,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    await store.toggleDecided();
    await wrapper.vm.$nextTick();
    expect(wrapper.findComponent({ name: "UndoControl" }).exists()).toBe(true);
    expect(wrapper.findComponent({ name: "ActionReceipt" }).exists()).toBe(
      true,
    );
    wrapper.unmount();
  });

  // A reopened verdict is rescan-proof server-side, but the group may not
  // match the CURRENT lens (rescanned away, tier switched off since). The
  // reload must land on an honest empty state, never a crash or stale count.
  it("tolerates an undone group that does not reappear in the current lens", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    stackGroup.mockResolvedValue({ batch_id: "srv-1" });
    await store.stack(store.groups[0]);
    expect(store.hasGroups).toBe(false);

    __operationStoreMock.nextUndo = { id: 7, op_type: "dedup.stack" };
    listGroups.mockClear();
    listGroups.mockResolvedValue({
      groups: [],
      total: 0,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    await runUndoAction("undo");
    await flushPromises();
    await wrapper.vm.$nextTick();

    expect(store.groups).toHaveLength(0);
    expect(store.focusIndex).toBe(-1);
    expect(store.total).toBe(0);
    expect(wrapper.find(".qdone").exists()).toBe(true);
    wrapper.unmount();
  });

  // Undoing an unrelated change must not yank a triage in progress back to
  // the top: the reload is scoped to dedup operations.
  it("leaves the queue alone for an undo that touched nothing dedup", async () => {
    const { wrapper } = await mountQueue([group("g1"), group("g2")]);
    __operationStoreMock.nextUndo = { id: 3, op_type: "tags.edit" };
    listGroups.mockClear();
    await runUndoAction("undo");
    await flushPromises();
    expect(listGroups).not.toHaveBeenCalled();
    wrapper.unmount();
  });
});

describe("DuplicateQueue - verdicts inside Compare", () => {
  function compare(wrapper) {
    return wrapper.findComponent({ name: "DedupCompareDialog" });
  }

  function pressC() {
    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "c", bubbles: true }),
    );
  }

  // The regression this pins: a verdict from Compare's footer closed the
  // dialog, so triaging a run of groups there meant reopening it per group.
  // Now the store's auto-advance flips the dialog to the next group in place,
  // and it closes only when the queue has nothing left.
  it("a footer verdict advances to the next group with the dialog open", async () => {
    const { wrapper } = await mountQueue([
      group("g1"),
      group("g2"),
      group("g3"),
    ]);
    stackGroup.mockResolvedValue({});
    pressC();
    await wrapper.vm.$nextTick();
    expect(compare(wrapper).props("open")).toBe(true);

    compare(wrapper).vm.$emit("stack");
    await flushPromises();
    expect(compare(wrapper).props("open")).toBe(true);
    expect(compare(wrapper).props("group").signature).toBe("g2");

    keepGroupSeparate.mockResolvedValue({});
    compare(wrapper).vm.$emit("keep-separate");
    await flushPromises();
    expect(compare(wrapper).props("open")).toBe(true);
    expect(compare(wrapper).props("group").signature).toBe("g3");
    wrapper.unmount();
  });

  it("the verdict on the LAST group closes the dialog", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    stackGroup.mockResolvedValue({});
    pressC();
    await wrapper.vm.$nextTick();
    expect(compare(wrapper).props("open")).toBe(true);

    compare(wrapper).vm.$emit("stack");
    await flushPromises();
    expect(store.hasGroups).toBe(false);
    expect(compare(wrapper).props("open")).toBe(false);
    wrapper.unmount();
  });

  // Enter and S while Compare is open must do exactly what the footer
  // buttons do: decide, advance in place, close only at the end.
  it("keyboard Enter and S advance the same way", async () => {
    const { wrapper, store } = await mountQueue([group("g1"), group("g2")]);
    stackGroup.mockResolvedValue({});
    pressC();
    await wrapper.vm.$nextTick();

    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
    await flushPromises();
    expect(compare(wrapper).props("open")).toBe(true);
    expect(store.focusedGroup.signature).toBe("g2");

    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
    await flushPromises();
    expect(store.hasGroups).toBe(false);
    expect(compare(wrapper).props("open")).toBe(false);
    wrapper.unmount();
  });

  // Up/Down leaf through the queue from inside Compare: the dialog renders
  // the focused group, so a focus move flips it in place.
  it("ArrowDown and ArrowUp switch the compared group, clamped at the ends", async () => {
    const { wrapper, store } = await mountQueue([
      group("g1"),
      group("g2"),
      group("g3"),
    ]);
    pressC();
    await wrapper.vm.$nextTick();
    expect(compare(wrapper).props("group").signature).toBe("g1");

    const key = (name) =>
      document.dispatchEvent(
        new window.KeyboardEvent("keydown", { key: name, bubbles: true }),
      );
    key("ArrowDown");
    await wrapper.vm.$nextTick();
    expect(compare(wrapper).props("open")).toBe(true);
    expect(compare(wrapper).props("group").signature).toBe("g2");

    key("ArrowUp");
    await wrapper.vm.$nextTick();
    expect(compare(wrapper).props("group").signature).toBe("g1");
    // The queue never wraps; neither does Compare.
    key("ArrowUp");
    await wrapper.vm.$nextTick();
    expect(compare(wrapper).props("group").signature).toBe("g1");

    // A verdict after an arrow switch hits the group being SHOWN.
    key("ArrowDown");
    stackGroup.mockResolvedValue({});
    key("Enter");
    await flushPromises();
    expect(stackGroup).toHaveBeenCalledWith("g2", expect.anything());
    expect(store.groups.map((g) => g.signature)).toEqual(["g1", "g3"]);
    wrapper.unmount();
  });

  // A failed verdict changes nothing: same group, dialog still up, failure
  // reported. Advancing past a group that was NOT resolved would bury it.
  it("stays on the same group when the verdict fails", async () => {
    const { wrapper } = await mountQueue([group("g1"), group("g2")]);
    stackGroup.mockRejectedValue(new Error("locked"));
    pressC();
    await wrapper.vm.$nextTick();

    compare(wrapper).vm.$emit("stack");
    await flushPromises();
    expect(compare(wrapper).props("open")).toBe(true);
    expect(compare(wrapper).props("group").signature).toBe("g1");
    expect(errorSpy).toHaveBeenCalled();
    wrapper.unmount();
  });
});

describe("DuplicateQueue - End means the true end", () => {
  // The regression this pins: End focused the last LOADED row, so on a paging
  // queue it had to be pressed once per page. The scroll track is already
  // sized from the server total, so one press pins the scroll to the real
  // bottom while the store pages the rest in and lands the focus there.
  it("one End press lands on the true last group with nothing left to page", async () => {
    const page1 = Array.from({ length: 20 }, (_, i) => group(`g${i + 1}`));
    const { wrapper, store } = await mountQueue(page1, { total: 60 });
    let served = 20;
    listGroups.mockImplementation(async ({ limit }) => {
      const size = Math.min(limit, 60 - served);
      const next = Array.from({ length: size }, (_, i) =>
        group(`g${served + i + 1}`),
      );
      served += size;
      return {
        groups: next,
        total: 60,
        offset: served,
        limit,
        scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
      };
    });

    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "End", bubbles: true }),
    );
    for (let i = 0; i < 6; i += 1) {
      await flushPromises();
      await wrapper.vm.$nextTick();
    }

    expect(store.groups.length).toBe(60);
    expect(store.focusIndex).toBe(59);
    expect(store.hasMore).toBe(false);
    expect(store.endChaseActive).toBe(false);
    // The completion is narrated as an ordinary focus move onto the last row.
    expect(wrapper.find('[data-testid="dedup-announcement"]').text()).toContain(
      "Group 60 of 60",
    );
    wrapper.unmount();
  });

  // Over a LARGE gap the total tells End exactly which cards to fetch: one
  // offset request for the tail page, the window rebased onto it, no walk
  // through the middle and no skeletons streaming past.
  it("End over a large gap jumps straight to the tail page", async () => {
    const page1 = Array.from({ length: 20 }, (_, i) => group(`g${i + 1}`));
    const { wrapper, store } = await mountQueue(page1, { total: 200 });
    listGroups.mockClear();
    listGroups.mockImplementation(async ({ offset = 0, limit }) => {
      const size = Math.max(0, Math.min(limit, 200 - offset));
      return {
        groups: Array.from({ length: size }, (_, i) =>
          group(`g${offset + i + 1}`),
        ),
        total: 200,
        offset,
        limit,
        scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
      };
    });

    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "End", bubbles: true }),
    );
    for (let i = 0; i < 6; i += 1) {
      await flushPromises();
      await wrapper.vm.$nextTick();
    }

    // ONE tail request, offset-only - never a cursor alongside it.
    expect(listGroups).toHaveBeenCalledTimes(1);
    const tail = listGroups.mock.calls[0][0];
    expect(tail.offset).toBe(180);
    expect(tail.cursor).toBeUndefined();
    expect(store.windowStart).toBe(180);
    expect(store.groups.length).toBe(20);
    expect(store.focusIndex).toBe(199);
    // The tail's cards are mounted at their absolute indices.
    const indices = wrapper.vm.windowedGroups.map((e) => e.index);
    expect(indices).toContain(199);
    expect(wrapper.find('[data-testid="dedup-announcement"]').text()).toContain(
      "Group 200 of 200",
    );
    wrapper.unmount();
  });

  it("scrolling up from the jumped tail backfills the page above, spacers intact", async () => {
    const page1 = Array.from({ length: 20 }, (_, i) => group(`g${i + 1}`));
    const { wrapper, store } = await mountQueue(page1, { total: 200 });
    listGroups.mockImplementation(async ({ offset = 0, limit }) => {
      const size = Math.max(0, Math.min(limit, 200 - offset));
      return {
        groups: Array.from({ length: size }, (_, i) =>
          group(`g${offset + i + 1}`),
        ),
        total: 200,
        offset,
        limit,
        scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
      };
    });
    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "End", bubbles: true }),
    );
    for (let i = 0; i < 6; i += 1) {
      await flushPromises();
      await wrapper.vm.$nextTick();
    }
    expect(store.windowStart).toBe(180);

    // The user drags up past the window's start: the previous page prepends,
    // the window grows upwards, and the track's height does not move.
    const trackRows = () => {
      const spacers = wrapper
        .findAll(".qspacer")
        .reduce((px, s) => px + parseFloat(s.element.style.height || "0"), 0);
      return spacers / PITCH + wrapper.vm.windowedGroups.length;
    };
    const list = wrapper.find(".qlist");
    list.element.scrollTop = 170 * PITCH;
    await list.trigger("scroll");
    await flushPromises();
    await wrapper.vm.$nextTick();

    expect(store.windowStart).toBe(160);
    expect(store.groups.length).toBe(40);
    expect(store.groups[0].signature).toBe("g161");
    expect(trackRows()).toBe(200);
    // The rows around the scroll position are mounted, absolute indices.
    expect(wrapper.vm.windowedGroups.map((e) => e.index)).toContain(170);
    wrapper.unmount();
  });

  it("Home after an End jump returns to the top window", async () => {
    const page1 = Array.from({ length: 20 }, (_, i) => group(`g${i + 1}`));
    const { wrapper, store } = await mountQueue(page1, { total: 200 });
    listGroups.mockImplementation(async ({ offset = 0, limit }) => {
      const size = Math.max(0, Math.min(limit, 200 - offset));
      return {
        groups: Array.from({ length: size }, (_, i) =>
          group(`g${offset + i + 1}`),
        ),
        total: 200,
        offset,
        limit,
        scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
      };
    });
    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "End", bubbles: true }),
    );
    for (let i = 0; i < 6; i += 1) {
      await flushPromises();
      await wrapper.vm.$nextTick();
    }
    expect(store.windowStart).toBe(180);

    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "Home", bubbles: true }),
    );
    for (let i = 0; i < 6; i += 1) {
      await flushPromises();
      await wrapper.vm.$nextTick();
    }
    expect(store.windowStart).toBe(0);
    expect(store.focusIndex).toBe(0);
    expect(store.groups[0].signature).toBe("g1");
    expect(wrapper.vm.windowedGroups.map((e) => e.index)).toContain(0);
    wrapper.unmount();
  });

  it("End with everything loaded focuses the last row at once, as before", async () => {
    const { wrapper, store } = await mountQueue([
      group("g1"),
      group("g2"),
      group("g3"),
    ]);
    listGroups.mockClear();
    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "End", bubbles: true }),
    );
    await wrapper.vm.$nextTick();
    expect(store.focusIndex).toBe(2);
    expect(listGroups).not.toHaveBeenCalled();
    wrapper.unmount();
  });

  it("Home mid-chase cancels the jump and the user's position wins", async () => {
    const page1 = Array.from({ length: 20 }, (_, i) => group(`g${i + 1}`));
    const { wrapper, store } = await mountQueue(page1, { total: 60 });
    let release;
    listGroups.mockImplementation(
      () =>
        new Promise((resolve) => {
          release = resolve;
        }),
    );

    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "End", bubbles: true }),
    );
    expect(store.endChaseActive).toBe(true);
    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "Home", bubbles: true }),
    );
    expect(store.endChaseActive).toBe(false);

    // The page already on the wire lands, but the focus stays where the user
    // put it: a chase that yanked them back down would be worse than the bug.
    release({
      groups: [group("g21")],
      total: 60,
      offset: 21,
      limit: 200,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    await flushPromises();
    await wrapper.vm.$nextTick();
    expect(store.focusIndex).toBe(0);
    wrapper.unmount();
  });

  it("a scroll away from the tail mid-chase cancels it", async () => {
    const page1 = Array.from({ length: 20 }, (_, i) => group(`g${i + 1}`));
    const { wrapper, store } = await mountQueue(page1, { total: 60 });
    listGroups.mockImplementation(() => new Promise(() => {}));

    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "End", bubbles: true }),
    );
    expect(store.endChaseActive).toBe(true);
    // Let the pin land before the user drags away from it.
    await wrapper.vm.$nextTick();
    await wrapper.vm.$nextTick();

    const list = wrapper.find(".qlist");
    list.element.scrollTop = 0;
    await list.trigger("scroll");
    expect(store.endChaseActive).toBe(false);
    wrapper.unmount();
  });
});

describe("DuplicateQueue - the Decided page", () => {
  it("lists decided groups with their verdict and clears one on demand", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    listGroups.mockResolvedValue({
      groups: [
        { ...group("g9"), verdict: "keep_separate", decided_at: "2026-07-29" },
        { ...group("g8"), verdict: "stacked", decided_at: "2026-07-29" },
      ],
      total: 2,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });

    await store.toggleDecided();
    await wrapper.vm.$nextTick();

    // The decided request is explicit, so the two pages can never blur.
    expect(listGroups).toHaveBeenLastCalledWith(
      expect.objectContaining({ decided: true }),
    );
    // BOTH verdict kinds land here with their state and the way back.
    const rows = wrapper.findAllComponents({ name: "DedupGroupRow" });
    expect(rows[0].text()).toContain("Kept separate");
    expect(rows[0].text()).toContain("Clear decision");
    expect(rows[1].text()).toContain("Stacked");
    expect(rows[1].text()).toContain("Clear decision");
    const row = rows[0];
    // A decided row gives no verdicts: Enter must be inert here.
    stackGroup.mockResolvedValue({});
    await store.stack(store.groups[0]);
    expect(stackGroup).not.toHaveBeenCalled();

    reopenGroup.mockResolvedValue({
      signature: "g9",
      previous_verdict: "keep_separate",
      group_returned_to_queue: true,
    });
    row.vm.$emit("clear-decision");
    await flushPromises();
    // The Decided page clears through the bulk path, which stamps a client
    // gesture batch id so multi-row clears reverse as one undo step.
    expect(reopenGroup).toHaveBeenCalledWith("g9", {
      batchId: expect.stringMatching(/^cli-/),
    });
    wrapper.unmount();
  });

  // Escape peels one layer at a time, and the Decided flip is a layer: one
  // press returns to the review queue, keyboard handed straight back.
  it("Escape on the Decided flip returns to the review queue with focus", async () => {
    const { wrapper, store } = await mountQueue([group("g1"), group("g2")]);
    listGroups.mockResolvedValue({
      groups: [
        { ...group("g9"), verdict: "keep_separate", decided_at: "2026-07-30" },
      ],
      total: 1,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    await store.toggleDecided();
    await wrapper.vm.$nextTick();
    expect(store.showingDecided).toBe(true);

    listGroups.mockResolvedValue({
      groups: [group("g1"), group("g2")],
      total: 2,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    await flushPromises();

    expect(store.showingDecided).toBe(false);
    expect(store.groups.map((g) => g.signature)).toEqual(["g1", "g2"]);
    expect(document.activeElement).toBe(wrapper.find(".dq").element);
    wrapper.unmount();
  });

  // A dialog or popover on top still takes precedence: Escape closes IT
  // first, and only the next press leaves Decided.
  it("Escape peels an open popover before leaving Decided", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    listGroups.mockResolvedValue({
      groups: [
        { ...group("g9"), verdict: "stacked", decided_at: "2026-07-30" },
      ],
      total: 1,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    await store.toggleDecided();
    await wrapper.vm.$nextTick();
    await wrapper
      .find('.dq-tier-wrap [data-testid="dedup-tier-trigger"]')
      .trigger("click");

    await wrapper.find(".dq").trigger("keydown", { key: "Escape" });
    expect(wrapper.findComponent({ name: "DedupTierMenu" }).exists()).toBe(
      false,
    );
    expect(store.showingDecided).toBe(true);

    document.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
    await flushPromises();
    expect(store.showingDecided).toBe(false);
    wrapper.unmount();
  });

  // The backend orders the decided listing newest-decision-first; the client
  // renders the SERVER's order and never re-sorts it.
  it("renders decided rows in the server's order", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    listGroups.mockResolvedValue({
      groups: [
        { ...group("g5"), verdict: "keep_separate", decided_at: "2026-07-30" },
        { ...group("g2"), verdict: "stacked", decided_at: "2026-07-29" },
        { ...group("g9"), verdict: "keep_separate", decided_at: "2026-07-28" },
      ],
      total: 3,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    await store.toggleDecided();
    await wrapper.vm.$nextTick();

    expect(store.groups.map((g) => g.signature)).toEqual(["g5", "g2", "g9"]);
    expect(
      wrapper
        .findAllComponents({ name: "DedupGroupRow" })
        .map((row) => row.props("group").signature),
    ).toEqual(["g5", "g2", "g9"]);
    wrapper.unmount();
  });

  it("multi-selects decided groups and clears every selected decision", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    listGroups.mockResolvedValue({
      groups: [
        { ...group("g8"), verdict: "keep_separate", decided_at: "2026-07-29" },
        { ...group("g9"), verdict: "stacked", decided_at: "2026-07-29" },
      ],
      total: 2,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    await store.toggleDecided();
    await wrapper.vm.$nextTick();

    const rows = wrapper.findAll(".grow");
    await rows[0].trigger("click", { ctrlKey: true });
    await rows[1].trigger("click", { ctrlKey: true });
    await wrapper.vm.$nextTick();
    expect(wrapper.find(".qselchip").text()).toContain(
      "Clear decision applies",
    );

    reopenGroup.mockResolvedValue({ group_returned_to_queue: true });
    const clearButtons = wrapper.findAll(
      '[data-testid="dedup-clear-decision"]',
    );
    const bulkClear = clearButtons.find((b) =>
      b.text().includes("Clear 2 decisions"),
    );
    expect(bulkClear).toBeTruthy();
    await bulkClear.trigger("click");
    await flushPromises();
    expect(reopenGroup).toHaveBeenCalledTimes(2);
    wrapper.unmount();
  });
});

describe("DuplicateQueue - filters in the URL", () => {
  it("restores near, threshold and the Decided view from the query", async () => {
    // A full refresh lands here with only the URL; the selection must come
    // back exactly, clamped by the same rules the tier menu enforces.
    routeMock.query = {
      near: "1",
      embedding: "0",
      threshold: "0.8",
      view: "decided",
    };
    getPolicy.mockResolvedValue({
      defaults: {
        near_enabled: false,
        embedding_enabled: false,
        threshold: 0.9,
      },
      bounds: BOUNDS,
    });
    listGroups.mockResolvedValue({
      groups: [],
      total: 0,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    getCounts.mockResolvedValue({
      unresolved_groups: 0,
      by_tier: {},
      scopes: [],
      scan: { status: "complete" },
    });
    const store = useDedupStore();
    const wrapper = mount(DuplicateQueue, {
      ...globalOpts,
      attachTo: document.body,
    });
    await flushPromises();

    expect(store.nearEnabled).toBe(true);
    expect(store.embeddingEnabled).toBe(false);
    expect(store.threshold).toBe(0.8);
    expect(store.showingDecided).toBe(true);
    wrapper.unmount();
  });

  // The regression this pins (user report): the params were mirrored INTO the
  // URL, but a full reload dropped them again - the mirror ran on the policy
  // landing, one microtask before openQueue adopted the URL's filters, read
  // the still-default gate as "the user chose the defaults", and replaced the
  // URL without its filter params while the store was only just adopting
  // them. The filtersRestored gate keeps the mirror silent until then.
  it("a full reload keeps the filter params in the URL", async () => {
    routeMock.query = { near: "1", embedding: "0", threshold: "0.8" };
    getPolicy.mockResolvedValue({
      defaults: {
        near_enabled: false,
        embedding_enabled: false,
        threshold: 0.9,
      },
      bounds: BOUNDS,
    });
    listGroups.mockResolvedValue({
      groups: [group("g1")],
      total: 1,
      offset: 0,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    getCounts.mockResolvedValue({
      unresolved_groups: 1,
      by_tier: {},
      scopes: [],
      scan: { status: "complete" },
    });
    const store = useDedupStore();
    const wrapper = mount(DuplicateQueue, {
      ...globalOpts,
      attachTo: document.body,
    });
    await flushPromises();
    await wrapper.vm.$nextTick();
    await flushPromises();

    // The filters are in force...
    expect(store.nearEnabled).toBe(true);
    expect(store.embeddingEnabled).toBe(false);
    expect(store.threshold).toBe(0.8);
    // ...and no mirror write ever stripped them from the address.
    for (const call of routerReplace.mock.calls) {
      expect(call[0].query.near).toBe("1");
      expect(call[0].query.threshold).toBe("0.8");
    }
    wrapper.unmount();
  });

  // The regression this pins (user report): with the open queue EMPTY,
  // flipping to Decided flashed the decided rows and then fell back to
  // "Queue clear". The mirror's own replace() write (view=decided) re-fired
  // the scope watcher - a getter returning a fresh array is compared by
  // identity, so EVERY query write refired it - and syncQueueToRoute, holding
  // no rows, fell through its fast path into a full openQueue, which
  // force-reset the flip and reloaded the open queue over the decided rows.
  // With rows in the queue the fast path absorbed the refire, which is why
  // only the empty queue ever showed it.
  it("flipping to Decided on an empty queue survives its own mirror write", async () => {
    getPolicy.mockResolvedValue({
      defaults: {
        near_enabled: false,
        embedding_enabled: false,
        threshold: 0.9,
      },
      bounds: BOUNDS,
    });
    listGroups.mockImplementation(async (args) =>
      args?.decided
        ? {
            groups: [group("g1")],
            total: 1,
            offset: 0,
            limit: 20,
            scan: {
              status: "complete",
              scanned_pictures: 1,
              total_pictures: 1,
            },
          }
        : {
            groups: [],
            total: 0,
            offset: 0,
            limit: 20,
            scan: {
              status: "complete",
              scanned_pictures: 1,
              total_pictures: 1,
            },
          },
    );
    getCounts.mockResolvedValue({
      unresolved_groups: 0,
      by_tier: {},
      scopes: [],
      scan: { status: "complete" },
    });
    // A real replace() lands in the route the component watches; the feedback
    // loop this test pins cannot fire against a write-only stub.
    routerReplace.mockImplementation((to) => {
      routeMock.query = to.query;
    });
    const store = useDedupStore();
    const wrapper = mount(DuplicateQueue, {
      ...globalOpts,
      attachTo: document.body,
    });
    await flushPromises();
    expect(wrapper.text()).toContain("Queue clear");
    startScan.mockClear();

    await wrapper.find('[data-testid="decided-toggle"]').trigger("click");
    await flushPromises();
    await wrapper.vm.$nextTick();
    await flushPromises();

    // The flip is a reload of the SAME queue, never a reopen: pre-fix the
    // mirror's write refired syncQueueToRoute, whose openQueue restarted the
    // scan, reset the tallies, and - depending on how the resulting cascade
    // interleaved - reloaded the open queue over the decided rows.
    expect(startScan).not.toHaveBeenCalled();
    expect(store.showingDecided).toBe(true);
    expect(store.groups).toHaveLength(1);
    expect(wrapper.text()).not.toContain("Queue clear");
    wrapper.unmount();
  });

  it("mirrors a filter change into the URL with replace, not push", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    routerReplace.mockReset();

    await store.setTierEnabled("near", true);
    await wrapper.vm.$nextTick();

    const { query } = routerReplace.mock.calls.at(-1)[0];
    expect(query.near).toBe("1");
    expect(query.embedding).toBe("0");
    wrapper.unmount();
  });

  it("drops the filter params when the selection returns to the defaults", async () => {
    const { wrapper, store } = await mountQueue([group("g1")]);
    await store.setTierEnabled("near", true);
    await wrapper.vm.$nextTick();
    routeMock.query = routerReplace.mock.calls.at(-1)[0].query;
    routerReplace.mockReset();

    await store.setTierEnabled("near", false);
    await wrapper.vm.$nextTick();

    const { query } = routerReplace.mock.calls.at(-1)[0];
    expect(query.near).toBeUndefined();
    expect(query.threshold).toBeUndefined();
    wrapper.unmount();
  });
});

describe("DuplicateQueue - multi-select", () => {
  it("ctrl+click selects, the buttons rename, and one verdict takes all", async () => {
    const { wrapper } = await mountQueue([
      group("g1"),
      group("g2"),
      group("g3"),
    ]);
    const rows = wrapper.findAll(".grow");
    await rows[0].trigger("click", { ctrlKey: true });
    await rows[1].trigger("click", { ctrlKey: true });
    await wrapper.vm.$nextTick();

    // The bulk scope is stated twice: once in the header, once on the very
    // buttons that will act.
    expect(wrapper.find(".qselchip").text()).toContain("2 groups selected");
    const stackBtn = wrapper
      .findAll(".grow")[0]
      .find('[data-testid="dedup-stack"]');
    expect(stackBtn.text()).toContain("Stack 2 groups");

    applyVerdictBatch.mockResolvedValue({
      batch_id: "cli-visible",
      results: [{}, {}],
    });
    await stackBtn.trigger("click");
    await wrapper.vm.$nextTick();
    await wrapper.vm.$nextTick();

    expect(applyVerdictBatch).toHaveBeenCalledTimes(1);
    const [actions, options] = applyVerdictBatch.mock.calls[0];
    expect(actions.map((action) => action.signature)).toEqual(["g1", "g2"]);
    expect(options.batchId).toMatch(/^cli-/);
  });

  it("keeps rows and announcement stable until the selected stack gesture settles", async () => {
    const { wrapper, store } = await mountQueue([
      group("g1"),
      group("g2"),
      group("g3"),
    ]);
    const rows = wrapper.findAll(".grow");
    await rows[0].trigger("click", { ctrlKey: true });
    await rows[2].trigger("click", { ctrlKey: true });
    let resolveBatch;
    applyVerdictBatch.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveBatch = resolve;
        }),
    );

    await wrapper
      .findAll(".grow")[0]
      .find('[data-testid="dedup-stack"]')
      .trigger("click");
    await vi.waitFor(() => expect(resolveBatch).toBeTypeOf("function"));
    await wrapper.vm.$nextTick();

    expect(store.groups.map((entry) => entry.signature)).toEqual([
      "g1",
      "g2",
      "g3",
    ]);
    expect(wrapper.findAll(".grow")).toHaveLength(3);
    expect(
      wrapper.find('[data-testid="dedup-announcement"]').text(),
    ).not.toContain("Stacked");

    resolveBatch({ batch_id: "cli-visible", results: [{}, {}] });
    await flushPromises();
    await wrapper.vm.$nextTick();
    expect(store.groups.map((entry) => entry.signature)).toEqual(["g2"]);
    expect(wrapper.findAll(".grow")).toHaveLength(1);
    expect(wrapper.find('[data-testid="dedup-announcement"]').text()).toContain(
      "Stacked 2 groups",
    );
    wrapper.unmount();
  });

  it("shift+click selects the range from the focus, Escape clears it", async () => {
    const { wrapper, store } = await mountQueue([
      group("g1"),
      group("g2"),
      group("g3"),
    ]);
    const rows = wrapper.findAll(".grow");
    await rows[0].trigger("click");
    await rows[2].trigger("click", { shiftKey: true });
    expect(store.selectionCount).toBe(3);

    await wrapper.trigger("keydown", { key: "Escape" });
    expect(store.selectionCount).toBe(0);
    // Clearing the selection must not cost the user their place.
    expect(store.focusIndex).toBe(2);
  });

  it("every selected row wears the Enter/S chips; C stays on the focus", async () => {
    const { wrapper } = await mountQueue([
      group("g1"),
      group("g2"),
      group("g3"),
    ]);
    const rows = wrapper.findAll(".grow");
    await rows[0].trigger("click", { ctrlKey: true });
    await rows[1].trigger("click", { ctrlKey: true });
    await wrapper.vm.$nextTick();

    const fresh = wrapper.findAll(".grow");
    // Both selected rows say where Enter acts, because it acts on both.
    expect(fresh[0].find('[data-testid="dedup-stack"] kbd').exists()).toBe(
      true,
    );
    expect(fresh[1].find('[data-testid="dedup-stack"] kbd').exists()).toBe(
      true,
    );
    expect(fresh[2].find('[data-testid="dedup-stack"] kbd').exists()).toBe(
      false,
    );
    // Compare opens ONE group, so its chip stays with the keyboard cursor.
    expect(fresh[0].find('[data-testid="dedup-compare"] kbd').exists()).toBe(
      false,
    );
    expect(fresh[1].find('[data-testid="dedup-compare"] kbd').exists()).toBe(
      true,
    );
    // The old explicit label is gone.
    expect(wrapper.text()).not.toContain("Keyboard acts here");
  });

  it("a verdict on an unselected group stays single", async () => {
    const { wrapper, store } = await mountQueue([
      group("g1"),
      group("g2"),
      group("g3"),
    ]);
    const rows = wrapper.findAll(".grow");
    await rows[0].trigger("click", { ctrlKey: true });
    await rows[1].trigger("click", { ctrlKey: true });

    stackGroup.mockResolvedValue({});
    // The third row is OUTSIDE the selection: its button must say and do the
    // single-group thing.
    const outsideBtn = wrapper
      .findAll(".grow")[2]
      .find('[data-testid="dedup-stack"]');
    expect(outsideBtn.text()).not.toContain("groups");
    await outsideBtn.trigger("click");
    await wrapper.vm.$nextTick();
    expect(stackGroup).toHaveBeenCalledTimes(1);
    expect(store.selectionCount).toBe(2);
  });
});

describe("DuplicateQueue - the render window", () => {
  it("follows the scroll, not just the keyboard focus", async () => {
    // The regression this pins: the window was anchored to focusIndex alone,
    // so a mouse user scrolling a 327-group queue saw ~9 rows and then blank
    // spacer for the rest.
    const many = Array.from({ length: 20 }, (_, i) => group(`g${i + 1}`));
    const { wrapper } = await mountQueue(many);
    const before = wrapper.vm.windowedGroups.map((e) => e.index);
    expect(before).not.toContain(19);

    const list = wrapper.find(".qlist");
    // ~row 15 at the estimate pitch (happy-dom never refines the measure).
    list.element.scrollTop = 15 * PITCH;
    await list.trigger("scroll");
    await wrapper.vm.$nextTick();
    const after = wrapper.vm.windowedGroups.map((e) => e.index);
    expect(after).toContain(19);
    // Scroll-anchored, not a union with the focus window: the mounted count
    // must stay a constant, so the head of the queue unmounts behind us.
    expect(after).not.toContain(0);
  });

  it("sizes the scroll track for the whole queue, not the pages loaded so far", async () => {
    // The regression this pins: the spacers stood for the LOADED rows, so the
    // track grew every time a page landed. The thumb shrank and jumped under
    // the user's hand and "the bottom" moved each time they reached it.
    const page1 = Array.from({ length: 20 }, (_, i) => group(`g${i + 1}`));
    const { wrapper, store } = await mountQueue(page1, { total: 200 });

    // What the track stands for, in rows: the two spacers plus the rows that
    // are actually mounted between them (which have no height in happy-dom).
    const trackRows = () => {
      const spacers = wrapper
        .findAll(".qspacer")
        .reduce((px, s) => px + parseFloat(s.element.style.height || "0"), 0);
      return spacers / PITCH + wrapper.vm.windowedGroups.length;
    };
    expect(trackRows()).toBe(200);

    listGroups.mockResolvedValue({
      groups: Array.from({ length: 20 }, (_, i) => group(`g${i + 21}`)),
      total: 200,
      offset: 20,
      limit: 20,
      scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
    });
    const list = wrapper.find(".qlist");
    list.element.scrollTop = 15 * PITCH;
    await list.trigger("scroll");
    await flushPromises();
    await wrapper.vm.$nextTick();

    expect(store.groups.length).toBe(40);
    expect(trackRows()).toBe(200);
  });

  it("keeps paging while the scroll sits past the rows it holds", async () => {
    // A drag into the reserved-but-unloaded tail fires ONE scroll event. Without
    // a second trigger on arrival, the chase stalls a page short of the user.
    const page1 = Array.from({ length: 20 }, (_, i) => group(`g${i + 1}`));
    const { wrapper, store } = await mountQueue(page1, { total: 200 });
    let served = 20;
    listGroups.mockImplementation(async () => {
      const next = Array.from({ length: 20 }, (_, i) =>
        group(`g${served + i + 1}`),
      );
      served += 20;
      return {
        groups: next,
        total: 200,
        offset: served,
        limit: 20,
        scan: { status: "complete", scanned_pictures: 1, total_pictures: 1 },
      };
    });

    const list = wrapper.find(".qlist");
    list.element.scrollTop = 70 * PITCH;
    await list.trigger("scroll");
    for (let i = 0; i < 6; i += 1) {
      await flushPromises();
      await wrapper.vm.$nextTick();
    }
    // It walked out to the scroll position and then stopped, rather than
    // fetching one page or running to the end of the queue.
    expect(store.groups.length).toBeGreaterThanOrEqual(86);
    expect(store.groups.length).toBeLessThan(200);
  });

  it("every mounted row may decode thumbnails", async () => {
    const { wrapper } = await mountQueue([
      group("g1"),
      group("g2"),
      group("g3"),
    ]);
    for (const entry of wrapper.vm.windowedGroups) {
      expect(entry.loadThumbnails).toBe(true);
    }
  });
});

describe("DuplicateQueue - the tier gate", () => {
  // Nothing about the ladder is hardcoded: the ids, the prerequisites and which
  // tier cannot be switched off all arrive from GET /dedup/policy.
  it("renders the tiers the server published, with tier 1 locked", async () => {
    const { wrapper } = await mountQueue([group("g1")], {
      byTier: { exact: 1204, near: 96, embedding: 9 },
    });
    wrapper.find('[data-testid="dedup-tier-trigger"]').trigger("click");
    await wrapper.vm.$nextTick();

    const rows = wrapper.findAll(".tierrow");
    expect(rows).toHaveLength(3);
    expect(rows[0].text()).toContain("always included");
    // Tier 3 is unreachable until tier 2 is on, so it must not be pressable.
    expect(rows[2].attributes("disabled")).toBeDefined();
    wrapper.unmount();
  });
});

describe("DuplicateQueue - a read-only session", () => {
  // Navigation stays live because reading the queue is not a verdict; the bulk
  // action is a verdict, so it goes.
  it("hides the bulk auto-stack button", async () => {
    getCounts.mockResolvedValue({
      unresolved_groups: 1,
      by_tier: { exact: 12 },
      tiers: [],
      scan: { state: "idle" },
    });
    readOnlyRef.value = true;
    const { wrapper, store } = await mountQueue([group("g1")]);
    store.exactCount = 12;
    await wrapper.vm.$nextTick();
    expect(wrapper.find('[data-testid="auto-stack-trigger"]').exists()).toBe(
      false,
    );
    wrapper.unmount();
  });

  it("tells Compare that the session cannot act", async () => {
    readOnlyRef.value = true;
    const { wrapper } = await mountQueue([group("g1")]);
    expect(
      wrapper.findComponent({ name: "DedupCompareDialog" }).props("readOnly"),
    ).toBe(true);
    wrapper.unmount();
  });
});

describe("DuplicateQueue - the Auto-stack dry run", () => {
  // The dry run can be slow on a large library, and the dialog it answers can
  // be closed, reopened or left behind before it lands. Its figures are the
  // user's consent, so they must belong to the dialog on screen, and the run
  // that consent starts must act on the scope those figures described.
  const mountWithExactGroups = () =>
    mountQueue([group("g1")], { byTier: { exact: 3 } });

  beforeEach(() => {
    // A started run is followed by a timed poll; fake timers keep it from
    // outliving the case.
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("keeps the newer opening's figures when the older reply lands last", async () => {
    const first = deferred();
    const second = deferred();
    previewAutoStackExact
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const { wrapper } = await mountWithExactGroups();

    await autoStackTrigger(wrapper).trigger("click");
    await autoStackTrigger(wrapper).trigger("click");
    // The newer opening cancels the request it replaced.
    expect(previewAutoStackExact.mock.calls[0][0].signal.aborted).toBe(true);
    expect(previewAutoStackExact.mock.calls[1][0].signal.aborted).toBe(false);

    second.resolve({ dry_run: true, groups: 2, pictures: 4 });
    await flushPromises();
    first.resolve({ dry_run: true, groups: 99, pictures: 999 });
    await flushPromises();

    expect(autoStackDialog(wrapper).props("preview")).toEqual({
      dry_run: true,
      groups: 2,
      pictures: 4,
    });
    expect(autoStackDialog(wrapper).props("loading")).toBe(false);
    wrapper.unmount();
  });

  it("cancels the dry run when the dialog closes and ignores what lands after", async () => {
    const pending = deferred();
    previewAutoStackExact.mockReturnValueOnce(pending.promise);
    const { wrapper } = await mountWithExactGroups();

    await autoStackTrigger(wrapper).trigger("click");
    const { signal } = previewAutoStackExact.mock.calls[0][0];
    expect(signal.aborted).toBe(false);

    autoStackDialog(wrapper).vm.$emit("close");
    await flushPromises();
    expect(signal.aborted).toBe(true);

    pending.resolve({ dry_run: true, groups: 5, pictures: 10 });
    await flushPromises();
    expect(autoStackDialog(wrapper).props("open")).toBe(false);
    expect(autoStackDialog(wrapper).props("preview")).toBe(null);
    expect(autoStackDialog(wrapper).props("loading")).toBe(false);
    wrapper.unmount();
  });

  it("cancels the dry run when the queue is left", async () => {
    previewAutoStackExact.mockReturnValueOnce(deferred().promise);
    const { wrapper } = await mountWithExactGroups();

    await autoStackTrigger(wrapper).trigger("click");
    const { signal } = previewAutoStackExact.mock.calls[0][0];
    wrapper.unmount();
    expect(signal.aborted).toBe(true);
  });

  it("starts the run on the scope it previewed after the queue moved to another", async () => {
    previewAutoStackExact.mockResolvedValueOnce({
      dry_run: true,
      groups: 3,
      pictures: 6,
    });
    startAutoStackRun.mockResolvedValueOnce({
      run: runFixture(),
      retryAfterMs: 2000,
    });
    const { wrapper, store } = await mountWithExactGroups();

    await autoStackTrigger(wrapper).trigger("click");
    await flushPromises();
    expect(previewAutoStackExact).toHaveBeenLastCalledWith(
      expect.objectContaining({ scopeType: "global", scopeId: null }),
    );

    // The sidebar's "Find duplicates in..." moves the queue while the dialog
    // still shows the library-wide figures.
    routeMock.query = { scope: "project", scope_id: "7" };
    await flushPromises();
    expect([store.scopeType, store.scopeId]).toEqual(["project", "7"]);

    autoStackDialog(wrapper).vm.$emit("confirm");
    await flushPromises();
    // The dry run never writes: only the run's own route does.
    expect(previewAutoStackExact).toHaveBeenCalledTimes(1);
    expect(startAutoStackRun).toHaveBeenCalledTimes(1);
    expect(startAutoStackRun).toHaveBeenCalledWith({
      scopeType: "global",
      scopeId: null,
      requestId: expect.stringMatching(/^cli-test-\d+$/),
    });
    // The start answers once the run is queued, so the dialog goes and the
    // banner takes over.
    expect(autoStackDialog(wrapper).props("open")).toBe(false);
    expect(wrapper.find('[data-testid="auto-stack-banner"]').exists()).toBe(
      true,
    );
    expect(announced(wrapper)).toBe(
      "Auto-stack started in the background. Its progress shows above the queue.",
    );
    wrapper.unmount();
  });

  it("hands a failed dry run to the dialog as the reason it failed", async () => {
    previewAutoStackExact.mockRejectedValueOnce(
      Object.assign(new Error("Request failed with status code 422"), {
        response: {
          status: 422,
          data: {
            detail: [{ msg: "fake-reason", input: "fake-echoed-input" }],
          },
        },
      }),
    );
    const { wrapper } = await mountWithExactGroups();

    await autoStackTrigger(wrapper).trigger("click");
    await flushPromises();

    expect(autoStackDialog(wrapper).props("previewError")).toEqual({
      kind: "validation",
      status: 422,
      detail: "fake-reason",
      code: null,
      run: null,
    });
    expect(autoStackDialog(wrapper).props("preview")).toBe(null);
    expect(autoStackDialog(wrapper).props("loading")).toBe(false);
    wrapper.unmount();
  });
});

describe("DuplicateQueue - starting the Auto-stack run", () => {
  // The reported failure: the whole tier ran inside one request, the client
  // gave up on it after 60 s, and the queue could only say the outcome was not
  // confirmed. The start now answers as soon as the run is queued; an
  // unanswered start is sent again under the same request id, which the server
  // answers with the run the first attempt began, never a second run.
  const OLD_NOTICE = /did not answer, so its outcome is not confirmed/;

  async function openAndPreview() {
    previewAutoStackExact.mockResolvedValue({
      dry_run: true,
      groups: 3,
      pictures: 6,
    });
    const mounted = await mountQueue([group("g1")], { byTier: { exact: 3 } });
    await autoStackTrigger(mounted.wrapper).trigger("click");
    await flushPromises();
    return mounted;
  }

  /** Confirm the open dialog, and run out the store's two resend waits. */
  async function confirmAndWaitOutResends(wrapper) {
    autoStackDialog(wrapper).vm.$emit("confirm");
    await flushPromises();
    await vi.advanceTimersByTimeAsync(1000);
    await vi.advanceTimersByTimeAsync(3000);
    await flushPromises();
  }

  const requestIds = () =>
    startAutoStackRun.mock.calls.map(([args]) => args.requestId);

  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it.each([
    ["the client's timeout", "timeout of 60000ms exceeded", "ECONNABORTED"],
    ["a refused or lost connection", "Network Error", "ERR_NETWORK"],
  ])(
    "keeps the dialog open on an unanswered start, and confirming again resends it: %s",
    async (_label, message, code) => {
      const kind = code === "ECONNABORTED" ? "timeout" : "network";
      startAutoStackRun.mockRejectedValue(
        Object.assign(new Error(message), { code }),
      );
      const { wrapper } = await openAndPreview();
      const notices = useNoticeStore();

      await confirmAndWaitOutResends(wrapper);

      // Sent three times as one request.
      expect(startAutoStackRun).toHaveBeenCalledTimes(3);
      expect(new Set(requestIds()).size).toBe(1);
      expect(autoStackDialog(wrapper).props("open")).toBe(true);
      expect(autoStackDialog(wrapper).props("startError")).toMatchObject({
        kind,
      });
      expect(announced(wrapper)).not.toMatch(OLD_NOTICE);
      expect(notices.info).not.toHaveBeenCalled();
      expect(notices.error).not.toHaveBeenCalled();

      // The fourth try is the same request too, and its answer is the run.
      const [sent] = requestIds();
      startAutoStackRun.mockReset();
      startAutoStackRun.mockResolvedValueOnce({
        run: runFixture(),
        retryAfterMs: 2000,
      });
      autoStackDialog(wrapper).vm.$emit("confirm");
      await flushPromises();
      expect(requestIds()).toEqual([sent]);
      expect(autoStackDialog(wrapper).props("open")).toBe(false);
      expect(wrapper.find('[data-testid="auto-stack-banner"]').exists()).toBe(
        true,
      );
      wrapper.unmount();
    },
  );

  it("mints a new request id for each opening of the dialog", async () => {
    startAutoStackRun.mockResolvedValue({
      run: runFixture({ status: "complete", active: false, processed: 120 }),
      retryAfterMs: null,
    });
    const { wrapper } = await openAndPreview();
    autoStackDialog(wrapper).vm.$emit("confirm");
    await flushPromises();
    // The id named a run that had already ended by the time it was answered.
    expect(announced(wrapper)).toBe(
      "That Auto-stack run has already ended. Its result shows above the queue.",
    );

    await autoStackTrigger(wrapper).trigger("click");
    await flushPromises();
    autoStackDialog(wrapper).vm.$emit("confirm");
    await flushPromises();
    const [first, second] = requestIds();
    expect(first).toMatch(/^cli-test-\d+$/);
    expect(second).toMatch(/^cli-test-\d+$/);
    expect(second).not.toBe(first);
    wrapper.unmount();
  });

  it("says an unanswered start a closed dialog no longer shows, and looks for its run", async () => {
    startAutoStackRun.mockRejectedValue(
      Object.assign(new Error("timeout of 60000ms exceeded"), {
        code: "ECONNABORTED",
      }),
    );
    const { wrapper } = await openAndPreview();
    const notices = useNoticeStore();
    expect(getLatestAutoStackRun).toHaveBeenCalledTimes(1);

    autoStackDialog(wrapper).vm.$emit("confirm");
    await flushPromises();
    autoStackDialog(wrapper).vm.$emit("close");
    await flushPromises();
    // The first attempt did start a run; only its answer was lost.
    getLatestAutoStackRun.mockResolvedValueOnce(runFixture());
    await vi.advanceTimersByTimeAsync(1000);
    await vi.advanceTimersByTimeAsync(3000);
    await flushPromises();

    const sentence =
      "No answer came from the server, so it is not known whether the run started. Its progress shows above the queue if it did.";
    expect(notices.error).toHaveBeenCalledWith(sentence);
    expect(announced(wrapper)).toBe(sentence);
    expect(announced(wrapper)).not.toMatch(OLD_NOTICE);
    expect(getLatestAutoStackRun).toHaveBeenCalledTimes(2);
    expect(wrapper.find('[data-testid="auto-stack-banner"]').exists()).toBe(
      true,
    );
    wrapper.unmount();
  });

  it("does not resend a refusal, and shows the run it names", async () => {
    startAutoStackRun.mockRejectedValueOnce(
      refusal(409, {
        code: "auto_stack_run_active",
        message: "Run 3 is already active.",
        run: runFixture({ id: 3 }),
      }),
    );
    const { wrapper, store } = await openAndPreview();
    await confirmAndWaitOutResends(wrapper);

    expect(startAutoStackRun).toHaveBeenCalledTimes(1);
    expect(autoStackDialog(wrapper).props("startError")).toMatchObject({
      kind: "conflict",
      code: "auto_stack_run_active",
    });
    expect(store.autoStackRun.id).toBe(3);
    expect(wrapper.find('[data-testid="auto-stack-banner"]').exists()).toBe(
      true,
    );
    wrapper.unmount();
  });

  it("tells the dialog a start is on the wire, and holds the trigger meanwhile", async () => {
    const pending = deferred();
    startAutoStackRun.mockReturnValueOnce(pending.promise);
    const { wrapper } = await openAndPreview();
    expect(autoStackDialog(wrapper).props("busy")).toBe(false);

    autoStackDialog(wrapper).vm.$emit("confirm");
    await flushPromises();
    expect(autoStackDialog(wrapper).props("busy")).toBe(true);
    expect(autoStackTrigger(wrapper).attributes("aria-disabled")).toBe("true");
    expect(tipText(autoStackTrigger(wrapper))).toBe(
      "Waiting for the server to answer the last Auto-stack request.",
    );

    pending.resolve({ run: runFixture(), retryAfterMs: 2000 });
    await flushPromises();
    expect(autoStackDialog(wrapper).props("busy")).toBe(false);
    wrapper.unmount();
  });

  it("holds the trigger while a run is going, without opening the dialog", async () => {
    getLatestAutoStackRun.mockResolvedValueOnce(runFixture());
    const { wrapper } = await mountQueue([group("g1")], {
      byTier: { exact: 3 },
    });
    await flushPromises();

    const trigger = autoStackTrigger(wrapper);
    // Held, not disabled: it keeps focus and its tooltip says why.
    expect(trigger.attributes("aria-disabled")).toBe("true");
    expect(trigger.attributes("disabled")).toBeUndefined();
    expect(tipText(trigger)).toBe(
      "An Auto-stack run is already going. Its progress shows above the queue.",
    );
    await trigger.trigger("click");
    await flushPromises();
    expect(previewAutoStackExact).not.toHaveBeenCalled();
    expect(autoStackDialog(wrapper).props("open")).toBe(false);
    wrapper.unmount();
  });

  it("offers the trigger when no run is going", async () => {
    const { wrapper } = await mountQueue([group("g1")], {
      byTier: { exact: 3 },
    });
    await flushPromises();
    const trigger = autoStackTrigger(wrapper);
    expect(trigger.attributes("aria-disabled")).toBeUndefined();
    expect(tipText(trigger)).toBe(trigger.attributes("aria-label"));
    wrapper.unmount();
  });
});

describe("DuplicateQueue - the Auto-stack run banner", () => {
  // The banner is the run's only surface once the start has answered: it
  // follows the run, stops it, undoes it, and is put away once it has ended.
  const banner = (wrapper) => wrapper.find('[data-testid="auto-stack-banner"]');

  /** Mount over a run the session finds when the queue opens. */
  async function mountWithRun(run) {
    getLatestAutoStackRun.mockResolvedValueOnce(run);
    const mounted = await mountQueue([group("g1")], { byTier: { exact: 3 } });
    await flushPromises();
    return mounted;
  }

  /** Press a banner control the way the keyboard would leave focus on it. */
  async function press(wrapper, testId) {
    const button = wrapper.find(`[data-testid="${testId}"]`);
    expect(button.exists()).toBe(true);
    button.element.focus();
    await button.trigger("click");
    await flushPromises();
  }

  const queueHasFocus = (wrapper) =>
    document.activeElement === wrapper.find(".dq").element;

  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("looks for the newest run on arriving, and shows it", async () => {
    const { wrapper } = await mountWithRun(runFixture());
    expect(getLatestAutoStackRun).toHaveBeenCalledTimes(1);
    expect(banner(wrapper).exists()).toBe(true);
    wrapper.unmount();
  });

  it("shows no banner when there is no run", async () => {
    const { wrapper } = await mountWithRun(null);
    expect(banner(wrapper).exists()).toBe(false);
    wrapper.unmount();
  });

  it("neither looks for a run nor shows one in a read-only session", async () => {
    readOnlyRef.value = true;
    const { wrapper, store } = await mountQueue([group("g1")]);
    await flushPromises();
    expect(getLatestAutoStackRun).not.toHaveBeenCalled();
    store.autoStackRun = runFixture();
    await flushPromises();
    expect(banner(wrapper).exists()).toBe(false);
    wrapper.unmount();
  });

  it.each([
    [
      "an apply run",
      runFixture(),
      runFixture({ cancel_requested: true }),
      "Auto-stack stops after the current step. The stacks it made so far stay.",
    ],
    [
      "an undo",
      runFixture({ id: 8, kind: "revert", target_run_id: 7 }),
      runFixture({
        id: 8,
        kind: "revert",
        target_run_id: 7,
        cancel_requested: true,
      }),
      "The undo stops after the current step. The groups it put back stay in the queue.",
    ],
    [
      "an apply run that had already ended",
      runFixture(),
      runFixture({ status: "complete", active: false, processed: 120 }),
      "Auto-stack had already ended. Its result shows above the queue.",
    ],
    [
      "an undo that had already ended",
      runFixture({ id: 8, kind: "revert", target_run_id: 7 }),
      runFixture({
        id: 8,
        kind: "revert",
        target_run_id: 7,
        status: "complete",
        active: false,
      }),
      "The undo had already ended. Its result shows above the queue.",
    ],
  ])(
    "stops %s and gives the queue the keyboard back",
    async (_label, run, answer, said) => {
      cancelAutoStackRun.mockResolvedValueOnce({
        run: answer,
        retryAfterMs: 2000,
      });
      const { wrapper } = await mountWithRun(run);

      await press(wrapper, "auto-stack-stop");

      expect(cancelAutoStackRun).toHaveBeenCalledWith(run.id);
      expect(announced(wrapper)).toBe(said);
      expect(queueHasFocus(wrapper)).toBe(true);
      wrapper.unmount();
    },
  );

  it("hands the banner the request on the wire, and whether the run can be read", async () => {
    const pending = deferred();
    cancelAutoStackRun.mockReturnValueOnce(pending.promise);
    getAutoStackRun.mockRejectedValue(
      Object.assign(new Error("Network Error"), { code: "ERR_NETWORK" }),
    );
    const { wrapper } = await mountWithRun(runFixture());
    const props = () =>
      wrapper.findComponent({ name: "DedupAutoStackBanner" }).props();
    expect(props()).toMatchObject({ action: null, unreachable: false });

    await vi.advanceTimersByTimeAsync(2000);
    await flushPromises();
    expect(props().unreachable).toBe(true);

    await press(wrapper, "auto-stack-stop");
    expect(props().action).toBe("cancel");
    pending.resolve({
      run: runFixture({ cancel_requested: true }),
      retryAfterMs: 2000,
    });
    await flushPromises();
    expect(props()).toMatchObject({ action: null, unreachable: false });
    wrapper.unmount();
  });

  it("undoes an ended run as a run of its own, which the banner follows", async () => {
    revertAutoStackRun.mockResolvedValueOnce({
      run: runFixture({ id: 8, kind: "revert", target_run_id: 7 }),
      retryAfterMs: 2000,
    });
    const { wrapper, store } = await mountWithRun(
      runFixture({ status: "complete", active: false, can_revert: true }),
    );

    await press(wrapper, "auto-stack-revert");

    expect(revertAutoStackRun).toHaveBeenCalledWith(7, {
      requestId: expect.stringMatching(/^cli-test-\d+$/),
    });
    expect(store.autoStackRun).toMatchObject({ id: 8, kind: "revert" });
    expect(announced(wrapper)).toBe(
      "Undoing Auto-stack in the background. Its progress shows above the queue.",
    );
    expect(queueHasFocus(wrapper)).toBe(true);
    wrapper.unmount();
  });

  it("says when the undo it asked for had already ended", async () => {
    revertAutoStackRun.mockResolvedValueOnce({
      run: runFixture({
        id: 8,
        kind: "revert",
        target_run_id: 7,
        status: "complete",
        active: false,
      }),
      retryAfterMs: null,
    });
    const { wrapper } = await mountWithRun(
      runFixture({ status: "complete", active: false, can_revert: true }),
    );
    await press(wrapper, "auto-stack-revert");
    expect(announced(wrapper)).toBe(
      "That undo has already ended. Its result shows above the queue.",
    );
    wrapper.unmount();
  });

  it("sends an unanswered undo again under the same id, and an answered one never", async () => {
    revertAutoStackRun.mockRejectedValue(
      Object.assign(new Error("Network Error"), { code: "ERR_NETWORK" }),
    );
    const { wrapper } = await mountWithRun(
      runFixture({ status: "complete", active: false, can_revert: true }),
    );
    const notices = useNoticeStore();
    const ids = () => revertAutoStackRun.mock.calls.map(([, o]) => o.requestId);

    await press(wrapper, "auto-stack-revert");
    await vi.advanceTimersByTimeAsync(1000);
    await vi.advanceTimersByTimeAsync(3000);
    await flushPromises();
    expect(notices.error).toHaveBeenCalledWith(
      "No answer came from the server, so it is not known whether the undo started. Press it again to retry: it sends the same request, which cannot start a second undo.",
    );
    const [unanswered] = ids();
    expect(new Set(ids())).toEqual(new Set([unanswered]));

    // The next press repeats that request; a refusal is an answer.
    revertAutoStackRun.mockReset();
    revertAutoStackRun.mockRejectedValueOnce(
      refusal(500, { code: null, message: "Fake reason." }),
    );
    await press(wrapper, "auto-stack-revert");
    expect(ids()).toEqual([unanswered]);
    expect(notices.error).toHaveBeenLastCalledWith(
      "Could not start the undo: Fake reason.",
    );

    // So the press after that is a new request.
    revertAutoStackRun.mockReset();
    revertAutoStackRun.mockRejectedValueOnce(
      refusal(500, { code: null, message: "Fake reason." }),
    );
    await press(wrapper, "auto-stack-revert");
    expect(ids()).toHaveLength(1);
    expect(ids()[0]).not.toBe(unanswered);
    wrapper.unmount();
  });

  it("puts an ended run's banner away and gives the queue the keyboard back", async () => {
    const { wrapper, store } = await mountWithRun(
      runFixture({ status: "complete", active: false, can_revert: true }),
    );
    await press(wrapper, "auto-stack-dismiss");
    expect(store.autoStackRunDismissed).toBe(true);
    expect(banner(wrapper).exists()).toBe(false);
    expect(queueHasFocus(wrapper)).toBe(true);
    wrapper.unmount();
  });

  it.each([
    [
      "stop",
      "auto-stack-stop",
      runFixture(),
      refusal(403, "Forbidden"),
      "You don't have permission to stack duplicates here.",
    ],
    [
      "stop",
      "auto-stack-stop",
      runFixture(),
      refusal(500, { code: null, message: "fake-reason" }),
      "Could not stop the run: fake-reason.",
    ],
    [
      "stop",
      "auto-stack-stop",
      runFixture(),
      refusal(500, { code: null, message: "Fake reason?" }),
      "Could not stop the run: Fake reason?",
    ],
    [
      "stop",
      "auto-stack-stop",
      runFixture(),
      refusal(500, null),
      "Could not stop the run on the server.",
    ],
    [
      "undo",
      "auto-stack-revert",
      runFixture({ status: "complete", active: false, can_revert: true }),
      refusal(409, {
        code: "auto_stack_run_active",
        message: "Run 9 is already active.",
        run: null,
      }),
      "The undo can start once the Auto-stack run above the queue has ended.",
    ],
    [
      "undo",
      "auto-stack-revert",
      runFixture({ status: "complete", active: false, can_revert: true }),
      refusal(409, {
        code: "nothing_to_revert",
        message: "Run 7 has nothing left to revert.",
        run: null,
      }),
      "Nothing is left to undo from that Auto-stack run.",
    ],
  ])(
    "words a refused %s for the person, not the server (%#)",
    async (_action, testId, run, err, sentence) => {
      cancelAutoStackRun.mockRejectedValueOnce(err);
      revertAutoStackRun.mockRejectedValueOnce(err);
      const { wrapper } = await mountWithRun(run);
      const notices = useNoticeStore();

      await press(wrapper, testId);

      expect(notices.error).toHaveBeenCalledWith(sentence);
      expect(announced(wrapper)).toBe(sentence);
      wrapper.unmount();
    },
  );

  it("words a closed dialog's refused start by the run it names", async () => {
    previewAutoStackExact.mockResolvedValue({
      dry_run: true,
      groups: 3,
      pictures: 6,
    });
    const pending = deferred();
    startAutoStackRun.mockReturnValueOnce(pending.promise);
    const { wrapper } = await mountQueue([group("g1")], {
      byTier: { exact: 3 },
    });
    const notices = useNoticeStore();
    await autoStackTrigger(wrapper).trigger("click");
    await flushPromises();
    autoStackDialog(wrapper).vm.$emit("confirm");
    await flushPromises();
    autoStackDialog(wrapper).vm.$emit("close");
    await flushPromises();

    pending.reject(
      refusal(409, {
        code: "auto_stack_run_active",
        message: "Run 3 is already active.",
        run: null,
      }),
    );
    await flushPromises();
    expect(notices.error).toHaveBeenCalledWith(
      "Another Auto-stack run is already going, and its progress shows above the queue.",
    );
    // A refusal is an answer: nothing is looked for again.
    expect(getLatestAutoStackRun).toHaveBeenCalledTimes(1);
    wrapper.unmount();
  });

  // The banner is not a live region, since its figures move on every poll, so
  // the end of the run it follows is said in the queue's own.
  it.each([
    [
      runFixture(),
      { status: "complete", applied: 120 },
      "Auto-stack finished. 120 groups stacked.",
    ],
    [
      runFixture(),
      { status: "complete", applied: 123456 },
      "Auto-stack finished. 123,456 groups stacked.",
    ],
    [
      runFixture(),
      { status: "cancelled", applied: 1 },
      "Auto-stack stopped. 1 group stacked.",
    ],
    [
      runFixture(),
      { status: "interrupted" },
      "Auto-stack was interrupted when the server restarted. 40 groups stacked.",
    ],
    [
      runFixture(),
      { status: "failed" },
      "Auto-stack stopped on an error. 40 groups stacked.",
    ],
    [
      revertRun(),
      { status: "complete" },
      "Auto-stack undone. 40 groups are back in the queue.",
    ],
    [
      revertRun(),
      { status: "cancelled", applied: 1 },
      "The undo stopped. 1 group is back in the queue.",
    ],
    [
      revertRun(),
      { status: "blocked" },
      "The undo stopped at a step that later work was done on top of. 40 groups are back in the queue.",
    ],
    [
      revertRun(),
      { status: "interrupted" },
      "The undo was interrupted when the server restarted. 40 groups are back in the queue.",
    ],
    [
      revertRun(),
      { status: "failed" },
      "The undo stopped on an error. 40 groups are back in the queue.",
    ],
  ])("says how the followed run ended (%#)", async (run, ended, said) => {
    getAutoStackRun.mockResolvedValueOnce({
      run: { ...run, ...ended, active: false },
      retryAfterMs: null,
    });
    const { wrapper } = await mountWithRun(run);
    expect(announced(wrapper)).toBe("");

    await vi.advanceTimersByTimeAsync(2000);
    await flushPromises();

    expect(getAutoStackRun).toHaveBeenCalledWith(run.id);
    expect(announced(wrapper)).toBe(said);
    wrapper.unmount();
  });

  it("says nothing while the run is still going", async () => {
    getAutoStackRun.mockResolvedValueOnce({
      run: runFixture({ processed: 80, applied: 80 }),
      retryAfterMs: 2000,
    });
    const { wrapper } = await mountWithRun(runFixture());
    await vi.advanceTimersByTimeAsync(2000);
    await flushPromises();
    expect(getAutoStackRun).toHaveBeenCalledTimes(1);
    expect(announced(wrapper)).toBe("");
    wrapper.unmount();
  });

  it("says nothing for a run found already ended", async () => {
    const { wrapper } = await mountWithRun(
      runFixture({ status: "complete", active: false, applied: 120 }),
    );
    expect(banner(wrapper).exists()).toBe(true);
    expect(announced(wrapper)).toBe("");
    wrapper.unmount();
  });
});

describe("DuplicateQueue: a picture scrapheaped elsewhere", () => {
  /** A `pictures_changed` frame, as `useUpdatesSocket` hands it to the store. */
  const removed = (picture_ids) => ({
    type: "pictures_changed",
    change_kind: "removed",
    picture_ids,
    source: "ui",
  });

  const rowIds = (wrapper) =>
    wrapper
      .findAll("[data-testid^='dedup-group-']")
      .map((el) => el.attributes("data-testid"));

  // The reported bug, at the surface it was reported on: the row for a group
  // that is now one picture stayed on screen, drawing an empty slot beside it.
  it("takes a group thinned below two off the screen without a reload", async () => {
    const { wrapper, store } = await mountQueue([
      group("g1"),
      group("g2"),
      group("g3"),
    ]);
    expect(rowIds(wrapper)).toEqual([
      "dedup-group-g1",
      "dedup-group-g2",
      "dedup-group-g3",
    ]);
    listGroups.mockClear();

    store.applyPictureEvent(
      removed([store.groups[1].candidates[0].picture_id]),
    );
    await wrapper.vm.$nextTick();

    expect(rowIds(wrapper)).toEqual(["dedup-group-g1", "dedup-group-g3"]);
    // Surgical, not a rebuild: the window and the user's place in it survive.
    expect(listGroups).not.toHaveBeenCalled();
    wrapper.unmount();
  });

  // Over-filtering is its own regression, and so is a tile that outlives its
  // picture: the row stays, one tile lighter.
  it("keeps a group that still has two, minus the deleted tile", async () => {
    const { wrapper, store } = await mountQueue([group("g1", 3)]);
    const gone = store.groups[0].candidates[2].picture_id;
    expect(wrapper.findAll(".gunit")).toHaveLength(3);

    store.applyPictureEvent(removed([gone]));
    await wrapper.vm.$nextTick();

    expect(rowIds(wrapper)).toEqual(["dedup-group-g1"]);
    expect(wrapper.findAll(".gunit")).toHaveLength(2);
    expect(wrapper.html()).not.toContain(`/pictures/thumbnails/${gone}.webp`);
    wrapper.unmount();
  });
});

// --- The row's expansion band (D4) -------------------------------------------
//
// A deck stands for a whole existing stack and the row shows one picture of it.
// The band is where the rest can be looked at, and the invariant that makes it
// safe is the queue's, not the row's: ONE band, on the FOCUSED row. Both scroll
// spacers are sized from a single uniform row pitch, so a second
// variable-height row breaks the arithmetic the whole track is built from.

/** A group of one deck (stack 12, four deep) and one loose picture. */
function deckGroup(signature = "d1", stackId = 12) {
  return {
    signature,
    tier: "near",
    confidence: 0.91,
    member_count: 2,
    cover_picture_id: null,
    why: [],
    candidates: [
      { picture_id: stackId * 10 + 3, stack_id: stackId },
      { picture_id: stackId * 10 + 7 },
    ],
    stacks: {
      [stackId]: {
        stack_id: stackId,
        member_count: 4,
        leader_picture_id: stackId * 10 + 1,
        leader_thumbnail_version: "vlead",
        matched_picture_ids: [stackId * 10 + 3],
        stackable: true,
        blocked_by_sets: [],
      },
    },
  };
}

/** The whole member list, as `GET /dedup/stacks/{id}/members` serves it. */
function memberPage(stackId) {
  return {
    stack_id: stackId,
    member_count: 4,
    members: Array.from({ length: 4 }, (_, i) => ({
      picture_id: stackId * 10 + i + 1,
      thumbnail_version: `v${i}`,
      position: i,
    })),
    next_offset: null,
  };
}

/** The badges of every mounted row, in render order. */
function badges(wrapper) {
  return wrapper.findAll('[data-testid="stack-badge"]');
}

function bands(wrapper) {
  return wrapper.findAll('[data-testid="dedup-row-expansion"]');
}

describe("DuplicateQueue: the expansion band", () => {
  beforeEach(() => {
    listStackMembers.mockImplementation((stackId) =>
      Promise.resolve(memberPage(stackId)),
    );
  });

  // Lazy by contract: the payload sizes the stack and names its leader, and
  // the members are a separate read the user asks for. Nothing is fetched
  // until a badge is pressed.
  it("reads a deck's members only when its badge is pressed", async () => {
    const { wrapper } = await mountQueue([deckGroup()]);
    expect(listStackMembers).not.toHaveBeenCalled();
    expect(bands(wrapper)).toHaveLength(0);

    await badges(wrapper)[0].trigger("click");
    expect(listStackMembers).toHaveBeenCalledWith(12, { limit: 200 });
    await flushPromises();

    const band = bands(wrapper)[0];
    expect(band.exists()).toBe(true);
    // The whole stack, not the one member this group named.
    expect(band.findAll('[data-testid="stack-member"]')).toHaveLength(4);
    wrapper.unmount();
  });

  // Inline in the strip would nest a second horizontal scroller on the same
  // axis, and would break the unit reading the row exists to create.
  it("opens below the row's columns, inside the row", async () => {
    const { wrapper } = await mountQueue([deckGroup()]);
    await badges(wrapper)[0].trigger("click");
    await flushPromises();

    const row = wrapper.find(".grow").element;
    const band = bands(wrapper)[0].element;
    const strip = wrapper.find(".gstrip").element;
    expect(row.contains(band)).toBe(true);
    expect(strip.contains(band)).toBe(false);
    expect(
      strip.compareDocumentPosition(band) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    wrapper.unmount();
  });

  // The hard constraint. A second open band is a second variable-height row,
  // and the spacers are sized from one uniform pitch.
  it("keeps at most one band in the whole queue", async () => {
    const { wrapper } = await mountQueue([
      deckGroup("d1", 12),
      deckGroup("d2", 20),
    ]);
    await badges(wrapper)[0].trigger("click");
    await flushPromises();
    expect(bands(wrapper)).toHaveLength(1);
    expect(badges(wrapper)[0].attributes("aria-expanded")).toBe("true");

    await badges(wrapper)[1].trigger("click");
    await flushPromises();
    expect(bands(wrapper)).toHaveLength(1);
    expect(badges(wrapper)[0].attributes("aria-expanded")).toBe("false");
    expect(badges(wrapper)[1].attributes("aria-expanded")).toBe("true");
    wrapper.unmount();
  });

  // Pressing the same badge again closes it, without a second read.
  it("closes on a second press", async () => {
    const { wrapper } = await mountQueue([deckGroup()]);
    await badges(wrapper)[0].trigger("click");
    await flushPromises();
    await badges(wrapper)[0].trigger("click");
    await flushPromises();
    expect(bands(wrapper)).toHaveLength(0);
    expect(listStackMembers).toHaveBeenCalledTimes(1);
    wrapper.unmount();
  });

  // The band lives on the focused row and nowhere else, so the keyboard cursor
  // takes it with it. This is what keeps the pitch sampled from collapsed rows.
  it("collapses when the focus moves off its row", async () => {
    const { wrapper, store } = await mountQueue([
      deckGroup("d1", 12),
      deckGroup("d2", 20),
    ]);
    await badges(wrapper)[0].trigger("click");
    await flushPromises();
    expect(bands(wrapper)).toHaveLength(1);

    await wrapper.find(".dq").trigger("keydown", { key: "ArrowDown" });
    await flushPromises();
    expect(store.focusIndex).toBe(1);
    expect(bands(wrapper)).toHaveLength(0);
    wrapper.unmount();
  });

  // A verdict advances the cursor onto a different group under the same index,
  // so the collapse cannot be keyed on the index alone.
  it("collapses when a verdict advances onto another group", async () => {
    const { wrapper, store } = await mountQueue([
      deckGroup("d1", 12),
      deckGroup("d2", 20),
    ]);
    stackGroup.mockResolvedValue({ signature: "d1", picture_ids: [121, 127] });
    await badges(wrapper)[0].trigger("click");
    await flushPromises();
    expect(bands(wrapper)).toHaveLength(1);

    await store.stack(store.groups[0]);
    await flushPromises();
    expect(store.focusIndex).toBe(0);
    expect(store.focusedGroup.signature).toBe("d2");
    expect(bands(wrapper)).toHaveLength(0);
    wrapper.unmount();
  });

  // Disclosure, not a mode: the verdict a user was about to give is the same
  // verdict a moment after they opened a deck to check it.
  it("leaves the verdict live, and Enter does what it would have done", async () => {
    const { wrapper } = await mountQueue([deckGroup(), group("g2")]);
    stackGroup.mockResolvedValue({ signature: "d1", picture_ids: [121, 127] });
    await badges(wrapper)[0].trigger("click");
    await flushPromises();

    const stackButton = wrapper.find('[data-testid="dedup-stack"]');
    expect(stackButton.attributes("disabled")).toBeUndefined();
    // The label still names the verdict's outcome over the row's units.
    expect(stackButton.text()).toContain("Add 1 to stack of 4");

    await wrapper.find(".dq").trigger("keydown", { key: "Enter" });
    await flushPromises();
    expect(stackGroup).toHaveBeenCalledWith(
      "d1",
      expect.objectContaining({ coverPictureId: 121 }),
    );
    wrapper.unmount();
  });

  // The digits address the strip's tiles whether the band is open or not: one
  // key meaning two things on one screen is how a key stops being trusted.
  it("keeps the digits on the units while a band is open", async () => {
    const { wrapper, store } = await mountQueue([deckGroup()]);
    await badges(wrapper)[0].trigger("click");
    await flushPromises();

    await wrapper.find(".dq").trigger("keydown", { key: "2" });
    expect(store.coverIdFor(store.groups[0])).toBe(127);
    await wrapper.find(".dq").trigger("keydown", { key: "1" });
    // The deck's LEADER, not one of the members the band just put on screen.
    expect(store.coverIdFor(store.groups[0])).toBe(121);
    wrapper.unmount();
  });

  // E is the keyboard's way in, and it is a read gesture: no verdict, no
  // change to the cover, nothing but the band.
  it("opens and closes the focused group's stack with E", async () => {
    const { wrapper } = await mountQueue([deckGroup()]);
    await wrapper.find(".dq").trigger("keydown", { key: "e" });
    await flushPromises();
    expect(bands(wrapper)).toHaveLength(1);
    expect(wrapper.find('[data-testid="dedup-announcement"]').text()).toContain(
      "Showing the 4 pictures in this stack",
    );

    await wrapper.find(".dq").trigger("keydown", { key: "e" });
    await flushPromises();
    expect(bands(wrapper)).toHaveLength(0);
    wrapper.unmount();
  });

  // Not a dead key on a group of loose pictures: there is nothing folded away,
  // and saying so is what stops the user pressing it again.
  it("says so when the focused group holds no stack", async () => {
    const { wrapper } = await mountQueue([group("g1")]);
    await wrapper.find(".dq").trigger("keydown", { key: "e" });
    await flushPromises();
    expect(bands(wrapper)).toHaveLength(0);
    expect(wrapper.find('[data-testid="dedup-announcement"]').text()).toContain(
      "no stack",
    );
    wrapper.unmount();
  });

  it("shows a reading state while the members are in flight", async () => {
    let settle;
    listStackMembers.mockImplementation(
      () => new Promise((resolve) => (settle = resolve)),
    );
    const { wrapper } = await mountQueue([deckGroup()]);
    await badges(wrapper)[0].trigger("click");
    await wrapper.vm.$nextTick();

    const band = bands(wrapper)[0];
    expect(band.find('[role="status"]').text()).toContain(
      "Reading the pictures in this stack",
    );
    expect(band.findAll('[data-testid="stack-member"]')).toHaveLength(0);

    settle(memberPage(12));
    await flushPromises();
    expect(
      bands(wrapper)[0].findAll('[data-testid="stack-member"]'),
    ).toHaveLength(4);
    wrapper.unmount();
  });

  // A failure to DISCLOSE is not a failure to decide, and the retry sits with
  // the sentence that says so.
  it("reports a failed read, keeps the verdict, and retries", async () => {
    listStackMembers.mockRejectedValueOnce(new Error("boom"));
    const { wrapper } = await mountQueue([deckGroup()]);
    await badges(wrapper)[0].trigger("click");
    await flushPromises();

    const band = bands(wrapper)[0];
    expect(band.find('[role="alert"]').text()).toContain(
      "The verdict buttons still work",
    );
    expect(
      wrapper.find('[data-testid="dedup-stack"]').attributes("disabled"),
    ).toBeUndefined();

    listStackMembers.mockImplementation((stackId) =>
      Promise.resolve(memberPage(stackId)),
    );
    await band.find(".gexp-state--error button").trigger("click");
    await flushPromises();
    expect(
      bands(wrapper)[0].findAll('[data-testid="stack-member"]'),
    ).toHaveLength(4);
    wrapper.unmount();
  });

  // A stack that answered with nothing is a failed read, not an empty stack:
  // the route 404s rather than serving an empty membership.
  it("treats an empty member list as a failed read", async () => {
    listStackMembers.mockResolvedValue({ stack_id: 12, members: [] });
    const { wrapper } = await mountQueue([deckGroup()]);
    await badges(wrapper)[0].trigger("click");
    await flushPromises();
    expect(bands(wrapper)[0].find('[role="alert"]').exists()).toBe(true);
    wrapper.unmount();
  });

  // Reading is not a verdict, so the band is offered in a share session too,
  // and it carries nothing that could write there or anywhere else.
  it("stays available, and read-only, in a read-only session", async () => {
    readOnlyRef.value = true;
    const { wrapper } = await mountQueue([deckGroup()]);
    await badges(wrapper)[0].trigger("click");
    await flushPromises();

    const band = bands(wrapper)[0];
    expect(band.exists()).toBe(true);
    expect(band.find('[data-testid="stack-unstack"]').exists()).toBe(false);
    for (const member of band.findAll('[data-testid="stack-member"]')) {
      expect(member.attributes("disabled")).toBeDefined();
    }
    wrapper.unmount();
  });
});

// ── The Mixed stacks page (design D5) ───────────────────────────────────────
//
// A third page of the SAME destination, which is the whole reason it is not a
// route and not a sidebar row: the queue stays standing behind it with its
// focus intact, so the two-way shortcut can offer a return that restores it.

/** A queue group that folds in one existing stack, so a deck is drawn. */
function groupWithStack(signature, stackId, over = {}) {
  const g = group(signature, 2);
  g.candidates[0].stack_id = stackId;
  g.stacks = {
    [String(stackId)]: {
      stack_id: stackId,
      member_count: 5,
      leader_picture_id: 7,
      leader_thumbnail_version: null,
      matched_picture_ids: [g.candidates[0].picture_id],
      stackable: true,
      blocked_by_sets: [],
      ...over,
    },
  };
  return g;
}

/** Mount the queue and flip to the Mixed stacks page. */
async function openMixedPage(wrapper, store) {
  await store.showMixedStacks();
  await wrapper.vm.$nextTick();
  await flushPromises();
}

describe("DuplicateQueue: the Mixed stacks page", () => {
  // The list is deliberately lazy because its first read can score the whole
  // library. Its entry cannot depend on that unread list, though: the first
  // press is what requests it, then the returned rows and count appear here.
  it("offers the third page cold and loads it on the first press", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([mixedStack({ stack_id: 1 }), mixedStack({ stack_id: 2 })]),
    );
    const { wrapper } = await mountQueue([group("g1")]);

    const toggle = wrapper.find('[data-testid="mixed-toggle"]');
    expect(toggle.exists()).toBe(true);
    expect(toggle.text()).toContain("Mixed stacks");
    expect(listMixedStacks).not.toHaveBeenCalled();

    await toggle.trigger("click");
    await flushPromises();

    expect(listMixedStacks).toHaveBeenCalledTimes(1);
    expect(wrapper.find('[data-testid="mixed-stacks"]').exists()).toBe(true);
    expect(wrapper.findAll('[data-testid^="mixed-stack-"]')).toHaveLength(2);
    expect(wrapper.find(".qtitle").text()).toBe("2 mixed stacks");
    expect(wrapper.find('[data-testid="mixed-toggle"]').text()).toContain(
      "Back to review",
    );
  });

  // Ranked worst first by the server, and printed in that order with no
  // numerals: the order IS the ranking.
  it("lists the rows in the server's ranked order, without rank numerals", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([
        mixedStack({ stack_id: 3, stranded_picture_ids: [1, 2] }),
        mixedStack({ stack_id: 1, stranded_picture_ids: [4] }),
        mixedStack({
          stack_id: 2,
          stranded_picture_ids: [],
          suggested_action: "unstack",
        }),
      ]),
    );
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);
    const rows = wrapper.findAll('[data-testid^="mixed-stack-"]');
    expect(rows.map((r) => r.attributes("data-testid"))).toEqual([
      "mixed-stack-3",
      "mixed-stack-1",
      "mixed-stack-2",
    ]);
    expect(wrapper.find(".mlist").text()).not.toMatch(/\b1\.\s/);
  });

  // The list is a function of the threshold, not of a constant: 26 at the
  // default 0.90 and 9 at the 0.65 floor on the owner's library.
  it("rebinds the list when the threshold slider moves", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage(
        Array.from({ length: 3 }, (_, i) => mixedStack({ stack_id: i + 1 })),
      ),
    );
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);
    expect(wrapper.findAll('[data-testid^="mixed-stack-"]')).toHaveLength(3);
    expect(wrapper.text()).toContain("90% similar");

    listMixedStacks.mockResolvedValue(
      mixedPage([mixedStack({ stack_id: 1 })], { threshold: 0.65, total: 1 }),
    );
    await store.setThreshold(0.65);
    await flushPromises();
    expect(listMixedStacks).toHaveBeenLastCalledWith(
      expect.objectContaining({ threshold: 0.65 }),
    );
    expect(wrapper.findAll('[data-testid^="mixed-stack-"]')).toHaveLength(1);
    expect(wrapper.text()).toContain("65% similar");
  });

  // Mirrors the shipped "No decided groups" construction, and carries its own
  // way back for the same reason: the header toggle is not where the user is
  // looking when the list runs out.
  it("mirrors the Decided page's empty state, with its own way back", async () => {
    listMixedStacks.mockResolvedValue(mixedPage([]));
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);
    const empty = wrapper.find(".qdone");
    expect(empty.text()).toContain("No mixed stacks");
    expect(empty.text()).toContain("Back to review");
    await empty.find("button").trigger("click");
    expect(store.showingMixed).toBe(false);
  });

  // A failed read is not an empty library, and must never be reported as one.
  it("reports a failed read rather than claiming there are none", async () => {
    listMixedStacks.mockRejectedValue(new Error("boom"));
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);
    expect(wrapper.text()).toContain("Could not check the stacks");
    expect(wrapper.text()).not.toContain("No mixed stacks");
  });
});

describe("DuplicateQueue: the Mixed stacks actions", () => {
  it("splits with the ids the row showed, and the row goes", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([mixedStack({ stack_id: 42, stranded_picture_ids: [11] })]),
    );
    splitMixedStack.mockResolvedValue({
      stack_id: 42,
      split_picture_ids: [11],
      remaining_picture_ids: [7, 8, 9, 10],
      stack_dissolved: false,
      batch_id: "srv-1",
    });
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);
    await wrapper.find(".gact button").trigger("click");
    await flushPromises();
    expect(splitMixedStack).toHaveBeenCalledWith(
      42,
      expect.objectContaining({ pictureIds: [11] }),
    );
    expect(unstackMixedStack).not.toHaveBeenCalled();
    expect(wrapper.findAll('[data-testid^="mixed-stack-"]')).toHaveLength(0);
    expect(wrapper.find('[data-testid="dedup-announcement"]').text()).toContain(
      "Ctrl+Z",
    );
  });

  // The primary names its outcome, and the name is a function of the MARKS:
  // `Unstack all N` the moment they would leave fewer than two members. Both
  // outcomes travel as one split call, so there is never a case where the label
  // and the request disagree about what is being done.
  it("unstacks, through the same split call, when nothing would be left", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([
        mixedStack({
          stack_id: 7,
          member_count: 2,
          member_ids: [7, 8],
          component_sizes: [1, 1],
          components: [[7], [8]],
          largest_component_size: 1,
          stranded_picture_ids: [7, 8],
          suggested_action: "unstack",
        }),
      ]),
    );
    splitMixedStack.mockResolvedValue({
      stack_id: 7,
      split_picture_ids: [7, 8],
      remaining_picture_ids: [],
      stack_dissolved: true,
      batch_id: "srv-2",
    });
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);
    expect(wrapper.find(".gact").text()).toContain("Unstack all 2");
    await wrapper.find(".gact button").trigger("click");
    await flushPromises();
    expect(splitMixedStack).toHaveBeenCalledWith(
      7,
      expect.objectContaining({ pictureIds: [7, 8] }),
    );
    expect(unstackMixedStack).not.toHaveBeenCalled();
    expect(wrapper.findAll('[data-testid^="mixed-stack-"]')).toHaveLength(0);
    // Reported from the response, never from the prediction.
    expect(wrapper.find('[data-testid="dedup-announcement"]').text()).toContain(
      "removed the stack",
    );
  });

  // A failed action leaves the row exactly where it was, and says so: the
  // page's promise is that nothing changes until it says it did.
  it("keeps the row and reports the failure when an action fails", async () => {
    listMixedStacks.mockResolvedValue(mixedPage([mixedStack()]));
    splitMixedStack.mockRejectedValue(new Error("nope"));
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);
    await wrapper.find(".gact button").trigger("click");
    await flushPromises();
    expect(wrapper.findAll('[data-testid^="mixed-stack-"]')).toHaveLength(1);
    expect(errorSpy).toHaveBeenCalled();
    expect(errorSpy.mock.calls.at(-1)[0]).toContain(
      "Check the connection or server log",
    );
    expect(errorSpy.mock.calls.at(-1)[0]).not.toContain("Nothing was changed");
  });

  // The 423 the whole lock contract exists for. Nothing was written, so the row
  // stays; it is marked with what the server named, so the button stops
  // offering an outcome that cannot happen; and the sentence names the SET,
  // because that is the only thing the user can go and act on.
  it("keeps a 423-refused row, names the set and locks the action", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([mixedStack({ stack_id: 42 })]),
    );
    splitMixedStack.mockRejectedValue({
      response: {
        status: 423,
        data: {
          detail: {
            code: "pictures_locked",
            action: "split a stack",
            sets: [{ id: 3, name: "Frozen" }],
            picture_ids: [11],
          },
        },
      },
    });
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);
    await wrapper.find(".gact button").trigger("click");
    await flushPromises();

    expect(wrapper.findAll('[data-testid^="mixed-stack-"]')).toHaveLength(1);
    expect(errorSpy.mock.calls.at(-1)[0]).toContain("locked set 'Frozen'");
    expect(wrapper.find('[data-testid="dedup-announcement"]').text()).toContain(
      "locked set 'Frozen'",
    );
    // The row now carries the refusal's own truth, so the reason is on screen
    // and the second press never reaches the server.
    const note = wrapper.find(".mqlock");
    expect(note.text()).toBe("Frozen by locked set 'Frozen'");
    expect(note.classes()).toContain("mqlock--flash");
    const primary = wrapper.find(".gact button");
    expect(primary.attributes("aria-disabled")).toBe("true");
    expect(primary.attributes("aria-describedby")).toBe(note.attributes("id"));

    splitMixedStack.mockClear();
    await primary.trigger("click");
    await flushPromises();
    expect(splitMixedStack).not.toHaveBeenCalled();
    expect(errorSpy.mock.calls.at(-1)[0]).toContain("locked set 'Frozen'");
  });

  // A row the LIST already knew was frozen never issues the doomed call at all,
  // and the press is answered rather than reading as a dead control.
  it("answers a press on a row the list already served as frozen", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([
        mixedStack({
          stack_id: 42,
          stackable: false,
          blocked_by_sets: [{ id: 3, name: "Frozen" }],
        }),
      ]),
    );
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);
    expect(wrapper.find(".mqlock").text()).toBe(
      "Frozen by locked set 'Frozen'",
    );
    await wrapper.find(".gact button").trigger("click");
    await flushPromises();
    expect(splitMixedStack).not.toHaveBeenCalled();
    expect(unstackMixedStack).not.toHaveBeenCalled();
    expect(wrapper.findAll('[data-testid^="mixed-stack-"]')).toHaveLength(1);
    expect(errorSpy.mock.calls.at(-1)[0]).toContain("locked set 'Frozen'");
  });

  // Over-blocking is its own regression: a live row keeps its action, and the
  // action still reaches the server.
  it("leaves a stackable row's action live", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([
        mixedStack({ stack_id: 42, stackable: true, blocked_by_sets: [] }),
      ]),
    );
    splitMixedStack.mockResolvedValue({
      stack_id: 42,
      split_picture_ids: [11],
      remaining_picture_ids: [7, 8, 9, 10],
      stack_dissolved: false,
      batch_id: "srv-1",
    });
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);
    expect(wrapper.find(".mqlock").exists()).toBe(false);
    const primary = wrapper.find(".gact button");
    expect(primary.attributes("aria-disabled")).toBeUndefined();
    await primary.trigger("click");
    await flushPromises();
    expect(splitMixedStack).toHaveBeenCalled();
    expect(wrapper.findAll('[data-testid^="mixed-stack-"]')).toHaveLength(0);
  });

  // Keep is what makes the list drainable, and it is the one action here that
  // is NOT undoable: it changes no picture, so DELETE is the way back and the
  // page has to offer it where the row used to be.
  it("Keep removes the row and offers the DELETE that brings it back", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([mixedStack({ stack_id: 42 })]),
    );
    keepMixedStack.mockResolvedValue({
      stack_id: 42,
      dismissed: true,
      created: true,
    });
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);
    const buttons = wrapper.findAll(".gact button");
    await buttons[1].trigger("click");
    await flushPromises();
    expect(keepMixedStack).toHaveBeenCalledWith(42);
    expect(wrapper.findAll('[data-testid^="mixed-stack-"]')).toHaveLength(0);

    // The offer, and the way back it carries.
    const notices = useNoticeStore();
    const offer = notices.info.mock.calls.at(-1);
    expect(offer[0]).toContain("Kept this stack");
    expect(offer[1].action.label).toBe("Undo keep");

    clearMixedStackKeep.mockResolvedValue({ stack_id: 42, removed: 1 });
    listMixedStacks.mockResolvedValue(
      mixedPage([mixedStack({ stack_id: 42 })]),
    );
    await offer[1].action.handler();
    await flushPromises();
    expect(clearMixedStackKeep).toHaveBeenCalledWith(42);
    expect(wrapper.findAll('[data-testid^="mixed-stack-"]')).toHaveLength(1);
  });
});

// ── The Mixed stacks page as a QUEUE ───────────────────────────────────────
//
// The owner rejected the first cut as under-equipped: no zoom, no Compare, no
// individual selection, no threshold, no multi-select, no keyboard. It is now
// the third queue, reusing the review queue's machinery rather than being a
// bespoke list, so what is pinned here is the machinery arriving intact.

describe("DuplicateQueue: the Mixed stacks queue's keyboard", () => {
  // A queue-trained user reads S as Stack and would mean Split; the two acts
  // are opposites. The key is claimed either way (it must never run the primary
  // by accident, and it must never fall through to the app shell) and answered
  // out loud rather than doing nothing, which would read as a broken key.
  it("answers S instead of running the primary", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([mixedStack({ stack_id: 42, stranded_picture_ids: [11] })]),
    );
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);
    splitMixedStack.mockClear();

    document.dispatchEvent(new KeyboardEvent("keydown", { key: "s" }));
    await flushPromises();

    expect(splitMixedStack).not.toHaveBeenCalled();
    expect(unstackMixedStack).not.toHaveBeenCalled();
    expect(wrapper.findAll('[data-testid^="mixed-stack-"]')).toHaveLength(1);
    const said = wrapper.find('[data-testid="dedup-announcement"]').text();
    expect(said).toContain("S means Stack in the review queue");
    expect(said).toContain("Split off 1");
    expect(said).toContain("press Enter");
  });

  // The digits address the same tiles on both queues and mean the same thing,
  // "point at this one". On this page pointing is all they do; the mark is X, a
  // second and deliberate press.
  it("moves the member cursor with a digit and marks it with X", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([mixedStack({ stack_id: 42, stranded_picture_ids: [11] })]),
    );
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);
    expect(wrapper.find(".gact button").text()).toContain("Split off 1");

    document.dispatchEvent(new KeyboardEvent("keydown", { key: "3" }));
    await flushPromises();
    const units = wrapper.findAll(".gunit");
    expect(units[2].classes()).toContain("gunit--cursor");
    // Pointing changes nothing about the outcome.
    expect(wrapper.find(".gact button").text()).toContain("Split off 1");

    document.dispatchEvent(new KeyboardEvent("keydown", { key: "x" }));
    await flushPromises();
    expect(wrapper.find(".gact button").text()).toContain("Split off 2");
    expect(wrapper.findAll(".gunit")[2].find(".gthumb").classes()).toContain(
      "gthumb--marked",
    );

    // Symmetric: the same key takes the mark straight back off.
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "x" }));
    await flushPromises();
    expect(wrapper.find(".gact button").text()).toContain("Split off 1");
  });

  // The engine's marks and the user's are one list, so X unmarks a member the
  // SERVER marked exactly as it unmarks one the user did.
  it("unmarks an engine mark with the same key", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([mixedStack({ stack_id: 42, stranded_picture_ids: [11] })]),
    );
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);
    // Member 11 is the fifth tile, and the engine opened with it marked.
    expect(wrapper.findAll(".gunit")[4].find(".gthumb").classes()).toContain(
      "gthumb--marked",
    );

    document.dispatchEvent(new KeyboardEvent("keydown", { key: "5" }));
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "x" }));
    await flushPromises();

    expect(
      wrapper.findAll(".gunit")[4].find(".gthumb").classes(),
    ).not.toContain("gthumb--marked");
    // Nothing marked means the only outcome left is to free the whole stack.
    expect(wrapper.find(".gact button").text()).toContain("Unstack all 5");
  });
});

describe("DuplicateQueue: the Mixed stacks queue's selection", () => {
  // Only Keep acts in bulk: the primary's outcome differs per row, so a bulk
  // primary could not name what it was about to do.
  it("says Keep is the only bulk action, and acts on the whole selection", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([
        mixedStack({ stack_id: 1 }),
        mixedStack({ stack_id: 2 }),
        mixedStack({ stack_id: 3 }),
      ]),
    );
    keepMixedStack.mockResolvedValue({ stack_id: 1, dismissed: true });
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);

    const rows = wrapper.findAll('[data-testid^="mixed-stack-"]');
    await rows[0].trigger("click", { ctrlKey: true });
    await rows[1].trigger("click", { ctrlKey: true });
    expect(wrapper.find(".qselchip").text()).toContain(
      "2 rows selected: Keep applies to all",
    );

    // The primary keeps naming ONE row's outcome while the selection holds.
    const buttons = rows[1].findAll(".gact button");
    expect(buttons[0].text()).toContain("Split off 1");
    expect(buttons[0].text()).not.toContain("2 stacks");
    expect(buttons[1].text()).toContain("Keep 2 stacks");

    keepMixedStack.mockClear();
    await buttons[1].trigger("click");
    await flushPromises();
    expect(keepMixedStack.mock.calls.map((c) => c[0]).sort()).toEqual([1, 2]);
    expect(wrapper.findAll('[data-testid^="mixed-stack-"]')).toHaveLength(1);
  });

  // Ctrl+A names what the selection can do, rather than leaving the user to
  // discover that the primary did not follow.
  it("names Keep when Ctrl+A takes the page", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([mixedStack({ stack_id: 1 }), mixedStack({ stack_id: 2 })]),
    );
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);
    document.dispatchEvent(
      new KeyboardEvent("keydown", { key: "a", ctrlKey: true }),
    );
    await flushPromises();
    const said = wrapper.find('[data-testid="dedup-announcement"]').text();
    expect(said).toContain("Selected all 2 stacks");
    expect(said).toContain("Keep applies to all of them");
  });
});

describe("DuplicateQueue: the Mixed stacks threshold header", () => {
  // The count is the sentence's SUBJECT, not a figure beside a caption: one
  // fact, so the two cannot drift apart. The band is sticky inside the list's
  // own scroller, because every row is a verdict relative to that number.
  it("states the count and the threshold as one sentence, and moves both", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage(
        Array.from({ length: 3 }, (_, i) => mixedStack({ stack_id: i + 1 })),
      ),
    );
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);

    const head = wrapper.find(".mixed-head");
    expect(head.exists()).toBe(true);
    expect(head.find(".mixed-lede").text()).toContain(
      "3 stacks don't hang together at 90% similar",
    );
    // The slider is the SHIPPED control, so its label, step and formatting
    // cannot differ from the tier popover's copy of the same number.
    expect(head.find(".dth-input").exists()).toBe(true);

    listMixedStacks.mockResolvedValue(
      mixedPage([mixedStack({ stack_id: 1 })], { threshold: 0.65, total: 1 }),
    );
    await head.find(".dth-input").trigger("change");
    await store.setThreshold(0.65);
    await flushPromises();

    expect(wrapper.find(".mixed-lede").text()).toContain(
      "1 stack doesn't hang together at 65% similar",
    );
    expect(wrapper.findAll('[data-testid^="mixed-stack-"]')).toHaveLength(1);
  });
});

describe("DuplicateQueue: Compare on the Mixed stacks queue", () => {
  // The zoom is the single largest thing this page gains by being a queue, and
  // a second dialog would be a second copy of it. What matters at this level is
  // that the dialog is handed the SAME marks the row is drawing, so a mark made
  // in one place is the mark the other shows.
  it("opens the shared dialog in mixed mode over the focused row's marks", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([mixedStack({ stack_id: 42, stranded_picture_ids: [11] })]),
    );
    const { wrapper, store } = await mountQueue([group("g1")]);
    await openMixedPage(wrapper, store);

    document.dispatchEvent(new KeyboardEvent("keydown", { key: "c" }));
    await flushPromises();

    const dialog = wrapper.findComponent({ name: "DedupCompareDialog" });
    expect(dialog.props("open")).toBe(true);
    expect(dialog.props("mode")).toBe("mixed");
    expect(dialog.props("mixedStack").stack_id).toBe(42);
    expect(dialog.props("markedIds")).toEqual([11]);
    expect(dialog.props("primaryLabel")).toBe("Split off 1");
    expect(dialog.props("primaryIcon")).toBe("call-split");

    // A mark made from Compare is the row's mark: one list, one gesture.
    dialog.vm.$emit("toggle-mark", 9);
    await flushPromises();
    expect(dialog.props("markedIds")).toEqual([11, 9]);
    expect(dialog.props("primaryLabel")).toBe("Split off 2");
    expect(wrapper.find(".gact button").text()).toContain("Split off 2");
  });
});

describe("DuplicateQueue: the warning chip on a deck in the queue", () => {
  it("marks only the deck whose stack is the strong case", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([mixedStack({ stack_id: 12, stranded_picture_ids: [99] })]),
    );
    const { wrapper, store } = await mountQueue([
      groupWithStack("g1", 12),
      groupWithStack("g2", 13),
    ]);
    await store.loadMixedStacks();
    await wrapper.vm.$nextTick();
    const badges = wrapper.findAll('[data-testid="stack-badge"]');
    expect(badges).toHaveLength(2);
    expect(badges[0].attributes("data-flagged")).toBe("true");
    expect(badges[1].attributes("data-flagged")).toBeUndefined();
  });

  // The soft cases never reach the flag set at all: the store keeps only the
  // stacks with a stranded member, because a mark on one tile in eight becomes
  // a warning field.
  it("leaves a soft case unmarked", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([
        mixedStack({
          stack_id: 12,
          stranded_picture_ids: [],
          suggested_action: "unstack",
        }),
      ]),
    );
    const { wrapper, store } = await mountQueue([groupWithStack("g1", 12)]);
    await store.loadMixedStacks();
    await wrapper.vm.$nextTick();
    expect(
      wrapper.find('[data-testid="stack-badge"]').attributes("data-flagged"),
    ).toBeUndefined();
  });

  // The rule the design is most explicit about. A mixed stack is one a user may
  // legitimately want to add to.
  it("never blocks the verdict on the row it sits in", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([mixedStack({ stack_id: 12, stranded_picture_ids: [99] })]),
    );
    stackGroup.mockResolvedValue({
      signature: "g1",
      verdict: "stacked",
      picture_ids: [1, 2],
      batch_id: "srv-9",
    });
    const { wrapper, store } = await mountQueue([groupWithStack("g1", 12)]);
    await store.loadMixedStacks();
    await wrapper.vm.$nextTick();
    const stackButton = wrapper.find('[data-testid="dedup-stack"]');
    expect(stackButton.attributes("disabled")).toBeUndefined();
    await stackButton.trigger("click");
    await flushPromises();
    expect(stackGroup).toHaveBeenCalled();
  });
});

describe("DuplicateQueue: the two-way shortcut", () => {
  // Queue to page: the queue's focus is deliberately untouched, which is what
  // makes the return a restore rather than a fresh arrival.
  it("goes from a flagged deck's open band to that stack's row", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([
        mixedStack({ stack_id: 11 }),
        mixedStack({ stack_id: 12, stranded_picture_ids: [99] }),
      ]),
    );
    listStackMembers.mockResolvedValue({
      stack_id: 12,
      member_count: 5,
      members: [{ picture_id: 7, is_leader: true }],
      next_offset: null,
    });
    const { wrapper, store } = await mountQueue([
      group("g0"),
      groupWithStack("g1", 12),
    ]);
    await store.loadMixedStacks();
    store.setFocus(1);
    await wrapper.vm.$nextTick();
    // Open the band: the badge is the disclosure, and the band is where the
    // shortcut lives, because the corner is already spoken for.
    await wrapper.find('[data-testid="stack-badge"]').trigger("click");
    await flushPromises();
    await wrapper.find(".gexp-flag button").trigger("click");
    await flushPromises();
    expect(store.showingMixed).toBe(true);
    expect(store.mixedFocusStackId).toBe("12");
    // The row that was jumped to says so, or the jump reads as a dead press on
    // a list of near-identical rows.
    expect(wrapper.find('[data-testid="mixed-stack-12"]').classes()).toContain(
      "grow--revealed",
    );
    // And the queue is exactly where it was left.
    expect(store.focusIndex).toBe(1);
  });

  // Page to queue: offered only when a LOADED group holds the stack, so the
  // control never promises a landing it cannot make.
  it("goes from a row back to the duplicate group the stack appears in", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([mixedStack({ stack_id: 12 }), mixedStack({ stack_id: 99 })]),
    );
    const { wrapper, store } = await mountQueue([
      group("g0"),
      groupWithStack("g1", 12),
    ]);
    await openMixedPage(wrapper, store);
    const held = wrapper.find('[data-testid="mixed-stack-12"]');
    const orphan = wrapper.find('[data-testid="mixed-stack-99"]');
    expect(held.text()).toContain("In the queue");
    // No loaded group holds stack 99, so the row does not offer the jump.
    expect(orphan.text()).not.toContain("In the queue");

    await held.findAll(".gact button").at(-1).trigger("click");
    await flushPromises();
    expect(store.showingMixed).toBe(false);
    expect(store.focusIndex).toBe(1);
    expect(wrapper.find('[data-testid="dedup-group-g1"]').exists()).toBe(true);
  });

  // A page of the same destination, not a route away: leaving it costs no
  // reload and no place in the queue.
  it("returns with one Escape, restoring the queue's focus", async () => {
    listMixedStacks.mockResolvedValue(mixedPage([mixedStack()]));
    const { wrapper, store } = await mountQueue([group("g0"), group("g1")]);
    store.setFocus(1);
    await openMixedPage(wrapper, store);
    listGroups.mockClear();
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await flushPromises();
    expect(store.showingMixed).toBe(false);
    expect(store.focusIndex).toBe(1);
    expect(listGroups).not.toHaveBeenCalled();
  });

  // While the page is up the REVIEW queue's rows are not on screen, so its
  // verdict keys must never reach them: Enter belongs to the page in front of
  // the user, which is now a queue of its own.
  // Asserted on this queue's own state rather than on the shared api mock:
  // earlier cases in this file attach their queues to the document and never
  // unmount them, so a document-level key reaches all of them.
  it("gives Enter to the Mixed queue, never to the group underneath", async () => {
    listMixedStacks.mockResolvedValue(
      mixedPage([mixedStack({ stack_id: 42, stranded_picture_ids: [11] })]),
    );
    splitMixedStack.mockResolvedValue({
      stack_id: 42,
      split_picture_ids: [11],
      remaining_picture_ids: [7, 8, 9, 10],
      stack_dissolved: false,
      batch_id: "srv-1",
    });
    const { wrapper, store } = await mountQueue([group("g0")]);
    await openMixedPage(wrapper, store);
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    await flushPromises();
    expect(splitMixedStack).toHaveBeenCalledWith(
      42,
      expect.objectContaining({ pictureIds: [11] }),
    );
    // The group behind the page is untouched, and the page is still up.
    expect(store.groups).toHaveLength(1);
    expect(store.showingMixed).toBe(true);
    expect(wrapper.findAll('[data-testid^="mixed-stack-"]')).toHaveLength(0);
  });
});

// ── The route from the queue-clear screen to the stacks ────────────────────
//
// The end-of-task surface is the one place this is offered: the toolbar would
// put it in front of someone mid-triage. The library fact arrives lazily with
// the optional Mixed stacks read, so a cold queue never starts an all-stack
// score just to decide whether to show a shortcut.
describe("DuplicateQueue: the route to the stacks", () => {
  /** Mount straight onto the queue-clear screen. */
  async function mountCleared({
    liveStackCount = 209,
    loadStackFacts = true,
  } = {}) {
    listMixedStacks.mockResolvedValue(
      mixedPage([], { live_stack_count: liveStackCount }),
    );
    const { wrapper, store } = await mountQueue([]);
    await flushPromises();
    expect(wrapper.text()).toContain("Queue clear");
    if (loadStackFacts) {
      await store.loadMixedStacks();
      await flushPromises();
    }
    return { wrapper, store };
  }

  /** The queue-clear screen's stacks button, or undefined when not offered. */
  function stacksButton(wrapper) {
    return wrapper
      .findAll(".qdone button")
      .find((b) => b.text().includes("Review your stacks"));
  }

  it("does not cold-load the optional all-stack facts on queue startup", async () => {
    const { wrapper } = await mountCleared({ loadStackFacts: false });
    expect(listMixedStacks).not.toHaveBeenCalled();
    expect(stacksButton(wrapper)).toBeUndefined();
  });

  // Once the optional page has supplied the library fact, the route is not
  // gated on this session's verdict tally: old stacks count too.
  it("is offered after the library's stack facts have been loaded", async () => {
    const { wrapper, store } = await mountCleared();
    expect(store.stackedCount).toBe(0);
    expect(stacksButton(wrapper)).toBeTruthy();
    expect(wrapper.text()).toContain("209 stacks hold");
  });

  it("is not offered when the library holds no multi-picture stack", async () => {
    const { wrapper } = await mountCleared({ liveStackCount: 0 });
    expect(stacksButton(wrapper)).toBeUndefined();
  });

  // A one-click path from a satisfying "Queue clear" screen into a confirm for
  // hundreds of deletions is how you get a bad afternoon. It lands in All
  // Pictures with the stacked filter applied and nothing else happening.
  it("goes to the place, not to the action", async () => {
    const { wrapper } = await mountCleared();
    await stacksButton(wrapper).trigger("click");

    expect(routerPush).toHaveBeenCalledTimes(1);
    expect(routerPush).toHaveBeenCalledWith({
      path: "/",
      query: { stack_state: "stacked" },
    });
    // Nothing is selected and nothing is armed: the push carries no picture or
    // stack ids at all, and no confirm was opened on the way out.
    const [to] = routerPush.mock.calls[0];
    expect(Object.keys(to.query)).toEqual(["stack_state"]);
    expect(JSON.stringify(to)).not.toMatch(/picture|stack_id|ids/);
    expect(wrapper.text()).not.toMatch(/Scrapheap/);
  });

  // A real route change, so it is reloadable and Back returns to the queue.
  // The URL mirror uses replace() for its own writes; this must not.
  it("pushes a real history entry rather than replacing the queue's", async () => {
    const { wrapper } = await mountCleared();
    routerReplace.mockReset();
    await stacksButton(wrapper).trigger("click");
    expect(routerReplace).not.toHaveBeenCalled();
  });
});
