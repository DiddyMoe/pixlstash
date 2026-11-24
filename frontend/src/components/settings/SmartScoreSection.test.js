// Settings › Smart score & filters › Penalised Tags.
//
// Removing a penalised tag with X takes it off this library's pictures and
// stops the tagger adding it in any library, and nothing brings the removed
// tags back. So the behaviours pinned here are: X asks first and does nothing
// when declined; the chip goes only once the server agreed; one question is
// open at a time; the progress and the outcome reach the notice stack; an
// unfinished cleanup offers a retry; and adding a removed tag back asks first
// and lifts the removal instead of only setting a weight.
//
// Every edit, a removal included, is saved one at a time against the list the
// server last accepted: a chip being removed is busy, a read that started
// before an accepted edit cannot put the old list back, and a pane reopened
// while an edit is saving reads the list that edit leaves.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import { createPinia, setActivePinia } from "pinia";

vi.mock("vuetify/components", () => ({
  VIcon: { name: "v-icon", template: "<i><slot /></i>" },
  VSwitch: {
    name: "v-switch",
    props: ["modelValue", "disabled", "label"],
    template: '<input type="checkbox" />',
  },
  // Tooltip.vue wraps VTooltip: render the activator only, as a closed tip does.
  VTooltip: {
    name: "VTooltip",
    setup: (_p, { slots }) => () => slots.activator?.({ props: {} }),
  },
}));

vi.mock("../../api/config", () => ({
  getUserConfig: vi.fn(),
  patchUserConfig: vi.fn(),
}));

vi.mock("../../api/users", () => ({
  putPenalisedTag: vi.fn(),
  deletePenalisedTag: vi.fn(),
  getPenalisedTagCleanup: vi.fn(),
  startPenalisedTagCleanup: vi.fn(),
}));

const confirmMock = vi.fn();
vi.mock("../../composables/useConfirm", () => ({
  useConfirm: () => ({ confirm: confirmMock }),
}));

import SmartScoreSection from "./SmartScoreSection.vue";
import { getUserConfig, patchUserConfig } from "../../api/config";
import {
  deletePenalisedTag,
  getPenalisedTagCleanup,
  putPenalisedTag,
  startPenalisedTagCleanup,
} from "../../api/users";
import { useNoticeStore } from "../../stores/useNoticeStore";

const REMOVE_QUESTION = {
  title: "Remove 'flux chin'?",
  message:
    "Removes this tag from all pictures in this library and stops the tagger " +
    "adding it in any of your libraries. Pictures stay; tags cannot be restored.",
  confirmLabel: "Remove",
  cancelLabel: "Cancel",
  danger: true,
};

/** A cleanup job as the server reports it. */
function job(overrides = {}) {
  return {
    id: "test-job-1",
    names: ["flux chin"],
    state: "queued",
    pictures: 0,
    scrapheap: 0,
    locked: 0,
    removed: { tags: 0, predictions: 0, suggestions: 0 },
    sidecar_failed: 0,
    ...overrides,
  };
}

/** The cleanup status route's body. */
function cleanupStatus(overrides = {}) {
  return {
    library_uuid: "test-library",
    excluded: [],
    remaining: 0,
    locked: 0,
    state: "idle",
    jobs: [],
    ...overrides,
  };
}

/** The DELETE response for 'flux chin'. */
function removedFluxChin(cleanup = job()) {
  return {
    smart_score_penalised_tags: { blurry: 4 },
    excluded: ["flux chin"],
    cleanup,
  };
}


function mountPane() {
  return mount(SmartScoreSection, {
    props: { open: true },
    global: {
      plugins: [pinia],
      stubs: {
        VIcon: true,
        AppSelect: {
          props: ["modelValue", "disabled", "options"],
          template: '<select class="test-select" :disabled="disabled" />',
        },
        AppInput: {
          props: ["modelValue"],
          template:
            '<input class="test-input" :value="modelValue" @input="$emit(\'update:modelValue\', $event.target.value)" />',
        },
        AppButton: {
          props: ["disabled", "loading"],
          template:
            '<button class="test-add" :disabled="disabled" @click="$emit(\'click\', $event)"><slot /></button>',
        },
      },
    },
  });
}

/** Let a click's question, its queued save and the answer settle. */
async function settle(wrapper) {
  for (let i = 0; i < 10; i += 1) await nextTick();
  return wrapper;
}

/** Let the poll's timer fire and its request settle. */
async function poll(wrapper) {
  await vi.advanceTimersByTimeAsync(1000);
  return settle(wrapper);
}

function chipFor(wrapper, tag) {
  return wrapper
    .findAll(".s-chip")
    .find((chip) => chip.find(".s-chip__label").text() === tag);
}

async function pressX(wrapper, tag) {
  await chipFor(wrapper, tag).find(".s-chip__remove").trigger("click");
  return settle(wrapper);
}

/** The first "Add tag" row is the penalised list's. */
async function addTag(wrapper, text) {
  await wrapper.findAll(".test-input")[0].setValue(text);
  await wrapper.findAll(".test-add")[0].trigger("click");
  return settle(wrapper);
}

function noticeKeyed(key) {
  return useNoticeStore().notices.find((n) => n.key === key);
}

function noticeSummary() {
  return useNoticeStore().notices.map((n) => [
    n.level,
    n.text,
    n.action?.label ?? null,
  ]);
}

let pinia;
let consoleError;

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function config(penalised) {
  return {
    smart_score_penalised_tags: penalised,
    hidden_tags: [],
    apply_tag_filter: false,
  };
}

/** A DELETE answer that started no cleanup job. */
function removedAnswer(penalised, excluded) {
  return { smart_score_penalised_tags: penalised, excluded, cleanup: null };
}

const AppSelectStub = {
  name: "AppSelect",
  props: ["modelValue", "options", "disabled", "compact"],
  emits: ["update:modelValue"],
  template: '<select class="app-select-stub" :disabled="disabled" />',
};

const AddTagRowStub = {
  name: "SettingsAddTagRow",
  props: ["modelValue", "placeholder", "btn"],
  emits: ["update:modelValue", "add"],
  template: '<div class="add-tag-row-stub" />',
};

function mountSection() {
  return mount(SmartScoreSection, {
    props: { open: true },
    global: {
      plugins: [pinia],
      stubs: {
        SettingsSection: { template: "<section><slot /></section>" },
        SettingsAddTagRow: AddTagRowStub,
        AppSelect: AppSelectStub,
        Tooltip: true,
      },
    },
  });
}

async function mountLoaded(penalised) {
  getUserConfig.mockResolvedValueOnce(config(penalised));
  const wrapper = mountSection();
  await flushPromises();
  return wrapper;
}

function labels(wrapper) {
  return wrapper.findAll(".s-chip__label").map((node) => node.text());
}

function chip(wrapper, tag) {
  const found = wrapper
    .findAll(".s-chip")
    .find((node) => node.find(".s-chip__label").text() === tag);
  if (!found) throw new Error(`no chip for ${tag}`);
  return found;
}

function removeButton(wrapper, tag) {
  return chip(wrapper, tag).find(".s-chip__remove");
}

function weightSelect(wrapper, tag) {
  return chip(wrapper, tag).findComponent(AppSelectStub);
}

function weights(wrapper) {
  return Object.fromEntries(
    wrapper
      .findAll(".s-chip")
      .map((node) => [
        node.find(".s-chip__label").text(),
        node.findComponent(AppSelectStub).props("modelValue"),
      ]),
  );
}

function savedLists() {
  return patchUserConfig.mock.calls.map(
    ([partial]) => partial.smart_score_penalised_tags,
  );
}

/** A fresh store and mocks with nothing queued from an earlier test. */
function resetEnvironment() {
  pinia = createPinia();
  setActivePinia(pinia);
  vi.resetAllMocks();
  consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
  getPenalisedTagCleanup.mockResolvedValue(cleanupStatus());
  confirmMock.mockResolvedValue(true);
}

describe("SmartScoreSection penalised tags", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    resetEnvironment();
    getUserConfig.mockResolvedValue({
      smart_score_penalised_tags: { "flux chin": 3, blurry: 4 },
      hidden_tags: [],
      apply_tag_filter: false,
    });
    patchUserConfig.mockResolvedValue({});
    deletePenalisedTag.mockResolvedValue(removedFluxChin());
  });

  afterEach(() => {
    consoleError.mockRestore();
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  it("says that removing a tag also removes it from pictures", async () => {
    const wrapper = await settle(mountPane());
    expect(wrapper.text()).toContain(
      "Tags listed here lower Smart Score. Removing one also removes it from your pictures.",
    );
  });

  it("asks before removing, then removes the tag and shows the progress", async () => {
    const wrapper = await settle(mountPane());
    await pressX(wrapper, "flux chin");

    expect(confirmMock).toHaveBeenCalledWith(REMOVE_QUESTION);
    expect(deletePenalisedTag).toHaveBeenCalledWith("flux chin");
    expect(patchUserConfig).not.toHaveBeenCalled();
    expect(chipFor(wrapper, "flux chin")).toBeUndefined();
    expect(chipFor(wrapper, "blurry")).toBeDefined();
    expect(noticeKeyed("penalised-cleanup:test-job-1")).toMatchObject({
      level: "info",
      text: "Removing 'flux chin' from pictures…",
      timeout: 0,
    });
  });

  it("does nothing when the question is declined", async () => {
    confirmMock.mockResolvedValue(false);
    const wrapper = await settle(mountPane());
    await pressX(wrapper, "flux chin");

    expect(confirmMock).toHaveBeenCalledTimes(1);
    expect(deletePenalisedTag).not.toHaveBeenCalled();
    expect(patchUserConfig).not.toHaveBeenCalled();
    expect(chipFor(wrapper, "flux chin")).toBeDefined();
    expect(noticeSummary()).toEqual([]);
  });

  it("keeps the chip and says why when the server refuses", async () => {
    deletePenalisedTag.mockRejectedValue({
      response: { status: 500, data: { detail: "test-refusal" } },
    });
    const wrapper = await settle(mountPane());
    await pressX(wrapper, "flux chin");

    expect(chipFor(wrapper, "flux chin")).toBeDefined();
    expect(noticeSummary()).toEqual([["error", "test-refusal", null]]);
    expect(wrapper.text()).not.toContain("Saved.");
    expect(consoleError).toHaveBeenCalled();
  });

  it("asks one question at a time and sends an add made meanwhile after the removal, without the removed tag", async () => {
    let finish;
    deletePenalisedTag.mockReturnValue(
      new Promise((resolve) => {
        finish = resolve;
      }),
    );
    const wrapper = await settle(mountPane());
    const x = chipFor(wrapper, "flux chin").find(".s-chip__remove");
    await x.trigger("click");
    await x.trigger("click");
    await settle(wrapper);

    expect(confirmMock).toHaveBeenCalledTimes(1);
    expect(deletePenalisedTag).toHaveBeenCalledTimes(1);
    expect(chipFor(wrapper, "flux chin").find(".s-chip__meta").text()).toBe(
      "Removing…",
    );

    // Sent now, the add would carry the list with 'flux chin' still in it.
    await addTag(wrapper, "grainy");
    expect(patchUserConfig).not.toHaveBeenCalled();

    finish(removedFluxChin());
    await settle(wrapper);
    expect(chipFor(wrapper, "flux chin")).toBeUndefined();
    expect(patchUserConfig).toHaveBeenCalledWith({
      smart_score_penalised_tags: { blurry: 4, grainy: 3 },
    });
  });

  it("sends a removal asked for while a save of the list is on its way after that save", async () => {
    let finish;
    patchUserConfig.mockReturnValue(
      new Promise((resolve) => {
        finish = resolve;
      }),
    );
    deletePenalisedTag.mockResolvedValue({
      smart_score_penalised_tags: { blurry: 4, grainy: 3 },
      excluded: ["flux chin"],
      cleanup: job(),
    });
    const wrapper = await settle(mountPane());
    await addTag(wrapper, "grainy");
    // The save in flight carries 'flux chin'; landing after the removal it
    // would put the weight back.
    await pressX(wrapper, "flux chin");
    expect(confirmMock).toHaveBeenCalledTimes(1);
    expect(deletePenalisedTag).not.toHaveBeenCalled();

    finish({});
    await settle(wrapper);
    expect(deletePenalisedTag).toHaveBeenCalledWith("flux chin");
    expect(chipFor(wrapper, "flux chin")).toBeUndefined();
    expect(chipFor(wrapper, "grainy")).toBeDefined();
  });

  it("reports how many pictures the tag came off", async () => {
    getPenalisedTagCleanup
      .mockResolvedValueOnce(cleanupStatus())
      .mockResolvedValueOnce(
        cleanupStatus({
          excluded: ["flux chin"],
          state: "running",
          jobs: [job({ state: "running" })],
        }),
      )
      .mockResolvedValue(
        cleanupStatus({
          excluded: ["flux chin"],
          jobs: [job({ state: "completed", pictures: 12 })],
        }),
      );
    const wrapper = await settle(mountPane());
    await pressX(wrapper, "flux chin");

    await poll(wrapper);
    expect(noticeSummary()).toEqual([
      ["info", "Removing 'flux chin' from pictures…", null],
    ]);
    await poll(wrapper);
    expect(noticeSummary()).toEqual([
      ["success", "Removed 'flux chin' from 12 pictures.", null],
    ]);
  });

  it("says locked pictures kept the tag, without a retry that would skip them again", async () => {
    getPenalisedTagCleanup
      .mockResolvedValueOnce(cleanupStatus())
      .mockResolvedValue(
        cleanupStatus({
          excluded: ["flux chin"],
          locked: 2,
          jobs: [job({ state: "completed", pictures: 12, locked: 2 })],
        }),
      );
    const wrapper = await settle(mountPane());
    await pressX(wrapper, "flux chin");
    await poll(wrapper);

    expect(noticeSummary()).toEqual([
      [
        "warning",
        "Removed 'flux chin' from 12 pictures; 2 locked pictures were skipped.",
        null,
      ],
    ]);
  });

  it("offers a retry when the cleanup stopped, and follows the new job", async () => {
    getPenalisedTagCleanup
      .mockResolvedValueOnce(cleanupStatus())
      .mockResolvedValue(
        cleanupStatus({
          excluded: ["flux chin"],
          remaining: 7,
          state: "interrupted",
          jobs: [job({ state: "failed", pictures: 5 })],
        }),
      );
    startPenalisedTagCleanup.mockResolvedValue({
      cleanup: job({ id: "test-job-2" }),
    });
    const wrapper = await settle(mountPane());
    await pressX(wrapper, "flux chin");
    await poll(wrapper);

    expect(noticeSummary()).toEqual([
      [
        "warning",
        "Removing 'flux chin' from pictures stopped before it finished.",
        "Retry",
      ],
    ]);
    const store = useNoticeStore();
    store.invokeAction(store.notices[0].id);
    await settle(wrapper);

    expect(startPenalisedTagCleanup).toHaveBeenCalledTimes(1);
    expect(noticeSummary()).toEqual([
      ["info", "Removing 'flux chin' from pictures…", null],
    ]);
    expect(noticeKeyed("penalised-cleanup:test-job-2")).toBeDefined();
  });

  it("follows the job already running when a retry is refused for it", async () => {
    getPenalisedTagCleanup
      .mockResolvedValueOnce(
        cleanupStatus({
          excluded: ["blurry", "flux chin"],
          remaining: 3,
          state: "interrupted",
        }),
      )
      .mockResolvedValue(
        cleanupStatus({
          excluded: ["blurry", "flux chin"],
          state: "running",
          jobs: [
            job({
              id: "test-job-3",
              names: ["flux chin", "blurry"],
              state: "running",
            }),
          ],
        }),
      );
    startPenalisedTagCleanup.mockRejectedValue({
      response: { status: 409, data: { detail: "test-busy" } },
    });
    const wrapper = await settle(mountPane());
    const store = useNoticeStore();
    store.invokeAction(store.notices[0].id);
    await settle(wrapper);

    expect(noticeSummary()).toEqual([
      ["info", "Removing 'blurry' and 'flux chin' from pictures…", null],
    ]);
    expect(noticeKeyed("penalised-cleanup:test-job-3")).toBeDefined();
  });

  it("follows the job that took over when another removal superseded it", async () => {
    getPenalisedTagCleanup
      .mockResolvedValueOnce(cleanupStatus())
      .mockResolvedValue(
        cleanupStatus({
          excluded: ["blurry", "flux chin", "grainy"],
          state: "running",
          jobs: [
            job({
              id: "test-job-2",
              names: ["grainy", "flux chin", "blurry"],
              state: "running",
            }),
            job({ state: "superseded" }),
          ],
        }),
      );
    const wrapper = await settle(mountPane());
    await pressX(wrapper, "flux chin");
    await poll(wrapper);

    expect(noticeSummary()).toEqual([
      ["info", "Removing 'blurry' and 2 more tags from pictures…", null],
    ]);
    expect(noticeKeyed("penalised-cleanup:test-job-2")).toBeDefined();
  });

  it("follows a job once when a removal and an older job both lead to it", async () => {
    const both = job({
      id: "test-job-2",
      names: ["blurry", "flux chin"],
      state: "running",
    });
    deletePenalisedTag.mockResolvedValueOnce(removedFluxChin()).mockResolvedValueOnce({
      smart_score_penalised_tags: {},
      excluded: ["blurry", "flux chin"],
      cleanup: both,
    });
    getPenalisedTagCleanup
      .mockResolvedValueOnce(cleanupStatus())
      .mockResolvedValue(
        cleanupStatus({
          excluded: ["blurry", "flux chin"],
          state: "running",
          jobs: [both, job({ state: "superseded" })],
        }),
      );
    const wrapper = await settle(mountPane());
    await pressX(wrapper, "flux chin");
    await pressX(wrapper, "blurry");
    await poll(wrapper);
    await poll(wrapper);

    // One read on open, then one per job per second: test-job-1 once (it was
    // superseded), test-job-2 twice.
    expect(getPenalisedTagCleanup).toHaveBeenCalledTimes(4);
    expect(noticeSummary()).toEqual([
      ["info", "Removing 'blurry' and 'flux chin' from pictures…", null],
    ]);
  });

  it("says nothing more when adding the tag back stopped its cleanup", async () => {
    getPenalisedTagCleanup
      .mockResolvedValueOnce(cleanupStatus())
      .mockResolvedValue(cleanupStatus({ jobs: [job({ state: "superseded" })] }));
    const wrapper = await settle(mountPane());
    await pressX(wrapper, "flux chin");
    await poll(wrapper);

    expect(noticeSummary()).toEqual([]);
  });

  it("says the tag is off the pictures when the job was lost but nothing is left", async () => {
    getPenalisedTagCleanup
      .mockResolvedValueOnce(cleanupStatus())
      .mockResolvedValue(cleanupStatus({ excluded: ["flux chin"] }));
    const wrapper = await settle(mountPane());
    await pressX(wrapper, "flux chin");
    await poll(wrapper);

    expect(noticeSummary()).toEqual([
      ["success", "Removed 'flux chin' from your pictures.", null],
    ]);
  });

  it.each(["failed", "cancelled"])(
    "offers a retry, not success, when the job %s though nothing is left",
    async (state) => {
      getPenalisedTagCleanup
        .mockResolvedValueOnce(cleanupStatus())
        .mockResolvedValue(
          cleanupStatus({ excluded: ["flux chin"], jobs: [job({ state })] }),
        );
      const wrapper = await settle(mountPane());
      await pressX(wrapper, "flux chin");
      await poll(wrapper);

      expect(noticeSummary()).toEqual([
        [
          "warning",
          "Removing 'flux chin' from pictures stopped before it finished.",
          "Retry",
        ],
      ]);
    },
  );

  it("offers to check again when the progress could not be read", async () => {
    getPenalisedTagCleanup
      .mockResolvedValueOnce(cleanupStatus())
      .mockRejectedValueOnce({ response: { status: 503, data: {} } })
      .mockResolvedValue(
        cleanupStatus({
          excluded: ["flux chin"],
          state: "running",
          jobs: [job({ state: "running" })],
        }),
      );
    const wrapper = await settle(mountPane());
    await pressX(wrapper, "flux chin");
    await poll(wrapper);

    expect(noticeSummary()).toEqual([
      ["warning", "Could not check on removing 'flux chin'.", "Check again"],
    ]);
    expect(consoleError).toHaveBeenCalled();

    const store = useNoticeStore();
    store.invokeAction(store.notices[0].id);
    await settle(wrapper);
    expect(noticeSummary()).toEqual([
      ["info", "Removing 'flux chin' from pictures…", null],
    ]);
    await poll(wrapper);
    expect(getPenalisedTagCleanup).toHaveBeenCalledTimes(3);
  });

  it("offers to finish removing tags still left in this library", async () => {
    getPenalisedTagCleanup.mockResolvedValue(
      cleanupStatus({
        excluded: ["flux chin"],
        remaining: 1,
        state: "interrupted",
      }),
    );
    startPenalisedTagCleanup.mockResolvedValue({
      cleanup: job({ id: "test-job-2" }),
    });
    const wrapper = await settle(mountPane());

    expect(noticeSummary()).toEqual([
      [
        "warning",
        "Removed tags are still on 1 picture in this library.",
        "Remove from this library",
      ],
    ]);
    const store = useNoticeStore();
    store.invokeAction(store.notices[0].id);
    await settle(wrapper);
    expect(startPenalisedTagCleanup).toHaveBeenCalledTimes(1);
    expect(noticeKeyed("penalised-cleanup:test-job-2")).toBeDefined();
  });

  it("asks before adding a removed tag back, then lifts the removal", async () => {
    getUserConfig.mockResolvedValue({
      smart_score_penalised_tags: { blurry: 4 },
      hidden_tags: [],
      apply_tag_filter: false,
    });
    getPenalisedTagCleanup.mockResolvedValue(
      cleanupStatus({ excluded: ["flux chin"] }),
    );
    putPenalisedTag.mockResolvedValue({
      smart_score_penalised_tags: { blurry: 4, "flux chin": 3 },
      excluded: [],
      cleanup: job({ id: "test-job-4", names: ["blurry"] }),
    });
    const wrapper = await settle(mountPane());
    await addTag(wrapper, " Flux Chin ");

    expect(confirmMock).toHaveBeenCalledWith({
      title: "Add 'flux chin' back?",
      message:
        "The tagger can add it to pictures again. Tags removed earlier are not restored.",
      confirmLabel: "Add back",
      cancelLabel: "Cancel",
    });
    expect(putPenalisedTag).toHaveBeenCalledWith("flux chin", 3);
    expect(patchUserConfig).not.toHaveBeenCalled();
    expect(chipFor(wrapper, "flux chin")).toBeDefined();
    // The re-add restarts the cleanup of the other removed tags.
    expect(noticeKeyed("penalised-cleanup:test-job-4")).toMatchObject({
      text: "Removing 'blurry' from pictures…",
    });
  });

  it("leaves a removed tag removed when adding it back is declined", async () => {
    confirmMock.mockResolvedValue(false);
    getPenalisedTagCleanup.mockResolvedValue(
      cleanupStatus({ excluded: ["grainy"] }),
    );
    const wrapper = await settle(mountPane());
    await addTag(wrapper, "grainy");

    expect(confirmMock).toHaveBeenCalledTimes(1);
    expect(putPenalisedTag).not.toHaveBeenCalled();
    expect(patchUserConfig).not.toHaveBeenCalled();
    expect(chipFor(wrapper, "grainy")).toBeUndefined();
  });

  it("knows a tag removed here is removed when it is typed again", async () => {
    putPenalisedTag.mockResolvedValue({
      smart_score_penalised_tags: { blurry: 4, "flux chin": 3 },
      excluded: [],
      cleanup: null,
    });
    const wrapper = await settle(mountPane());
    await pressX(wrapper, "flux chin");
    await addTag(wrapper, "flux chin");

    expect(confirmMock).toHaveBeenCalledTimes(2);
    expect(putPenalisedTag).toHaveBeenCalledWith("flux chin", 3);
    expect(patchUserConfig).not.toHaveBeenCalled();
  });

  it("adds a tag that was never removed without asking", async () => {
    const wrapper = await settle(mountPane());
    await addTag(wrapper, "grainy");

    expect(confirmMock).not.toHaveBeenCalled();
    expect(putPenalisedTag).not.toHaveBeenCalled();
    expect(patchUserConfig).toHaveBeenCalledWith({
      smart_score_penalised_tags: { blurry: 4, "flux chin": 3, grainy: 3 },
    });
  });
});

describe("SmartScoreSection: Penalised Tags edits save in order", () => {
  beforeEach(resetEnvironment);
  afterEach(() => consoleError.mockRestore());

  it("two quick weight edits both stick: the second is sent after the first settles, with the first's weight", async () => {
    const wrapper = await mountLoaded({ "test-a": 3, "test-b": 4 });
    const first = deferred();
    const second = deferred();
    patchUserConfig
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);

    weightSelect(wrapper, "test-a").vm.$emit("update:modelValue", 5);
    weightSelect(wrapper, "test-b").vm.$emit("update:modelValue", 1);
    await flushPromises();
    expect(savedLists()).toEqual([{ "test-a": 5, "test-b": 4 }]);

    first.resolve({});
    await flushPromises();
    expect(savedLists()).toEqual([
      { "test-a": 5, "test-b": 4 },
      { "test-a": 5, "test-b": 1 },
    ]);

    second.resolve({});
    await flushPromises();
    expect(weights(wrapper)).toEqual({ "test-a": 5, "test-b": 1 });
  });

  it("a chip whose removal is being saved cannot be removed again", async () => {
    const wrapper = await mountLoaded({ "test-a": 3, "test-b": 4 });
    deletePenalisedTag.mockReturnValueOnce(deferred().promise);

    await removeButton(wrapper, "test-a").trigger("click");
    await flushPromises();

    const busy = removeButton(wrapper, "test-a");
    expect(busy.attributes("disabled")).toBeDefined();
    expect(busy.attributes("aria-busy")).toBe("true");
    expect(weightSelect(wrapper, "test-a").props("disabled")).toBe(true);
    expect(removeButton(wrapper, "test-b").attributes("disabled")).toBeUndefined();
  });

  it("X behind a weight edit on the same chip waits for it, and the chip goes on the server's answer", async () => {
    const wrapper = await mountLoaded({ "test-a": 3, "test-b": 4 });
    const weightSave = deferred();
    patchUserConfig.mockReturnValueOnce(weightSave.promise);
    deletePenalisedTag.mockResolvedValueOnce(
      removedAnswer({ "test-b": 4 }, ["test-a"]),
    );

    weightSelect(wrapper, "test-a").vm.$emit("update:modelValue", 5);
    await removeButton(wrapper, "test-a").trigger("click");
    await flushPromises();
    // The removal waits for the weight save, so it cannot be overtaken by it.
    expect(savedLists()).toEqual([{ "test-a": 5, "test-b": 4 }]);
    expect(deletePenalisedTag).not.toHaveBeenCalled();

    weightSave.resolve({});
    await flushPromises();
    expect(deletePenalisedTag).toHaveBeenCalledWith("test-a");
    expect(weights(wrapper)).toEqual({ "test-b": 4 });
  });

  it("a weight edit made while a removal saves is applied to the list the removal left", async () => {
    const wrapper = await mountLoaded({ "test-a": 3, "test-b": 4 });
    const removal = deferred();
    deletePenalisedTag.mockReturnValueOnce(removal.promise);
    patchUserConfig.mockResolvedValueOnce({});

    await removeButton(wrapper, "test-a").trigger("click");
    await flushPromises();
    weightSelect(wrapper, "test-b").vm.$emit("update:modelValue", 5);
    await flushPromises();
    expect(patchUserConfig).not.toHaveBeenCalled();

    removal.resolve(removedAnswer({ "test-b": 4 }, ["test-a"]));
    await flushPromises();
    // Built from the list on screen at the click, it would put 'test-a' back.
    expect(savedLists()).toEqual([{ "test-b": 5 }]);
    expect(weights(wrapper)).toEqual({ "test-b": 5 });
  });

  it("a failed removal keeps its chip, says so, and the edit queued behind it still saves", async () => {
    const wrapper = await mountLoaded({ "test-a": 3, "test-b": 4 });
    const failed = deferred();
    deletePenalisedTag.mockReturnValueOnce(failed.promise);
    patchUserConfig.mockResolvedValueOnce({});

    await removeButton(wrapper, "test-a").trigger("click");
    await flushPromises();
    weightSelect(wrapper, "test-b").vm.$emit("update:modelValue", 5);
    failed.reject({ response: { data: { detail: "test-refused" } } });
    await flushPromises();

    expect(savedLists()).toEqual([{ "test-a": 3, "test-b": 5 }]);
    expect(labels(wrapper)).toEqual(["test-a", "test-b"]);
    expect(removeButton(wrapper, "test-a").attributes("disabled")).toBeUndefined();
    const notices = useNoticeStore().notices;
    expect(
      notices.some((n) => n.level === "error" && n.text === "test-refused"),
    ).toBe(true);
  });

  it("an add made before the first read lands keeps the saved list", async () => {
    const firstRead = deferred();
    getUserConfig
      .mockReturnValueOnce(firstRead.promise)
      .mockResolvedValueOnce(config({ "test-a": 3, "test-b": 4 }));
    patchUserConfig.mockResolvedValue({});
    const wrapper = mountSection();

    const addRow = wrapper.findComponent(AddTagRowStub);
    addRow.vm.$emit("update:modelValue", "test-new");
    await wrapper.vm.$nextTick();
    addRow.vm.$emit("add");
    await flushPromises();

    expect(savedLists()).toEqual([{ "test-a": 3, "test-b": 4, "test-new": 3 }]);
    expect(labels(wrapper)).toEqual(["test-a", "test-b", "test-new"]);
  });
});

describe("SmartScoreSection: reading the list on open", () => {
  beforeEach(resetEnvironment);
  afterEach(() => consoleError.mockRestore());

  it("a read that started before an accepted removal does not bring the tag back", async () => {
    const wrapper = await mountLoaded({ "test-a": 3, "test-b": 4 });
    // Reopened while still mounted (during the close animation): a new read.
    const reread = deferred();
    getUserConfig.mockReturnValueOnce(reread.promise);
    await wrapper.setProps({ open: false });
    await wrapper.setProps({ open: true });
    await flushPromises();
    expect(getUserConfig).toHaveBeenCalledTimes(2);

    deletePenalisedTag.mockResolvedValueOnce(
      removedAnswer({ "test-b": 4 }, ["test-a"]),
    );
    await removeButton(wrapper, "test-a").trigger("click");
    await flushPromises();
    expect(labels(wrapper)).toEqual(["test-b"]);

    reread.resolve(config({ "test-a": 3, "test-b": 4 }));
    await flushPromises();
    expect(labels(wrapper)).toEqual(["test-b"]);
  });

  it("closing aborts the read in flight and its late answer is ignored", async () => {
    const wrapper = await mountLoaded({ "test-a": 3 });
    const reread = deferred();
    getUserConfig.mockReturnValueOnce(reread.promise);
    await wrapper.setProps({ open: false });
    await wrapper.setProps({ open: true });
    await flushPromises();
    const { signal } = getUserConfig.mock.calls[1][0];
    expect(signal.aborted).toBe(false);

    await wrapper.setProps({ open: false });
    expect(signal.aborted).toBe(true);

    reread.resolve(config({ "test-z": 5 }));
    await flushPromises();
    expect(labels(wrapper)).toEqual(["test-a"]);
  });
});

describe("SmartScoreSection: reopening Settings while an edit is saving", () => {
  // Closing Settings unmounts the pane and reopening mounts a new one, in the
  // same session. The server keeps one list and each save replaces all of it.
  let persisted;
  let saves;

  beforeEach(() => {
    resetEnvironment();
    persisted = { "test-a": 3, "test-b": 4, "test-c": 2 };
    saves = [];
    getUserConfig.mockImplementation(async () => config({ ...persisted }));
    patchUserConfig.mockImplementation((partial) => {
      const save = deferred();
      saves.push({
        accept() {
          persisted = { ...partial.smart_score_penalised_tags };
          save.resolve({});
        },
        refuse() {
          save.reject({ response: { data: { detail: "test-refused" } } });
        },
      });
      return save.promise;
    });
  });
  afterEach(() => consoleError.mockRestore());

  it.each([
    ["is saved", true, { "test-a": 5, "test-b": 1, "test-c": 2 }],
    ["is refused", false, { "test-a": 3, "test-b": 1, "test-c": 2 }],
  ])(
    "a weight edit in the reopened pane, after the closed pane's edit %s, keeps what that edit left",
    async (_outcome, accepted, expected) => {
      const closed = mountSection();
      await flushPromises();
      weightSelect(closed, "test-a").vm.$emit("update:modelValue", 5);
      await flushPromises();
      closed.unmount();

      const reopened = mountSection();
      await flushPromises();
      if (accepted) saves[0].accept();
      else saves[0].refuse();
      await flushPromises();

      weightSelect(reopened, "test-b").vm.$emit("update:modelValue", 1);
      await flushPromises();
      saves[1].accept();
      await flushPromises();

      expect(persisted).toEqual(expected);
    },
  );
});
