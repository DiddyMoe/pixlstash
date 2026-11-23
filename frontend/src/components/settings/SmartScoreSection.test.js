// Settings › Smart score & filters › Penalised Tags.
//
// Removing a penalised tag with X takes it off this library's pictures and
// stops the tagger adding it in any library, and nothing brings the removed
// tags back. So the behaviours pinned here are: X asks first and does nothing
// when declined; the chip goes only once the server agreed; one removal is in
// flight at a time; the progress and the outcome reach the notice stack; an
// unfinished cleanup offers a retry; and adding a removed tag back asks first
// and lifts the removal instead of only setting a weight.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mount } from "@vue/test-utils";
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

let pinia;
let consoleError;

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

async function settle(wrapper) {
  await nextTick();
  await nextTick();
  await nextTick();
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

beforeEach(() => {
  vi.useFakeTimers();
  pinia = createPinia();
  setActivePinia(pinia);
  vi.clearAllMocks();
  consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
  getUserConfig.mockResolvedValue({
    smart_score_penalised_tags: { "flux chin": 3, blurry: 4 },
    hidden_tags: [],
    apply_tag_filter: false,
  });
  patchUserConfig.mockResolvedValue({});
  getPenalisedTagCleanup.mockResolvedValue(cleanupStatus());
  confirmMock.mockResolvedValue(true);
  deletePenalisedTag.mockResolvedValue(removedFluxChin());
});

afterEach(() => {
  consoleError.mockRestore();
  vi.clearAllTimers();
  vi.useRealTimers();
});

describe("SmartScoreSection penalised tags", () => {
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

  it("sends one removal at a time and holds the list still meanwhile", async () => {
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
    await pressX(wrapper, "blurry");

    expect(confirmMock).toHaveBeenCalledTimes(1);
    expect(deletePenalisedTag).toHaveBeenCalledTimes(1);
    expect(chipFor(wrapper, "flux chin").find(".s-chip__meta").text()).toBe(
      "Removing…",
    );
    expect(
      wrapper
        .findAll(".test-select")
        .map((select) => select.attributes("disabled") !== undefined),
    ).toEqual([true, true]);

    // An add now would send the old list, the tag being removed included.
    await addTag(wrapper, "grainy");
    expect(patchUserConfig).not.toHaveBeenCalled();
    expect(wrapper.findAll(".test-input")[0].element.value).toBe("grainy");

    finish(removedFluxChin());
    await settle(wrapper);
    expect(chipFor(wrapper, "flux chin")).toBeUndefined();
  });

  it("does not remove a tag while a save of the list is still on its way", async () => {
    let finish;
    patchUserConfig.mockReturnValue(
      new Promise((resolve) => {
        finish = resolve;
      }),
    );
    const wrapper = await settle(mountPane());
    await addTag(wrapper, "grainy");
    // The save in flight carries 'flux chin'; landing after the removal it
    // would put the weight back.
    await pressX(wrapper, "flux chin");
    expect(confirmMock).not.toHaveBeenCalled();
    expect(deletePenalisedTag).not.toHaveBeenCalled();

    finish({});
    await settle(wrapper);
    await pressX(wrapper, "flux chin");
    expect(deletePenalisedTag).toHaveBeenCalledWith("flux chin");
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
