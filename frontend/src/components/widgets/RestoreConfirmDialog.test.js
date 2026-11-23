// The restore dialog shows a dry run of one snapshot above a Restore button
// that runs the snapshot picked last. These tests pin that the two never part:
// a preview read for a snapshot picked earlier must not land over the one
// picked since, whichever order the replies come back in.

import { describe, it, expect, vi, beforeEach } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

import { useSnapshotsStore } from "../../stores/useSnapshotsStore";

const previewRestoreBatch = vi.fn();
const executeRestoreBatch = vi.fn();
vi.mock("../../api/snapshots", () => ({
  previewRestoreBatch: (...a) => previewRestoreBatch(...a),
  executeRestoreBatch: (...a) => executeRestoreBatch(...a),
}));

import RestoreConfirmDialog from "./RestoreConfirmDialog.vue";

const AppDialogStub = {
  props: ["open", "title", "size"],
  template:
    "<div v-if='open' class='dlg'><slot />" +
    "<footer class='dlg-foot'><slot name='footer' /></footer></div>",
};

const AppButtonStub = {
  props: ["disabled", "loading", "variant", "size", "iconLeft"],
  emits: ["click"],
  template:
    '<button :disabled="disabled || loading" @click="$emit(\'click\')"><slot /></button>',
};

const SNAPSHOTS = [
  {
    id: 5,
    kind: "MANUAL",
    label: "fake-before",
    created_at: "2026-01-01T00:00:00Z",
    is_compatible: true,
  },
  {
    id: 6,
    kind: "MANUAL",
    label: "fake-after",
    created_at: "2026-01-02T00:00:00Z",
    is_compatible: true,
  },
];

function previewOf(snapshot, picturesToRevert) {
  return {
    snapshot,
    summary: { pictures_to_revert: picturesToRevert },
    resources: [],
    warnings: [],
  };
}

/** A preview the test answers by hand, to choose the order replies land in. */
function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

async function openOnPicker() {
  useSnapshotsStore().snapshots = SNAPSHOTS;
  const wrapper = mount(RestoreConfirmDialog, {
    // A per-picture restore: no full-vault acknowledgement stands between a
    // landed preview and an enabled Restore button.
    props: { open: false, resources: [{ type: "picture", id: 1 }] },
    global: {
      stubs: {
        AppDialog: AppDialogStub,
        AppButton: AppButtonStub,
        Tooltip: true,
        "v-chip": true,
        "v-icon": true,
        "v-progress-circular": true,
        "v-alert": true,
        "v-checkbox": true,
      },
    },
  });
  // The dialog resets and loads on the open transition, not on mount.
  await wrapper.setProps({ open: true });
  return wrapper;
}

function buttonWith(wrapper, text) {
  return wrapper.findAll("button").find((b) => b.text().includes(text));
}

/** Pick the first snapshot, go back, pick the second: two previews in flight. */
async function pickTwice(wrapper) {
  await wrapper.findAll(".restore-picker-row")[0].trigger("click");
  await buttonWith(wrapper, "Back").trigger("click");
  await wrapper.findAll(".restore-picker-row")[1].trigger("click");
}

describe("RestoreConfirmDialog - a preview read for an earlier pick", () => {
  let earlier;
  let later;

  beforeEach(() => {
    setActivePinia(createPinia());
    previewRestoreBatch.mockReset();
    executeRestoreBatch.mockReset();
    earlier = deferred();
    later = deferred();
    previewRestoreBatch
      .mockReturnValueOnce(earlier.promise)
      .mockReturnValueOnce(later.promise);
  });

  it.each([
    ["answers", (reply) => reply.resolve(previewOf(SNAPSHOTS[0], 111))],
    ["fails", (reply) => reply.reject(new Error("fake-preview-failed"))],
  ])("does not land over the later pick when it %s last", async (_o, land) => {
    const wrapper = await openOnPicker();
    await pickTwice(wrapper);
    expect(previewRestoreBatch.mock.calls.map((call) => call[0])).toEqual([
      5, 6,
    ]);

    later.resolve(previewOf(SNAPSHOTS[1], 222));
    await flushPromises();
    land(earlier);
    await flushPromises();

    expect(wrapper.find(".restore-preview-cp-label").text()).toBe(
      "fake-after",
    );
    expect(wrapper.find(".restore-summary-value").text()).toBe("222");
    expect(wrapper.text()).not.toContain("fake-preview-failed");
    expect(
      buttonWith(wrapper, "Restore").attributes("disabled"),
    ).toBeUndefined();
  });

  it("keeps loading, with Restore off, when it lands before the later pick's", async () => {
    const wrapper = await openOnPicker();
    await pickTwice(wrapper);

    earlier.resolve(previewOf(SNAPSHOTS[0], 111));
    await flushPromises();

    expect(wrapper.text()).toContain("Loading preview…");
    expect(wrapper.text()).not.toContain("fake-before");
    expect(buttonWith(wrapper, "Restore").attributes("disabled")).toBe("");

    later.resolve(previewOf(SNAPSHOTS[1], 222));
    await flushPromises();
    expect(wrapper.find(".restore-preview-cp-label").text()).toBe(
      "fake-after",
    );
  });
});
