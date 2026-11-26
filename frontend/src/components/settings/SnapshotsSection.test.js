// Settings › Recovery: when the snapshots section reads from the server.
//
// The section sits in a tab of the settings dialog, which mounts every pane at
// once. What is pinned is that a pane whose tab is not showing reads nothing,
// that the first showing reads the list, the restore status and the daily
// setting, and that coming back to the tab does not read them again. Also that
// the Automatic snapshots switch takes no toggle until that setting is read,
// and that deleting a snapshot asks in the app's dialog and deletes nothing on
// no.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";

vi.mock("vuetify/components", () => ({
  VSwitch: {
    name: "VSwitch",
    props: ["disabled"],
    template: '<input :disabled="disabled" />',
  },
  VProgressLinear: { template: "<i />" },
  VProgressCircular: { template: "<i />" },
  VIcon: { template: "<i />" },
  VTooltip: { template: "<span />" },
}));

const store = vi.hoisted(() => ({
  snapshots: [],
  loading: false,
  activeJob: null,
  error: null,
  dailySnapshotsEnabled: true,
  snapshotSettingsLoading: true,
  fetchSnapshots: vi.fn(),
  fetchStatus: vi.fn(),
  fetchSnapshotSettings: vi.fn(),
  deleteSnapshot: vi.fn(),
}));
// Reactive, so a change of the read flag reaches a mounted section the way the
// Pinia store's does.
vi.mock("../../stores/useSnapshotsStore", async () => {
  const { reactive } = await import("vue");
  const reactiveStore = reactive(store);
  return { useSnapshotsStore: () => reactiveStore };
});

import SnapshotsSection from "./SnapshotsSection.vue";
import { useSnapshotsStore } from "../../stores/useSnapshotsStore";
import {
  activeConfirm,
  registerConfirmHost,
  resolveConfirm,
  unregisterConfirmHost,
} from "../../composables/useConfirm";

/** How many times each of the section's three reads has run. */
function reads() {
  return [
    store.fetchSnapshots.mock.calls.length,
    store.fetchStatus.mock.calls.length,
    store.fetchSnapshotSettings.mock.calls.length,
  ];
}

beforeEach(() => {
  store.snapshotSettingsLoading = true;
  store.fetchSnapshots.mockReset().mockResolvedValue(undefined);
  store.fetchStatus.mockReset().mockResolvedValue(undefined);
  store.fetchSnapshotSettings.mockReset().mockResolvedValue(undefined);
  store.deleteSnapshot.mockReset().mockResolvedValue(undefined);
});

afterEach(() => {
  unregisterConfirmHost();
  store.snapshots = [];
  vi.restoreAllMocks();
});

describe("a hidden snapshots section reads nothing until it is shown", () => {
  it("reads the list, the status and the daily setting once, on first being shown", async () => {
    const wrapper = mount(SnapshotsSection, {
      props: { open: true, active: false },
    });
    await flushPromises();
    expect(reads()).toEqual([0, 0, 0]);

    await wrapper.setProps({ active: true });
    await flushPromises();
    expect(reads()).toEqual([1, 1, 1]);

    await wrapper.setProps({ active: false });
    await wrapper.setProps({ active: true });
    await flushPromises();
    expect(reads()).toEqual([1, 1, 1]);

    wrapper.unmount();
  });

  it("reads at once when the dialog opens on its tab", async () => {
    const wrapper = mount(SnapshotsSection, {
      props: { open: true, active: true },
    });
    await flushPromises();
    expect(reads()).toEqual([1, 1, 1]);

    wrapper.unmount();
  });
});

describe("the Automatic snapshots switch waits for the saved setting", () => {
  it("is disabled while the setting is being read, and enabled once it is", async () => {
    const wrapper = mount(SnapshotsSection, {
      props: { open: true, active: true },
    });
    await flushPromises();
    const toggle = wrapper.findComponent({ name: "VSwitch" });
    expect(toggle.props("disabled")).toBe(true);

    useSnapshotsStore().snapshotSettingsLoading = false;
    await nextTick();
    expect(toggle.props("disabled")).toBe(false);

    wrapper.unmount();
  });
});

describe("deleting a snapshot asks first", () => {
  it("asks in the app's dialog, and deletes nothing on no", async () => {
    registerConfirmHost();
    const nativeConfirm = vi.spyOn(window, "confirm");
    store.snapshots = [
      {
        id: 41,
        kind: "manual",
        label: "before the cleanup",
        created_at: "2026-11-20T10:00:00Z",
        is_compatible: true,
        picture_count: 3,
        byte_size: 2048,
      },
    ];
    const wrapper = mount(SnapshotsSection, {
      props: { open: true, active: true },
    });
    await flushPromises();
    const del = wrapper.find('.snapshot-row-actions button[aria-label="Delete"]');
    expect(del.exists()).toBe(true);

    await del.trigger("click");
    await flushPromises();
    expect(activeConfirm.value.options).toMatchObject({
      title: 'Delete snapshot "before the cleanup"?',
      message: "The library can no longer be restored to this point.",
      warning: "There is no undo for this.",
      confirmLabel: "Delete snapshot",
      danger: true,
    });
    resolveConfirm(false);
    await flushPromises();
    expect(store.deleteSnapshot).not.toHaveBeenCalled();

    await del.trigger("click");
    await flushPromises();
    expect(store.deleteSnapshot).not.toHaveBeenCalled();
    resolveConfirm(true);
    await flushPromises();
    expect(store.deleteSnapshot).toHaveBeenCalledWith(41);
    expect(nativeConfirm).not.toHaveBeenCalled();

    wrapper.unmount();
  });
});
