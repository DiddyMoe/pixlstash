// useSnapshotsStore.test.js - the daily-snapshot setting's read state.
//
// The Automatic snapshots switch is disabled while the saved value is being
// read, so a toggle made before the read lands cannot be overwritten by it.
// What is pinned here is the flag that drives that: true while the read is on
// the wire, false once it has settled either way, and owned by the current
// session only.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { setActivePinia, createPinia } from "pinia";

const api = vi.hoisted(() => ({
  getSnapshotSettings: vi.fn(),
  setDailySnapshotsEnabled: vi.fn(),
}));
vi.mock("../api/serverConfig", () => api);
vi.mock("../api/snapshots", () => ({
  listSnapshots: vi.fn(),
  getSnapshotStatus: vi.fn(),
}));
vi.mock("../utils/apiClient", () => ({
  onSessionReset: () => () => {},
}));

import { useSnapshotsStore } from "./useSnapshotsStore";

/** Make each getSnapshotSettings call return a promise the test settles. */
function deferReads() {
  const reads = [];
  api.getSnapshotSettings.mockImplementation(
    () =>
      new Promise((resolve, reject) => {
        reads.push({ resolve, reject });
      }),
  );
  return reads;
}

let warn;
beforeEach(() => {
  setActivePinia(createPinia());
  api.getSnapshotSettings.mockReset();
  api.setDailySnapshotsEnabled.mockReset();
  warn = vi.spyOn(console, "warn").mockImplementation(() => {});
});
afterEach(() => {
  warn.mockRestore();
});

describe("snapshotSettingsLoading tracks the read of the daily setting", () => {
  it("is true while the read is pending, then false with the saved value shown", async () => {
    const reads = deferReads();
    const store = useSnapshotsStore();

    const pending = store.fetchSnapshotSettings();
    expect(store.snapshotSettingsLoading).toBe(true);

    reads[0].resolve({ daily_snapshots: false });
    await pending;
    expect(store.snapshotSettingsLoading).toBe(false);
    expect(store.dailySnapshotsEnabled).toBe(false);
  });

  it("is false again after a failed read", async () => {
    const reads = deferReads();
    const store = useSnapshotsStore();

    const pending = store.fetchSnapshotSettings();
    expect(store.snapshotSettingsLoading).toBe(true);

    reads[0].reject(new Error("test-read-failed"));
    await pending;
    expect(store.snapshotSettingsLoading).toBe(false);
  });

  it("is cleared by a session reset while the read is pending", async () => {
    const reads = deferReads();
    const store = useSnapshotsStore();

    const pending = store.fetchSnapshotSettings();
    store.reset();
    expect(store.snapshotSettingsLoading).toBe(false);

    // The previous session's read lands and is dropped.
    reads[0].resolve({ daily_snapshots: false });
    await pending;
    expect(store.dailySnapshotsEnabled).toBe(true);
  });

  it("is not cleared by the previous session's read while the next one is pending", async () => {
    const reads = deferReads();
    const store = useSnapshotsStore();

    const previous = store.fetchSnapshotSettings();
    store.reset();
    const current = store.fetchSnapshotSettings();

    reads[0].resolve({ daily_snapshots: false });
    await previous;
    expect(store.snapshotSettingsLoading).toBe(true);

    reads[1].resolve({ daily_snapshots: false });
    await current;
    expect(store.snapshotSettingsLoading).toBe(false);
    expect(store.dailySnapshotsEnabled).toBe(false);
  });
});
