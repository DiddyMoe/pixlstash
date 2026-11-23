import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { ref } from "vue";

const isReadOnly = ref(false);
vi.mock("../utils/apiClient", () => ({
  API_BASE_URL: "/api/v1",
  get isReadOnly() {
    return isReadOnly;
  },
}));

vi.mock("../api/config", () => ({
  patchUserConfig: vi.fn(),
}));

import { patchUserConfig } from "../api/config";
import { useUserPrefsStore } from "./useUserPrefsStore";

beforeEach(() => {
  setActivePinia(createPinia());
  isReadOnly.value = false;
  patchUserConfig.mockReset();
});

describe("useUserPrefsStore consent persistence", () => {
  it("persists the whole consent choice in one request", async () => {
    patchUserConfig.mockResolvedValue({ status: "success" });
    const prefs = useUserPrefsStore();
    const decision = {
      check_for_updates: true,
      telemetry_send_install_id: true,
      telemetry_consent_prompted: true,
    };

    expect(await prefs.saveTelemetry(decision)).toBe(true);

    expect(patchUserConfig).toHaveBeenCalledTimes(1);
    expect(patchUserConfig).toHaveBeenCalledWith(decision);
    expect(prefs.checkForUpdates).toBe(true);
    expect(prefs.telemetrySendInstallId).toBe(true);
    expect(prefs.telemetryConsentPrompted).toBe(true);
  });

  it("applies none of the choice locally when the request fails", async () => {
    patchUserConfig.mockRejectedValue(new Error("offline"));
    const prefs = useUserPrefsStore();

    expect(
      await prefs.saveTelemetry({
        check_for_updates: true,
        telemetry_send_install_id: true,
        telemetry_consent_prompted: true,
      }),
    ).toBe(false);

    expect(prefs.checkForUpdates).toBe(null);
    expect(prefs.telemetrySendInstallId).toBe(false);
    expect(prefs.telemetryConsentPrompted).toBe(false);
  });
});

describe("useUserPrefsStore penalised-tag edit queue", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("runs edits one at a time in order, and a refused edit does not stop the ones behind it", async () => {
    const errorLog = vi.spyOn(console, "error").mockImplementation(() => {});
    const prefs = useUserPrefsStore();
    const ran = [];
    let refuseFirst;
    const first = prefs.queuePenalisedEdit(() => {
      ran.push("first");
      return new Promise((_resolve, reject) => {
        refuseFirst = reject;
      });
    });
    const second = prefs.queuePenalisedEdit(async () => {
      ran.push("second");
      return "test-second-saved";
    });

    await flushPromises();
    expect(ran).toEqual(["first"]);

    refuseFirst(new Error("test-refused"));
    await expect(first).rejects.toThrow("test-refused");
    await expect(second).resolves.toBe("test-second-saved");
    expect(ran).toEqual(["first", "second"]);
    await expect(prefs.penalisedEditsSettled()).resolves.toBeUndefined();
    expect(errorLog).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ message: "test-refused" }),
    );
  });

  it("reports settled only once every edit queued so far has finished", async () => {
    const prefs = useUserPrefsStore();
    let finish;
    prefs.queuePenalisedEdit(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    let settled = false;
    prefs.penalisedEditsSettled().then(() => {
      settled = true;
    });

    await flushPromises();
    expect(settled).toBe(false);

    finish();
    await flushPromises();
    expect(settled).toBe(true);
  });
});
