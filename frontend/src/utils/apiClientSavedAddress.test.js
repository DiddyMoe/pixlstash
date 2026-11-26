// A saved ComfyUI address that no longer passes the backend's check answers
// 409 with one fixed detail. The API client turns that one 409, and no other,
// into a notice that says where to fix it and opens Settings > ComfyUI.

import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

const { axiosInstance, responseInterceptors } = vi.hoisted(() => {
  const responseInterceptors = [];
  return {
    responseInterceptors,
    axiosInstance: {
      get: vi.fn().mockResolvedValue({ data: {} }),
      post: vi.fn().mockResolvedValue({ data: {} }),
      interceptors: {
        request: { use: vi.fn() },
        response: {
          use: vi.fn((onFulfilled, onRejected) =>
            responseInterceptors.push(onRejected),
          ),
        },
      },
    },
  };
});

vi.mock("axios", () => ({ default: { create: () => axiosInstance } }));

import { setSettingsOpener } from "./apiClient";
import { errorDetail, errorMessage, requestFailure } from "./apiError";
import { useNoticeStore } from "../stores/useNoticeStore";

// The backend's `SAVED_ADDRESS_INVALID` (pixlstash/utils/comfyui_url.py).
const SAVED_ADDRESS_INVALID =
  "The saved ComfyUI address is not valid, so nothing was sent to it. " +
  "Set the address again in Settings.";

/**
 * Run a rejected response through every response interceptor, as axios does.
 * @returns the error a caller's catch then receives.
 */
async function reject(status, detail) {
  const error = { response: { status, data: { detail } }, config: { url: "/x" } };
  for (const onRejected of responseInterceptors) {
    await expect(onRejected(error)).rejects.toBe(error);
  }
  return error;
}

describe("the saved-address 409 prompt", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    setSettingsOpener(null);
  });

  it("shows one card that opens Settings on the ComfyUI entry", async () => {
    const opened = [];
    setSettingsOpener((tab) => opened.push(tab));

    await reject(409, SAVED_ADDRESS_INVALID);
    await reject(409, SAVED_ADDRESS_INVALID);

    const { notices } = useNoticeStore();
    expect(notices).toHaveLength(1);
    const [notice] = notices;
    expect(notice.level).toBe("error");
    expect(notice.count).toBe(2);
    expect(notice.text).toContain("Settings > ComfyUI");
    expect(notice.action.label).toBe("Open Settings");

    notice.action.handler();
    expect(opened).toEqual(["comfyui"]);
  });

  it("names the place without a dead button before the shell registers", async () => {
    await reject(409, SAVED_ADDRESS_INVALID);

    const [notice] = useNoticeStore().notices;
    expect(notice.text).toContain("Settings > ComfyUI");
    expect(notice.action).toBeNull();
  });

  it.each([
    [409, "A run with that name already exists."],
    [409, undefined],
    [400, SAVED_ADDRESS_INVALID],
    [502, SAVED_ADDRESS_INVALID],
  ])("shows nothing for a %i with another detail or status (%s)", async (status, detail) => {
    setSettingsOpener(() => {});
    await reject(status, detail);
    expect(useNoticeStore().notices).toHaveLength(0);
  });
});

// The card says why; a caller's catch says what failed, and must not say the
// card's sentence a second time beside it.
describe("a caller's catch after the saved-address 409", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    setSettingsOpener(null);
  });

  it("gets no server sentence, so its own copy is what shows", async () => {
    const error = await reject(409, SAVED_ADDRESS_INVALID);

    expect(errorDetail(error)).toBe("");
    expect(errorMessage(error, "Could not install the pack.")).toBe(
      "Could not install the pack.",
    );
    expect(requestFailure(error)).toMatchObject({ kind: "conflict", detail: "" });
  });

  it("keeps another 409's sentence, which no card has shown", async () => {
    const error = await reject(409, "A run with that name already exists.");

    expect(errorDetail(error)).toBe("A run with that name already exists.");
    expect(errorMessage(error, "Could not save.")).toBe(
      "A run with that name already exists.",
    );
    expect(requestFailure(error).detail).toBe(
      "A run with that name already exists.",
    );
  });
});
