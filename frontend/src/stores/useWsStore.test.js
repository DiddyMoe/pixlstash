// useWsStore across a sign-out.
//
// isUploadInProgress makes the socket take every `picture_imported` as the
// echo of this tab's own upload. An upload in flight at sign-out never reports
// its end to this session, so the flag has to go with the session, or the next
// one drops its imports.

import { beforeEach, describe, it, expect, vi } from "vitest";
import { setActivePinia, createPinia } from "pinia";

const { axiosInstance } = vi.hoisted(() => ({
  axiosInstance: {
    get: vi.fn().mockResolvedValue({ data: {} }),
    post: vi.fn().mockResolvedValue({ data: {} }),
    patch: vi.fn().mockResolvedValue({ data: {} }),
    put: vi.fn().mockResolvedValue({ data: {} }),
    delete: vi.fn().mockResolvedValue({ data: {} }),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
  },
}));

// The real apiClient owns `notifySessionReset`, the chokepoint the store
// registers on (as in sessionReset.test.js). Only the transport is stubbed.
vi.mock("axios", () => ({ default: { create: () => axiosInstance } }));

import { notifySessionReset } from "../utils/apiClient";
import { useWsStore } from "./useWsStore";

beforeEach(() => {
  setActivePinia(createPinia());
});

describe("useWsStore on a session reset", () => {
  it("forgets an upload the old session had in flight", () => {
    const store = useWsStore();
    store.isUploadInProgress = true;

    notifySessionReset("test");

    expect(store.isUploadInProgress).toBe(false);
  });

  it("leaves an upload alone while the session lasts", () => {
    const store = useWsStore();
    store.isUploadInProgress = true;
    expect(store.isUploadInProgress).toBe(true);
  });
});
