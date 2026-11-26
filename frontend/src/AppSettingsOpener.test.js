// The saved-address notice (utils/apiClient.js) offers "Open Settings" only once
// the app shell has handed it an opener. App.vue owns the settings dialog, so it
// registers `openSettingsDialog` when it mounts; the notice's button then opens
// Settings on the ComfyUI entry (`comfyui`, UserSettingsDialog's nav id, which
// UserSettingsDialog.test.js proves lands on the ComfyUI pane).
//
// The real API client runs here over a stand-in axios, so the 409 goes through
// the same response interceptor a page load would.

import { describe, it, expect, beforeEach, vi } from "vitest";
import { setActivePinia, createPinia } from "pinia";
import { mount, flushPromises } from "@vue/test-utils";

const nav = vi.hoisted(() => ({ route: null }));
vi.mock("vue-router", async () => {
  const { reactive } = await vi.importActual("vue");
  const { vi: vitest } = await import("vitest");
  nav.route = reactive({ path: "/", name: "all-pictures", params: {}, query: {} });
  return {
    useRoute: () => nav.route,
    useRouter: () => ({
      push: vitest.fn().mockResolvedValue(undefined),
      replace: vitest.fn().mockResolvedValue(undefined),
      currentRoute: { value: nav.route },
    }),
  };
});

vi.mock("vuetify", () => ({
  useTheme: () => ({ global: { name: { value: "dark" } } }),
}));

const { axiosInstance, responseInterceptors } = vi.hoisted(() => {
  const responseInterceptors = [];
  const answer = () => Promise.resolve({ data: {} });
  return {
    responseInterceptors,
    axiosInstance: {
      get: answer,
      post: answer,
      patch: answer,
      put: answer,
      delete: answer,
      interceptors: {
        request: { use: () => {} },
        response: {
          use: (onFulfilled, onRejected) => responseInterceptors.push(onRejected),
        },
      },
    },
  };
});
vi.mock("axios", () => ({ default: { create: () => axiosInstance } }));

vi.mock("./api/config", () => ({
  getUserConfig: vi.fn(async () => ({
    data: {
      sort: "date",
      thumbnail: 256,
      check_for_updates: false,
      telemetry_send_install_id: false,
      telemetry_consent_prompted: true,
    },
  })),
  patchUserConfig: vi.fn(async () => ({ data: {} })),
}));

vi.mock("./api/telemetry", () => ({
  getInstallId: vi.fn(async () => ({ is_new_install: false })),
}));

import App from "./App.vue";
import { setSettingsOpener } from "./utils/apiClient";
import { useNoticeStore } from "./stores/useNoticeStore";

// The backend's `SAVED_ADDRESS_INVALID` (pixlstash/utils/comfyui_url.py).
const SAVED_ADDRESS_INVALID =
  "The saved ComfyUI address is not valid, so nothing was sent to it. " +
  "Set the address again in Settings.";

// What App.vue's `openSettingsDialog(tab)` hands the sidebar, which owns the
// dialog and passes the tab on as its `initial-tab`.
const settingsOpened = [];
const SideBarStub = {
  name: "SideBar",
  template: "<div />",
  methods: {
    refreshSidebar() {},
    async offerLoosePictures() {},
    openSettingsDialog(tab) {
      settingsOpened.push(tab);
    },
    startLocalImport() {},
  },
};

async function mountApp() {
  const wrapper = mount(App, {
    shallow: true,
    global: {
      stubs: { SideBar: SideBarStub },
      config: {
        compilerOptions: { isCustomElement: (tag) => tag.startsWith("v-") },
      },
    },
  });
  await flushPromises();
  return wrapper;
}

/** Run a rejected response through every response interceptor, as axios does. */
async function reject(status, detail) {
  const error = { response: { status, data: { detail } }, config: { url: "/x" } };
  for (const onRejected of responseInterceptors) {
    await expect(onRejected(error)).rejects.toBe(error);
  }
}

beforeEach(() => {
  window.localStorage.clear();
  setActivePinia(createPinia());
  setSettingsOpener(null);
  settingsOpened.length = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({ json: async () => ({ version: "2.0.0" }) })),
  );
  vi.stubGlobal("WebSocket", class {
    static OPEN = 1;
    constructor() {
      this.readyState = 1;
    }
    send() {}
    close() {}
  });
  vi.spyOn(console, "warn").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});
});

describe("the saved-address notice, once the app is up", () => {
  it("opens Settings on the ComfyUI entry", async () => {
    const wrapper = await mountApp();

    await reject(409, SAVED_ADDRESS_INVALID);

    const notice = useNoticeStore().notices.find(
      (n) => n.key === "saved-comfyui-address-invalid",
    );
    // Wrong if the shell never registered: the notice then has no button.
    expect(notice.action?.label).toBe("Open Settings");
    notice.action.handler();
    expect(settingsOpened).toEqual(["comfyui"]);
    wrapper.unmount();
  });
});
