// The sidebar's three deletes (a project, people, picture sets) ask through
// the app's confirm dialog, not the browser's `window.confirm`: a native box
// is unstyled, names no action on its button ("OK"), and blocks the page. Each
// asks before it deletes, deletes nothing on Cancel, and deletes on confirm.

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { setActivePinia, createPinia } from "pinia";
import { mount, flushPromises } from "@vue/test-utils";
import { ref } from "vue";

const apiGet = vi.fn();
const apiDelete = vi.fn();

vi.mock("../../utils/apiClient", async () => {
  const { ref: makeRef } = await import("vue");
  return {
    apiClient: {
      get: (...args) => apiGet(...args),
      post: vi.fn().mockResolvedValue({ data: {} }),
      patch: vi.fn().mockResolvedValue({ data: {} }),
      put: vi.fn().mockResolvedValue({ data: {} }),
      delete: (...args) => apiDelete(...args),
    },
    onSessionReset: () => () => {},
    activateShareToken: vi.fn(),
    appendShareToken: (url) => url,
    checkLoginStatus: vi.fn(),
    checkSession: vi.fn(),
    isAuthenticated: makeRef(true),
    isReadOnly: makeRef(false),
    sessionContext: makeRef(null),
    login: vi.fn(),
    logout: vi.fn(),
    newOperationBatchId: () => "cli-test",
    operationBatchHeaders: () => undefined,
    setRequestClientId: vi.fn(),
    notifySessionReset: vi.fn(),
    toBackendWebSocketUrl: () => "",
    API_BASE_URL: "/api/v1",
  };
});

vi.mock("vuetify/components", async () => {
  const { vuetifyComponentStubs } = await import("../../testing/vuetifyStubs");
  return vuetifyComponentStubs();
});

vi.mock("vue-router", () => ({
  useRoute: () => ({ query: {}, params: {}, path: "/", name: "all-pictures" }),
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    currentRoute: ref({ query: {} }),
  }),
}));

import SideBar from "./SideBar.vue";
import {
  activeConfirm,
  registerConfirmHost,
  resolveConfirm,
  unregisterConfirmHost,
} from "../../composables/useConfirm";

const PROJECT = { id: 3, name: "Book" };
const PERSON = { id: 5, name: "Ada", project_id: null };
const SET = { id: 9, name: "Holiday", project_id: null };

function respond(url) {
  const u = String(url ?? "");
  if (u.includes("/summary")) return { data: { image_count: 0 } };
  if (u.startsWith("/characters")) return { data: [PERSON] };
  if (u.startsWith("/projects")) return { data: [PROJECT] };
  if (u.startsWith("/picture_sets")) return { data: [SET] };
  return { data: [] };
}

async function mountSidebar() {
  const wrapper = mount(SideBar, {
    shallow: true,
    props: { backendUrl: "/api/v1" },
    global: {
      config: {
        compilerOptions: { isCustomElement: (tag) => tag.startsWith("v-") },
      },
    },
  });
  wrapper.vm.refreshSidebar();
  for (let i = 0; i < 5; i += 1) await flushPromises();
  return wrapper;
}

const DELETES = [
  {
    what: "a project",
    run: (vm) => vm.deleteProjectById(PROJECT),
    url: "/projects/3",
    asked: {
      title: 'Delete project "Book"?',
      message: "Its people, sets and pictures stay in the library.",
      warning: "Its attachments are deleted.",
      confirmLabel: "Delete project",
      danger: true,
    },
  },
  {
    what: "a character",
    run: (vm) => vm.deleteCharactersByIds([PERSON.id]),
    url: "/characters/5",
    asked: {
      title: 'Delete character "Ada"?',
      message:
        "Their pictures stay in the library, and the faces named as them go back to unnamed.",
      confirmLabel: "Delete character",
      danger: true,
    },
  },
  {
    what: "a picture set",
    run: (vm) => vm.deleteSetsByIds([SET.id]),
    url: "/picture_sets/9",
    asked: {
      title: 'Delete picture set "Holiday"?',
      message: "Its pictures leave the set and stay in the library.",
      warning: "",
      confirmLabel: "Delete set",
      danger: true,
    },
  },
];

let nativeConfirm;

beforeEach(() => {
  setActivePinia(createPinia());
  window.localStorage.clear();
  apiGet.mockReset().mockImplementation((url) => Promise.resolve(respond(url)));
  apiDelete.mockReset().mockResolvedValue({ data: {} });
  nativeConfirm = vi.spyOn(window, "confirm").mockReturnValue(true);
  vi.spyOn(console, "warn").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});
  registerConfirmHost();
});

afterEach(() => {
  unregisterConfirmHost();
  vi.restoreAllMocks();
});

describe.each(DELETES)("deleting $what", ({ run, url, asked }) => {
  it("asks in the app's dialog, naming the action, and deletes nothing yet", async () => {
    const wrapper = await mountSidebar();
    const pending = run(wrapper.vm);
    await flushPromises();

    expect(activeConfirm.value?.options).toMatchObject(asked);
    expect(nativeConfirm).not.toHaveBeenCalled();
    expect(apiDelete).not.toHaveBeenCalled();

    resolveConfirm(false);
    await pending;
    wrapper.unmount();
  });

  it("deletes nothing on Cancel", async () => {
    const wrapper = await mountSidebar();
    const pending = run(wrapper.vm);
    await flushPromises();
    resolveConfirm(false);
    await pending;
    await flushPromises();

    expect(apiDelete).not.toHaveBeenCalled();
    wrapper.unmount();
  });

  it("deletes once confirmed", async () => {
    const wrapper = await mountSidebar();
    const pending = run(wrapper.vm);
    await flushPromises();
    resolveConfirm(true);
    await pending;
    await flushPromises();

    expect(apiDelete.mock.calls.map(([u]) => u)).toEqual([url]);
    wrapper.unmount();
  });
});

describe("deleting several at once", () => {
  it("counts them in the question and on the button", async () => {
    const wrapper = await mountSidebar();
    const pending = wrapper.vm.deleteSetsByIds([SET.id, 10]);
    await flushPromises();

    expect(activeConfirm.value?.options).toMatchObject({
      title: "Delete 2 picture sets?",
      message: "Their pictures leave the sets and stay in the library.",
      confirmLabel: "Delete 2 sets",
    });

    resolveConfirm(false);
    await pending;
    wrapper.unmount();
  });
});
