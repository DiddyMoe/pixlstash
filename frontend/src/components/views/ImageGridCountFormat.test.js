// Counts the grid shows to people read as grouped figures ("1,200"), the way
// every other count in the app does (utils/filterChips.js formatCount); the
// singular/plural choice still reads the raw number. The plugin-run and
// export sentences are pinned in their own files; this one holds the pieces
// that need a rendered button to read.

import { describe, it, expect, beforeEach, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { setActivePinia, createPinia } from "pinia";
import { nextTick, ref } from "vue";
import { useWsStore } from "../../stores/useWsStore.js";

const apiGet = vi.fn();
const apiPost = vi.fn();

vi.mock("../../utils/apiClient", async () => {
  const { ref: makeRef, computed: makeComputed } = await import("vue");
  return {
    onSessionReset: () => () => {},
    apiClient: {
      get: (...args) => apiGet(...args),
      post: (...args) => apiPost(...args),
      patch: vi.fn(),
      put: vi.fn(),
      delete: vi.fn(),
    },
    activateShareToken: vi.fn(),
    appendShareToken: (url) => url,
    checkLoginStatus: vi.fn(),
    checkSession: vi.fn(),
    isAuthenticated: makeRef(true),
    isReadOnly: makeComputed(() => false),
    login: vi.fn(),
    logout: vi.fn(),
    sessionContext: makeRef({ scope: "ALL" }),
    setRequestClientId: vi.fn(),
    API_BASE_URL: "/api/v1",
  };
});

vi.mock("vuetify/components", async () => {
  const { vuetifyComponentStubs } = await import("../../testing/vuetifyStubs");
  return vuetifyComponentStubs();
});

vi.mock("vue-router", () => ({
  useRoute: () => ({ query: {}, params: {}, path: "/", name: "grid" }),
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    currentRoute: ref({ query: {} }),
  }),
}));

import ImageGrid from "./ImageGrid.vue";

beforeEach(() => {
  setActivePinia(createPinia());
  apiGet.mockReset();
  apiPost.mockReset();
  apiGet.mockResolvedValue({ data: { pictures: [], count: 0, total: 0 } });
  apiPost.mockResolvedValue({ data: {} });
});

function mountGrid() {
  return mount(ImageGrid, {
    shallow: true,
    global: {
      // The pill's label is its slot; a bare shallow stub drops it.
      stubs: { AppButton: { template: "<button><slot /></button>" } },
      config: {
        compilerOptions: { isCustomElement: (tag) => tag.startsWith("v-") },
      },
    },
    props: { backendUrl: "/api/v1" },
  });
}

describe("ImageGrid - counts shown as grouped figures", () => {
  it("groups the new-pictures pill's count", async () => {
    const wrapper = mountGrid();
    useWsStore().addPendingExternalImportIds(
      Array.from({ length: 1200 }, (_, i) => i + 1),
    );
    await nextTick();

    const pill = wrapper.get('[data-testid="pending-imports-pill"]');
    expect(pill.text().replace(/\s+/g, " ")).toBe("1,200 new pictures - Load");
    wrapper.unmount();
  });

  it("still reads the raw number for the singular", async () => {
    const wrapper = mountGrid();
    useWsStore().addPendingExternalImportIds([7]);
    await nextTick();

    const pill = wrapper.get('[data-testid="pending-imports-pill"]');
    expect(pill.text().replace(/\s+/g, " ")).toBe("1 new picture - Load");
    wrapper.unmount();
  });
});
