// The sidebar's top-level category counts, and the zero they must not swallow.
//
// The Scrapheap badge read `categoryCounts[SCRAPHEAP] || ""` while every
// sibling badge in the same file reads `?? ""`. `||` is falsy-tested, so a
// count of exactly 0 rendered as an empty string: an empty Scrapheap looked
// identical to one whose count had not loaded.
//
// That is the distinction `utils/sidebarCounts.js` exists as a separate module
// to protect ("a real `0` is an answer and still writes", pinned by
// `sidebarCounts.test.js`), and it is the only count binding in the sidebar
// that broke it. `scrapheapIsEmpty` nearby does use a falsy test, but it gates
// a disabled context-menu item, where 0 and "not loaded" genuinely mean the
// same thing; a displayed number is the case where they do not.
//
// Both directions: a real 0 renders "0", a genuinely absent count renders
// nothing, and the sibling badges keep behaving exactly as they did.

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { setActivePinia, createPinia } from "pinia";
import { mount, flushPromises } from "@vue/test-utils";
import { ref } from "vue";

const apiGet = vi.fn();

vi.mock("../../utils/apiClient", async () => {
  const { ref: makeRef } = await import("vue");
  return {
    apiClient: {
      get: (...args) => apiGet(...args),
      post: vi.fn().mockResolvedValue({ data: {} }),
      patch: vi.fn().mockResolvedValue({ data: {} }),
      put: vi.fn().mockResolvedValue({ data: {} }),
      delete: vi.fn().mockResolvedValue({ data: {} }),
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
import { useEntityListsStore } from "../../stores/useEntityListsStore";
import { useFilterStore } from "../../stores/useFilterStore";
import { useProjectStore } from "../../stores/useProjectStore";
import { useUserPrefsStore } from "../../stores/useUserPrefsStore";
import { viewQueryParams } from "../../utils/viewQueryParams";

const SCRAPHEAP_ID = "SCRAPHEAP";
const ALL_ID = "ALL";

function respond(url) {
  const u = String(url ?? "");
  if (u.includes("/characters")) return { data: [] };
  if (u.includes("/projects")) return { data: [] };
  if (u.includes("/picture_sets")) return { data: [] };
  if (u.includes("/summary")) return { data: { image_count: 0 } };
  return { data: [] };
}

function mountSidebar() {
  return mount(SideBar, {
    shallow: true,
    props: { backendUrl: "/api/v1" },
    global: {
      config: {
        compilerOptions: { isCustomElement: (tag) => tag.startsWith("v-") },
      },
    },
  });
}

/**
 * The text of a top-level category row's trailing count badge.
 *
 * Located by the row's visible label, so the assertion follows the row a user
 * reads rather than a DOM index.
 */
function countTextFor(wrapper, label) {
  const row = wrapper
    .findAll(".sidebar-list-item")
    .find((el) => el.find(".sidebar-list-label").exists() &&
      el.find(".sidebar-list-label").text() === label);
  if (!row) return null;
  const badge = row.find(".sidebar-list-count");
  return badge.exists() ? badge.text() : null;
}

/** Put a count straight into the component's own state and re-render. */
async function setCounts(wrapper, counts) {
  wrapper.vm.categoryCounts = { ...wrapper.vm.categoryCounts, ...counts };
  await wrapper.vm.$nextTick();
}

beforeEach(() => {
  setActivePinia(createPinia());
  apiGet.mockReset().mockImplementation((url) => Promise.resolve(respond(url)));
  vi.spyOn(console, "warn").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("the Scrapheap count badge", () => {
  it("renders a real zero", async () => {
    // An empty Scrapheap is a fact worth stating, and it is the state the badge
    // spends most of its life in. Before the fix this rendered "".
    const wrapper = mountSidebar();
    await flushPromises();
    await setCounts(wrapper, { [SCRAPHEAP_ID]: 0 });

    expect(countTextFor(wrapper, "Scrapheap")).toBe("0");

    wrapper.unmount();
  });

  it("renders nothing when the count has not arrived", async () => {
    // The other direction, and the only case that may be blank: `undefined` is
    // "not answered yet", not "zero". A fix that rendered "undefined" or "0"
    // here would be its own regression.
    const wrapper = mountSidebar();
    await flushPromises();
    await setCounts(wrapper, { [SCRAPHEAP_ID]: undefined });

    expect(countTextFor(wrapper, "Scrapheap")).toBe("");

    wrapper.unmount();
  });

  it("still renders a non-zero count unchanged", async () => {
    const wrapper = mountSidebar();
    await flushPromises();
    await setCounts(wrapper, { [SCRAPHEAP_ID]: 42 });

    expect(countTextFor(wrapper, "Scrapheap")).toBe("42");

    wrapper.unmount();
  });
});

describe("the sibling category badges are unchanged", () => {
  it("All media keeps rendering its zero and its number", async () => {
    // These already used `?? ""`. Asserted so the Scrapheap change is visibly a
    // convergence on the existing convention rather than a new rule.
    const wrapper = mountSidebar();
    await flushPromises();

    await setCounts(wrapper, { [ALL_ID]: 0 });
    expect(countTextFor(wrapper, "All media")).toBe("0");

    await setCounts(wrapper, { [ALL_ID]: 7 });
    expect(countTextFor(wrapper, "All media")).toBe("7");

    await setCounts(wrapper, { [ALL_ID]: undefined });
    expect(countTextFor(wrapper, "All media")).toBe("");

    wrapper.unmount();
  });

  it("both badges agree on how they treat zero", async () => {
    // The single assertion the original divergence would have failed.
    const wrapper = mountSidebar();
    await flushPromises();
    await setCounts(wrapper, { [ALL_ID]: 0, [SCRAPHEAP_ID]: 0 });

    expect(countTextFor(wrapper, "Scrapheap")).toBe(
      countTextFor(wrapper, "All media"),
    );

    wrapper.unmount();
  });
});

// A count whose refresh fails keeps the number it last had, marked stale with
// a tooltip saying so, and the sidebar reads again on a backoff until it comes
// back. Before, the catch only logged: the badge kept an old number that
// looked exactly like a current one, and nothing read it again until some
// unrelated refresh happened to.
describe("a count that could not be refreshed", () => {
  const STALE_TIP = "Couldn't refresh this count";
  const COUNT_READS = {
    "All media": "/characters/ALL/summary",
    Scrapheap: "/characters/SCRAPHEAP/summary",
    Favourites: "/pictures/count?favourite=1",
    "All not in a project": "/projects/UNASSIGNED/summary",
  };
  let failing;
  let answer;
  let people;

  function countResponse(url) {
    const u = String(url ?? "");
    if (u === "/characters") return Promise.resolve({ data: people });
    const read = Object.values(COUNT_READS).find((r) => u.startsWith(r));
    if (!read) return Promise.resolve(respond(url));
    if (failing.has(read)) return Promise.reject(new Error("example-503"));
    return Promise.resolve({ data: { image_count: answer, count: answer } });
  }

  function badgeFor(wrapper, label) {
    const row = wrapper
      .findAll(".sidebar-list-item")
      .find((el) => el.find(".sidebar-list-label").text() === label);
    return row.find(".sidebar-list-count");
  }

  function isMarkedStale(badge) {
    const tip = badge.find("tooltip-stub");
    return (
      badge.classes().includes("sidebar-list-count--stale") &&
      tip.exists() &&
      tip.attributes("text") === STALE_TIP
    );
  }

  async function settle() {
    for (let i = 0; i < 5; i += 1) await flushPromises();
  }

  function readsOf(url) {
    return apiGet.mock.calls.filter(([u]) => String(u).startsWith(url)).length;
  }

  beforeEach(() => {
    failing = new Set();
    answer = 12;
    people = [];
    apiGet.mockImplementation(countResponse);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it.each(Object.keys(COUNT_READS))(
    "%s keeps its last number, marked stale, then clears once it reads",
    async (label) => {
      const wrapper = mountSidebar();
      if (label === "All not in a project") wrapper.vm.chooseScope("none");
      wrapper.vm.refreshSidebar();
      await settle();
      expect(badgeFor(wrapper, label).text()).toBe("12");
      expect(isMarkedStale(badgeFor(wrapper, label))).toBe(false);

      failing.add(COUNT_READS[label]);
      answer = 30;
      wrapper.vm.refreshSidebar();
      await settle();
      expect(badgeFor(wrapper, label).text()).toBe("12");
      expect(isMarkedStale(badgeFor(wrapper, label))).toBe(true);

      failing.clear();
      wrapper.vm.refreshSidebar();
      await settle();
      expect(badgeFor(wrapper, label).text()).toBe("30");
      expect(isMarkedStale(badgeFor(wrapper, label))).toBe(false);

      wrapper.unmount();
    },
  );

  it.each([
    ["any", "All media"],
    ["none", "All not in a project"],
  ])(
    "hands the project dropdown its %s count's stale mark",
    async (key, label) => {
      const wrapper = mountSidebar();
      const slot = () => wrapper.findComponent({ name: "SidebarScopeSlot" });
      wrapper.vm.refreshSidebar();
      await settle();
      expect(slot().exists()).toBe(true);
      expect(slot().props("counts").stale).toEqual({ any: false, none: false });

      failing.add(COUNT_READS[label]);
      wrapper.vm.refreshSidebar();
      await settle();
      expect(slot().props("counts").stale).toEqual({
        any: key === "any",
        none: key === "none",
      });
      expect(slot().props("counts")[key]).toBe(12);

      failing.clear();
      wrapper.vm.refreshSidebar();
      await settle();
      expect(slot().props("counts").stale).toEqual({ any: false, none: false });

      wrapper.unmount();
    },
  );

  it("reads again after 2 s, then 4 s, and stops once the count is back", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const read = COUNT_READS["All media"];
    failing.add(read);
    const wrapper = mountSidebar();
    wrapper.vm.refreshSidebar();
    await settle();
    expect(isMarkedStale(badgeFor(wrapper, "All media"))).toBe(true);
    const first = readsOf(read);

    vi.advanceTimersByTime(1999);
    await settle();
    expect(readsOf(read)).toBe(first);
    vi.advanceTimersByTime(1);
    await settle();
    expect(readsOf(read)).toBe(first + 1);

    // Still failing: the next read waits twice as long.
    vi.advanceTimersByTime(3999);
    await settle();
    expect(readsOf(read)).toBe(first + 1);
    failing.clear();
    vi.advanceTimersByTime(1);
    await settle();
    expect(readsOf(read)).toBe(first + 2);
    expect(badgeFor(wrapper, "All media").text()).toBe("12");
    expect(isMarkedStale(badgeFor(wrapper, "All media"))).toBe(false);

    // Back: nothing is scheduled any more.
    vi.advanceTimersByTime(120000);
    await settle();
    expect(readsOf(read)).toBe(first + 2);

    wrapper.unmount();
  });

  it("does not read again once the sidebar is gone", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const read = COUNT_READS.Scrapheap;
    failing.add(read);
    const wrapper = mountSidebar();
    wrapper.vm.refreshSidebar();
    await settle();
    expect(isMarkedStale(badgeFor(wrapper, "Scrapheap"))).toBe(true);
    const reads = readsOf(read);
    wrapper.unmount();

    vi.advanceTimersByTime(60000);
    await settle();
    expect(readsOf(read)).toBe(reads);
  });

  // A read held open until the test lets it go, answering `answer` then.
  function holdRead(read) {
    let release;
    const gate = new Promise((resolve) => {
      release = resolve;
    });
    apiGet.mockImplementation((url) =>
      String(url ?? "").startsWith(read)
        ? gate.then(() => countResponse(url))
        : countResponse(url),
    );
    return () => release();
  }

  it("reads again when the people list moves under a pass and no newer pass starts", async () => {
    // Another surface (the add-to control, the run dialog, the picture picker)
    // refetches the people list while a sidebar pass is waiting on a count.
    // That pass's numbers are discarded as stale; before, nothing read them
    // again, so the stale mark stayed up for good.
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const read = COUNT_READS["All media"];
    failing.add(read);
    const wrapper = mountSidebar();
    wrapper.vm.refreshSidebar();
    await settle();
    expect(isMarkedStale(badgeFor(wrapper, "All media"))).toBe(true);

    failing.clear();
    answer = 30;
    const release = holdRead(read);
    vi.advanceTimersByTime(2000);
    await settle();
    people = [{ id: 7, name: "test-person", project_id: null }];
    await useEntityListsStore().refresh("characters");
    expect(useEntityListsStore().characters.map((c) => c.id)).toEqual([7]);
    release();
    await settle();

    expect(badgeFor(wrapper, "All media").text()).toBe("30");
    expect(isMarkedStale(badgeFor(wrapper, "All media"))).toBe(false);

    wrapper.unmount();
  });

  it("schedules nothing from a pass still in flight when the sidebar goes", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const read = COUNT_READS.Scrapheap;
    failing.add(read);
    const release = holdRead(read);
    const wrapper = mountSidebar();
    wrapper.vm.refreshSidebar();
    await settle();
    wrapper.unmount();
    release();
    await settle();
    const reads = readsOf(read);

    vi.advanceTimersByTime(120000);
    await settle();
    expect(readsOf(read)).toBe(reads);
  });
});

describe("the category badges separate thousands", () => {
  it("as the rest of the app's counts do", async () => {
    const wrapper = mountSidebar();
    await flushPromises();
    await setCounts(wrapper, { [ALL_ID]: 12345, [SCRAPHEAP_ID]: 1000 });

    expect(countTextFor(wrapper, "All media")).toBe("12,345");
    expect(countTextFor(wrapper, "Scrapheap")).toBe("1,000");

    wrapper.unmount();
  });
});

// With hidden tags on, the grid and the Details panel leave hidden pictures
// out of every place, so the sidebar's place counts do too: each read sends
// what `viewQueryParams` sends for the view its row opens, its scope aside (the
// summary's path names it).
describe("the place counts follow the hidden-tags preference", () => {
  // `filters` and `project` are what the row's grid adds to All media's view;
  // `scope` is what the read's path says instead of a parameter.
  const PLACES = {
    "All media": {
      isRead: (u) => u === "/characters/ALL/summary",
      view: ALL_ID,
      scope: [],
    },
    Scrapheap: {
      isRead: (u) => u === "/characters/SCRAPHEAP/summary",
      view: SCRAPHEAP_ID,
      scope: ["only_deleted=true"],
    },
    // The Inbox's reads are `/pictures/count` too; this one is the heart's.
    Favourites: {
      isRead: (u) =>
        u.startsWith("/pictures/count?") &&
        new URLSearchParams(u.split("?")[1]).get("favourite") === "1",
      view: ALL_ID,
      filters: { favouriteOnlyFilter: true },
      scope: [],
    },
    "All not in a project": {
      isRead: (u) => u === "/projects/UNASSIGNED/summary",
      view: ALL_ID,
      project: { projectViewMode: "project", selectedProjectId: null },
      scope: ["project_id=UNASSIGNED"],
    },
  };

  const pairsOf = (params) =>
    [...new URLSearchParams(params ?? {}).entries()].map(
      ([key, value]) => `${key}=${value}`,
    );

  /** What a read asked for: its URL's query and its `params`, sorted. */
  function sentPairs(url, options) {
    const [, query] = String(url ?? "").split("?");
    return [...pairsOf(query), ...pairsOf(options?.params)].sort();
  }

  async function settle() {
    for (let i = 0; i < 5; i += 1) await flushPromises();
  }

  function lastSent(isRead) {
    const call = apiGet.mock.calls.filter(([u]) => isRead(String(u))).at(-1);
    return sentPairs(...call);
  }

  function badgeFor(wrapper, label) {
    const row = wrapper
      .findAll(".sidebar-list-item")
      .find((el) => el.find(".sidebar-list-label").text() === label);
    return row.find(".sidebar-list-count").text();
  }

  function mountFor(label) {
    const wrapper = mountSidebar();
    if (label === "All not in a project") wrapper.vm.chooseScope("none");
    return wrapper;
  }

  for (const [label, place] of Object.entries(PLACES)) {
    const { isRead, view, scope } = place;
    for (const on of [false, true]) {
      it(`${label} asks for the grid's view, hidden tags ${on ? "on" : "off"}`, async () => {
        useUserPrefsStore().applyTagFilter = on;
        useUserPrefsStore().hiddenTags = on ? ["example-hidden"] : [];
        const wrapper = mountFor(label);
        wrapper.vm.refreshSidebar();
        await settle();

        const expected = viewQueryParams({
          setIds: [],
          characterIds: [],
          selection: { selectedCharacter: view },
          project: place.project ?? useProjectStore(),
          filters: { ...useFilterStore().$state, ...place.filters },
          applyTagFilter: useUserPrefsStore().applyTagFilter,
          referenceFolderId: null,
          filePathPrefix: null,
          importSourceFolder: null,
          inboxPeriod: null,
        });
        expect([...scope, ...lastSent(isRead)].sort()).toEqual(
          pairsOf(expected).sort(),
        );
        expect(expected.has("apply_tag_filter")).toBe(on);

        wrapper.unmount();
      });
    }

    it(`${label} drops a read made before the preference changed`, async () => {
      // The read answers with the number for the setting it was sent under,
      // and the one sent with the preference off is held until it has changed.
      let release;
      const gate = new Promise((resolve) => {
        release = resolve;
      });
      apiGet.mockImplementation((url, options) => {
        if (!isRead(String(url ?? ""))) return Promise.resolve(respond(url));
        const sentOn = sentPairs(url, options).includes("apply_tag_filter=true");
        const answer = sentOn ? 30 : 12;
        const body = { data: { image_count: answer, count: answer } };
        return sentOn ? Promise.resolve(body) : gate.then(() => body);
      });
      const wrapper = mountFor(label);
      wrapper.vm.refreshSidebar();
      await settle();

      // Turned on while the first read is out; that read lands afterwards.
      useUserPrefsStore().hiddenTags = ["example-hidden"];
      useUserPrefsStore().applyTagFilter = true;
      release();
      await settle();

      expect(badgeFor(wrapper, label)).toBe("30");
      expect(lastSent(isRead)).toContain("apply_tag_filter=true");

      wrapper.unmount();
    });
  }
});

// The Scrapheap row's menu disables Empty Scrapheap when its badge reads 0,
// as the grid's own Empty button does. With hidden tags on, that 0 can mean
// "every picture here is hidden", so the disabled item's tooltip must not
// claim the heap is empty.
describe("the Empty Scrapheap menu item", () => {
  function mountWithMenu() {
    return mount(SideBar, {
      shallow: true,
      props: { backendUrl: "/api/v1" },
      global: {
        config: {
          compilerOptions: { isCustomElement: (tag) => tag.startsWith("v-") },
        },
        stubs: { teleport: { template: "<div><slot /></div>" } },
      },
    });
  }

  async function openMenuFor(wrapper, count) {
    await setCounts(wrapper, { [SCRAPHEAP_ID]: count });
    const row = wrapper
      .findAll(".sidebar-list-item")
      .find((el) => el.find(".sidebar-list-label").text() === "Scrapheap");
    await row.trigger("contextmenu");
    const item = wrapper
      .findAll(".sidebar-ctx-menu .ctx-item")
      .find((el) => el.text().includes("Empty Scrapheap"));
    return { item, tip: item.find("tooltip-stub").attributes("text") };
  }

  it.each([
    [false, "Scrapheap is already empty"],
    [true, "No pictures to empty while hidden tags are on"],
  ])(
    "at 0 with hidden tags %s is disabled and says %j",
    async (on, expected) => {
      useUserPrefsStore().applyTagFilter = on;
      const wrapper = mountWithMenu();
      await flushPromises();

      const { item, tip } = await openMenuFor(wrapper, 0);

      expect(item.attributes("disabled")).toBeDefined();
      expect(tip).toBe(expected);

      wrapper.unmount();
    },
  );

  it.each([false, true])(
    "with pictures to empty is enabled and has no tooltip, hidden tags %s",
    async (on) => {
      useUserPrefsStore().applyTagFilter = on;
      const wrapper = mountWithMenu();
      await flushPromises();

      const { item, tip } = await openMenuFor(wrapper, 3);

      expect(item.attributes("disabled")).toBeUndefined();
      expect(tip).toBe("");

      wrapper.unmount();
    },
  );
});
