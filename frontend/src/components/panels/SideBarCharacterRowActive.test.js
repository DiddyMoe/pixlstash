// The character row's active highlight, on the one path that reaches its
// single-selection fallback.
//
// The row's active class is
//
//   selectedCharacterIdSet.size > 0
//     ? selectedCharacterIdSet.has(char.id)
//     : selectionStore.selectedCharacter === char.id
//
// and the `else` half used to read a BARE `selectedCharacter`, an identifier
// that is not declared in the component's `<script setup>` at all. It resolved
// to `undefined`, so the comparison was always false and the branch could never
// light a row.
//
// That branch is not dead code. Every ordinary selection path writes the scalar
// and the id list together, so the set is non-empty and the ternary takes its
// first arm:
//
//   * `SideBar.selectCharacter` emits `ids: [numId]` for a real person, and
//     `useAppNavigation` writes both (`useAppNavigation.js:112-113`);
//   * the route parser derives `ids` from `params.id` when `?ids=` is absent
//     (`useViewStore.parseIds`), so `/character/7` yields `[7]` as well.
//
// The exception is `App.vue:391`: a share token scoped to a CHARACTER sets
// `selectionStore.selectedCharacter = ctx.resource_id` and never touches
// `selectedCharacterIds`, which stays at its `[]` default. That session renders
// the People section (it is gated `scopedResourceType !== 'picture_set'`), so
// the one person the token exists for was the one row that could never show as
// selected. That is the state this file pins.

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { shallowRef } from "vue";
import { setActivePinia, createPinia } from "pinia";
import { mount, flushPromises } from "@vue/test-utils";

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

// Mutable, so a test can put the sidebar on a route other than the default.
const nav = vi.hoisted(() => ({ route: null }));
vi.mock("vue-router", async () => {
  const { reactive } = await vi.importActual("vue");
  const { vi: vitest } = await import("vitest");
  nav.route = reactive({
    query: {},
    params: {},
    path: "/",
    name: "all-pictures",
  });
  return {
    useRoute: () => nav.route,
    useRouter: () => ({
      push: vitest.fn(),
      replace: vitest.fn(),
      currentRoute: { value: { query: {} } },
    }),
  };
});

import { isReadOnly, sessionContext } from "../../utils/apiClient";
import { useSidebarRefresh } from "../../composables/useSidebarRefresh";
import { useFilterStore } from "../../stores/useFilterStore";
import { useSelectionStore } from "../../stores/useSelectionStore";
import { useSidebarStore } from "../../stores/useSidebarStore";
import { useEntityListsStore } from "../../stores/useEntityListsStore";
import unknownPerson from "../../assets/unknown-person.png";
import SideBar from "./SideBar.vue";

/**
 * The element's own tooltip text: its direct `Tooltip` child, stubbed by the
 * shallow mount. `undefined` when it has none or the text is empty, because an
 * empty text is a disabled tip.
 */
function tipText(el) {
  const tip = el
    .findAll("tooltip-stub")
    .find((t) => t.element.parentElement === el.element);
  return tip?.attributes("text") || undefined;
}

const ADA = { id: 7, name: "Ada", image_count: 3, project_image_count: 3 };
const GRACE = { id: 9, name: "Grace", image_count: 5, project_image_count: 5 };
const CHARACTERS = [ADA, GRACE];

function respond(url) {
  const u = String(url ?? "");
  if (u.includes("/characters")) return { data: CHARACTERS };
  if (u.includes("/projects")) return { data: [] };
  if (u.includes("/picture_sets")) return { data: [] };
  if (u.includes("/summary")) return { data: { image_count: 3 } };
  return { data: [] };
}

async function mountSidebar({ docked = false, teleport = docked } = {}) {
  if (docked) useSidebarStore().sidebarDocked = true;
  const options = {
    shallow: true,
    props: { backendUrl: "/api/v1" },
    global: {
      config: {
        compilerOptions: { isCustomElement: (tag) => tag.startsWith("v-") },
      },
      ...(teleport ? { stubs: { teleport: false } } : {}),
    },
  };
  if (docked) options.attachTo = document.body;
  const wrapper = mount(SideBar, options);
  wrapper.vm.refreshSidebar();
  for (let i = 0; i < 5; i += 1) await flushPromises();
  return wrapper;
}

/**
 * Is the People-section row for `name` rendered with the active class?
 *
 * Located by the row's own tooltip, which the template builds from the person's
 * name, so the assertion is tied to the row a user would actually click rather
 * than to a DOM position. Returns `null` when no such row was rendered, which
 * would otherwise read as a passing "not active".
 */
function characterRowIsActive(wrapper, name) {
  const row = wrapper
    .findAll(".sidebar-list-item")
    .find((el) => String(tipText(el) ?? "").startsWith(`${name} (`));
  if (!row) return null;
  return row.classes().includes("active");
}

function characterRow(wrapper, name) {
  return wrapper
    .findAll(".sidebar-list-item")
    .find((el) => String(tipText(el) ?? "").startsWith(`${name} (`));
}

let consoleWarn;

beforeEach(() => {
  setActivePinia(createPinia());
  isReadOnly.value = false;
  sessionContext.value = null;
  apiGet.mockReset().mockImplementation((url) => Promise.resolve(respond(url)));
  consoleWarn = vi.spyOn(console, "warn").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
  document.body.innerHTML = "";
});

describe("the character row's single-selection fallback", () => {
  it("exposes selection state and keyboard activation semantics", async () => {
    const selection = useSelectionStore();
    const wrapper = await mountSidebar();
    selection.selectedCharacter = ADA.id;
    selection.selectedCharacterIds = [];
    await wrapper.vm.$nextTick();

    const adaRow = characterRow(wrapper, ADA.name);
    const graceRow = characterRow(wrapper, GRACE.name);
    expect(adaRow.attributes()).toMatchObject({
      role: "button",
      tabindex: "0",
      "aria-pressed": "true",
    });
    expect(graceRow.attributes("aria-pressed")).toBe("false");

    await graceRow.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("select-character")?.at(-1)?.[0]).toMatchObject({
      id: GRACE.id,
      ids: [GRACE.id],
    });

    wrapper.unmount();
  });

  it("lights the row when only the scalar selection is set", async () => {
    // Exactly the `App.vue:391` share-token state: the scalar names a real
    // person, the id list was never written. Before the fix the bare
    // `selectedCharacter` was `undefined` here and no row could go active.
    const selection = useSelectionStore();
    const wrapper = await mountSidebar();
    selection.selectedCharacter = ADA.id;
    selection.selectedCharacterIds = [];
    await wrapper.vm.$nextTick();

    expect(characterRowIsActive(wrapper, ADA.name)).toBe(true);
    // And only that row.
    expect(characterRowIsActive(wrapper, GRACE.name)).toBe(false);

    wrapper.unmount();
  });

  it("does not light a row when the scalar names nobody", async () => {
    // The other direction: "All Pictures" is not a person, so no person row may
    // claim the highlight. A fallback that matched loosely would light one.
    const selection = useSelectionStore();
    const wrapper = await mountSidebar();
    selection.selectedCharacter = "ALL";
    selection.selectedCharacterIds = [];
    await wrapper.vm.$nextTick();

    expect(characterRowIsActive(wrapper, ADA.name)).toBe(false);
    expect(characterRowIsActive(wrapper, GRACE.name)).toBe(false);

    wrapper.unmount();
  });

  it("still prefers the id set when the ordinary paths populate it", async () => {
    // The first arm of the ternary is what every normal selection uses; the fix
    // must not disturb it. Scalar and set deliberately disagree here so the
    // assertion can only pass if the SET won.
    const selection = useSelectionStore();
    const wrapper = await mountSidebar();
    selection.selectedCharacter = ADA.id;
    selection.selectedCharacterIds = [GRACE.id];
    await wrapper.vm.$nextTick();

    expect(characterRowIsActive(wrapper, GRACE.name)).toBe(true);
    expect(characterRowIsActive(wrapper, ADA.name)).toBe(false);

    wrapper.unmount();
  });

  it("lights every row of a genuine multi-selection", async () => {
    const selection = useSelectionStore();
    const wrapper = await mountSidebar();
    selection.selectedCharacter = ADA.id;
    selection.selectedCharacterIds = [ADA.id, GRACE.id];
    await wrapper.vm.$nextTick();

    expect(characterRowIsActive(wrapper, ADA.name)).toBe(true);
    expect(characterRowIsActive(wrapper, GRACE.name)).toBe(true);

    wrapper.unmount();
  });

  it("resolves the identifier, so no render warning is logged", async () => {
    // The cheap signal the bug produced: Vue warns once per render on a
    // template reference it cannot resolve.
    const selection = useSelectionStore();
    const wrapper = await mountSidebar();
    selection.selectedCharacter = ADA.id;
    selection.selectedCharacterIds = [];
    await wrapper.vm.$nextTick();

    const resolutionWarnings = consoleWarn.mock.calls.filter(([msg]) =>
      String(msg ?? "").includes("selectedCharacter"),
    );
    expect(resolutionWarnings).toEqual([]);

    wrapper.unmount();
  });
});

describe("the shelf's two routes are one destination", () => {
  // `SideBar` keeps its own `isModelsView`, separate from the one
  // `useAppNavigation` exports, and adding the runs tab broke this half
  // silently: on `/models/runs` the underlying picture selection lit a row of
  // its own, which is the two-active-destinations defect the guard exists to
  // prevent. Both now read `MODEL_SHELF_ROUTES`, so the pair cannot drift
  // apart again. (That the runs route lights the Models workspace is the title
  // bar's now: TitleBar.test.js.)
  it("does not let a picture selection light a second destination there", async () => {
    nav.route.name = "models-runs";
    useSelectionStore().selectedCharacter = ADA.id;
    const wrapper = await mountSidebar();
    const active = wrapper.findAll(".sidebar-list-item.active");
    const labels = active.map((el) => el.text());
    expect(labels.filter((l) => l.includes("Ada"))).toHaveLength(0);
    wrapper.unmount();
    nav.route.name = "all-pictures";
  });
});

describe("the docked rail's node flyout (#2448)", () => {
  it("uses menu semantics, arrow keys, and Escape back to its button", async () => {
    const wrapper = await mountSidebar({ docked: true });
    const trigger = wrapper
      .findAll(".sidebar-collapsed-item")
      .find((b) => b.attributes("aria-label") === "Any project");

    expect(trigger.element.tagName).toBe("BUTTON");
    expect(trigger.attributes()).toMatchObject({
      "aria-haspopup": "menu",
      "aria-expanded": "false",
    });

    await trigger.trigger("click");
    await flushPromises();

    const menu = document.querySelector(".sidebar-rail-flyout");
    expect(menu.getAttribute("role")).toBe("menu");
    expect(menu.getAttribute("aria-label")).toBe("Any project");
    // Limit view to (radios), then the people of its scope (Any project:
    // everyone) as plain items.
    const people = Array.from(menu.querySelectorAll('[role="menuitem"]'));
    const labels = people.map((item) => item.querySelector(".ctx-label-text").textContent);
    expect(labels).toEqual(
      expect.arrayContaining(["Ada", "Grace"]),
    );
    const items = Array.from(menu.querySelectorAll(".ctx-item"));
    expect(items[0].getAttribute("role")).toBe("menuitemradio");
    // Every row carries a glyph and every heading after the first a rule, so
    // the groups do not run together as one flat list.
    expect(items.every((item) => item.querySelector(".ctx-icon")?.textContent.trim())).toBe(true);
    const later = Array.from(menu.querySelectorAll(".section-label")).slice(1);
    expect(later.length).toBeGreaterThan(0);
    expect(
      later.every((label) => label.previousElementSibling?.classList.contains("ctx-sep")),
    ).toBe(true);
    expect(document.activeElement).toBe(items[0]);

    items[0].dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "ArrowDown",
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.activeElement).toBe(items[1]);

    items[1].dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Escape",
        bubbles: true,
        cancelable: true,
      }),
    );
    await flushPromises();
    expect(document.querySelector(".sidebar-rail-flyout")).toBeNull();
    expect(document.activeElement).toBe(trigger.element);

    wrapper.unmount();
  });
});

describe("Create with LoRA… in a person's or a set's context menu", () => {
  // The row opened the Run popup on anybody, and the popup then said there
  // was no LoRA to run. It is now live only once a LoRA is known to be
  // attached to the person or set the menu was opened on.
  const PORTRAITS = { id: 4, name: "Portraits", picture_count: 2 };
  const LORA = { sha256: "a".repeat(64), name: "ada-v1" };
  const NO_LORA =
    "No LoRA is attached to Ada. Assign one from the Models shelf.";

  /** Right-click `row` and return the menu's Create with LoRA… button. */
  async function loraItem(row) {
    await row.trigger("contextmenu");
    await flushPromises();
    return loraButton();
  }

  /**
   * The row as it is now: re-queried, since a re-render may replace it. The
   * menu is teleported to <body>, so it is not under the wrapper.
   */
  function loraButton() {
    return Array.from(
      document.querySelectorAll(".sidebar-ctx-menu .ctx-item"),
    ).find((el) => /Create with LoRA|No LoRA attached/.test(el.textContent));
  }

  function itemTip(el) {
    return el.querySelector("tooltip-stub")?.getAttribute("text");
  }

  /** Serve `/adapters` from `answer(params)`; everything else as before. */
  function serveAdapters(answer) {
    apiGet.mockImplementation((url, config) => {
      const u = String(url ?? "");
      if (u.includes("/adapters")) {
        return Promise.resolve(answer(config?.params ?? {})).then((rows) => ({
          data: { adapters: rows },
        }));
      }
      if (u.includes("/picture_sets")) {
        return Promise.resolve({ data: [PORTRAITS] });
      }
      return Promise.resolve(respond(url));
    });
  }

  beforeEach(() => {
    useFilterStore().comfyuiConfigured = true;
  });

  it("is disabled, with the reason, for a person with no LoRA", async () => {
    serveAdapters(() => []);
    const wrapper = await mountSidebar({ teleport: true });
    const item = await loraItem(characterRow(wrapper, ADA.name));

    expect(item.disabled).toBe(true);
    // Said in the row itself, not only in a tooltip a keyboard cannot reach.
    expect(item.textContent).toContain("No LoRA attached");
    expect(itemTip(item)).toBe(NO_LORA);
    // Both attachable file kinds were asked about, for this person.
    const asked = apiGet.mock.calls
      .filter(([url]) => String(url).includes("/adapters"))
      .map(([, config]) => config.params);
    expect(asked).toEqual([
      { file_kind: "adapter", character_id: ADA.id },
      { file_kind: "unknown", character_id: ADA.id },
    ]);
    wrapper.unmount();
  });

  it("is live for a person with a LoRA attached", async () => {
    serveAdapters((params) => (params.file_kind === "adapter" ? [LORA] : []));
    const wrapper = await mountSidebar({ teleport: true });
    const item = await loraItem(characterRow(wrapper, ADA.name));

    expect(item.disabled).toBe(false);
    expect(item.textContent).toContain("Create with LoRA…");
    expect(itemTip(item)).toBe("Run a workflow with Ada's LoRA");
    wrapper.unmount();
  });

  it("counts a LoRA the shelf has not classified", async () => {
    serveAdapters((params) => (params.file_kind === "unknown" ? [LORA] : []));
    const wrapper = await mountSidebar({ teleport: true });
    const item = await loraItem(characterRow(wrapper, ADA.name));

    expect(item.disabled).toBe(false);
    wrapper.unmount();
  });

  it("stays disabled until the answer lands, and when it cannot be read", async () => {
    let fail;
    serveAdapters(() => new Promise((_, reject) => (fail = reject)));
    const wrapper = await mountSidebar({ teleport: true });
    const item = await loraItem(characterRow(wrapper, ADA.name));

    expect(item.disabled).toBe(true);
    expect(itemTip(item)).toBe("Checking for an attached LoRA…");

    fail(new Error("shelf unavailable"));
    await flushPromises();
    expect(loraButton().disabled).toBe(true);
    expect(itemTip(loraButton())).toBe("Could not read the Models shelf.");
    wrapper.unmount();
  });

  it("does not let a late answer for one person enable another's row", async () => {
    // Ada's read is still out when the menu is reopened on Grace, who has no
    // LoRA. Ada's answer must not make Grace's row live.
    const pending = [];
    serveAdapters((params) =>
      params.character_id === ADA.id
        ? new Promise((resolve) => pending.push(resolve))
        : [],
    );
    const wrapper = await mountSidebar({ teleport: true });
    await loraItem(characterRow(wrapper, ADA.name));
    await loraItem(characterRow(wrapper, GRACE.name));
    pending.forEach((resolve) => resolve([LORA]));
    await flushPromises();

    expect(loraButton().disabled).toBe(true);
    expect(itemTip(loraButton())).toBe(
      "No LoRA is attached to Grace. Assign one from the Models shelf.",
    );
    wrapper.unmount();
  });

  it("makes no read while the row is not offered, then asks once it is", async () => {
    // ComfyUI's configuration lands after start-up. A menu opened before it
    // has made no read, so the row that then appears must not be live on the
    // last menu's answer, and must not sit at "Checking…" for good either.
    let answer = () => [LORA];
    serveAdapters((params) => answer(params));
    const wrapper = await mountSidebar({ teleport: true });
    expect((await loraItem(characterRow(wrapper, GRACE.name))).disabled).toBe(
      false,
    );
    apiGet.mockClear();

    useFilterStore().comfyuiConfigured = false;
    expect(await loraItem(characterRow(wrapper, ADA.name))).toBeUndefined();
    expect(
      apiGet.mock.calls.filter(([url]) => String(url).includes("/adapters")),
    ).toEqual([]);

    const pending = [];
    answer = () => new Promise((resolve) => pending.push(resolve));
    useFilterStore().comfyuiConfigured = true;
    await flushPromises();
    expect(loraButton().disabled).toBe(true);
    expect(itemTip(loraButton())).toBe("Checking for an attached LoRA…");

    // The read it was owed, for the person the menu is open on.
    expect(
      apiGet.mock.calls
        .filter(([url]) => String(url).includes("/adapters"))
        .map(([, config]) => config.params.character_id),
    ).toEqual([ADA.id, ADA.id]);
    pending.forEach((resolve) => resolve([LORA]));
    await flushPromises();
    expect(loraButton().disabled).toBe(false);
    wrapper.unmount();
  });

  it("follows the same rule on a picture set", async () => {
    let attached = [];
    serveAdapters((params) =>
      params.set_id === PORTRAITS.id && params.file_kind === "adapter"
        ? attached
        : [],
    );
    const wrapper = await mountSidebar({ teleport: true });
    // `characterRow` finds a row by its tooltip, which a set's row shares.
    const row = characterRow(wrapper, PORTRAITS.name);

    const item = await loraItem(row);
    expect(item.disabled).toBe(true);
    expect(itemTip(item)).toBe(
      "No LoRA is attached to Portraits. Assign one from the Models shelf.",
    );

    attached = [LORA];
    expect((await loraItem(row)).disabled).toBe(false);
    wrapper.unmount();
  });
});

describe("a coordinated refresh with the sidebar mounted", () => {
  // The paths that read the shared people, set and project lists. Thumbnail,
  // summary and count reads are other routes and are not counted here.
  const LIST_URLS = ["/characters", "/picture_sets", "/projects"];

  function listReads() {
    return apiGet.mock.calls
      .map(([url]) => String(url ?? ""))
      .filter((url) => LIST_URLS.includes(url))
      .sort();
  }

  /** The app's refresh coordinator, pointed at a mounted sidebar. */
  function coordinatorFor(sidebar) {
    let api;
    const host = mount({
      setup() {
        api = useSidebarRefresh({ sidebarRef: shallowRef(sidebar.vm) });
        return () => null;
      },
    });
    return { api, host };
  }

  it("reads each shared list once, through the sidebar", async () => {
    const sidebar = await mountSidebar();
    const { api, host } = coordinatorFor(sidebar);
    apiGet.mockClear();

    api.refreshSidebar();
    for (let i = 0; i < 5; i += 1) await flushPromises();

    // The sidebar's own readers refetch all three lists. A second invalidation
    // of the same lists from the coordinator would queue a trailing read of
    // each behind the first.
    expect(listReads()).toEqual(LIST_URLS);

    host.unmount();
    sidebar.unmount();
  });

  it("shows the project counts of a trailing read, not of the read a refresh joined", async () => {
    const sidebar = await mountSidebar();
    const { api, host } = coordinatorFor(sidebar);
    const projectReads = [];
    apiGet.mockImplementation((url) => {
      if (url === "/projects") {
        return new Promise((resolve) => projectReads.push(resolve));
      }
      return Promise.resolve(respond(url));
    });

    // A second refresh while the first one's project read is out queues a
    // trailing read; both refreshes' count passes join the first read.
    api.refreshSidebar();
    api.refreshSidebar();
    projectReads.shift()({
      data: [{ id: 31, name: "test-project", image_count: 11 }],
    });
    for (let i = 0; i < 5; i += 1) await flushPromises();
    expect(projectReads).toHaveLength(1);

    projectReads.shift()({
      data: [{ id: 31, name: "test-project", image_count: 99 }],
    });
    for (let i = 0; i < 5; i += 1) await flushPromises();

    expect(useEntityListsStore().projects[0].image_count).toBe(99);
    expect(sidebar.vm.scopeCounts.byId[31]).toBe(99);

    host.unmount();
    sidebar.unmount();
  });
});

describe("character thumbnail object URLs", () => {
  // Each person's thumbnail is an object URL over a fetched blob, and the blob
  // stays in memory until the URL is revoked. jsdom has no blob URLs, so both
  // halves are stand-ins that record what the sidebar asked for.
  const THUMBNAIL_PATH = /^\/characters\/(\d+)\/thumbnail$/;
  let created;
  let revoked;
  let people;
  /** Thumbnail reads still waiting for an answer, oldest first. */
  let thumbnailReads;
  let originalCreate;
  let originalRevoke;
  /** The sidebar whose rows a revoke must no longer be showing. */
  let watched;
  let revokedWhileShown;

  beforeEach(() => {
    created = [];
    revoked = [];
    watched = null;
    revokedWhileShown = [];
    people = CHARACTERS;
    thumbnailReads = [];
    originalCreate = URL.createObjectURL;
    originalRevoke = URL.revokeObjectURL;
    URL.createObjectURL = vi.fn((blob) => {
      const url = `blob:test-${blob.person}-${created.length + 1}`;
      created.push(url);
      return url;
    });
    URL.revokeObjectURL = vi.fn((url) => {
      revoked.push(url);
      // Revoked while a row still points at it: that image can no longer load.
      if (watched?.html().includes(url)) revokedWhileShown.push(url);
    });
    apiGet.mockImplementation((url) => {
      const u = String(url ?? "");
      const thumbnail = THUMBNAIL_PATH.exec(u);
      if (thumbnail) {
        const id = Number(thumbnail[1]);
        return new Promise((resolve, reject) => {
          thumbnailReads.push({
            id,
            answer: () => resolve({ data: { person: id } }),
            fail: reject,
          });
        });
      }
      if (u === "/characters") return Promise.resolve({ data: people });
      return Promise.resolve(respond(url));
    });
  });

  afterEach(() => {
    URL.createObjectURL = originalCreate;
    URL.revokeObjectURL = originalRevoke;
  });

  /** Answer every thumbnail read, including the ones a freed slot starts. */
  async function answerThumbnails() {
    while (thumbnailReads.length) {
      thumbnailReads.splice(0).forEach((read) => read.answer());
      await flushPromises();
    }
    // The revokes wait for the render that stops showing the old URL.
    await flushPromises();
  }

  async function refresh(wrapper) {
    wrapper.vm.refreshSidebar();
    for (let i = 0; i < 5; i += 1) await flushPromises();
  }

  function urlsFor(person) {
    return created.filter((url) => url.startsWith(`blob:test-${person.id}-`));
  }

  function shownThumbnail(wrapper, name) {
    return characterRow(wrapper, name)
      .find("img.sidebar-character-thumb")
      .attributes("src");
  }

  it("revokes every URL a later refresh replaces, and shows the new ones", async () => {
    const wrapper = await mountSidebar();
    watched = wrapper;
    await answerThumbnails();
    const [adaFirst] = urlsFor(ADA);
    const [graceFirst] = urlsFor(GRACE);
    expect(shownThumbnail(wrapper, ADA.name)).toBe(adaFirst);

    await refresh(wrapper);
    await answerThumbnails();

    expect([...revoked].sort()).toEqual([adaFirst, graceFirst].sort());
    expect(revokedWhileShown).toEqual([]);
    expect(shownThumbnail(wrapper, ADA.name)).toBe(urlsFor(ADA)[1]);
    expect(shownThumbnail(wrapper, GRACE.name)).toBe(urlsFor(GRACE)[1]);
    watched = null;
    wrapper.unmount();
  });

  it("revokes the thumbnail of a person who is no longer listed", async () => {
    const wrapper = await mountSidebar();
    watched = wrapper;
    await answerThumbnails();
    const [adaFirst] = urlsFor(ADA);
    const [graceUrl] = urlsFor(GRACE);

    people = [ADA];
    await refresh(wrapper);
    await answerThumbnails();

    expect(characterRow(wrapper, GRACE.name)).toBeUndefined();
    expect([...revoked].sort()).toEqual([adaFirst, graceUrl].sort());
    expect(revokedWhileShown).toEqual([]);
    watched = null;
    wrapper.unmount();
  });

  it("installs nothing for a person removed while their read was out", async () => {
    const wrapper = await mountSidebar();
    expect(thumbnailReads.map((read) => read.id)).toEqual([ADA.id, GRACE.id]);

    people = [ADA];
    await refresh(wrapper);
    await answerThumbnails();

    expect(urlsFor(GRACE)).toEqual([]);
    wrapper.unmount();
  });

  it("drops the face it had when a read fails, and shows the placeholder", async () => {
    // The route answers 404 for a person with no usable face, so the face from
    // before is stale, not a better answer than the placeholder.
    const wrapper = await mountSidebar();
    watched = wrapper;
    await answerThumbnails();
    const [adaUrl] = urlsFor(ADA);

    await refresh(wrapper);
    thumbnailReads.splice(0).forEach((read) =>
      read.id === ADA.id ? read.fail(new Error("test-no-face")) : read.answer(),
    );
    await flushPromises();
    await flushPromises();

    expect(shownThumbnail(wrapper, ADA.name)).toBe(unknownPerson);
    expect(revoked).toContain(adaUrl);
    expect(revokedWhileShown).toEqual([]);
    expect(urlsFor(ADA)).toEqual([adaUrl]);
    watched = null;
    wrapper.unmount();
  });

  it("installs nothing from a read that a newer one superseded", async () => {
    const wrapper = await mountSidebar();
    await refresh(wrapper);
    const [olderAda, olderGrace, newerAda, newerGrace] = thumbnailReads.splice(0);
    expect([olderAda.id, olderGrace.id, newerAda.id, newerGrace.id]).toEqual([
      ADA.id,
      GRACE.id,
      ADA.id,
      GRACE.id,
    ]);

    newerAda.answer();
    newerGrace.answer();
    await flushPromises();
    olderAda.answer();
    olderGrace.answer();
    await flushPromises();
    await flushPromises();

    expect(created).toEqual([`blob:test-${ADA.id}-1`, `blob:test-${GRACE.id}-2`]);
    expect(shownThumbnail(wrapper, ADA.name)).toBe(`blob:test-${ADA.id}-1`);
    expect(shownThumbnail(wrapper, GRACE.name)).toBe(`blob:test-${GRACE.id}-2`);
    wrapper.unmount();
  });

  it("creates nothing after unmount, and revokes what it was showing", async () => {
    const wrapper = await mountSidebar();
    await answerThumbnails();
    const shown = [...created];

    // One refresh has its thumbnail reads out; a second is still waiting for
    // the people list, so its reads would only be asked for after unmount.
    await refresh(wrapper);
    expect(thumbnailReads.map((read) => read.id)).toEqual([ADA.id, GRACE.id]);
    wrapper.vm.refreshSidebar();
    wrapper.unmount();
    for (let i = 0; i < 5; i += 1) await flushPromises();
    await answerThumbnails();

    expect(created).toEqual(shown);
    expect([...revoked].sort()).toEqual([...shown].sort());
  });

  it("keeps the thumbnails shown after a session reset when an older people read lands", async () => {
    const wrapper = await mountSidebar();
    await answerThumbnails();
    const peopleReads = [];
    const answer = apiGet.getMockImplementation();
    apiGet.mockImplementation((url, config) =>
      url === "/characters"
        ? new Promise((resolve) => peopleReads.push(resolve))
        : answer(url, config),
    );

    // One refresh is still waiting for the people list when the session
    // resets; a refresh after the reset gets its list first and installs
    // fresh thumbnails.
    await refresh(wrapper);
    const beforeReset = peopleReads.shift();
    useEntityListsStore().reset();
    await refresh(wrapper);
    peopleReads.shift()({ data: CHARACTERS });
    for (let i = 0; i < 5; i += 1) await flushPromises();
    await answerThumbnails();
    const adaCurrent = urlsFor(ADA).at(-1);
    expect(shownThumbnail(wrapper, ADA.name)).toBe(adaCurrent);

    // The read from before the reset belongs to the previous session.
    beforeReset({ data: CHARACTERS });
    for (let i = 0; i < 5; i += 1) await flushPromises();

    expect(shownThumbnail(wrapper, ADA.name)).toBe(adaCurrent);
    wrapper.unmount();
  });

  it("revokes every URL on a session reset, and installs nothing from a read still out", async () => {
    const wrapper = await mountSidebar();
    await answerThumbnails();
    const shown = [...created];

    // A refresh has its thumbnail reads out when the session resets.
    await refresh(wrapper);
    useEntityListsStore().reset();
    await flushPromises();
    await answerThumbnails();

    expect(created).toEqual(shown);
    expect([...revoked].sort()).toEqual([...shown].sort());
    wrapper.unmount();
  });

  it("keeps at most six thumbnail reads out at once", async () => {
    people = Array.from({ length: 20 }, (_, i) => ({
      id: 100 + i,
      name: `test-person-${i}`,
      image_count: 1,
      project_image_count: 1,
    }));
    const ids = people.map((person) => person.id);
    const wrapper = await mountSidebar();

    expect(thumbnailReads.map((read) => read.id)).toEqual(ids.slice(0, 6));
    // A slot that frees starts the next person in line, not a burst.
    thumbnailReads.shift().answer();
    await flushPromises();
    expect(thumbnailReads.map((read) => read.id)).toEqual(ids.slice(1, 7));

    await answerThumbnails();
    expect(created).toEqual(ids.map((id, i) => `blob:test-${id}-${i + 1}`));
    wrapper.unmount();
  });

  it("skips the queued reads that a newer refresh superseded", async () => {
    people = Array.from({ length: 20 }, (_, i) => ({
      id: 100 + i,
      name: `test-person-${i}`,
      image_count: 1,
      project_image_count: 1,
    }));
    const ids = people.map((person) => person.id);
    const asked = [];
    const ask = apiGet.getMockImplementation();
    apiGet.mockImplementation((url, config) => {
      const thumbnail = THUMBNAIL_PATH.exec(String(url ?? ""));
      if (thumbnail) asked.push(Number(thumbnail[1]));
      return ask(url, config);
    });
    const wrapper = await mountSidebar();
    await refresh(wrapper);
    await answerThumbnails();

    // The first refresh's six reads were already out; its other fourteen were
    // still queued and give way to the second refresh's.
    expect(asked).toEqual([...ids.slice(0, 6), ...ids]);
    wrapper.unmount();
  });
});
