// Opening the lightbox reads the people list, and only the list.
//
// The list feeds one thing here: the default name the "New person…" flow on a
// face row offers ("Character NNNN", the first one nobody has). Nothing in the
// lightbox draws a person's thumbnail, so a thumbnail download per person on
// every open is a request and a blob for nobody - and one past the HTTP cache,
// if it carries a buster.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { enableAutoUnmount, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

// Imported statically, not lazily inside the tests. `vi.mock` is hoisted
// above every import, so a lazy import buys no mock ordering. It only moves
// the cost of compiling this 5.7k-line SFC (~7s on a loaded machine) inside
// the first test's 5s timeout, which is what made this file flake in the full
// suite while passing on its own.
import Lightbox from "./Lightbox.vue";

// A test that fails mid-way must not leave a mounted lightbox behind: its
// window-level keydown listener would answer every later test in this file.
enableAutoUnmount(afterEach);

const PEOPLE = [
  { id: 7, name: "Character 0001" },
  { id: 9, name: "test-person-9" },
];

const apiGet = vi.fn(async (url) => {
  if (url === "/characters") return { data: PEOPLE };
  if (typeof url === "string" && url.includes("/workflow")) {
    const error = new Error("no workflow");
    error.response = { status: 404 };
    throw error;
  }
  if (typeof url === "string" && url.includes("/thumbnail")) {
    return { data: new Blob(["fake-face"], { type: "image/png" }) };
  }
  return { data: [] };
});

vi.mock("../../utils/apiClient", async () => {
  const { ref } = await import("vue");
  return {
    API_BASE_URL: "/api/v1",
    onSessionReset: () => () => {},
    sessionContext: { value: null },
    apiClient: { get: (...args) => apiGet(...args), post: vi.fn(), delete: vi.fn() },
    appendShareToken: (url) => url,
    isReadOnly: ref(false),
    setRequestClientId: vi.fn(),
  };
});

const STUBS = {
  RailTags: true,
  LightboxFilmstrip: true,
  LightboxDescriptionPanel: true,
  AddToEntityControl: true,
  CharacterEditor: true,
  StarRatingOverlay: true,
  PluginParametersUI: true,
  LightboxActionReceipt: true,
  "v-icon": true,
  "v-menu": true,
  "v-tooltip": true,
};

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));
const originalCreateObjectUrl = URL.createObjectURL;
const originalRevokeObjectUrl = URL.revokeObjectURL;

async function openLightbox() {
  const wrapper = mount(Lightbox, {
    props: {
      open: false,
      initialImageId: 3,
      allImages: [{ id: 3, format: "jpg", tags: [] }],
      backendUrl: "http://test",
      tagUpdate: { key: 0, pictureIds: [] },
      descriptionUpdate: { key: 0, pictureIds: [] },
      smartScoreUpdate: { key: 0, pictureIds: [] },
    },
    global: { stubs: STUBS },
    attachTo: document.body,
  });
  await wrapper.setProps({ open: true });
  await flush();
  await flush();
  return wrapper;
}

const peopleReads = () =>
  apiGet.mock.calls
    .map(([url]) => url)
    .filter((url) => typeof url === "string" && url.startsWith("/characters"));

beforeEach(() => {
  setActivePinia(createPinia());
  apiGet.mockClear();
  URL.createObjectURL = vi.fn(() => "blob:test-face");
  URL.revokeObjectURL = vi.fn();
});

afterEach(() => {
  URL.createObjectURL = originalCreateObjectUrl;
  URL.revokeObjectURL = originalRevokeObjectUrl;
  document.body.innerHTML = "";
});

describe("Lightbox people list", () => {
  it("reads the people list on open and downloads no one's thumbnail", async () => {
    await openLightbox();

    expect(peopleReads()).toEqual(["/characters"]);
    expect(URL.createObjectURL).not.toHaveBeenCalled();
  });

  it("names a new person from the list it read", async () => {
    const wrapper = await openLightbox();

    await wrapper.vm.openCreatePersonForFace({ id: 41 }, "");

    // "Character 0001" is taken, so the default is the next one in the series.
    expect(wrapper.vm.createPersonCharacter.name).toBe("Character 0002");
  });
});
