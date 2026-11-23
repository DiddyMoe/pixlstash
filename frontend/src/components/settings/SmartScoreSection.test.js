// The Penalised Tags list in Settings: edits are saved one at a time against
// the list the server last accepted, a chip being removed is busy, a read that
// started before an accepted edit cannot put the old list back, and a pane
// reopened while an edit is saving reads the list that edit leaves.

import { describe, it, expect, vi, beforeEach } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

vi.mock("vuetify/components", () => ({
  VIcon: { template: "<i><slot /></i>" },
  VSwitch: { template: "<input />" },
}));
vi.mock("../../api/config", () => ({
  getUserConfig: vi.fn(),
  patchUserConfig: vi.fn(),
}));

import { getUserConfig, patchUserConfig } from "../../api/config";
import { useNoticeStore } from "../../stores/useNoticeStore";
import SmartScoreSection from "./SmartScoreSection.vue";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function config(penalised) {
  return {
    smart_score_penalised_tags: penalised,
    hidden_tags: [],
    apply_tag_filter: false,
  };
}

const AppSelectStub = {
  name: "AppSelect",
  props: ["modelValue", "options", "disabled", "compact"],
  emits: ["update:modelValue"],
  template: '<select class="app-select-stub" :disabled="disabled" />',
};

const AddTagRowStub = {
  name: "SettingsAddTagRow",
  props: ["modelValue", "placeholder", "btn"],
  emits: ["update:modelValue", "add"],
  template: '<div class="add-tag-row-stub" />',
};

let pinia;

function mountSection() {
  return mount(SmartScoreSection, {
    props: { open: true },
    global: {
      plugins: [pinia],
      stubs: {
        SettingsSection: { template: "<section><slot /></section>" },
        SettingsAddTagRow: AddTagRowStub,
        AppSelect: AppSelectStub,
        Tooltip: true,
      },
    },
  });
}

async function mountLoaded(penalised) {
  getUserConfig.mockResolvedValueOnce(config(penalised));
  const wrapper = mountSection();
  await flushPromises();
  return wrapper;
}

function labels(wrapper) {
  return wrapper.findAll(".s-chip__label").map((node) => node.text());
}

function chip(wrapper, tag) {
  const found = wrapper
    .findAll(".s-chip")
    .find((node) => node.find(".s-chip__label").text() === tag);
  if (!found) throw new Error(`no chip for ${tag}`);
  return found;
}

function removeButton(wrapper, tag) {
  return chip(wrapper, tag).find(".s-chip__remove");
}

function weightSelect(wrapper, tag) {
  return chip(wrapper, tag).findComponent(AppSelectStub);
}

function weights(wrapper) {
  return Object.fromEntries(
    wrapper
      .findAll(".s-chip")
      .map((node) => [
        node.find(".s-chip__label").text(),
        node.findComponent(AppSelectStub).props("modelValue"),
      ]),
  );
}

function savedLists() {
  return patchUserConfig.mock.calls.map(
    ([partial]) => partial.smart_score_penalised_tags,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  getUserConfig.mockReset();
  patchUserConfig.mockReset();
  pinia = createPinia();
  setActivePinia(pinia);
});

describe("SmartScoreSection: Penalised Tags edits", () => {
  it("two quick removals both stick: the second is sent after the first settles, without the first tag", async () => {
    const wrapper = await mountLoaded({ "test-a": 3, "test-b": 4, "test-c": 2 });
    const first = deferred();
    const second = deferred();
    patchUserConfig
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);

    await removeButton(wrapper, "test-a").trigger("click");
    await removeButton(wrapper, "test-b").trigger("click");
    await flushPromises();

    expect(savedLists()).toEqual([{ "test-b": 4, "test-c": 2 }]);

    first.resolve({});
    await flushPromises();
    expect(savedLists()).toEqual([
      { "test-b": 4, "test-c": 2 },
      { "test-c": 2 },
    ]);

    second.resolve({});
    await flushPromises();
    expect(labels(wrapper)).toEqual(["test-c"]);
  });

  it("a chip whose removal is being saved cannot be removed again", async () => {
    const wrapper = await mountLoaded({ "test-a": 3, "test-b": 4 });
    patchUserConfig.mockReturnValueOnce(deferred().promise);

    await removeButton(wrapper, "test-a").trigger("click");

    const busy = removeButton(wrapper, "test-a");
    expect(busy.attributes("disabled")).toBeDefined();
    expect(busy.attributes("aria-busy")).toBe("true");
    expect(weightSelect(wrapper, "test-a").props("disabled")).toBe(true);
    expect(removeButton(wrapper, "test-b").attributes("disabled")).toBeUndefined();
  });

  it("a weight edit followed by X on the same chip: the removal wins and other weights survive", async () => {
    const wrapper = await mountLoaded({ "test-a": 3, "test-b": 4 });
    const weightSave = deferred();
    patchUserConfig
      .mockReturnValueOnce(weightSave.promise)
      .mockResolvedValueOnce({});

    weightSelect(wrapper, "test-a").vm.$emit("update:modelValue", 5);
    await removeButton(wrapper, "test-a").trigger("click");
    await flushPromises();
    // The removal waits for the weight save, so it cannot be overtaken by it.
    expect(savedLists()).toEqual([{ "test-a": 5, "test-b": 4 }]);

    weightSave.resolve({});
    await flushPromises();
    expect(savedLists()).toEqual([
      { "test-a": 5, "test-b": 4 },
      { "test-b": 4 },
    ]);
    expect(weights(wrapper)).toEqual({ "test-b": 4 });
  });

  it("a weight edit followed by X on another chip keeps the new weight", async () => {
    const wrapper = await mountLoaded({ "test-a": 3, "test-b": 4 });
    const weightSave = deferred();
    patchUserConfig
      .mockReturnValueOnce(weightSave.promise)
      .mockResolvedValueOnce({});

    weightSelect(wrapper, "test-a").vm.$emit("update:modelValue", 5);
    await removeButton(wrapper, "test-b").trigger("click");
    weightSave.resolve({});
    await flushPromises();

    expect(savedLists()[1]).toEqual({ "test-a": 5 });
    expect(weights(wrapper)).toEqual({ "test-a": 5 });
  });

  it("a failed removal keeps its chip, says so, and the removal queued behind it still saves", async () => {
    const wrapper = await mountLoaded({ "test-a": 3, "test-b": 4, "test-c": 2 });
    const failed = deferred();
    patchUserConfig
      .mockReturnValueOnce(failed.promise)
      .mockResolvedValueOnce({});

    await removeButton(wrapper, "test-a").trigger("click");
    await removeButton(wrapper, "test-b").trigger("click");
    failed.reject({ response: { data: { detail: "test-refused" } } });
    await flushPromises();

    expect(savedLists()[1]).toEqual({ "test-a": 3, "test-c": 2 });
    expect(labels(wrapper)).toEqual(["test-a", "test-c"]);
    expect(removeButton(wrapper, "test-a").attributes("disabled")).toBeUndefined();
    expect(wrapper.text()).not.toContain("Saved.");
    expect(wrapper.text()).toContain("test-refused");
    const notices = useNoticeStore().notices;
    expect(
      notices.some((n) => n.level === "error" && n.text === "test-refused"),
    ).toBe(true);
  });

  it("an add made before the first read lands keeps the saved list", async () => {
    const firstRead = deferred();
    getUserConfig
      .mockReturnValueOnce(firstRead.promise)
      .mockResolvedValueOnce(config({ "test-a": 3, "test-b": 4 }));
    patchUserConfig.mockResolvedValue({});
    const wrapper = mountSection();

    const addRow = wrapper.findComponent(AddTagRowStub);
    addRow.vm.$emit("update:modelValue", "test-new");
    await wrapper.vm.$nextTick();
    addRow.vm.$emit("add");
    await flushPromises();

    expect(savedLists()).toEqual([{ "test-a": 3, "test-b": 4, "test-new": 3 }]);
    expect(labels(wrapper)).toEqual(["test-a", "test-b", "test-new"]);
  });
});

describe("SmartScoreSection: reading the list on open", () => {
  it("a read that started before an accepted removal does not bring the tag back", async () => {
    const wrapper = await mountLoaded({ "test-a": 3, "test-b": 4 });
    // Reopened while still mounted (during the close animation): a new read.
    const reread = deferred();
    getUserConfig.mockReturnValueOnce(reread.promise);
    await wrapper.setProps({ open: false });
    await wrapper.setProps({ open: true });
    await flushPromises();
    expect(getUserConfig).toHaveBeenCalledTimes(2);

    patchUserConfig.mockResolvedValueOnce({});
    await removeButton(wrapper, "test-a").trigger("click");
    await flushPromises();
    expect(labels(wrapper)).toEqual(["test-b"]);

    reread.resolve(config({ "test-a": 3, "test-b": 4 }));
    await flushPromises();
    expect(labels(wrapper)).toEqual(["test-b"]);
  });

  it("closing aborts the read in flight and its late answer is ignored", async () => {
    const wrapper = await mountLoaded({ "test-a": 3 });
    const reread = deferred();
    getUserConfig.mockReturnValueOnce(reread.promise);
    await wrapper.setProps({ open: false });
    await wrapper.setProps({ open: true });
    await flushPromises();
    const { signal } = getUserConfig.mock.calls[1][0];
    expect(signal.aborted).toBe(false);

    await wrapper.setProps({ open: false });
    expect(signal.aborted).toBe(true);

    reread.resolve(config({ "test-z": 5 }));
    await flushPromises();
    expect(labels(wrapper)).toEqual(["test-a"]);
  });
});

describe("SmartScoreSection: reopening Settings while an edit is saving", () => {
  // Closing Settings unmounts the pane and reopening mounts a new one, in the
  // same session. The server keeps one list and each save replaces all of it.
  let persisted;
  let saves;

  beforeEach(() => {
    persisted = { "test-a": 3, "test-b": 4, "test-c": 2 };
    saves = [];
    getUserConfig.mockImplementation(async () => config({ ...persisted }));
    patchUserConfig.mockImplementation((partial) => {
      const save = deferred();
      saves.push({
        accept() {
          persisted = { ...partial.smart_score_penalised_tags };
          save.resolve({});
        },
        refuse() {
          save.reject({ response: { data: { detail: "test-refused" } } });
        },
      });
      return save.promise;
    });
  });

  it.each([
    ["is saved", true, { "test-c": 2 }],
    ["is refused", false, { "test-a": 3, "test-c": 2 }],
  ])(
    "a removal in the reopened pane, after the closed pane's removal %s, keeps what that removal left",
    async (_outcome, accepted, expected) => {
      const closed = mountSection();
      await flushPromises();
      await removeButton(closed, "test-a").trigger("click");
      await flushPromises();
      closed.unmount();

      const reopened = mountSection();
      await flushPromises();
      if (accepted) saves[0].accept();
      else saves[0].refuse();
      await flushPromises();

      await removeButton(reopened, "test-b").trigger("click");
      await flushPromises();
      saves[1].accept();
      await flushPromises();

      expect(persisted).toEqual(expected);
    },
  );
});
