// Workflow names come from one listing, however many ids are asked for, and a
// name read under one credential must not land under the next.

import { describe, it, expect, vi, beforeEach } from "vitest";
import { flushPromises } from "@vue/test-utils";

vi.mock("../api/workflows", () => ({ listWorkflowCards: vi.fn() }));

import { listWorkflowCards } from "../api/workflows";
import { useWorkflowNames } from "./useWorkflowNames";
import { notifySessionReset } from "../utils/apiClient";

const listing = (...cards) => ({ cards, one_offs: 0, hidden: 0 });

describe("useWorkflowNames", () => {
  beforeEach(() => {
    notifySessionReset();
    listWorkflowCards.mockReset();
  });

  it("names every asked workflow from one listing", async () => {
    listWorkflowCards.mockResolvedValue(
      listing(
        { id: "auto:a", name: "Portrait" },
        { id: "auto:b", name: "Portrait (2)" },
        { id: "c".repeat(32), name: "Hidden one-off" },
      ),
    );
    const { nameOf } = useWorkflowNames();
    expect(nameOf("auto:a")).toBeNull();
    expect(nameOf("auto:b")).toBeNull();
    expect(nameOf("c".repeat(32))).toBeNull();
    await flushPromises();
    expect(listWorkflowCards).toHaveBeenCalledTimes(1);
    // Hidden and one-off workflows too: a missing-model note can name either.
    expect(listWorkflowCards).toHaveBeenCalledWith({
      includeHidden: true,
      includeOneOffs: true,
    });
    expect(nameOf("auto:a")).toBe("Portrait");
    expect(nameOf("auto:b")).toBe("Portrait (2)");
    expect(nameOf("c".repeat(32))).toBe("Hidden one-off");
    // Asked again, nothing is read again.
    await flushPromises();
    expect(listWorkflowCards).toHaveBeenCalledTimes(1);
  });

  it("reads once more for an id first asked while the listing was out", async () => {
    let release;
    listWorkflowCards.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = () => resolve(listing({ id: "auto:a", name: "Portrait" }));
        }),
    );
    listWorkflowCards.mockResolvedValue(
      listing(
        { id: "auto:a", name: "Portrait" },
        { id: "auto:new", name: "Just made" },
      ),
    );
    const { nameOf } = useWorkflowNames();
    nameOf("auto:a");
    nameOf("auto:gone");
    await flushPromises();
    nameOf("auto:new");
    release();
    await flushPromises();
    expect(nameOf("auto:new")).toBe("Just made");
    // auto:gone was in the first listing's question and not in its answer:
    // it is not read for again.
    expect(nameOf("auto:gone")).toBeNull();
    await flushPromises();
    expect(listWorkflowCards).toHaveBeenCalledTimes(2);
  });

  it("drops a read that lands after a session reset", async () => {
    let release;
    listWorkflowCards.mockImplementation(
      () =>
        new Promise((resolve) => {
          release = () => resolve(listing({ id: "auto:late", name: "Old name" }));
        }),
    );
    const { nameOf } = useWorkflowNames();
    expect(nameOf("auto:late")).toBeNull();
    await flushPromises();
    notifySessionReset();
    release();
    await flushPromises();
    expect(nameOf("auto:late")).toBeNull();
  });
});
