import { describe, it, expect } from "vitest";

import {
  SCRAPHEAP_GESTURES,
  picturesPhrase,
  scrapheapRunFailureSentence,
  scrapheapRunOutcomeNotice,
} from "./scrapheapRun";

const G = SCRAPHEAP_GESTURES;

describe("utils/scrapheapRun - picturesPhrase", () => {
  it("counts one picture, many, and grouped digits", () => {
    expect(picturesPhrase(1)).toBe("1 picture");
    expect(picturesPhrase(0)).toBe("0 pictures");
    expect(picturesPhrase(50123)).toBe(`${(50123).toLocaleString()} pictures`);
  });

  it("reads a missing or negative count as none", () => {
    expect(picturesPhrase(undefined)).toBe("0 pictures");
    expect(picturesPhrase(-3)).toBe("0 pictures");
  });
});

describe("utils/scrapheapRun - scrapheapRunFailureSentence", () => {
  it("says nothing for a request the session reset dropped", () => {
    expect(
      scrapheapRunFailureSentence("start", { kind: "cancelled" }, G.EMPTY),
    ).toBeNull();
  });

  it("says a request is still on the wire", () => {
    expect(
      scrapheapRunFailureSentence("start", { kind: "busy" }, G.EMPTY),
    ).toBe("The last Scrapheap request is still waiting for an answer.");
  });

  it("points at the bar when another run holds the heap", () => {
    expect(
      scrapheapRunFailureSentence(
        "start",
        { kind: "conflict", status: 409, code: "scrapheap_run_active" },
        G.RESTORE_ALL,
      ),
    ).toBe(
      "Another Scrapheap run is going, and its progress shows above the " +
        "grid. This can start once it ends.",
    );
  });

  it("does not claim a start failed when no answer came", () => {
    expect(
      scrapheapRunFailureSentence("start", { kind: "timeout" }, G.EMPTY),
    ).toBe(
      "No answer came from the server, so it is not known whether it " +
        "started. If it did, its progress shows above the grid.",
    );
  });

  it("says a run is going when the follow-up read found one", () => {
    expect(
      scrapheapRunFailureSentence(
        "start",
        { kind: "network", run: { id: "run-1", active: true } },
        G.EMPTY,
      ),
    ).toBe(
      "No answer came from the server, but a Scrapheap run is going, and " +
        "its progress shows above the grid.",
    );
  });

  it("says a stop with no answer can be repeated", () => {
    expect(scrapheapRunFailureSentence("stop", { kind: "timeout" })).toBe(
      "No answer came from the server, so it is not known whether the run " +
        "is stopping. Press Stop again to retry; a repeated stop is harmless.",
    );
  });

  it("names a missing permission", () => {
    expect(
      scrapheapRunFailureSentence(
        "start",
        { kind: "forbidden", status: 403 },
        G.PURGE_SELECTION,
      ),
    ).toBe("You don't have permission to change the Scrapheap.");
  });

  it("leads with what each gesture could not do, then the server's words", () => {
    const detail = { kind: "server", status: 503, detail: "Nothing started" };
    expect(scrapheapRunFailureSentence("start", detail, G.EMPTY)).toBe(
      "Couldn't empty the scrapheap. Nothing started.",
    );
    expect(scrapheapRunFailureSentence("start", detail, G.PURGE_SELECTION)).toBe(
      "Couldn't delete those pictures. Nothing started.",
    );
    expect(scrapheapRunFailureSentence("start", detail, G.RESTORE_ALL)).toBe(
      "Couldn't restore the scrapheap. Nothing started.",
    );
    expect(
      scrapheapRunFailureSentence("start", detail, G.RESTORE_SELECTION),
    ).toBe("Couldn't restore those pictures. Nothing started.");
    expect(scrapheapRunFailureSentence("stop", detail)).toBe(
      "Couldn't stop the run. Nothing started.",
    );
  });

  it("keeps the server's own full stop, and asks again when it said nothing", () => {
    expect(
      scrapheapRunFailureSentence(
        "start",
        { kind: "conflict", status: 409, detail: "Token expired." },
        G.EMPTY,
      ),
    ).toBe("Couldn't empty the scrapheap. Token expired.");
    expect(
      scrapheapRunFailureSentence(
        "start",
        { kind: "server", status: 500, detail: "  " },
        G.EMPTY,
      ),
    ).toBe("Couldn't empty the scrapheap. Please try again.");
  });
});

describe("utils/scrapheapRun - scrapheapRunOutcomeNotice", () => {
  const purge = (fields) => ({ kind: "purge", total: 0, ...fields });
  const restore = (fields) => ({ kind: "restore", total: 0, ...fields });

  it("reports an emptied heap", () => {
    expect(
      scrapheapRunOutcomeNotice(
        purge({ status: "complete", total: 3, deleted_count: 3 }),
        G.EMPTY,
      ),
    ).toEqual({
      level: "success",
      message: "Emptied the Scrapheap: 3 pictures deleted forever.",
    });
  });

  it("reports a selection deleted forever", () => {
    expect(
      scrapheapRunOutcomeNotice(
        purge({ status: "complete", total: 1, deleted_count: 1 }),
        G.PURGE_SELECTION,
      ),
    ).toEqual({ level: "success", message: "Deleted 1 picture forever." });
  });

  it("names what a purge kept, one sentence per reason, singular and plural", () => {
    expect(
      scrapheapRunOutcomeNotice(
        purge({
          status: "complete",
          total: 7,
          deleted_count: 4,
          skipped_count: 1,
          skipped_locked_count: 2,
        }),
        G.EMPTY,
      ).message,
    ).toBe(
      "Emptied the Scrapheap: 4 pictures deleted forever. " +
        "1 reference-folder original was kept. " +
        "2 pictures in locked sets were kept.",
    );
    expect(
      scrapheapRunOutcomeNotice(
        purge({
          status: "complete",
          total: 3,
          deleted_count: 0,
          skipped_count: 2,
          skipped_locked_count: 1,
        }),
        G.EMPTY,
      ).message,
    ).toBe(
      "Emptied the Scrapheap: 0 pictures deleted forever. " +
        "2 reference-folder originals were kept. " +
        "1 picture in locked sets was kept.",
    );
  });

  it("reports a restore, what it found gone, and its History steps", () => {
    expect(
      scrapheapRunOutcomeNotice(
        restore({
          status: "complete",
          total: 12000,
          restored_count: 11998,
          skipped_restored: 2,
          segment_batch_ids: ["b1", "b2", "b3"],
        }),
        G.RESTORE_ALL,
      ),
    ).toEqual({
      level: "success",
      message:
        `Restored ${(11998).toLocaleString()} pictures from the Scrapheap. ` +
        "2 pictures had already left it. " +
        "History records it as 3 steps, each undone on its own.",
    });
  });

  it("says nothing of History for a restore that is one step", () => {
    expect(
      scrapheapRunOutcomeNotice(
        restore({
          status: "complete",
          total: 2,
          restored_count: 2,
          segment_batch_ids: ["b1"],
        }),
        G.RESTORE_SELECTION,
      ).message,
    ).toBe("Restored 2 pictures from the Scrapheap.");
  });

  it("reports a stopped purge and what is left", () => {
    expect(
      scrapheapRunOutcomeNotice(
        purge({ status: "cancelled", total: 50000, deleted_count: 1500 }),
        G.EMPTY,
      ),
    ).toEqual({
      level: "info",
      message:
        `Deleting stopped. Deleted ${(1500).toLocaleString()} of ` +
        `${(50000).toLocaleString()} pictures forever; the rest are still ` +
        "in the Scrapheap.",
    });
  });

  it("does not claim anything is left when the stop came after the last slice", () => {
    expect(
      scrapheapRunOutcomeNotice(
        restore({ status: "cancelled", total: 4, restored_count: 4 }),
        G.RESTORE_ALL,
      ).message,
    ).toBe("Restoring stopped. Restored 4 of 4 pictures.");
  });

  it("warns of a run the restart interrupted", () => {
    expect(
      scrapheapRunOutcomeNotice(
        restore({ status: "interrupted", total: 10, restored_count: 3 }),
        G.RESTORE_ALL,
      ),
    ).toEqual({
      level: "warning",
      message:
        "Restoring was interrupted when the server restarted. Restored 3 of " +
        "10 pictures; the rest are still in the Scrapheap.",
    });
  });

  it("reports a failed run with the server's reason", () => {
    expect(
      scrapheapRunOutcomeNotice(
        purge({
          status: "failed",
          total: 10,
          deleted_count: 5,
          error: "disk I/O error",
        }),
        G.EMPTY,
      ),
    ).toEqual({
      level: "error",
      message:
        "Deleting stopped on an error. Deleted 5 of 10 pictures forever; " +
        "the rest are still in the Scrapheap. disk I/O error.",
    });
  });

  it("reports a failed run that gave no reason without a dangling space", () => {
    expect(
      scrapheapRunOutcomeNotice(
        purge({ status: "failed", total: 2, deleted_count: 0 }),
        G.PURGE_SELECTION,
      ).message,
    ).toBe(
      "Deleting stopped on an error. Deleted 0 of 2 pictures forever; the " +
        "rest are still in the Scrapheap.",
    );
  });
});
