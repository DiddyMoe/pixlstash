import { describe, it, expect } from "vitest";

import {
  SCRAPHEAP_GESTURES,
  picturesPhrase,
  scrapheapRunFailureSentence,
  scrapheapRunOutcomeNotice,
  scrapheapRunProgress,
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

  it("passes on a restore's refusal of a start where the start was made", () => {
    // TaskRunner.submit's refusal while a restore holds the TaskRunner,
    // answered as a 409 by server.py; nothing was created.
    const refused = {
      kind: "conflict",
      status: 409,
      code: null,
      detail:
        "Not started: a library restore is in progress. Try again once it has finished.",
      run: null,
    };
    expect(scrapheapRunFailureSentence("start", refused, G.EMPTY)).toBe(
      "Couldn't empty the scrapheap. Not started: a library restore is in " +
        "progress. Try again once it has finished.",
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

  // An undo's own refusals say what the code means, not "Couldn't undo the
  // restore." and the server's message.
  it("says why an undo was refused", () => {
    const undo = (code) =>
      scrapheapRunFailureSentence(
        "undo",
        { kind: "conflict", code, detail: "fake-server-message", run: null },
        G.UNDO_RESTORE,
      );
    expect(undo("nothing_to_undo")).toBe(
      "That restore has already been undone.",
    );
    expect(undo("not_a_restore_run")).toBe("Only a restore can be undone.");
    expect(undo("scrapheap_run_not_found")).toBe(
      "That restore is no longer on record; undo its steps in History instead.",
    );
    expect(undo("some_other_code")).toBe(
      "Couldn't undo the restore. fake-server-message.",
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

  // A purge destroys only what its confirmation showed, so what
  // was binned while the dialog was open is still there, and is said so.
  it("says what was binned after the confirmation is still in the heap", () => {
    expect(
      scrapheapRunOutcomeNotice(
        purge({
          status: "complete",
          total: 3,
          deleted_count: 3,
          skipped_count: 1,
          arrived_since: 1,
        }),
        G.EMPTY,
      ).message,
    ).toBe(
      "Emptied the Scrapheap: 3 pictures deleted forever. " +
        "1 reference-folder original was kept. " +
        "1 picture binned since you confirmed is still in the Scrapheap.",
    );
    expect(
      scrapheapRunOutcomeNotice(
        purge({
          status: "cancelled",
          total: 50000,
          deleted_count: 1500,
          arrived_since: 2000,
        }),
        G.EMPTY,
      ).message,
    ).toBe(
      `Deleting stopped. Deleted ${(1500).toLocaleString()} of ` +
        `${(50000).toLocaleString()} pictures forever; the rest are still ` +
        `in the Scrapheap. ${(2000).toLocaleString()} pictures binned since ` +
        "you confirmed are still in the Scrapheap.",
    );
    expect(
      scrapheapRunOutcomeNotice(
        purge({
          status: "error",
          total: 2,
          deleted_count: 2,
          error: "Disk full",
          arrived_since: 2,
        }),
        G.EMPTY,
      ).message,
    ).toBe(
      "Deleting stopped on an error. Deleted 2 of 2 pictures forever. " +
        "Disk full. 2 pictures binned since you confirmed are still in the " +
        "Scrapheap.",
    );
  });

  it("says nothing of late arrivals when there were none", () => {
    for (const arrived_since of [0, null, undefined]) {
      expect(
        scrapheapRunOutcomeNotice(
          purge({
            status: "interrupted",
            total: 4,
            deleted_count: 1,
            arrived_since,
          }),
          G.EMPTY,
        ).message,
      ).toBe(
        "Deleting was interrupted when the server restarted. Deleted 1 of " +
          "4 pictures forever; the rest are still in the Scrapheap.",
      );
    }
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

  it("says a restore, not a restart, interrupted a run the snapshot held", () => {
    expect(
      scrapheapRunOutcomeNotice(
        restore({
          status: "interrupted",
          error_code: "library_restored",
          total: 10,
          restored_count: 3,
        }),
        G.RESTORE_ALL,
      ),
    ).toEqual({
      level: "warning",
      message:
        "Restoring was interrupted when the library was restored to a " +
        "snapshot taken before it finished. Restored 3 of 10 pictures; the " +
        "rest are still in the Scrapheap.",
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

describe("utils/scrapheapRun - an undo of a restore", () => {
  // An undo's total and processed count the restore's History steps.
  const undo = (fields) => ({
    kind: "undo",
    total: 5,
    processed: 2,
    undone_count: 0,
    target_run_id: "run-1",
    ...fields,
  });

  it("says the pictures went back in the Scrapheap, never that they were restored", () => {
    expect(
      scrapheapRunOutcomeNotice(
        undo({ status: "complete", processed: 5, undone_count: 1200 }),
        G.UNDO_RESTORE,
      ),
    ).toEqual({
      level: "success",
      message: "Put 1,200 pictures back in the Scrapheap.",
    });
  });

  it("names the History entry that blocked it, in the server's words", () => {
    const blockedBy =
      "Undo stopped: a later change in History (\u201ctest-label\u201d, entry 7) must be undone first.";
    expect(
      scrapheapRunOutcomeNotice(
        undo({
          status: "blocked",
          undone_count: 400,
          error_code: "history_moved_on",
          error: blockedBy,
        }),
        G.UNDO_RESTORE,
      ),
    ).toEqual({ level: "warning", message: blockedBy });
  });

  it("says why the undo refused a part, and what it put back", () => {
    expect(
      scrapheapRunOutcomeNotice(
        undo({
          status: "blocked",
          undone_count: 400,
          error_code: "undo_refused",
          error: "fake-server-sentence",
        }),
        G.UNDO_RESTORE,
      ),
    ).toEqual({
      level: "warning",
      message:
        "Undo stopped: a picture of this restore is now in a locked set or " +
        "was deleted forever. Put 400 pictures back in the Scrapheap; the " +
        "rest stay restored.",
    });
  });

  it("says a restore interrupted an undo the snapshot held", () => {
    expect(
      scrapheapRunOutcomeNotice(
        undo({
          status: "interrupted",
          error_code: "library_restored",
          undone_count: 300,
        }),
        null,
      ).message,
    ).toBe(
      "Undo was interrupted when the library was restored to a snapshot " +
        "taken before it finished. Put 300 pictures back in the Scrapheap; " +
        "the rest stay restored.",
    );
  });

  it("reports a stopped, interrupted or failed undo with its tally", () => {
    const tally =
      "Put 300 pictures back in the Scrapheap; the rest stay restored.";
    expect(
      scrapheapRunOutcomeNotice(
        undo({ status: "cancelled", undone_count: 300 }),
        G.UNDO_RESTORE,
      ),
    ).toEqual({ level: "info", message: `Undo stopped. ${tally}` });
    expect(
      scrapheapRunOutcomeNotice(
        undo({ status: "interrupted", undone_count: 300 }),
        null,
      ),
    ).toEqual({
      level: "warning",
      message: `Undo was interrupted when the server restarted. ${tally}`,
    });
    expect(
      scrapheapRunOutcomeNotice(
        undo({ status: "failed", undone_count: 300, error: "fake-reason" }),
        null,
      ),
    ).toEqual({
      level: "error",
      message: `Undo stopped on an error. ${tally} fake-reason.`,
    });
  });

  it("says nothing stays restored when the undo stopped after its last step", () => {
    expect(
      scrapheapRunOutcomeNotice(
        undo({ status: "cancelled", processed: 5, undone_count: 1200 }),
        G.UNDO_RESTORE,
      ),
    ).toEqual({
      level: "info",
      message: "Undo stopped. Put 1,200 pictures back in the Scrapheap.",
    });
  });
});

describe("utils/scrapheapRun - a run's progress", () => {
  const going = (fields) => ({
    kind: "purge",
    status: "running",
    active: true,
    total: 1200,
    processed: 300,
    ...fields,
  });

  it("counts a purge or a restore in pictures, and says what Stop keeps", () => {
    expect(scrapheapRunProgress(going({ slice_max_pictures: 250 }))).toEqual({
      processed: 300,
      total: 1200,
      steps: false,
      determinate: true,
      percent: 25,
      counted: "300 of 1,200 pictures",
      stopTip:
        "Stops after the current batch of up to 250 pictures. Pictures " +
        "already deleted forever cannot be restored.",
    });
    expect(scrapheapRunProgress(going({ kind: "restore" })).stopTip).toBe(
      "Stops after the current batch. Pictures already restored stay restored.",
    );
  });

  it("counts an undo in the History steps it works through", () => {
    const undo = scrapheapRunProgress(
      going({ kind: "undo", total: 5, processed: 2, slice_max_pictures: 250 }),
    );
    expect(undo.steps).toBe(true);
    expect(undo.counted).toBe("step 2 of 5");
    expect(undo.percent).toBe(40);
    expect(undo.stopTip).toBe(
      "Stops after the History step it is on. Pictures already put back " +
        "stay in the Scrapheap.",
    );
  });

  it("has no percent while the run is queued, ended or has no total", () => {
    for (const fields of [
      { status: "pending" },
      { active: false, status: "complete" },
      { total: 0 },
    ]) {
      const progress = scrapheapRunProgress(going(fields));
      expect(progress.determinate).toBe(false);
      expect(progress.percent).toBe(0);
    }
    expect(scrapheapRunProgress(going({ processed: 1300 })).percent).toBe(100);
  });
});
