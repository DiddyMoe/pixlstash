// The copy around the one destructive action on the dedup surface. Every
// sentence here is a place the UI could lie, so each shape is pinned: the unit
// the menu counts in, the retention window it must READ rather than assume, and
// the wording that keeps a soft delete from sounding like reclaimed space.

import { describe, it, expect } from "vitest";

import {
  selectedKeepCoverOnlyStacks,
  keepCoverOnlyLockReason,
  keepCoverOnlyBytesSentence,
  keepCoverOnlyConfirmLabel,
  keepCoverOnlyIncompleteMessage,
  keepCoverOnlyMenuLabel,
  keepCoverOnlyPreviewFailure,
  keepCoverOnlyProgressSentence,
  keepCoverOnlyReceiptNote,
  keepCoverOnlyRetentionSentence,
  keepCoverOnlySkipNote,
  keepCoverOnlySkipReasons,
  keepCoverOnlySkippedCount,
  keepCoverOnlyTitle,
} from "./keepCoverOnly";

/** A collapsed stack leader tile: one row standing for `count` pictures. */
function leader(id, stackId, count) {
  return { id, stack_id: stackId, stack_count: count };
}

/** One member of an expanded stack: siblings sharing a `stack_id`, no count. */
function member(id, stackId) {
  return { id, stack_id: stackId };
}

/** A picture in no stack at all. */
function loose(id) {
  return { id, stack_id: null };
}

describe("selectedKeepCoverOnlyStacks", () => {
  it("names one stack per selected stack tile and ignores loose pictures", () => {
    const images = [leader(1, "10", 4), leader(2, "20", 3), loose(3)];
    const stacks = selectedKeepCoverOnlyStacks({
      selectedIds: [1, 2, 3],
      images,
    });
    expect(stacks.map((s) => s.id)).toEqual(["10", "20"]);
  });

  // A partial selection inside a stack still names the WHOLE stack, because
  // that is what the action does. The dialog says so; the count must agree.
  it("names a stack once however many of its members are selected", () => {
    const images = [member(1, "10"), member(2, "10"), member(3, "10")];
    const stacks = selectedKeepCoverOnlyStacks({
      selectedIds: [1, 2],
      images,
    });
    expect(stacks).toHaveLength(1);
    expect(stacks[0].pictureIds).toEqual([1, 2, 3]);
  });

  // A stack_id whose stack holds one live member has nothing to collapse; a
  // cover left behind by an earlier run is the ordinary way to get one. Counting
  // it would put a stack in the label that the server will only skip.
  it("skips a stack that holds a single picture", () => {
    const images = [leader(1, "10", 1)];
    expect(
      selectedKeepCoverOnlyStacks({ selectedIds: [1], images }),
    ).toEqual([]);
  });

  it("returns nothing for an empty or unmatched selection", () => {
    expect(selectedKeepCoverOnlyStacks({ selectedIds: [], images: [] })).toEqual(
      [],
    );
    expect(
      selectedKeepCoverOnlyStacks({ selectedIds: [99], images: [loose(1)] }),
    ).toEqual([]);
    expect(selectedKeepCoverOnlyStacks()).toEqual([]);
  });
});

describe("keepCoverOnlyLockReason", () => {
  const lockedSetNames = () => ["Portfolio 2026"];

  // A locked set refuses the WHOLE stack: stack membership reconciles to the
  // union of its members' sets, so removing one member is exactly the mutation
  // the lock forbids. One locked member is therefore enough to hold the stack.
  it("treats one locked member as holding its whole stack", () => {
    const reason = keepCoverOnlyLockReason({
      stacks: [{ id: "10", pictureIds: [1, 2, 3] }],
      isLocked: (id) => id === 3,
      lockedSetNames,
    });
    expect(reason).toContain("refuses the whole stack");
    expect(reason).toContain("Portfolio 2026");
  });

  // The gate fires only when the action provably cannot do anything. One
  // unlocked stack means there is work to do, and over-blocking is its own
  // regression: the dialog reports the skips.
  it("stays available while any named stack is unlocked", () => {
    expect(
      keepCoverOnlyLockReason({
        stacks: [
          { id: "10", pictureIds: [1, 2] },
          { id: "20", pictureIds: [3, 4] },
        ],
        isLocked: (id) => id === 1,
        lockedSetNames,
      }),
    ).toBeNull();
  });

  it("blocks only when every named stack is held", () => {
    expect(
      keepCoverOnlyLockReason({
        stacks: [
          { id: "10", pictureIds: [1, 2] },
          { id: "20", pictureIds: [3, 4] },
        ],
        isLocked: (id) => id === 1 || id === 4,
        lockedSetNames,
      }),
    ).toContain("Every selected stack is held");
  });

  it("says nothing when nothing is locked, or nothing is named", () => {
    expect(
      keepCoverOnlyLockReason({
        stacks: [{ id: "10", pictureIds: [1, 2] }],
        isLocked: () => false,
        lockedSetNames,
      }),
    ).toBeNull();
    expect(keepCoverOnlyLockReason({ stacks: [] })).toBeNull();
  });
});

describe("keepCoverOnlyMenuLabel", () => {
  // The action ignores loose pictures in a mixed selection. That is only honest
  // if the label counts the unit the action works in.
  it("counts stacks, not the tiles that were clicked", () => {
    expect(keepCoverOnlyMenuLabel({ stackCount: 3, selectedCount: 3 })).toBe(
      "Keep cover only (3 stacks)",
    );
    expect(keepCoverOnlyMenuLabel({ stackCount: 1, selectedCount: 1 })).toBe(
      "Keep cover only (1 stack)",
    );
  });

  // The eight pictures that will be left alone have to be visible BEFORE the
  // click, not discovered afterwards from a count that does not add up.
  it("reports partial eligibility when the selection holds loose pictures", () => {
    expect(keepCoverOnlyMenuLabel({ stackCount: 12, selectedCount: 20 })).toBe(
      "Keep cover only (12 of 20)",
    );
  });

  it("drops the count when the selection names no stack at all", () => {
    expect(keepCoverOnlyMenuLabel({ stackCount: 0, selectedCount: 5 })).toBe(
      "Keep cover only",
    );
    expect(keepCoverOnlyMenuLabel()).toBe("Keep cover only");
  });
});

describe("keepCoverOnlyTitle / keepCoverOnlyConfirmLabel", () => {
  // The title/button pairing is the dialog's safety property: one names what
  // survives, the other what goes.
  it("says what you keep and what you lose", () => {
    expect(keepCoverOnlyTitle(160)).toBe("Keep only the cover of 160 stacks");
    expect(keepCoverOnlyConfirmLabel(414)).toBe("Move 414 to the Scrapheap");
  });

  it("names no figure while the preview is unknown", () => {
    expect(keepCoverOnlyTitle(null)).toBe("Keep only the cover");
    expect(keepCoverOnlyConfirmLabel(null)).toBe("Move to the Scrapheap");
  });

  it("keeps the singular readable", () => {
    expect(keepCoverOnlyTitle(1)).toBe("Keep only the cover of 1 stack");
  });
});

describe("keepCoverOnlyRetentionSentence", () => {
  // DEFAULT_RETENTION_DAYS is null. Hardcoding "30 days" would be exactly the
  // class of error this dialog exists to avoid, so the "never" branch is the
  // one that has to be right first.
  it("says the Scrapheap never empties on its own when retention is off", () => {
    const sentence = keepCoverOnlyRetentionSentence(null);
    expect(sentence).toContain("never empties on its own");
    expect(sentence).not.toMatch(/\d/);
  });

  it("treats an absent setting as never, not as a window it invented", () => {
    expect(keepCoverOnlyRetentionSentence(undefined)).toContain(
      "never empties on its own",
    );
  });

  it("names the live window when the server carries one", () => {
    expect(keepCoverOnlyRetentionSentence(30)).toContain("after 30 days");
    expect(keepCoverOnlyRetentionSentence(90)).toContain("after 90 days");
  });
});

describe("keepCoverOnlyBytesSentence", () => {
  // Nothing is freed by a soft delete. The words that would claim otherwise are
  // the ones to watch for.
  it("says the bytes are held, never freed or reclaimed", () => {
    const sentence = keepCoverOnlyBytesSentence(1_234_567_890);
    expect(sentence).toContain("1.1 GB");
    expect(sentence).toContain("stays there until the Scrapheap is emptied");
    expect(sentence).not.toMatch(/free|reclaim|saved/i);
  });

  it("says nothing at all when there is nothing to say", () => {
    expect(keepCoverOnlyBytesSentence(0)).toBe("");
    expect(keepCoverOnlyBytesSentence(null)).toBe("");
  });
});

describe("keepCoverOnlySkipReasons / keepCoverOnlySkippedCount", () => {
  const PREVIEW = {
    stacks_selected: 20,
    stacks_eligible: 17,
    stacks_skipped_locked: 2,
    stacks_skipped_character_on_copy: 1,
    stacks_skipped_single_member: 0,
  };

  // A sum of three directly-counted disjoint buckets. Deriving it as
  // `stacks_selected - stacks_eligible` would produce a number no query ever
  // answered, which is how the neighbouring dialog got its 62-for-3.
  it("sums the named buckets rather than subtracting from the total", () => {
    expect(keepCoverOnlySkippedCount(PREVIEW)).toBe(3);
    // Same arithmetic holds when the server reports a bucket this build's copy
    // does not word, so the row can never disagree with the buckets under it.
    expect(
      keepCoverOnlySkippedCount({
        ...PREVIEW,
        stacks_skipped_single_member: 4,
      }),
    ).toBe(7);
  });

  it("names the locked refusal as whole-stack, and how to lift it", () => {
    const [locked] = keepCoverOnlySkipReasons(PREVIEW);
    expect(locked.key).toBe("locked");
    expect(locked.text).toContain("whole");
    expect(locked.text).toContain("Unlock the set");
  });

  it("skips the buckets that are empty", () => {
    expect(keepCoverOnlySkipReasons(PREVIEW).map((r) => r.key)).toEqual([
      "locked",
      "character_on_copy",
    ]);
    expect(
      keepCoverOnlySkipReasons({
        stacks_skipped_locked: 0,
        stacks_skipped_character_on_copy: 0,
        stacks_skipped_single_member: 0,
      }),
    ).toEqual([]);
  });
});

describe("keepCoverOnlySkipNote", () => {
  // The mutation's skip buckets are LISTS, unlike the preview's counts.
  it("counts the response's skipped-stack rows", () => {
    expect(
      keepCoverOnlySkipNote({
        stacks_skipped_locked: [{ stack_id: 1 }, { stack_id: 2 }],
        stacks_skipped_character_on_copy: [{ stack_id: 3 }],
      }),
    ).toBe("3 stacks skipped: 2 locked, 1 holding a person's only link.");
  });

  it("names the single reason when only one applies", () => {
    expect(
      keepCoverOnlySkipNote({ stacks_skipped_locked: [{ stack_id: 1 }] }),
    ).toBe("1 stack skipped: held by a locked picture set.");
  });

  it("stays silent when the run skipped nothing", () => {
    expect(
      keepCoverOnlySkipNote({
        stacks_skipped_locked: [],
        stacks_skipped_character_on_copy: [],
      }),
    ).toBe("");
    expect(keepCoverOnlySkipNote(null)).toBe("");
  });
});

describe("Keep recipes only copy", () => {
  it("counts a stack with nothing reproducible as one that won't change", () => {
    const preview = {
      stacks_skipped_locked: 1,
      stacks_skipped_nothing_reproducible: 2,
    };
    expect(keepCoverOnlySkippedCount(preview)).toBe(3);
    expect(keepCoverOnlySkipReasons(preview).map((r) => r.key)).toEqual([
      "nothing_reproducible",
      "locked",
    ]);
  });

  it("names unchanged stacks on the receipt", () => {
    expect(
      keepCoverOnlySkipNote({
        stacks_skipped_nothing_reproducible: [{}, {}],
        pictures_staying: 3,
      }),
    ).toBe(
      "2 stacks unchanged: nothing in them could be made again. 3 pictures stay: they could not be made again.",
    );
  });

  it("puts the pictures that stayed on the receipt", () => {
    expect(keepCoverOnlySkipNote({ pictures_staying: 2 })).toBe(
      "2 pictures stay: they could not be made again.",
    );
    expect(
      keepCoverOnlySkipNote({ stacks_skipped_locked: [{}], pictures_staying: 1 }),
    ).toBe(
      "1 stack skipped: held by a locked picture set. 1 picture stays: it could not be made again.",
    );
  });
});

// A selection past one request is collapsed in several, under one batch id.
// The receipt names one operation; these sentences carry the gesture.
describe("Selections larger than one request", () => {
  it("leads the receipt with the gesture's totals when it took several steps", () => {
    expect(
      keepCoverOnlyReceiptNote(
        {
          stacks_collapsed: 1201,
          pictures_moved: 3300,
          stacks_skipped_locked: [{}],
        },
        3,
      ),
    ).toBe(
      "In all, 3,300 pictures from 1,201 stacks moved to the Scrapheap. 1 stack skipped: held by a locked picture set.",
    );
  });

  // One step: the receipt's own figures are already the gesture's.
  it("adds no totals to a one-step gesture", () => {
    expect(
      keepCoverOnlyReceiptNote({ stacks_collapsed: 40, pictures_moved: 90 }, 1),
    ).toBe("");
    expect(
      keepCoverOnlyReceiptNote({ stacks_collapsed: 40, pictures_moved: 90 }),
    ).toBe("");
  });

  it("groups a merged skip count the way every other count is grouped", () => {
    const rows = Array.from({ length: 1200 }, () => ({}));
    expect(keepCoverOnlySkipNote({ stacks_skipped_locked: rows })).toBe(
      "1,200 stacks skipped: held by a locked picture set.",
    );
  });

  it("states progress as stacks done of the whole", () => {
    expect(
      keepCoverOnlyProgressSentence({ stacksDone: 500, stacksTotal: 1201 }),
    ).toBe("500 of 1,201 stacks done…");
    expect(keepCoverOnlyProgressSentence(null)).toBe("");
    expect(
      keepCoverOnlyProgressSentence({ stacksDone: 0, stacksTotal: 0 }),
    ).toBe("");
  });

  // Nothing happened: the reason is the whole story, and nothing claims a
  // partial collapse that did not occur.
  it("gives the server's reason alone when nothing was done", () => {
    expect(
      keepCoverOnlyIncompleteMessage({
        stacksDone: 0,
        stacksTotal: 1201,
        detail: "the selection changed since the preview; preview again",
      }),
    ).toBe(
      "Couldn't collapse those stacks. The selection changed since the preview; preview again.",
    );
  });

  // Some done: what happened, why it stopped, and both ways forward.
  it("says how far it got and how to finish or take it back", () => {
    expect(
      keepCoverOnlyIncompleteMessage({
        stacksDone: 500,
        stacksTotal: 1201,
        detail: "Database is busy.",
      }),
    ).toBe(
      "Collapsed 500 of 1,201 stacks, then stopped. Database is busy. The other 701 are still selected: run Keep cover only again to finish, or press Ctrl+Z to take back what was done.",
    );
  });

  // A lost answer may have committed: the message must not claim either way.
  it("never claims an outcome nobody saw", () => {
    const lost = keepCoverOnlyIncompleteMessage({
      stacksDone: 500,
      stacksTotal: 1201,
      detail: "",
      outcomeUnknown: true,
      reloaded: true,
    });
    expect(lost).toBe(
      "Collapsed 500 of 1,201 stacks, then lost the server's answer, so more of them may have collapsed too. The grid has been reloaded to show where things stand. Running Keep cover only again is safe: a stack already collapsed is skipped.",
    );
    expect(lost).not.toMatch(/still selected: run/);
    expect(
      keepCoverOnlyIncompleteMessage({
        stacksDone: 0,
        stacksTotal: 40,
        detail: "timeout of 60000ms exceeded",
        outcomeUnknown: true,
      }),
    ).toMatch(/^Lost the server's answer while collapsing those stacks/);
  });

  // Nor a reload nobody saw land.
  it("says the grid could not be reloaded when its re-read failed", () => {
    const lost = keepCoverOnlyIncompleteMessage({
      stacksDone: 500,
      stacksTotal: 1201,
      detail: "",
      outcomeUnknown: true,
      reloaded: false,
    });
    expect(lost).toBe(
      "Collapsed 500 of 1,201 stacks, then lost the server's answer, so more of them may have collapsed too. The grid could not be reloaded either, so reload the page to see where things stand. Running Keep cover only again is safe: a stack already collapsed is skipped.",
    );
    expect(lost).not.toMatch(/has been reloaded/);
  });
});

describe("keepCoverOnlyPreviewFailure", () => {
  // The reported screen: a server fault, and the only way on was to close the
  // dialog and rebuild the selection.
  it("offers a retry where one can help", () => {
    for (const kind of ["timeout", "network", "server", "conflict"]) {
      const failure = keepCoverOnlyPreviewFailure({ kind, detail: "" });
      expect(failure.retryable).toBe(true);
      expect(failure.sentence).toMatch(/so these counts are unknown\.$/);
    }
  });

  // The same request would be refused the same way again.
  it("says why instead when a retry would fail the same way", () => {
    expect(
      keepCoverOnlyPreviewFailure({
        kind: "validation",
        detail: "At most 2000 ids per request",
      }),
    ).toEqual({
      sentence: "The server refused the preview: At most 2000 ids per request.",
      retryable: false,
    });
    expect(keepCoverOnlyPreviewFailure({ kind: "forbidden" })).toEqual({
      sentence: "You don't have permission to change stacks here.",
      retryable: false,
    });
  });

  it("never tells the user to close the dialog to try again", () => {
    for (const kind of ["timeout", "network", "server", "validation"]) {
      expect(keepCoverOnlyPreviewFailure({ kind }).sentence).not.toMatch(
        /close/i,
      );
    }
  });

  it("says nothing about a preview the dialog cancelled itself", () => {
    expect(keepCoverOnlyPreviewFailure({ kind: "cancelled" })).toEqual({
      sentence: "",
      retryable: false,
    });
    expect(keepCoverOnlyPreviewFailure(null).sentence).toBe("");
  });
});
