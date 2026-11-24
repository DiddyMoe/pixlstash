// Keep-cover-only: the copy, as pure functions.
//
// Collapsing a stack to its cover is the one destructive action on the dedup
// surface, so the sentences around it are load-bearing and every one of them is
// a place the UI could lie. They live here, tested without a mounted view, for
// the same reason the dedup verdict copy does.
//
// Two rules run through the whole file, and both come from
// `docs/design/keep-cover-only.md`:
//
//   * **Nothing is freed.** A soft delete moves rows; the files stay on disk
//     until the Scrapheap is emptied. So the byte figure is a sentence about
//     what could later be reclaimed, never a figure block, and the words
//     "freed", "reclaimed" and "saved" do not appear.
//   * **The retention window is read, never assumed.** `scrapheap_retention_days`
//     defaults to `null`, which means the Scrapheap never empties on its own.
//     Hardcoding "30 days" would be exactly the class of error the confirm
//     dialog exists to avoid.

import { normalizeRetentionDays, retentionLabel } from "./retention";
import { getPictureStackId, getStackBadgeCount } from "./stack";

/**
 * The placeholder a figure shows while its number is unknown.
 *
 * An en dash at the figure's own size, not a spinner and never a zero: the
 * dialog must keep its height while the preview lands, and a zero would read as
 * "there is nothing to collapse" when the truth is "nobody has asked yet".
 * Matches the neighbouring auto-stack dialog.
 */
export const UNKNOWN_FIGURE = "–";

/** The action's name, once, so the menus and the dialog cannot drift. */
const KEEP_COVER_ONLY_LABEL = "Keep cover only";

/**
 * The glyph, once: the inverse of the mdi-layers-plus the user pressed to build
 * these stacks. Not `mdi-delete`, which would over-claim; nothing leaves disk.
 * The same glyph rides the menu item, the confirm button and the receipt, so
 * the operation is named identically at all three moments.
 */
export const KEEP_COVER_ONLY_ICON = "mdi-layers-minus";

/** The same glyph without the `mdi-` prefix, which is what AppButton takes. */
export const KEEP_COVER_ONLY_ICON_NAME = "layers-minus";

/**
 * The stacks a grid selection names, as `[{ id, pictureIds }]`.
 *
 * The action's unit is the **stack**, so this is what decides both what the
 * menu offers and what its label counts. Two things it deliberately does:
 *
 *   * **Loose pictures contribute nothing.** A picture with no `stack_id` names
 *     no stack, which is why a mixed selection can be acted on honestly at all.
 *   * **A stack with one live member is not a stack to collapse.** A collapsed
 *     leader tile carries the whole stack's `stack_count`, and an expanded
 *     stack shows its members as siblings sharing one `stack_id`; a `stack_id`
 *     with neither signal is skipped, so the label never counts a stack that has
 *     nothing to lose. The server's preview stays authoritative; this only
 *     decides what is offered.
 *
 * @param {Object} options
 * @param {Array<number|string>} options.selectedIds - the grid selection.
 * @param {Array<Object>} options.images - the grid's mounted picture objects.
 * @returns {Array<{id: string, pictureIds: number[]}>} one entry per stack, in
 *   the order the selection first named it.
 */
export function selectedKeepCoverOnlyStacks({ selectedIds, images } = {}) {
  const ids = (Array.isArray(selectedIds) ? selectedIds : [])
    .map((id) => Number(id))
    .filter((id) => Number.isFinite(id) && id > 0);
  if (!ids.length) return [];
  const pictures = Array.isArray(images) ? images : [];

  const byId = new Map();
  const membersByStack = new Map();
  for (const img of pictures) {
    if (!img || img.id == null) continue;
    byId.set(String(img.id), img);
    const stackId = getPictureStackId(img);
    if (!stackId) continue;
    const held = membersByStack.get(stackId);
    if (held) held.push(img);
    else membersByStack.set(stackId, [img]);
  }

  const stacks = new Map();
  for (const id of ids) {
    const stackId = getPictureStackId(byId.get(String(id)));
    if (!stackId || stacks.has(stackId)) continue;
    const members = membersByStack.get(stackId) || [];
    const collapsible =
      members.length > 1 || members.some((m) => getStackBadgeCount(m) > 1);
    if (!collapsible) continue;
    stacks.set(
      stackId,
      members
        .map((m) => Number(m.id))
        .filter((n) => Number.isFinite(n) && n > 0),
    );
  }
  return [...stacks].map(([id, pictureIds]) => ({ id, pictureIds }));
}

/**
 * Why Keep cover only is unavailable, or `null` when it is available.
 *
 * A locked picture set refuses the **whole** stack, never one member: stack
 * membership reconciles to the union of its members' sets, so removing a member
 * from a stack a locked set touches is exactly the mutation the lock forbids,
 * and a partial collapse would be the worst outcome available.
 *
 * The gate therefore fires only when EVERY named stack is locked, which is the
 * one case where the action provably cannot do anything. A mixed selection stays
 * enabled and the confirm dialog reports the skips, which is the same rule the
 * shipped Delete item follows.
 *
 * @param {Object} options
 * @param {Array<{pictureIds: number[]}>} options.stacks - from
 *   {@link selectedKeepCoverOnlyStacks}.
 * @param {(id: number) => boolean} options.isLocked
 * @param {(id: number) => string[]} options.lockedSetNames
 * @returns {string|null}
 */
export function keepCoverOnlyLockReason({
  stacks,
  isLocked,
  lockedSetNames,
} = {}) {
  const named = Array.isArray(stacks) ? stacks : [];
  if (!named.length) return null;
  const names = new Set();
  for (const stack of named) {
    const pictureIds = Array.isArray(stack?.pictureIds) ? stack.pictureIds : [];
    const lockedId = pictureIds.find((id) => isLocked?.(id));
    // One unlocked stack is enough: the run has work to do, and the dialog will
    // report whatever the server refuses.
    if (lockedId === undefined) return null;
    for (const name of lockedSetNames?.(lockedId) ?? []) names.add(name);
  }
  const joined = [...names].join(", ");
  // "Every selected stack" reads as a warning about a bulk action, which is
  // wrong for one stack; the two shapes keep the sentence about what is in
  // front of the user.
  const subject =
    named.length === 1 ? "This stack is" : "Every selected stack is";
  return (
    `Locked: ${subject} held by the locked set '${joined}'. A locked set ` +
    `refuses the whole stack, so unlock it first: right-click the set in the ` +
    `sidebar and choose Unlock.`
  );
}

/**
 * The menu item's label, which has to state the unit it acts in.
 *
 * The action's unit is the **stack**, so a mixed selection ignores the loose
 * pictures in it. That is only honest if the label counts stacks rather than
 * echoing the selection's own count, which is what these two shapes are for:
 *
 *   * everything selected is stacked → `Keep cover only (3 stacks)`;
 *   * some of it is not             → `Keep cover only (12 of 20)`, so the
 *     eight pictures that will be left alone are visible before the click.
 *
 * @param {Object} options
 * @param {number} options.stackCount - distinct stacks the selection names.
 * @param {number} options.selectedCount - pictures (tiles) selected.
 * @returns {string}
 */
export function keepCoverOnlyMenuLabel({
  stackCount,
  selectedCount,
  label = KEEP_COVER_ONLY_LABEL,
} = {}) {
  const stacks = Number(stackCount) || 0;
  const selected = Number(selectedCount) || 0;
  if (stacks <= 0) return label;
  // A selection whose every tile is a stack tile needs no "of": the two numbers
  // would be the same, and "3 of 3" reads as a warning about nothing.
  if (selected > stacks) {
    return `${label} (${stacks} of ${selected})`;
  }
  const noun = stacks === 1 ? "stack" : "stacks";
  return `${label} (${stacks} ${noun})`;
}

/**
 * The dialog title: what you KEEP.
 *
 * The title/button pairing is the dialog's safety property: the title names
 * what survives and the button names what goes, so this one never mentions the
 * Scrapheap. Falls back to the unnumbered form while the preview is unknown,
 * because a title is not a figure block and an en dash inside a sentence reads
 * as a typo.
 *
 * @param {number|null} stacksEligible - stacks that would actually collapse, or
 *   `null` while the preview is in flight or has failed.
 * @returns {string}
 */
export function keepCoverOnlyTitle(stacksEligible) {
  if (stacksEligible === null || stacksEligible === undefined) {
    return "Keep only the cover";
  }
  const count = Number(stacksEligible) || 0;
  const noun = count === 1 ? "stack" : "stacks";
  return `Keep only the cover of ${count.toLocaleString()} ${noun}`;
}

/**
 * The confirm button: what you LOSE.
 *
 * Built from the same number the headline figure renders, so the two cannot
 * disagree. While that number is unknown the label drops it rather than
 * printing a placeholder or a stale count; the button is disabled in that state
 * anyway, and a button that names a figure nobody has seen is the exact failure
 * the neighbouring auto-stack dialog shipped.
 *
 * @param {number|null} picturesMoving - pictures that would move, or `null`.
 * @returns {string}
 */
export function keepCoverOnlyConfirmLabel(picturesMoving) {
  if (picturesMoving === null || picturesMoving === undefined) {
    return "Move to the Scrapheap";
  }
  return `Move ${(Number(picturesMoving) || 0).toLocaleString()} to the Scrapheap`;
}

/**
 * What happens to a scrapheaped copy over time, from the LIVE setting.
 *
 * `null` (the default on a fresh install) means auto-purge is off entirely, so
 * the sentence has to say so: a copy sits there until the user empties the
 * Scrapheap themselves. Anything else names the window the server actually
 * carries. Nothing here is derived from a constant.
 *
 * @param {number|null|undefined} days - the preview's `scrapheap_retention_days`.
 * @returns {string}
 */
export function keepCoverOnlyRetentionSentence(days) {
  const window = normalizeRetentionDays(days);
  if (window === null) {
    return "The Scrapheap never empties on its own, so nothing goes until you empty it yourself.";
  }
  return `The Scrapheap empties itself after ${retentionLabel(window)}, and until then every copy can be restored.`;
}

/**
 * Bytes, at the same scale the rest of the app uses.
 * @param {number} bytes
 * @returns {string} e.g. `"1.2 GB"`.
 */
function humanBytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let scaled = value;
  let unit = 0;
  while (scaled >= 1024 && unit < units.length - 1) {
    scaled /= 1024;
    unit += 1;
  }
  return `${scaled.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

/**
 * What the copies are holding on disk, as a SENTENCE, deliberately.
 *
 * The general rule this is an instance of: a figure block is for what changes
 * now, a sentence is for what changes later. Nothing is freed by this action,
 * so promoting the byte count to a figure would read as a reclaim that has
 * happened. It has not, and with retention off it never will on its own.
 *
 * @param {number} bytes - the preview's `bytes_held_by_copies`.
 * @returns {string} `""` when there is nothing to say.
 */
export function keepCoverOnlyBytesSentence(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value <= 0) return "";
  return `Those copies hold ${humanBytes(value)} on disk, which stays there until the Scrapheap is emptied.`;
}

/**
 * One row per skip reason, so the dialog can name what it is leaving alone.
 *
 * The three buckets are counted directly by the server and are disjoint; this
 * only picks the non-empty ones and words them. A locked stack is refused
 * WHOLE, and the sentence says so, because a partial collapse is the worst
 * outcome available and the user needs to know it was not attempted.
 *
 * @param {Object|null} preview - the dry-run body.
 * @returns {Array<{key: string, text: string}>}
 */
export function keepCoverOnlySkipReasons(preview) {
  if (!preview) return [];
  const rows = [];
  const locked = Number(preview.stacks_skipped_locked) || 0;
  const character = Number(preview.stacks_skipped_character_on_copy) || 0;
  const single = Number(preview.stacks_skipped_single_member) || 0;
  const nothing = Number(preview.stacks_skipped_nothing_reproducible) || 0;
  if (nothing > 0) {
    rows.push({
      key: "nothing_reproducible",
      text: `${nothing} ${nothing === 1 ? "stack has" : "stacks have"} no picture besides the cover that could be made again.`,
    });
  }
  if (locked > 0) {
    rows.push({
      key: "locked",
      text:
        `${locked} ${locked === 1 ? "stack is" : "stacks are"} held whole by a locked ` +
        `picture set. Unlock the set to collapse ${locked === 1 ? "it" : "them"}.`,
    });
  }
  if (character > 0) {
    rows.push({
      key: "character_on_copy",
      text:
        `${character} ${character === 1 ? "stack keeps" : "stacks keep"} a person's only link on a ` +
        `copy, so collapsing would lose it.`,
    });
  }
  if (single > 0) {
    rows.push({
      key: "single_member",
      text: `${single} ${single === 1 ? "stack has" : "stacks have"} nothing left to collapse.`,
    });
  }
  return rows;
}

/**
 * Total stacks the run would leave alone.
 *
 * A SUM of three directly-counted, disjoint buckets; never `stacks_selected`
 * minus `stacks_eligible`. Deriving a bucket by subtraction is how a dialog
 * ends up reporting a number no query ever produced.
 *
 * @param {Object|null} preview
 * @returns {number}
 */
export function keepCoverOnlySkippedCount(preview) {
  if (!preview) return 0;
  return (
    (Number(preview.stacks_skipped_locked) || 0) +
    (Number(preview.stacks_skipped_character_on_copy) || 0) +
    (Number(preview.stacks_skipped_single_member) || 0) +
    (Number(preview.stacks_skipped_nothing_reproducible) || 0)
  );
}

/**
 * The receipt's second sentence: what the run did NOT touch.
 *
 * A skip belongs on the same pill as the move, not in a notice of its own: two
 * surfaces for one action means the user reads the reassuring half and dismisses
 * the half that needed a decision. The response's skip buckets are lists here
 * (the preview's are counts), so their lengths are the figures.
 *
 * @param {Object|null} result - the mutation's response body.
 * @returns {string} `""` when nothing was skipped.
 */
export function keepCoverOnlySkipNote(result) {
  if (!result) return "";
  const locked = Array.isArray(result.stacks_skipped_locked)
    ? result.stacks_skipped_locked.length
    : 0;
  const character = Array.isArray(result.stacks_skipped_character_on_copy)
    ? result.stacks_skipped_character_on_copy.length
    : 0;
  const nothing = Array.isArray(result.stacks_skipped_nothing_reproducible)
    ? result.stacks_skipped_nothing_reproducible.length
    : 0;
  const total = locked + character;
  const staying = Number(result.pictures_staying) || 0;
  const stayingNote =
    staying > 0
      ? `${staying.toLocaleString()} ${staying === 1 ? "picture stays" : "pictures stay"}: ${staying === 1 ? "it" : "they"} could not be made again.`
      : "";
  const nothingNote =
    nothing > 0
      ? `${nothing} ${nothing === 1 ? "stack" : "stacks"} unchanged: nothing in ${nothing === 1 ? "it" : "them"} could be made again.`
      : "";
  if (total <= 0) return [nothingNote, stayingNote].filter(Boolean).join(" ");
  const noun = total === 1 ? "stack" : "stacks";
  let note;
  if (locked && character) {
    note = `${total.toLocaleString()} ${noun} skipped: ${locked.toLocaleString()} locked, ${character.toLocaleString()} holding a person's only link.`;
  } else if (locked) {
    note = `${locked.toLocaleString()} ${noun} skipped: held by a locked picture set.`;
  } else {
    note = `${character.toLocaleString()} ${noun} skipped: a person's only link sits on a copy.`;
  }
  return [note, nothingNote, stayingNote].filter(Boolean).join(" ");
}

/**
 * The receipt's second sentence for a whole gesture.
 *
 * A selection larger than one request is collapsed as several operations under
 * one batch id, and the receipt names the newest of them with a `+N` for the
 * rest. Its figures are that one request's, so when there was more than one,
 * the gesture's own totals lead the note: "Kept the cover of 201 stacks" must
 * not be the last word on a Ctrl+A that collapsed 1,201.
 *
 * @param {Object|null} result - the merged collapse result.
 * @param {number} [steps=1] - operations the gesture recorded: requests that
 *   collapsed at least one stack.
 * @returns {string} `""` when there is nothing to add.
 */
export function keepCoverOnlyReceiptNote(result, steps = 1) {
  const skipNote = keepCoverOnlySkipNote(result);
  if (!result || !(Number(steps) > 1)) return skipNote;
  const stacks = Number(result.stacks_collapsed) || 0;
  const pictures = Number(result.pictures_moved) || 0;
  const totals = `In all, ${pictures.toLocaleString()} ${pictures === 1 ? "picture" : "pictures"} from ${stacks.toLocaleString()} ${stacks === 1 ? "stack" : "stacks"} moved to the Scrapheap.`;
  return [totals, skipNote].filter(Boolean).join(" ");
}

/**
 * Where a collapse of several requests has got to, for the dialog's status line.
 *
 * Shown only when the selection takes more than one request: one request is
 * answered within the button's own spinner, and a line that flashes "0 of 40"
 * for a moment says nothing.
 *
 * @param {{stacksDone: number, stacksTotal: number}|null} progress
 * @returns {string} `""` with no progress to report.
 */
export function keepCoverOnlyProgressSentence(progress) {
  if (!progress) return "";
  const total = Number(progress.stacksTotal) || 0;
  if (total <= 0) return "";
  const done = Math.min(Number(progress.stacksDone) || 0, total);
  return `${done.toLocaleString()} of ${total.toLocaleString()} stacks done…`;
}

/** A sentence the server or the transport gave, ending in a full stop. */
function asSentence(text) {
  const trimmed = String(text ?? "").trim();
  if (!trimmed) return "";
  const capital = trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
  return /[.!?…]$/.test(capital) ? capital : `${capital}.`;
}

/**
 * The notice for a collapse that did not finish.
 *
 * Says what happened, why, and what to do next, in that order, because the
 * three cases need different next steps:
 *
 *   * nothing done and the server said why: the selection and the dialog are
 *     untouched, so the reason is the whole story;
 *   * some requests done, then a refusal: the done part is real and one Ctrl+Z
 *     away, and the rest is still selected to run again;
 *   * the answer was lost (a timeout, a dropped connection): the request may
 *     have committed, so the grid is re-read and the message cannot claim
 *     either way. Rerunning is still safe: a collapsed stack has one live
 *     member left, and the server skips a stack of one without writing.
 *
 * @param {Object} details
 * @param {number} details.stacksDone - stacks in the requests that answered.
 * @param {number} details.stacksTotal - stacks in the whole selection.
 * @param {string} [details.detail] - the server's or the transport's reason.
 * @param {boolean} [details.outcomeUnknown] - the failed request got no answer.
 * @returns {string}
 */
export function keepCoverOnlyIncompleteMessage({
  stacksDone,
  stacksTotal,
  detail,
  outcomeUnknown = false,
}) {
  const done = Number(stacksDone) || 0;
  const total = Number(stacksTotal) || 0;
  const reason = asSentence(detail);
  if (outcomeUnknown) {
    const lead =
      done > 0
        ? `Collapsed ${done.toLocaleString()} of ${total.toLocaleString()} stacks, then lost the server's answer, so more of them may have collapsed too.`
        : "Lost the server's answer while collapsing those stacks, so some of them may have collapsed.";
    return [
      lead,
      reason,
      "The grid has been reloaded to show where things stand. Running Keep cover only again is safe: a stack already collapsed is skipped.",
    ]
      .filter(Boolean)
      .join(" ");
  }
  if (done <= 0) {
    return ["Couldn't collapse those stacks.", reason].filter(Boolean).join(" ");
  }
  const left = Math.max(total - done, 0);
  return [
    `Collapsed ${done.toLocaleString()} of ${total.toLocaleString()} stacks, then stopped.`,
    reason,
    `The other ${left.toLocaleString()} ${left === 1 ? "is" : "are"} still selected: run Keep cover only again to finish, or press Ctrl+Z to take back what was done.`,
  ]
    .filter(Boolean)
    .join(" ");
}

/**
 * What the dialog says when its preview could not be read, and whether a retry
 * can help.
 *
 * The counts are unknown either way, so the dialog shows no figure; the kind
 * decides the rest. A timeout, an unreachable server and a server fault are
 * worth a retry, so those offer one in place: "close this and try again" made
 * the user rebuild a Ctrl+A selection to ask the same question twice. A refused
 * request and a missing permission would fail the same way again, so those say
 * why instead. A cancelled preview says nothing: it was cancelled because the
 * dialog moved on.
 *
 * @param {{kind: string, detail?: string}|null} failure - from
 *   `requestFailure` in `utils/apiError.js`.
 * @returns {{sentence: string, retryable: boolean}}
 */
export function keepCoverOnlyPreviewFailure(failure) {
  if (!failure || failure.kind === "cancelled") {
    return { sentence: "", retryable: false };
  }
  const reason = String(failure.detail ?? "").trim();
  const ended = reason && !/[.!?…]$/.test(reason) ? `${reason}.` : reason;
  switch (failure.kind) {
    case "validation":
      return {
        sentence: ended
          ? `The server refused the preview: ${ended}`
          : "The server refused the preview.",
        retryable: false,
      };
    case "forbidden":
      return {
        sentence: "You don't have permission to change stacks here.",
        retryable: false,
      };
    case "timeout":
      return {
        sentence:
          "The preview took too long to answer, so these counts are unknown.",
        retryable: true,
      };
    case "network":
      return {
        sentence: "The server could not be reached, so these counts are unknown.",
        retryable: true,
      };
    default:
      return {
        sentence: [
          "The preview failed on the server, so these counts are unknown.",
          ended,
        ]
          .filter(Boolean)
          .join(" "),
        retryable: true,
      };
  }
}
