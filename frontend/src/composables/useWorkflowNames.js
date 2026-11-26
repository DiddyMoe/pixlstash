import { reactive } from "vue";

import { listWorkflowCards } from "../api/workflows";
import { onSessionReset } from "../utils/apiClient";

// Workflow id -> the name the Workflows grid prints (`card.name`, generated
// names already told apart server-side). Read from the grid's own listing,
// hidden and one-off workflows included, so any number of ids costs one
// request rather than one detail read each; shared, because the missing-model
// note and the Replace dialog ask for the same ids.
const names = reactive(new Map());
// Every id a caller has asked for: each is looked up once per session.
const asked = new Set();
// Ids asked for since the listing in flight (or about to be) was sent.
let waiting = new Set();
// The listing in flight, or null.
let reading = null;
// A listing is due at the end of this tick: the asks of one render share it.
let scheduled = false;
// A read that lands after a credential change belongs to the old one.
let session = 0;

onSessionReset(() => {
  session += 1;
  names.clear();
  asked.clear();
  waiting = new Set();
  reading = null;
});

function schedule() {
  if (reading || scheduled) return;
  scheduled = true;
  Promise.resolve().then(() => {
    scheduled = false;
    read();
  });
}

function read() {
  const mine = session;
  waiting = new Set();
  reading = listWorkflowCards({ includeHidden: true, includeOneOffs: true })
    .then(({ cards }) => {
      if (mine !== session) return;
      for (const card of cards) {
        if (card?.id && card?.name) names.set(card.id, card.name);
      }
    })
    .catch((err) => {
      console.warn("[workflows] could not read the workflow names", err);
    })
    .finally(() => {
      if (mine !== session) return;
      reading = null;
      // An id first asked while the listing was out may be a workflow made
      // after it was read: one more listing for those, never a loop, since a
      // second miss was asked before that listing went out.
      if ([...waiting].some((id) => !names.has(id))) schedule();
      else waiting = new Set();
    });
}

/**
 * The Workflows grid's name for each workflow, read on first ask.
 *
 * `nameOf(id)` is null until the read lands, and stays null for a workflow
 * the listing does not hold or when the read fails (logged), so a caller
 * always has its own fallback to print.
 */
export function useWorkflowNames() {
  function nameOf(id) {
    if (!id) return null;
    if (!asked.has(id)) {
      asked.add(id);
      waiting.add(id);
      schedule();
    }
    return names.get(id) ?? null;
  }
  return { nameOf };
}
