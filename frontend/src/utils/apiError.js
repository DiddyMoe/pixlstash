// The server's own explanation for a refusal, for the 70-odd places that show
// one.
//
// FastAPI answers a refusal with `detail`, and the app's habit was to inline
// `err?.response?.data?.detail || err?.message || "…"` at every catch site.
// That chain is wrong whenever `detail` is not a string: the backend serves the
// locked-set refusal as `{code, action, sets, picture_ids}` precisely so the
// client can write its own copy, and `object || fallback` takes the OBJECT -
// which reaches the user as "[object Object]", with the one refusal they could
// have acted on being the one that breaks.
//
// So the extraction lives here, once, and knows the difference.

import { isLockedRefusal, lockedSetsSentence } from "./dedup";

/**
 * The server's sentence for a rejection, or "" when it did not give one, or
 * when the saved-address notice has already shown it.
 *
 * Always a string. A structured locked-set refusal becomes the sentence naming
 * the sets; any other non-string detail becomes "", so the caller's own
 * fallback is what the user reads.
 *
 * @param {*} err - the rejection a catch block received.
 * @returns {string}
 */
export function errorDetail(err) {
  // The saved-address notice already said it (utils/apiClient.js).
  if (err?.savedAddressNoticed) return "";
  const detail = err?.response?.data?.detail;
  if (typeof detail === "string") return detail.trim();
  if (isLockedRefusal(err)) return lockedSetsSentence(detail?.sets);
  return "";
}

/**
 * The sentence to show the user for a failed request.
 *
 * The server's reason if it gave one, then the transport's message, then the
 * caller's own copy. Pass a fallback that says what failed - `err.message` on
 * its own is "Request failed with status code 500".
 *
 * @param {*} err
 * @param {string} fallback
 * @returns {string}
 */
export function errorMessage(err, fallback) {
  // Not the transport's "Request failed with status code 409" either: the
  // notice gave the reason, the caller's copy says what failed.
  if (err?.savedAddressNoticed) return fallback;
  return errorDetail(err) || err?.message || fallback;
}

/**
 * Why a request failed, as a kind a caller can word and act on.
 *
 * `errorDetail` answers "what did the server say"; this answers "what kind of
 * failure was it", which decides the next step. A timeout, an unreachable
 * server and a server fault are worth a retry; a refused request or a missing
 * permission are not, and need their reason said instead.
 *
 * Checked in this order because the cases overlap: an aborted or timed-out
 * request also arrives without a response, and only the remainder is a server
 * that could not be reached. `detail` is the server's own sentence, or for a
 * refused body (a list of field errors) the first error's message only: never
 * its `input`, which echoes the submitted value. A structured detail
 * (`{code, message, ...}`) gives its `message` and its machine-readable `code`.
 *
 * @param {*} err - the rejection a catch block received.
 * @returns {{kind: string, status: (number|null), detail: string,
 *   code: (string|null)}} `kind` is `cancelled` (the signal fired), `timeout`,
 *   `network` (no response), `validation` (400/422), `forbidden` (401/403),
 *   `conflict` (409) or `server` (any other status).
 */
export function requestFailure(err) {
  const status = Number(err?.response?.status) || null;
  const served = err?.response?.data?.detail;
  let detail = "";
  let code = null;
  if (typeof served === "string") {
    detail = served.trim();
  } else if (Array.isArray(served) && typeof served[0]?.msg === "string") {
    detail = served[0].msg.trim();
  } else if (served && typeof served === "object") {
    if (typeof served.message === "string") detail = served.message.trim();
    if (typeof served.code === "string") code = served.code;
  }
  // The saved-address notice already said it (utils/apiClient.js).
  if (err?.savedAddressNoticed) detail = "";
  let kind;
  if (
    err?.code === "ERR_CANCELED" ||
    err?.name === "CanceledError" ||
    err?.name === "AbortError"
  ) {
    kind = "cancelled";
  } else if (err?.code === "ECONNABORTED" || err?.code === "ETIMEDOUT") {
    kind = "timeout";
  } else if (!err?.response) {
    kind = "network";
  } else if (status === 400 || status === 422) {
    kind = "validation";
  } else if (status === 401 || status === 403) {
    kind = "forbidden";
  } else if (status === 409) {
    kind = "conflict";
  } else {
    kind = "server";
  }
  return { kind, status, detail, code };
}
