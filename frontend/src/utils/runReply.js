/**
 * A background run's answer: the run, plus the server's poll hint.
 *
 * Every long batch route (the exact-tier auto-stack, the Scrapheap's empty and
 * restore) is an asynchronous request-reply: the start answers 202 at once with
 * the run, and the client polls the run until it is no longer `active`. The
 * server sets the poll interval: an active run's answer carries `Retry-After`,
 * returned here as `retryAfterMs`, null when absent (an ended run has nothing
 * left to poll).
 *
 * Lives beside `unwrap` rather than in `utils/apiClient` for the same reason:
 * every `api/*` test mocks that module wholesale, and a helper there would be
 * mocked away with it.
 */

const RETRY_AFTER_HEADER = "retry-after";

/**
 * Read a `Retry-After` header as milliseconds.
 *
 * Only RFC 9110's `delay-seconds` form is read, because it is the only form
 * the run routes send. The HTTP-date form is deliberately not: `Date.parse` is
 * lenient enough to read "-2" or "1.5" as a date in 2001, which would turn a
 * malformed hint into a zero wait. Anything else is not a hint, and null
 * leaves the interval to the caller.
 *
 * @param {Object} [headers] - the axios response headers.
 * @returns {number|null}
 */
export function retryAfterMs(headers) {
  const raw =
    typeof headers?.get === "function"
      ? headers.get(RETRY_AFTER_HEADER)
      : headers?.[RETRY_AFTER_HEADER];
  if (raw == null) return null;
  const text = String(raw).trim();
  return /^\d+$/.test(text) ? Number(text) * 1000 : null;
}

/**
 * Turn a run route's axios response into `{run, retryAfterMs}`.
 *
 * @param {{data: Object, headers?: Object}} response
 * @returns {{run: Object, retryAfterMs: number|null}}
 */
export function runReply(response) {
  return { run: response.data, retryAfterMs: retryAfterMs(response.headers) };
}
