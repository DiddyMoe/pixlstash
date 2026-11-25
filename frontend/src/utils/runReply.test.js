import { describe, it, expect } from "vitest";

import { retryAfterMs, runReply } from "./runReply";

describe("utils/runReply - retryAfterMs", () => {
  it("reads a count of seconds", () => {
    expect(retryAfterMs({ "retry-after": "2" })).toBe(2000);
    expect(retryAfterMs({ "retry-after": " 0 " })).toBe(0);
  });

  it("reads axios's header object through its getter", () => {
    const headers = {
      get: (name) => (name === "retry-after" ? "4" : undefined),
    };
    expect(retryAfterMs(headers)).toBe(4000);
  });

  it("treats a missing or unreadable value as no hint", () => {
    expect(retryAfterMs(undefined)).toBeNull();
    expect(retryAfterMs({})).toBeNull();
    expect(retryAfterMs({ "retry-after": "soon" })).toBeNull();
    // `Date.parse` reads both of these as dates in 2001; taken as a date, a
    // malformed hint would become a zero wait and a tight poll.
    expect(retryAfterMs({ "retry-after": "-2" })).toBeNull();
    expect(retryAfterMs({ "retry-after": "1.5" })).toBeNull();
    // The run routes send seconds only, so a date is not read as a hint.
    expect(
      retryAfterMs({ "retry-after": "Mon, 23 Nov 2026 10:00:03 GMT" }),
    ).toBeNull();
  });
});

describe("utils/runReply - runReply", () => {
  it("pairs the body with the hint, and the very body object", () => {
    const run = { id: "run-1", active: true };
    const reply = runReply({ data: run, headers: { "retry-after": "3" } });
    expect(reply.run).toBe(run);
    expect(reply.retryAfterMs).toBe(3000);
  });

  it("answers no hint for an ended run that sends none", () => {
    expect(runReply({ data: { id: "run-2", active: false } })).toEqual({
      run: { id: "run-2", active: false },
      retryAfterMs: null,
    });
  });
});
