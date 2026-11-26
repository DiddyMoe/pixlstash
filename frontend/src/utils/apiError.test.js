import { describe, it, expect } from "vitest";
import { errorDetail, errorMessage, requestFailure } from "./apiError";

const err = (detail, status) => ({ response: { data: { detail }, status } });

describe("errorDetail", () => {
  it("returns the server's sentence, trimmed", () => {
    expect(errorDetail(err("  A stack needs two pictures.  "))).toBe(
      "A stack needs two pictures.",
    );
  });

  it("names the sets on a structured locked-set refusal", () => {
    const detail = { code: "locked_set", sets: [{ name: "Keepers" }] };
    expect(errorDetail(err(detail, 423))).toContain("Keepers");
  });

  // The bug this module exists for: `detail || fallback` takes the OBJECT when
  // the refusal is structured, and the user reads "[object Object]".
  it("never returns a non-string", () => {
    for (const detail of [{ code: "something_else" }, [{ msg: "x" }], 42, null])
      expect(typeof errorDetail(err(detail))).toBe("string");
  });

  it("is empty when there is no response at all", () => {
    expect(errorDetail(new Error("Network Error"))).toBe("");
    expect(errorDetail(undefined)).toBe("");
  });
});

describe("errorMessage", () => {
  it("prefers the server, then the transport, then the caller's copy", () => {
    expect(errorMessage(err("Refused."), "fallback")).toBe("Refused.");
    expect(errorMessage(new Error("Network Error"), "fallback")).toBe(
      "Network Error",
    );
    expect(errorMessage({}, "Could not save.")).toBe("Could not save.");
  });
});

describe("requestFailure", () => {
  // The kind decides the next step, so every branch is pinned: a retry helps
  // the first three and not the rest.
  it("tells the failures a retry can fix from those it cannot", () => {
    const kinds = [
      [{ code: "ECONNABORTED", message: "timeout of 60000ms exceeded" }, "timeout"],
      [new Error("Network Error"), "network"],
      [err("boom", 500), "server"],
      [err("too many ids", 400), "validation"],
      [err([{ msg: "field required" }], 422), "validation"],
      [err("no", 403), "forbidden"],
      [err("preview again", 409), "conflict"],
    ];
    for (const [failure, kind] of kinds) {
      expect(requestFailure(failure).kind).toBe(kind);
    }
  });

  // An abort arrives without a response, and must not read as "unreachable".
  it("reads a cancelled request as cancelled, not as a network failure", () => {
    expect(requestFailure({ code: "ERR_CANCELED" }).kind).toBe("cancelled");
    expect(requestFailure(new DOMException("x", "AbortError")).kind).toBe(
      "cancelled",
    );
  });

  // A 422's `input` echoes what was submitted; only the message is shown.
  it("takes a field error's message and never its echoed input", () => {
    const failure = requestFailure(
      err([{ msg: "Too many ids", input: [1, 2, 3] }], 422),
    );
    expect(failure).toEqual({
      kind: "validation",
      status: 422,
      detail: "Too many ids",
      code: null,
    });
  });

  it("keeps a structured refusal's code and message", () => {
    expect(
      requestFailure(err({ code: "plan_grew", message: "Preview again." }, 409)),
    ).toEqual({
      kind: "conflict",
      status: 409,
      detail: "Preview again.",
      code: "plan_grew",
    });
  });
});

// utils/apiClient.js marks the saved-address 409 once its notice is up; the
// sentence is then the card's alone.
describe("an error the saved-address notice has already shown", () => {
  const shown = () => ({
    ...err("The saved ComfyUI address is not valid.", 409),
    message: "Request failed with status code 409",
    savedAddressNoticed: true,
  });

  it("gives no detail and leaves the caller's copy", () => {
    expect(errorDetail(shown())).toBe("");
    expect(errorMessage(shown(), "Could not load the packs.")).toBe(
      "Could not load the packs.",
    );
    expect(requestFailure(shown())).toEqual({
      kind: "conflict",
      status: 409,
      detail: "",
      code: null,
    });
  });

  it("leaves an unmarked 409 its detail", () => {
    const plain = err("Preview again.", 409);
    expect(errorDetail(plain)).toBe("Preview again.");
    expect(errorMessage(plain, "fallback")).toBe("Preview again.");
    expect(requestFailure(plain).detail).toBe("Preview again.");
  });
});
