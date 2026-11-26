import { describe, it, expect } from "vitest";
import { isImeComposing } from "./imeComposition.js";

describe("isImeComposing", () => {
  it("is true while a composition is open", () => {
    expect(
      isImeComposing(
        new KeyboardEvent("keydown", { key: "Enter", isComposing: true }),
      ),
    ).toBe(true);
  });

  it("is true for the key that ends a composition in Safari (keyCode 229)", () => {
    expect(
      isImeComposing(
        new KeyboardEvent("keydown", { key: "Enter", keyCode: 229 }),
      ),
    ).toBe(true);
  });

  it("is false for a plain key", () => {
    expect(
      isImeComposing(
        new KeyboardEvent("keydown", { key: "Enter", keyCode: 13 }),
      ),
    ).toBe(false);
    expect(
      isImeComposing(new KeyboardEvent("keydown", { key: "Escape" })),
    ).toBe(false);
  });
});
