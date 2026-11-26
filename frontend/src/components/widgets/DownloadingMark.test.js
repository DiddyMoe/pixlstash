// The mark a model on its way wears: a download still waiting for one
// of the server's few download slots says "Queued", with its arrow held still,
// rather than claiming bytes are arriving. Which motion the class resolves to
// is a scoped stylesheet jsdom does not apply, so the class is as far as this
// reaches.

import { describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";

vi.mock("vuetify/components", async () => {
  const { vuetifyComponentStubs } = await import("../../testing/vuetifyStubs");
  return vuetifyComponentStubs();
});

import DownloadingMark from "./DownloadingMark.vue";

const mark = (props) => mount(DownloadingMark, { props }).find('[data-testid="downloading-mark"]');

describe("DownloadingMark", () => {
  it("says Queued for a queued download, and holds its arrow still", () => {
    const queued = mark({ status: "queued", percent: 0 });
    expect(queued.attributes("aria-label")).toBe("Queued");
    expect(queued.classes()).toContain("dlm--queued");
  });

  it("says how far a running download is, its arrow dropping", () => {
    const running = mark({ status: "running", percent: 40.4 });
    expect(running.attributes("aria-label")).toBe("Downloading, 40%");
    expect(running.classes()).not.toContain("dlm--queued");
  });

  it("says Downloading where no status is given", () => {
    expect(mark({}).attributes("aria-label")).toBe("Downloading");
  });
});
