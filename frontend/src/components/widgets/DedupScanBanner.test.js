// The streaming "still scanning" banner above the duplicate queue.
//
// The banner is what makes a partial queue honest, so the tests pin when it
// appears and disappears, that it never states a figure it does not have, and
// that its progress track is announced as a progress bar rather than as a
// decorative strip.

import { describe, it, expect } from "vitest";
import { mount } from "@vue/test-utils";

import DedupScanBanner from "./DedupScanBanner.vue";

const globalOpts = { global: { stubs: { "v-icon": true } } };

/** The store's normalised scan record, which is what the banner is given. */
function scan(overrides = {}) {
  return {
    status: "running",
    phase: "exact",
    step: { number: 1, of: 1, label: "Exact matches" },
    scanned: 1200,
    total: 48000,
    percent: 2.5,
    measured: true,
    buckets: 0,
    totalBuckets: 0,
    error: null,
    ...overrides,
  };
}

/** Whether the meter is sliding ("working, length unknown"). */
function slides(wrapper) {
  return wrapper
    .find(".scan-banner__fill")
    .classes()
    .includes("scan-banner__fill--indeterminate");
}

/** Text with runs of whitespace collapsed, so template wrapping is not asserted. */
function textOf(wrapper) {
  return wrapper.text().replace(/\s+/g, " ");
}

function mountBanner(overrides = {}) {
  return mount(DedupScanBanner, {
    ...globalOpts,
    props: { scan: scan(overrides) },
  });
}

describe("DedupScanBanner", () => {
  it("reports how far the comparison has got", () => {
    // Without the counts a short queue reads as "you have almost no
    // duplicates" rather than "the scan is 2% in".
    const text = textOf(mountBanner());
    expect(text).toContain("Still scanning.");
    expect(text).toContain(
      `${(1200).toLocaleString()} of ${(48000).toLocaleString()} pictures compared.`,
    );
    expect(text).toContain("Groups appear here as they are found.");
  });

  // Once candidate buckets exist they describe the expensive comparison phase,
  // even if picture enumeration has already reached its total.
  it("prefers candidate batches for a running person scan", () => {
    const text = textOf(
      mountBanner({
        phase: "near",
        step: { number: 2, of: 2, label: "Near-identical" },
        total: 8,
        scanned: 8,
        buckets: 3,
        totalBuckets: 12,
        percent: 25,
      }),
    );
    expect(text).toContain("3 of 12 candidate batches compared.");
    expect(text).not.toContain("pictures compared");
    expect(text).toContain("25%");
  });

  it("keeps a running person scan visible when its counters reach 100 percent", () => {
    expect(mountBanner({ percent: 100 }).find(".scan-banner").exists()).toBe(true);
  });

  it("renders nothing when the scan completed or is idle", () => {
    // A finished, idle or failed scan must not leave a banner frozen at 99
    // percent. Incomplete terminal states have their own persistent warning.
    for (const status of ["idle", "complete"]) {
      expect(
        mountBanner({ status, percent: 40 }).find(".scan-banner").exists(),
      ).toBe(false);
    }
  });

  it.each([
    ["partial", "Duplicate scan incomplete.", "pair limit reached"],
    ["failed", "Duplicate scan failed.", "worker stopped"],
  ])("keeps a %s scan visible as an incomplete warning", (status, copy, error) => {
    const wrapper = mountBanner({ status, error });
    expect(textOf(wrapper)).toContain(copy);
    expect(textOf(wrapper)).toContain(error);
    expect(wrapper.find(".scan-banner").attributes("role")).toBe("alert");
    expect(wrapper.find(".scan-banner__track").exists()).toBe(false);
  });

  it("announces a pending person scan as queued without inventing zero totals", () => {
    const wrapper = mountBanner({
      status: "pending",
      phase: null,
      step: null,
      scanned: 0,
      total: 0,
      buckets: 0,
      totalBuckets: 0,
      percent: 0,
      measured: false,
    });
    expect(textOf(wrapper)).toContain("Duplicate scan queued.");
    expect(textOf(wrapper)).toContain("Queued");
    expect(textOf(wrapper)).not.toContain("0 of 0");
    expect(wrapper.find(".scan-banner").attributes("role")).toBe("status");
    expect(
      wrapper.find(".scan-banner__track").attributes("aria-valuenow"),
    ).toBeUndefined();
    // Queued is waiting, not working: a sliding meter would claim otherwise.
    expect(slides(wrapper)).toBe(false);
  });

  it("announces a running person scan with unknown totals as starting", () => {
    const wrapper = mountBanner({
      status: "running",
      phase: null,
      step: null,
      scanned: 0,
      total: 0,
      buckets: 0,
      totalBuckets: 0,
      percent: 0,
      measured: false,
    });
    expect(textOf(wrapper)).toContain("Duplicate scan is starting.");
    expect(textOf(wrapper)).toContain("Starting");
    expect(textOf(wrapper)).not.toContain("0 of 0");
    expect(
      wrapper.find(".scan-banner__track").attributes("aria-valuenow"),
    ).toBeUndefined();
    expect(slides(wrapper)).toBe(true);
  });

  // The reported fault: an exact-only scan read "0 of 376,889" until it was
  // done. A figure the store did not measure must never be printed as one,
  // and a figure it did measure must be.
  it("prints a figure only when the store measured it", () => {
    const unmeasured = mountBanner({ scanned: 0, percent: 0, measured: false });
    expect(textOf(unmeasured)).not.toContain("pictures compared");
    expect(textOf(unmeasured)).not.toContain("0%");
    expect(slides(unmeasured)).toBe(true);

    const measured = mountBanner({ scanned: 12000, percent: 25 });
    expect(textOf(measured)).toContain(
      `${(12000).toLocaleString()} of ${(48000).toLocaleString()} pictures compared.`,
    );
    expect(slides(measured)).toBe(false);
  });

  it("says the near tier is building its batches before it can count them", () => {
    const wrapper = mountBanner({
      phase: "near",
      step: { number: 2, of: 2, label: "Near-identical" },
      scanned: 48000,
      percent: 0,
      measured: false,
      totalBuckets: 0,
    });
    const text = textOf(wrapper);
    expect(text).toContain("Sorting pictures into candidate batches to compare.");
    expect(text).toContain("Near-identical, step 2 of 2");
    // The exact tier's finished count is not this tier's progress.
    expect(text).not.toContain("pictures compared");
    expect(text).not.toContain("%");
    expect(
      wrapper.find(".scan-banner__track").attributes("aria-label"),
    ).toBe("Duplicate candidate batches being prepared");
    expect(
      wrapper.find(".scan-banner__track").attributes("aria-valuenow"),
    ).toBeUndefined();
    expect(slides(wrapper)).toBe(true);
  });

  // The embedding tier has no counter, and it writes its groups only once its
  // comparison is complete, so "groups appear as they are found" would be
  // false here.
  it("describes the same-scene tier without a figure it does not have", () => {
    const wrapper = mountBanner({
      phase: "embedding",
      step: { number: 3, of: 3, label: "Same scene" },
      percent: 0,
      measured: false,
    });
    const text = textOf(wrapper);
    expect(text).toContain("Comparing pictures for the same scene.");
    expect(text).toContain("These groups appear when this step finishes.");
    expect(text).not.toContain("Groups appear here as they are found.");
    expect(text).toContain("Same scene, step 3 of 3");
    expect(text).not.toContain("%");
    expect(
      wrapper.find(".scan-banner__track").attributes("aria-label"),
    ).toBe("Same-scene comparison in progress");
    expect(slides(wrapper)).toBe(true);
  });

  // Every tier counts from zero again, so the step is what stops a restarted
  // meter reading as progress going backwards.
  it("names the step beside the figure on a scan of several tiers", () => {
    const wrapper = mountBanner({
      step: { number: 1, of: 2, label: "Exact matches" },
      percent: 25,
    });
    expect(wrapper.find(".scan-banner__meta").text()).toBe(
      "Exact matches, step 1 of 2 · 25%",
    );
  });

  it("states only the figure on a scan of one tier", () => {
    expect(mountBanner({ percent: 25 }).find(".scan-banner__meta").text()).toBe(
      "25%",
    );
  });

  // The server reports no time estimate, and inventing one from a bucket rate
  // would be a guess presented as a fact. A wrong estimate is worse than none.
  it("states no time remaining, because the server does not know one", () => {
    expect(textOf(mountBanner())).not.toContain("min left");
  });

  it("announces the track as a progress bar", () => {
    // The track is the only progress readout; unlabelled it is invisible to a
    // screen reader (WCAG 1.3.1).
    const track = mountBanner({ percent: 42 }).find(".scan-banner__track");
    expect(track.attributes("role")).toBe("progressbar");
    expect(track.attributes("aria-label")).toBe("Duplicate pictures processed");
    expect(track.attributes("aria-valuenow")).toBe("42");
    expect(track.attributes("aria-valuemin")).toBe("0");
    expect(track.attributes("aria-valuemax")).toBe("100");
  });

  it("fills the track to the same figure it prints", () => {
    // A bar and a number that disagree make the whole readout untrustworthy.
    const wrapper = mountBanner({ percent: 42.4 });
    expect(textOf(wrapper)).toContain("42%");
    expect(wrapper.find(".scan-banner__fill").attributes("style")).toContain(
      "width: 42%",
    );
  });

  it("treats an unknown percentage with known totals as zero instead of NaN", () => {
    const wrapper = mountBanner({ percent: undefined, total: 12 });
    expect(textOf(wrapper)).toContain("0%");
    expect(
      wrapper.find(".scan-banner__track").attributes("aria-valuenow"),
    ).toBe("0");
  });
});
