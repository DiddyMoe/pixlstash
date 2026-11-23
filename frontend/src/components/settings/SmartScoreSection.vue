<script setup>
import { onMounted, ref, watch } from "vue";
import { getUserConfig, patchUserConfig } from "../../api/config";
import {
  deletePenalisedTag,
  getPenalisedTagCleanup,
  putPenalisedTag,
  startPenalisedTagCleanup,
} from "../../api/users";
import { useConfirm } from "../../composables/useConfirm";
import { useNoticeStore } from "../../stores/useNoticeStore";
import { VSwitch } from "vuetify/components";
import SettingsSection from "./SettingsSection.vue";
import SettingsChipGrid from "./SettingsChipGrid.vue";
import SettingsChip from "./SettingsChip.vue";
import SettingsAddTagRow from "./SettingsAddTagRow.vue";
import AppSelect from "../widgets/AppSelect.vue";
import { errorDetail } from "../../utils/apiError";

const props = defineProps({
  open: { type: Boolean, default: false },
});

const emit = defineEmits(["update:hidden-tags", "update:apply-tag-filter"]);

// First adoption of the central notice store (Phase 2 scaffold): a smart-score
// save failure raises a notice as well as the inline error. The notice is a
// no-op on screen until the maintainer's design pass mounts the notice host.
const noticeStore = useNoticeStore();
const { confirm } = useConfirm();

// ── Tag filter (hidden tags) - moved here from Behaviour ──────────────────────
const hiddenTags = ref([]);
const hiddenTagInput = ref("");
const hiddenTagsLoading = ref(false);
const hiddenTagsError = ref("");
const hiddenTagsSuccess = ref("");
const applyTagFilter = ref(false);
const applyTagFilterLoading = ref(false);

function normalizeHiddenTags(tags) {
  const values = Array.isArray(tags)
    ? tags
    : tags && typeof tags === "object"
      ? Object.keys(tags)
      : [];
  const seen = new Set();
  const cleaned = [];
  for (const tag of values) {
    if (tag == null) continue;
    const clean = String(tag).trim().toLowerCase();
    if (!clean || seen.has(clean)) continue;
    seen.add(clean);
    cleaned.push(clean);
  }
  return cleaned.sort((a, b) => a.localeCompare(b));
}

async function saveHiddenTags(nextTags) {
  hiddenTagsLoading.value = true;
  hiddenTagsError.value = "";
  hiddenTagsSuccess.value = "";
  try {
    const normalized = normalizeHiddenTags(nextTags);
    await patchUserConfig({ hidden_tags: normalized });
    hiddenTags.value = normalized;
    emit("update:hidden-tags", hiddenTags.value);
    hiddenTagsSuccess.value = "Saved.";
  } catch (e) {
    hiddenTagsError.value = errorDetail(e) || "Failed to update hidden tags.";
  } finally {
    hiddenTagsLoading.value = false;
    if (hiddenTagsSuccess.value) {
      setTimeout(() => {
        hiddenTagsSuccess.value = "";
      }, 2000);
    }
  }
}

async function addHiddenTag() {
  const trimmed = hiddenTagInput.value.trim().toLowerCase();
  if (!trimmed) return;
  const next = normalizeHiddenTags([...hiddenTags.value, trimmed]);
  hiddenTagInput.value = "";
  await saveHiddenTags(next);
}

async function removeHiddenTag(tag) {
  const next = normalizeHiddenTags(
    hiddenTags.value.filter((entry) => entry !== tag),
  );
  await saveHiddenTags(next);
}

async function setApplyTagFilter(value) {
  applyTagFilterLoading.value = true;
  hiddenTagsError.value = "";
  try {
    const nextValue = Boolean(value);
    await patchUserConfig({ apply_tag_filter: nextValue });
    applyTagFilter.value = nextValue;
    emit("update:apply-tag-filter", applyTagFilter.value);
  } catch (e) {
    hiddenTagsError.value = errorDetail(e) || "Failed to update tag filter.";
  } finally {
    applyTagFilterLoading.value = false;
  }
}

const smartScorePenalisedTags = ref([]);
const smartScoreTagInput = ref("");
const smartScoreTagsLoading = ref(false);
const smartScoreTagsError = ref("");
const smartScoreTagsSuccess = ref("");

// Removing a penalised tag (X) also takes it off this library's pictures and
// stops the tagger producing it; the server keeps the removed names
// (`excluded`) and runs the cleanup as a job this section follows.
const CLEANUP_POLL_MS = 1000;
const ACTIVE_CLEANUP_STATES = new Set(["queued", "running"]);
// One card for "removed tags are still on pictures", whichever job left them.
const UNFINISHED_KEY = "penalised-cleanup:unfinished";
const excludedTags = ref([]);
// The tag whose removal is in flight: its chip says so and the weights hold.
const removingTag = ref("");
// One removal or re-add at a time, from the question to the server's answer:
// a second confirm would replace the first, and an add sent meanwhile would
// carry the list from before the removal.
let changeInFlight = false;
const followedJobs = new Set();

const smartScoreImportanceOptions = [
  { value: 1, label: "Mild" },
  { value: 2, label: "Low" },
  { value: 3, label: "Moderate" },
  { value: 4, label: "High" },
  { value: 5, label: "Severe" },
];

function clampImportance(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return 3;
  return Math.min(5, Math.max(1, Math.round(num)));
}

function SmartScoreTags(tags) {
  const d = new Map();
  if (Array.isArray(tags)) {
    for (const item of tags) {
      if (item == null) continue;
      if (typeof item === "object") {
        const clean = String(item.tag || "")
          .trim()
          .toLowerCase();
        if (!clean) continue;
        d.set(clean, clampImportance(item.weight));
      } else {
        const clean = String(item).trim().toLowerCase();
        if (!clean) continue;
        d.set(clean, 3);
      }
    }
  } else if (tags && typeof tags === "object") {
    for (const [tag, weight] of Object.entries(tags)) {
      if (tag == null) continue;
      const clean = String(tag).trim().toLowerCase();
      if (!clean) continue;
      const nextWeight = clampImportance(weight);
      const existing = d.get(clean);
      if (existing == null || nextWeight > existing) {
        d.set(clean, nextWeight);
      }
    }
  }
  return Array.from(d.entries())
    .map(([tag, weight]) => ({ tag, weight }))
    .sort((a, b) => a.tag.localeCompare(b.tag));
}

function serializeSmartScoreTags(entries) {
  const d = SmartScoreTags(entries);
  const payload = {};
  for (const entry of d) {
    payload[entry.tag] = clampImportance(entry.weight);
  }
  return { d, payload };
}

async function fetchData() {
  smartScoreTagsLoading.value = true;
  smartScoreTagsError.value = "";
  smartScoreTagInput.value = "";
  smartScoreTagsSuccess.value = "";
  try {
    const config = await getUserConfig();
    smartScorePenalisedTags.value = SmartScoreTags(
      config?.smart_score_penalised_tags,
    );
    hiddenTags.value = normalizeHiddenTags(config?.hidden_tags);
    emit("update:hidden-tags", hiddenTags.value);
    applyTagFilter.value = Boolean(config?.apply_tag_filter);
    emit("update:apply-tag-filter", applyTagFilter.value);
  } catch {
    smartScoreTagsError.value = "Failed to load smart score settings.";
  } finally {
    smartScoreTagsLoading.value = false;
  }
  try {
    const status = await getPenalisedTagCleanup();
    excludedTags.value = status?.excluded || [];
    if (status?.state === "interrupted") reportResidue(status.remaining);
  } catch (e) {
    console.error("Reading the penalised-tag cleanup status failed", e);
  }
}

async function saveSmartScoreTags(nextTags) {
  smartScoreTagsLoading.value = true;
  smartScoreTagsError.value = "";
  smartScoreTagsSuccess.value = "";
  try {
    const { d, payload } = serializeSmartScoreTags(nextTags);
    await patchUserConfig({ smart_score_penalised_tags: payload });
    smartScorePenalisedTags.value = d;
    smartScoreTagsSuccess.value = "Saved.";
  } catch (e) {
    smartScoreTagsError.value =
      errorDetail(e) || "Failed to update smart score tags.";
    noticeStore.error(smartScoreTagsError.value);
  } finally {
    smartScoreTagsLoading.value = false;
    if (smartScoreTagsSuccess.value) {
      setTimeout(() => {
        smartScoreTagsSuccess.value = "";
      }, 2000);
    }
  }
}

async function addSmartScoreTag() {
  if (changeInFlight) return;
  const trimmed = smartScoreTagInput.value.trim().toLowerCase();
  if (!trimmed) return;
  if (excludedTags.value.includes(trimmed)) {
    await addRemovedTagBack(trimmed);
    return;
  }
  const next = SmartScoreTags([
    ...smartScorePenalisedTags.value,
    { tag: trimmed, weight: 3 },
  ]);
  smartScoreTagInput.value = "";
  await saveSmartScoreTags(next);
}

/** Let a removed tag back into tagging, penalised at the default weight. */
async function addRemovedTagBack(tag) {
  changeInFlight = true;
  try {
    const confirmed = await confirm({
      title: `Add '${tag}' back?`,
      message:
        "The tagger can add it to pictures again. Tags removed earlier are not restored.",
      confirmLabel: "Add back",
      cancelLabel: "Cancel",
    });
    if (!confirmed) return;
    smartScoreTagInput.value = "";
    smartScoreTagsLoading.value = true;
    smartScoreTagsError.value = "";
    smartScoreTagsSuccess.value = "";
    try {
      const result = await putPenalisedTag(tag, 3);
      smartScorePenalisedTags.value = SmartScoreTags(
        result?.smart_score_penalised_tags,
      );
      excludedTags.value = result?.excluded || [];
      smartScoreTagsSuccess.value = "Saved.";
      if (result?.cleanup) followCleanup(result.cleanup);
    } catch (e) {
      console.error(`Adding the removed tag '${tag}' back failed`, e);
      smartScoreTagsError.value = errorDetail(e) || `Could not add '${tag}' back.`;
      noticeStore.error(smartScoreTagsError.value);
    } finally {
      smartScoreTagsLoading.value = false;
      if (smartScoreTagsSuccess.value) {
        setTimeout(() => {
          smartScoreTagsSuccess.value = "";
        }, 2000);
      }
    }
  } finally {
    changeInFlight = false;
  }
}

/**
 * X: remove the tag from the list, from tagging and from this library's
 * pictures. The chip goes only once the server has done the first two.
 */
async function removeSmartScoreTag(tag) {
  // A save still in flight carries the list with this tag in it.
  if (changeInFlight || smartScoreTagsLoading.value) return;
  changeInFlight = true;
  try {
    const confirmed = await confirm({
      title: `Remove '${tag}'?`,
      message:
        "Removes this tag from all pictures in this library and stops the tagger " +
        "adding it in any of your libraries. Pictures stay; tags cannot be restored.",
      confirmLabel: "Remove",
      cancelLabel: "Cancel",
      danger: true,
    });
    if (!confirmed) return;
    removingTag.value = tag;
    smartScoreTagsError.value = "";
    smartScoreTagsSuccess.value = "";
    const result = await deletePenalisedTag(tag);
    smartScorePenalisedTags.value = SmartScoreTags(
      result?.smart_score_penalised_tags,
    );
    excludedTags.value = result?.excluded || [];
    if (result?.cleanup) followCleanup(result.cleanup);
  } catch (e) {
    console.error(`Removing the penalised tag '${tag}' failed`, e);
    smartScoreTagsError.value = errorDetail(e) || `Could not remove '${tag}'.`;
    noticeStore.error(smartScoreTagsError.value);
  } finally {
    removingTag.value = "";
    changeInFlight = false;
  }
}

function progressKey(job) {
  return `penalised-cleanup:${job.id}`;
}

/** 'a'; 'a' and 'b'; 'a' and 2 more tags. */
function tagsLabel(names) {
  const sorted = [...names].sort((a, b) => a.localeCompare(b));
  if (sorted.length === 1) return `'${sorted[0]}'`;
  if (sorted.length === 2) return `'${sorted[0]}' and '${sorted[1]}'`;
  return `'${sorted[0]}' and ${sorted.length - 1} more tags`;
}

function picturesLabel(count) {
  return `${count.toLocaleString()} picture${count === 1 ? "" : "s"}`;
}

/** Show a cleanup job's progress until it ends, once per job. */
function followCleanup(job) {
  if (followedJobs.has(job.id)) return;
  followedJobs.add(job.id);
  noticeStore.dismissByKey(UNFINISHED_KEY);
  showCleanupProgress(job);
}

function showCleanupProgress(job) {
  // Replace whatever card this job has (a failed check) instead of
  // coalescing onto it.
  noticeStore.dismissByKey(progressKey(job));
  noticeStore.push({
    level: "info",
    text: `Removing ${tagsLabel(job.names)} from pictures…`,
    timeout: 0,
    key: progressKey(job),
  });
  setTimeout(() => checkCleanup(job), CLEANUP_POLL_MS);
}

async function checkCleanup(job) {
  const label = tagsLabel(job.names);
  let status;
  try {
    status = await getPenalisedTagCleanup();
  } catch (e) {
    console.error(`Checking on the cleanup job ${job.id} failed`, e);
    noticeStore.dismissByKey(progressKey(job));
    noticeStore.warning(`Could not check on removing ${label}.`, {
      key: progressKey(job),
      action: {
        label: "Check again",
        handler: () => {
          showCleanupProgress(job);
        },
      },
    });
    return;
  }
  excludedTags.value = status?.excluded || [];
  const jobs = status?.jobs || [];
  const idx = jobs.findIndex((entry) => entry.id === job.id);
  const current = idx < 0 ? null : jobs[idx];
  if (current && ACTIVE_CLEANUP_STATES.has(current.state)) {
    setTimeout(() => checkCleanup(job), CLEANUP_POLL_MS);
    return;
  }
  noticeStore.dismissByKey(progressKey(job));
  if (current?.state === "completed") {
    const done = `Removed ${label} from ${picturesLabel(current.pictures)}`;
    if (current.locked > 0) {
      const locked = current.locked;
      noticeStore.warning(
        `${done}; ${locked.toLocaleString()} locked picture${locked === 1 ? " was" : "s were"} skipped.`,
      );
    } else {
      noticeStore.success(`${done}.`);
    }
    return;
  }
  // A later removal or re-add stopped this job and a newer one took its
  // names over (the list is newest first).
  const names = new Set(job.names);
  const newer = jobs
    .slice(0, idx < 0 ? 0 : idx)
    .filter((entry) => entry.names.some((name) => names.has(name)));
  if (newer.length) {
    for (const entry of newer) followCleanup(entry);
    return;
  }
  // A job that failed or was cancelled did not finish, whatever the count of
  // pictures left says.
  if (
    status?.state === "interrupted" ||
    current?.state === "failed" ||
    current?.state === "cancelled"
  ) {
    reportUnfinished(job.names);
    return;
  }
  // The job is gone (the server restarted) and nothing is left to remove.
  if (status?.state === "idle" && current?.state !== "superseded") {
    noticeStore.success(`Removed ${label} from your pictures.`);
  }
}

function reportUnfinished(names) {
  noticeStore.dismissByKey(UNFINISHED_KEY);
  noticeStore.warning(
    `Removing ${tagsLabel(names)} from pictures stopped before it finished.`,
    {
      key: UNFINISHED_KEY,
      action: {
        label: "Retry",
        handler: () => {
          retryCleanup();
        },
      },
    },
  );
}

/** Removed tags left on this library's pictures from an earlier session. */
function reportResidue(count) {
  noticeStore.dismissByKey(UNFINISHED_KEY);
  noticeStore.warning(
    `Removed tags are still on ${picturesLabel(count)} in this library.`,
    {
      key: UNFINISHED_KEY,
      action: {
        label: "Remove from this library",
        handler: () => {
          retryCleanup();
        },
      },
    },
  );
}

/** Start removing every removed tag from this library's pictures again. */
async function retryCleanup() {
  try {
    const result = await startPenalisedTagCleanup();
    followCleanup(result.cleanup);
  } catch (e) {
    if (e?.response?.status === 409) {
      // One is running already: follow it instead.
      await followActiveCleanups();
      return;
    }
    console.error("Restarting the penalised-tag cleanup failed", e);
    noticeStore.error(
      errorDetail(e) || "Could not restart removing tags from pictures.",
    );
  }
}

async function followActiveCleanups() {
  try {
    const status = await getPenalisedTagCleanup();
    excludedTags.value = status?.excluded || [];
    for (const entry of status?.jobs || []) {
      if (ACTIVE_CLEANUP_STATES.has(entry.state)) followCleanup(entry);
    }
  } catch (e) {
    console.error("Reading the running penalised-tag cleanup failed", e);
    noticeStore.error(
      errorDetail(e) || "Could not check on removing tags from pictures.",
    );
  }
}

async function updateSmartScoreTagWeight(tag, weight) {
  const next = SmartScoreTags(
    smartScorePenalisedTags.value.map((entry) =>
      entry.tag === tag ? { ...entry, weight: clampImportance(weight) } : entry,
    ),
  );
  await saveSmartScoreTags(next);
}

onMounted(fetchData);

watch(
  () => props.open,
  (isOpen) => {
    if (isOpen) fetchData();
  },
);
</script>

<template>
  <div class="smart-score">
    <SettingsSection
      class="smart-score__grow"
      title="Penalised Tags"
      desc="Tags listed here lower Smart Score. Removing one also removes it from your pictures."
      first
    >
      <SettingsChipGrid empty="No penalised tags yet.">
        <SettingsChip
          v-for="entry in smartScorePenalisedTags"
          :key="entry.tag"
          :label="entry.tag"
          :meta="removingTag === entry.tag ? 'Removing…' : ''"
          @remove="removeSmartScoreTag(entry.tag)"
        >
          <div class="smart-score-importance">
            <AppSelect
              compact
              :model-value="entry.weight"
              :options="smartScoreImportanceOptions"
              :disabled="smartScoreTagsLoading || Boolean(removingTag)"
              @update:model-value="
                (value) => updateSmartScoreTagWeight(entry.tag, value)
              "
            />
          </div>
        </SettingsChip>
      </SettingsChipGrid>

      <SettingsAddTagRow
        v-model="smartScoreTagInput"
        placeholder="Add penalised tag"
        btn="Add tag"
        @add="addSmartScoreTag"
      />

      <div
        v-if="smartScoreTagsError"
        class="smart-score-status smart-score-status--error"
      >
        {{ smartScoreTagsError }}
      </div>
      <div
        v-else-if="smartScoreTagsSuccess"
        class="smart-score-status smart-score-status--success"
      >
        {{ smartScoreTagsSuccess }}
      </div>
    </SettingsSection>

    <SettingsSection
      title="Tag Filter"
      desc="Tags listed here are filtered from the GUI entirely."
    >
      <div class="smart-score-filter-toggle">
        <v-switch
          v-model="applyTagFilter"
          color="primary"
          density="compact"
          hide-details
          :disabled="applyTagFilterLoading"
          label="Apply tag filter to all pictures and videos"
          @update:model-value="setApplyTagFilter"
        />
      </div>
      <SettingsChipGrid class="smart-score__capped" empty="No hidden tags yet.">
        <SettingsChip
          v-for="tag in hiddenTags"
          :key="tag"
          :label="tag"
          @remove="removeHiddenTag(tag)"
        />
      </SettingsChipGrid>
      <SettingsAddTagRow
        v-model="hiddenTagInput"
        placeholder="Add tag filter"
        btn="Add tag"
        @add="addHiddenTag"
      />
      <div
        v-if="hiddenTagsError"
        class="smart-score-status smart-score-status--error"
      >
        {{ hiddenTagsError }}
      </div>
      <div
        v-else-if="hiddenTagsSuccess"
        class="smart-score-status smart-score-status--success"
      >
        {{ hiddenTagsSuccess }}
      </div>
    </SettingsSection>
  </div>
</template>

<style scoped>
/* The penalised list keeps its full height until the pane runs out, then
   shrinks and scrolls inside itself (SettingsChipGrid), so the pane never
   scrolls. Only that section may shrink: a shrunk Tag filter section would
   spill its own rows past the pane. */
.smart-score {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.smart-score__grow {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

/* The hidden-tag list: four rows, then it scrolls in its own box. */
.smart-score__capped {
  max-height: calc(4 * (var(--control-h) + var(--space-1)));
}

/* Fixed-width wrapper for the inline importance select inside each chip,
   matching the proposal's 96px slot. */
.smart-score-importance {
  width: 96px;
  flex-shrink: 0;
}

.smart-score-filter-toggle {
  margin-bottom: var(--space-4);
}

.smart-score-status {
  font-size: var(--text-xs);
  margin-top: var(--space-2);
}

.smart-score-status--error {
  color: rgb(var(--v-theme-surface-error));
}

.smart-score-status--success {
  color: rgb(var(--v-theme-on-surface));
}
</style>
