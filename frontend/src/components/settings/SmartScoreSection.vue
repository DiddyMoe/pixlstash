<script setup>
import { onWatcherCleanup, reactive, ref, watch } from "vue";
import { getUserConfig, patchUserConfig } from "../../api/config";
import {
  deletePenalisedTag,
  getPenalisedTagCleanup,
  putPenalisedTag,
  startPenalisedTagCleanup,
} from "../../api/users";
import { useConfirm } from "../../composables/useConfirm";
import { useNoticeStore } from "../../stores/useNoticeStore";
import { useUserPrefsStore } from "../../stores/useUserPrefsStore";
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
// Owns the penalised-tag edit queue, which outlives this pane.
const prefs = useUserPrefsStore();
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

// The list on screen is always the last one the server accepted: a read that
// was applied, or an edit whose PATCH succeeded. Nothing is shown before then.
const smartScorePenalisedTags = ref([]);
const smartScoreTagInput = ref("");
const smartScoreTagsError = ref("");
const smartScoreTagsSuccess = ref("");
// Chips whose removal is queued or being saved; their X shows the spinner.
const removingTags = reactive(new Set());
// False until a read or an edit has given the list a known value, so an edit
// made before the first read lands cannot save a list built from nothing.
let penalisedLoaded = false;
// Bumped when a read starts and when an edit is accepted. A read applies its
// list only if nothing newer has happened since it started.
let listEpoch = 0;
let successTimer = null;

// Removing a penalised tag (X) also takes it off this library's pictures and
// stops the tagger producing it; the server keeps the removed names
// (`excluded`) and runs the cleanup as a job this section follows.
const CLEANUP_POLL_MS = 1000;
const ACTIVE_CLEANUP_STATES = new Set(["queued", "running"]);
// One card for "removed tags are still on pictures", whichever job left them.
const UNFINISHED_KEY = "penalised-cleanup:unfinished";
const excludedTags = ref([]);
// One question at a time: a second confirm would replace the first. The
// answer is queued like any other edit, so it saves after the edits before it.
let confirming = false;
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

function acceptPenalisedTags(tags) {
  smartScorePenalisedTags.value = tags;
  penalisedLoaded = true;
}

async function fetchData(signal) {
  const epoch = ++listEpoch;
  smartScoreTagsError.value = "";
  smartScoreTagInput.value = "";
  smartScoreTagsSuccess.value = "";
  try {
    // An edit still saving, perhaps from a pane closed before this one
    // opened, would make this read stale: read the list it leaves instead.
    await prefs.penalisedEditsSettled();
    if (signal.aborted) return;
    const config = await getUserConfig({ signal });
    // Closed (or reopened) while this read was out: a newer read owns the pane.
    if (signal.aborted) return;
    // An edit accepted since this read started is newer than what it returned.
    if (epoch === listEpoch) {
      acceptPenalisedTags(SmartScoreTags(config?.smart_score_penalised_tags));
    }
    hiddenTags.value = normalizeHiddenTags(config?.hidden_tags);
    emit("update:hidden-tags", hiddenTags.value);
    applyTagFilter.value = Boolean(config?.apply_tag_filter);
    emit("update:apply-tag-filter", applyTagFilter.value);
  } catch (e) {
    if (signal.aborted) return;
    console.error("Failed to load smart score settings:", e);
    smartScoreTagsError.value = "Failed to load smart score settings.";
  }
  if (signal.aborted) return;
  try {
    const status = await getPenalisedTagCleanup();
    if (signal.aborted) return;
    excludedTags.value = status?.excluded || [];
    if (status?.state === "interrupted") reportResidue(status.remaining);
  } catch (e) {
    console.error("Reading the penalised-tag cleanup status failed", e);
  }
}

function showSaved() {
  smartScoreTagsSuccess.value = "Saved.";
  clearTimeout(successTimer);
  successTimer = setTimeout(() => {
    smartScoreTagsSuccess.value = "";
  }, 2000);
}

// Runs one queued save. `save` sends it and resolves to the list the server
// accepted. It never rejects, so one failed save does not stop the saves
// behind it.
async function runPenalisedSave(save, failure) {
  try {
    const tags = await save();
    listEpoch += 1;
    acceptPenalisedTags(tags);
    showSaved();
  } catch (e) {
    // The list stays at the last accepted value: nothing was shown early.
    console.error(failure, e);
    smartScoreTagsError.value = errorDetail(e) || failure;
    noticeStore.error(smartScoreTagsError.value);
  }
}

function queuePenalisedSave(save, failure) {
  // A new click starts a new outcome; an earlier failure stays on screen
  // until then, because that edit really was not saved.
  smartScoreTagsError.value = "";
  smartScoreTagsSuccess.value = "";
  return prefs.queuePenalisedEdit(() => runPenalisedSave(save, failure));
}

// `edit` maps the last accepted list to the next one, at its turn in the
// queue, and the whole list is saved.
function queuePenalisedEdit(edit) {
  return queuePenalisedSave(async () => {
    if (!penalisedLoaded) {
      const config = await getUserConfig();
      acceptPenalisedTags(SmartScoreTags(config?.smart_score_penalised_tags));
    }
    const { d, payload } = serializeSmartScoreTags(
      edit(smartScorePenalisedTags.value),
    );
    await patchUserConfig({ smart_score_penalised_tags: payload });
    return d;
  }, "Failed to update smart score tags.");
}

// Applies a per-tag answer (remove or add back): the list, the removed names
// and the cleanup job it started.
function acceptPerTagResult(result) {
  excludedTags.value = result?.excluded || [];
  if (result?.cleanup) followCleanup(result.cleanup);
  return SmartScoreTags(result?.smart_score_penalised_tags);
}

// Asks, then runs `run` in the same turn as the answer, so nothing can slip
// in between a yes and what it starts.
async function askThen(options, run) {
  if (confirming) return;
  confirming = true;
  let confirmed;
  try {
    confirmed = await confirm(options);
  } finally {
    confirming = false;
  }
  if (confirmed) return run();
}

function addSmartScoreTag() {
  const trimmed = smartScoreTagInput.value.trim().toLowerCase();
  if (!trimmed) return;
  if (excludedTags.value.includes(trimmed)) return addRemovedTagBack(trimmed);
  smartScoreTagInput.value = "";
  return queuePenalisedEdit((list) => [...list, { tag: trimmed, weight: 3 }]);
}

/** Let a removed tag back into tagging, penalised at the default weight. */
function addRemovedTagBack(tag) {
  return askThen(
    {
      title: `Add '${tag}' back?`,
      message:
        "The tagger can add it to pictures again. Tags removed earlier are not restored.",
      confirmLabel: "Add back",
      cancelLabel: "Cancel",
    },
    () => {
      smartScoreTagInput.value = "";
      return queuePenalisedSave(
        async () => acceptPerTagResult(await putPenalisedTag(tag, 3)),
        `Could not add '${tag}' back.`,
      );
    },
  );
}

/**
 * X: remove the tag from the list, from tagging and from this library's
 * pictures. The chip goes only once the server has done the first two.
 */
function removeSmartScoreTag(tag) {
  if (removingTags.has(tag)) return;
  return askThen(
    {
      title: `Remove '${tag}'?`,
      message:
        "Removes this tag from all pictures in this library and stops the tagger " +
        "adding it in any of your libraries. Pictures stay; tags cannot be restored.",
      confirmLabel: "Remove",
      cancelLabel: "Cancel",
      danger: true,
    },
    () => {
      removingTags.add(tag);
      return queuePenalisedSave(
        async () => acceptPerTagResult(await deletePenalisedTag(tag)),
        `Could not remove '${tag}'.`,
      ).finally(() => removingTags.delete(tag));
    },
  );
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

function updateSmartScoreTagWeight(tag, weight) {
  return queuePenalisedEdit((list) =>
    list.map((entry) =>
      entry.tag === tag ? { ...entry, weight: clampImportance(weight) } : entry,
    ),
  );
}

// The dialog's content is unmounted after it closes and mounted again with
// `open` already true, so the read belongs to an immediate watcher. Closing,
// or reopening during the close animation, aborts the read in flight.
watch(
  () => props.open,
  (isOpen) => {
    if (!isOpen) return;
    const controller = new AbortController();
    onWatcherCleanup(() => controller.abort());
    fetchData(controller.signal);
  },
  { immediate: true },
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
          :meta="removingTags.has(entry.tag) ? 'Removing…' : ''"
          :busy="removingTags.has(entry.tag)"
          @remove="removeSmartScoreTag(entry.tag)"
        >
          <div class="smart-score-importance">
            <AppSelect
              compact
              :model-value="entry.weight"
              :options="smartScoreImportanceOptions"
              :disabled="removingTags.has(entry.tag)"
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
