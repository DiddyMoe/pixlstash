<script setup>
import { onWatcherCleanup, reactive, ref, watch } from "vue";
import { getUserConfig, patchUserConfig } from "../../api/config";
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
}

function showSaved() {
  smartScoreTagsSuccess.value = "Saved.";
  clearTimeout(successTimer);
  successTimer = setTimeout(() => {
    smartScoreTagsSuccess.value = "";
  }, 2000);
}

// Runs one queued edit. `edit` maps the last accepted list to the next one.
// It never rejects, so one failed edit does not stop the edits behind it.
async function runPenalisedEdit(edit) {
  try {
    if (!penalisedLoaded) {
      const config = await getUserConfig();
      acceptPenalisedTags(SmartScoreTags(config?.smart_score_penalised_tags));
    }
    const { d, payload } = serializeSmartScoreTags(
      edit(smartScorePenalisedTags.value),
    );
    await patchUserConfig({ smart_score_penalised_tags: payload });
    listEpoch += 1;
    acceptPenalisedTags(d);
    showSaved();
  } catch (e) {
    // The list stays at the last accepted value: nothing was shown early.
    smartScoreTagsError.value =
      errorDetail(e) || "Failed to update smart score tags.";
    noticeStore.error(smartScoreTagsError.value);
  }
}

function queuePenalisedEdit(edit) {
  // A new click starts a new outcome; an earlier failure stays on screen
  // until then, because that edit really was not saved.
  smartScoreTagsError.value = "";
  smartScoreTagsSuccess.value = "";
  return prefs.queuePenalisedEdit(() => runPenalisedEdit(edit));
}

function addSmartScoreTag() {
  const trimmed = smartScoreTagInput.value.trim().toLowerCase();
  if (!trimmed) return;
  smartScoreTagInput.value = "";
  return queuePenalisedEdit((list) => [...list, { tag: trimmed, weight: 3 }]);
}

function removeSmartScoreTag(tag) {
  removingTags.add(tag);
  return queuePenalisedEdit((list) =>
    list.filter((entry) => entry.tag !== tag),
  ).finally(() => removingTags.delete(tag));
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
      desc="Tags listed here reduce Smart Score when present on a picture. Adjust the importance to control how much they hurt the score."
      first
    >
      <SettingsChipGrid empty="No penalised tags yet.">
        <SettingsChip
          v-for="entry in smartScorePenalisedTags"
          :key="entry.tag"
          :label="entry.tag"
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
