<template>
  <div v-if="visible" class="selection-bar-overlay">
    <div class="selection-bar-content">
      <!-- ── A run under way: its progress and its Stop take the bar ──── -->
      <template v-if="running">
        <p class="run-note" data-testid="scrapheap-run-progress">
          <v-icon size="14" class="run-note__icon">{{
            restoring ? "mdi-restore" : "mdi-delete-forever-outline"
          }}</v-icon>
          <span v-if="stopping" class="run-note__line">
            Stopping after the current step: <b>{{ counts.done }}</b>
            {{ restoring ? "restored" : "deleted" }} so far.
          </span>
          <span v-else-if="pending" class="run-note__line">
            {{ restoring ? "Restore" : "Delete forever" }} is queued and starts
            when the server gets to it.
          </span>
          <span v-else class="run-note__line">
            {{ restoring ? "Restoring" : "Deleting forever" }}:
            <b>{{ counts.processed }}</b> of <b>{{ counts.total }}</b> pictures.
          </span>
          <span
            v-if="runUnreachable"
            class="run-note__warning"
            data-testid="scrapheap-run-unreachable"
          >
            No answer from the server; these are the last figures read.
          </span>
          <span v-else class="run-note__hint">You can keep working.</span>
        </p>
        <div class="selection-bar-actions">
          <span v-if="determinate && !stopping" class="run-meta">
            {{ percent }}%
          </span>
          <AppButton
            v-if="showStop"
            size="sm"
            :loading="runAction === 'cancel'"
            :disabled="runAction != null && runAction !== 'cancel'"
            data-testid="scrapheap-run-stop"
            @click="$emit('stop-run')"
          >
            Stop
          </AppButton>
        </div>
      </template>

      <!-- ── No run: the heap's own actions ───────────────────────────── -->
      <template v-else>
        <div class="selection-bar-left">
          <AppButton
            variant="primary"
            size="sm"
            :disabled="restoreDisabled"
            @click="$emit('restore-scrapheap')"
          >
            Restore all
          </AppButton>
          <!-- Active retention policy, stated where the action happens. Hidden
               until the policy is known so it never shows a guessed window. -->
          <p v-if="retentionLabel" class="retention-note">
            <v-icon size="14" class="retention-note__icon"
              >mdi-delete-clock-outline</v-icon
            >
            <span>Auto-empty: {{ retentionLabel }}</span>
            <span class="retention-note__sep" aria-hidden="true">·</span>
            <button
              type="button"
              class="retention-note__change"
              :aria-label="`Change the auto-empty window (currently ${retentionLabel})`"
              @click="$emit('open-retention-settings')"
            >
              change
            </button>
          </p>
        </div>
        <div class="selection-bar-actions">
          <AppButton
            variant="danger"
            size="sm"
            :disabled="disabled"
            @click="$emit('empty-scrapheap')"
          >
            Empty scrapheap
          </AppButton>
        </div>
      </template>
    </div>

    <!-- Pinned to the bar's bottom edge, so the bar itself is the meter. -->
    <span
      v-if="running"
      class="run-track"
      role="progressbar"
      :aria-label="restoring ? 'Restore progress' : 'Delete forever progress'"
      :aria-valuenow="determinate ? percent : null"
      :aria-valuetext="
        determinate
          ? `${counts.processed} of ${counts.total} pictures done`
          : null
      "
      aria-valuemin="0"
      aria-valuemax="100"
    >
      <span class="run-track__fill" :style="{ width: `${percent}%` }"></span>
    </span>

    <!-- Always mounted, so a change of state is announced. It says only that a
         run started or is stopping: progress moves on every poll, and how the
         run ended is the notice the grid raises, which announces itself. -->
    <span
      class="visually-hidden"
      role="status"
      aria-live="polite"
      data-testid="scrapheap-run-announcement"
      >{{ announcement }}</span
    >
  </div>
</template>

<script setup>
/**
 * Scrapheap action bar: restore-all / empty, plus the active auto-empty policy.
 *
 * While a background run empties or restores the heap (or Delete forever or
 * Restore works through a selection), the bar shows that run instead: what it
 * is doing, how far it has got, and a Stop. The run goes on on the server
 * whatever the page does, so this is how the page says the user may keep
 * working; how it ended is a notice the grid raises.
 *
 * Presentational only - the parent (ImageGrid) owns the retention value, the
 * run (`useScrapheapRunStore`) and the navigation to Settings, so this
 * component never touches a store or the API.
 */
import { computed } from "vue";
import { VIcon } from "vuetify/components";
import AppButton from "./AppButton.vue";

const props = defineProps({
  visible: { type: Boolean, default: false },
  disabled: { type: Boolean, default: false },
  restoreDisabled: { type: Boolean, default: false },
  /**
   * Human label for the active auto-empty window, e.g. "30 days" / "Never".
   * Empty string hides the policy line (policy not loaded yet).
   */
  retentionLabel: { type: String, default: "" },
  /**
   * The Scrapheap run the store follows, as the server serializes it, or null.
   * Shown only while `active`; `kind` is "purge" or "restore", `total` and
   * `processed` count pictures.
   */
  run: { type: Object, default: null },
  /** The last poll failed, so the figures are the last ones read. */
  runUnreachable: { type: Boolean, default: false },
  /** The run request on the wire: "start", "cancel" or null. */
  runAction: { type: String, default: null },
});

defineEmits([
  "empty-scrapheap",
  "restore-scrapheap",
  "stop-run",
  "open-retention-settings",
]);

/** A count as the sentence shows it: grouped digits, never NaN. */
function figure(value) {
  return Math.max(0, Number(value) || 0).toLocaleString();
}

const running = computed(() => Boolean(props.run?.active));
const restoring = computed(() => props.run?.kind === "restore");
const stopping = computed(
  () => running.value && Boolean(props.run?.cancel_requested),
);
const pending = computed(
  () => running.value && props.run?.status === "pending",
);

const total = computed(() => Math.max(0, Number(props.run?.total) || 0));
const processed = computed(() =>
  Math.max(0, Number(props.run?.processed) || 0),
);
const counts = computed(() => ({
  total: figure(total.value),
  processed: figure(processed.value),
  done: figure(
    restoring.value ? props.run?.restored_count : props.run?.deleted_count,
  ),
}));

/** Progress as a whole number in 0 to 100, known only once a total is. */
const determinate = computed(
  () => running.value && !pending.value && total.value > 0,
);
const percent = computed(() => {
  if (!determinate.value) return 0;
  return Math.min(
    100,
    Math.max(0, Math.round((processed.value / total.value) * 100)),
  );
});

/** Stop while the run goes on and has not been asked to already. */
const showStop = computed(
  () => running.value && !props.run?.cancel_requested,
);

const announcement = computed(() => {
  if (!running.value) return "";
  if (stopping.value) return "Stopping after the current step.";
  return restoring.value
    ? "Restoring pictures from the Scrapheap in the background."
    : "Deleting pictures forever in the background.";
});
</script>

<style scoped>
.selection-bar-overlay {
  position: absolute !important;
  left: 0;
  top: calc(var(--selbar-height, 48px) + var(--filter-strip-h, 0px));
  width: 100%;
  z-index: var(--z-sticky);
  background: rgba(var(--v-theme-background), 0.95);
  padding: 0 var(--space-3) var(--space-2);
  margin: 0;
  height: 30px;
  box-sizing: border-box;
  display: flex;
  align-items: center;
}
.selection-bar-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  min-width: 0;
}
.selection-bar-left {
  display: flex;
  align-items: center;
  gap: var(--space-4);
  min-width: 0;
}
.selection-bar-actions {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin-left: auto;
  flex-wrap: nowrap;
}
/* Active-policy line: quiet status text, not a control, with one inline link-
   styled button through to the setting that owns it. */
.retention-note {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin: 0;
  min-width: 0;
  font-size: var(--text-xs);
  line-height: var(--leading-snug);
  color: rgba(var(--v-theme-on-background), 0.7);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.retention-note__icon {
  flex-shrink: 0;
  opacity: 0.75;
}

.retention-note__sep {
  color: rgba(var(--v-theme-on-background), 0.4);
}

.retention-note__change {
  padding: 0;
  font: inherit;
  color: rgb(var(--v-theme-accent));
  text-decoration: underline;
  border-radius: var(--radius-sm);
}

/* The note clips its overflow, which would cut an outset ring. */
.retention-note__change:focus-visible {
  outline: none;
  box-shadow: var(--focus-ring-inset);
}

/* A run's line: one line in the bar's fixed height, so it clips rather than
   wraps. The progress sentence is the primary text; the reassurance after it
   is quiet, and gives way first. */
.run-note {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin: 0;
  min-width: 0;
  font-size: var(--text-xs);
  line-height: var(--leading-snug);
  color: rgb(var(--v-theme-on-background));
  white-space: nowrap;
  overflow: hidden;
}

.run-note__icon {
  flex-shrink: 0;
  color: rgba(
    var(--v-theme-on-background),
    var(--opacity-text-secondary)
  );
}

.run-note__line {
  flex-shrink: 0;
}

/* The counts are what the eye returns to, and tabular figures stop the
   sentence reflowing as they tick up. */
.run-note__line b {
  font-weight: var(--weight-semibold);
  font-variant-numeric: tabular-nums;
}

.run-note__hint,
.run-note__warning {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}

.run-note__hint {
  color: rgba(
    var(--v-theme-on-background),
    var(--opacity-text-secondary)
  );
}

.run-note__warning {
  color: rgb(var(--v-theme-surface-warning));
}

.run-meta {
  font-size: var(--text-xs);
  line-height: var(--leading-snug);
  color: rgba(
    var(--v-theme-on-background),
    var(--opacity-text-secondary)
  );
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.run-track {
  position: absolute;
  inset-inline: 0;
  bottom: 0;
  height: var(--countdown-h);
  background: rgba(var(--v-theme-on-background), 0.14);
  pointer-events: none;
}

.run-track__fill {
  display: block;
  height: 100%;
  background: rgb(var(--v-theme-accent));
  /* Progress arrives in polls, so the fill eases between them. The global
     reduced-motion rule collapses this. */
  transition: width var(--dur-2) var(--ease-standard);
}
</style>
