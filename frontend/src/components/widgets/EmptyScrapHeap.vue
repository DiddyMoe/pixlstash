<template>
  <div v-if="visible" class="selection-bar-overlay">
    <div class="selection-bar-content">
      <!-- ── A run under way: its progress and its Stop take the bar ──── -->
      <template v-if="running">
        <p class="run-note" data-testid="scrapheap-run-progress">
          <v-icon size="14" class="run-note__icon">{{ words.icon }}</v-icon>
          <span v-if="stopping" class="run-note__line">
            Stopping after {{ words.stopAfter }}: <b>{{ counts.done }}</b>
            {{ words.doneSoFar }}.
          </span>
          <span v-else-if="pending" class="run-note__line">
            {{ words.gesture }} is queued and starts when the server gets to it.
          </span>
          <!-- An undo counts the restore's History steps, not pictures. -->
          <span v-else-if="progress.steps" class="run-note__line">
            {{ words.doing }}: History step <b>{{ counts.processed }}</b> of
            <b>{{ counts.total }}</b>, <b>{{ counts.done }}</b> pictures so far.
          </span>
          <span v-else class="run-note__line">
            {{ words.doing }}: <b>{{ counts.processed }}</b> of
            <b>{{ counts.total }}</b> pictures.
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
            :disabled="runAction === 'cancel'"
            :tooltip="stopTip"
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
          <!-- The delete preview is on the wire: Delete forever waits for it
               before it can show what would be destroyed. -->
          <span
            v-if="checking"
            class="run-meta"
            data-testid="scrapheap-delete-checking"
          >
            Checking what would be destroyed…
          </span>
          <AppButton
            variant="danger"
            size="sm"
            :disabled="disabled || checking != null"
            :loading="checking === 'all'"
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
      :aria-label="words.progress"
      :aria-valuenow="determinate ? percent : null"
      :aria-valuetext="
        determinate
          ? `${progress.steps ? 'History ' : ''}${progress.counted} done`
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
import { scrapheapRunProgress } from "../../utils/scrapheapRun";
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
   * Shown only while `active`; `kind` is "purge", "restore" or "undo" (of a
   * restore run). `total` and `processed` count pictures, or for an undo the
   * restore's History steps (`scrapheapRunProgress`).
   */
  run: { type: Object, default: null },
  /** The last poll failed, so the figures are the last ones read. */
  runUnreachable: { type: Boolean, default: false },
  /** The run request on the wire: "start", "undo", "cancel" or null. */
  runAction: { type: String, default: null },
  /**
   * The Delete forever preview on the wire, by what it was asked about: "all"
   * (Empty scrapheap) or "selection", or null when none is.
   */
  checking: { type: String, default: null },
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

/**
 * How the bar names each kind of run, and the count of what it has done. An
 * undo puts a restore's pictures back in the Scrapheap, so it is worded as
 * that, never as a deletion.
 */
const RUN_WORDS = Object.freeze({
  purge: {
    icon: "mdi-delete-forever-outline",
    gesture: "Delete forever",
    doing: "Deleting forever",
    doneSoFar: "deleted so far, and they cannot be restored",
    doneField: "deleted_count",
    progress: "Delete forever progress",
    stopAfter: "the current batch",
    stopping:
      "Stopping after the current batch. Pictures already deleted cannot be restored.",
    background: "Deleting pictures forever in the background.",
  },
  restore: {
    icon: "mdi-restore",
    gesture: "Restore",
    doing: "Restoring",
    doneSoFar: "restored so far",
    doneField: "restored_count",
    progress: "Restore progress",
    stopAfter: "the current batch",
    stopping: "Stopping after the current batch.",
    background: "Restoring pictures from the Scrapheap in the background.",
  },
  undo: {
    icon: "mdi-undo",
    gesture: "Undo restore",
    doing: "Putting back in the Scrapheap",
    doneSoFar: "put back so far",
    doneField: "undone_count",
    progress: "Undo restore progress",
    stopAfter: "the History step it is on",
    stopping: "Stopping after the History step it is on.",
    background:
      "Putting restored pictures back in the Scrapheap in the background.",
  },
});

const running = computed(() => Boolean(props.run?.active));
const words = computed(() => RUN_WORDS[props.run?.kind] ?? RUN_WORDS.purge);
const stopping = computed(
  () => running.value && Boolean(props.run?.cancel_requested),
);
const pending = computed(
  () => running.value && props.run?.status === "pending",
);

const progress = computed(() => scrapheapRunProgress(props.run));
const counts = computed(() => ({
  total: figure(progress.value.total),
  processed: figure(progress.value.processed),
  done: figure(props.run?.[words.value.doneField]),
}));
const determinate = computed(() => progress.value.determinate);
const percent = computed(() => progress.value.percent);

/** Stop while the run goes on and has not been asked to already. */
const showStop = computed(
  () => running.value && !props.run?.cancel_requested,
);

/** What Stop does, said before it is pressed (`scrapheapRunProgress`). */
const stopTip = computed(() => progress.value.stopTip);

const announcement = computed(() => {
  if (!running.value) {
    return props.checking ? "Checking what Delete forever would destroy." : "";
  }
  if (stopping.value) return words.value.stopping;
  return words.value.background;
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
  color: rgba(var(--v-theme-on-background), var(--opacity-text-secondary));
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
