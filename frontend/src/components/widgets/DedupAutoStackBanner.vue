<script setup>
/**
 * The banner that follows a background Auto-stack run above the duplicate
 * queue.
 *
 * Auto-stack runs on the server in steps, for as long as the exact tier takes,
 * so the page that started it is only an observer: this banner is how that
 * observer says how far the run has got, that the user may keep working, and
 * what the run left behind once it ends. It is modelled on DedupScanBanner and
 * sits beside it, with one difference that changes its shape: it carries
 * controls (Stop, Undo run, Dismiss), so the progress track is clipped by its
 * own layer rather than by the banner, where it would clip their focus rings.
 *
 * It is deliberately not a live region. Progress moves on every poll, and a
 * region that re-reads a whole sentence every two seconds drowns out
 * everything else; the destination's one live region announces the run's
 * changes of state instead, and the progress bar carries the figures for
 * anyone who asks for them.
 *
 * Presentational: the store owns the run, and the parent turns the emitted
 * gestures into store actions.
 */
import { computed } from "vue";
import AppButton from "./AppButton.vue";

/** Run statuses that end with work left over that the user should hear about. */
const WARNING_STATES = new Set(["interrupted", "failed", "blocked"]);

const props = defineProps({
  /**
   * The run as the server serializes it (`serialize_run`): `kind` is "apply"
   * or "revert"; `total` / `processed` count groups for an apply run and
   * History steps for a revert; `applied` is groups stacked, or for a revert
   * groups unstacked; `pictures` is the pictures those stacks hold.
   */
  run: { type: Object, required: true },
  /** The last poll failed, so the figures are the last ones read. */
  unreachable: { type: Boolean, default: false },
  /** The run request on the wire: "start", "cancel", "revert" or null. */
  action: { type: String, default: null },
  /** Undo run is available: see `autoStackCanRevert` in the store. */
  canRevert: { type: Boolean, default: false },
});

const emit = defineEmits(["stop", "revert", "dismiss"]);

/** A count as the sentence shows it: grouped digits, never NaN. */
function figure(value) {
  const n = Number(value);
  return (Number.isFinite(n) ? n : 0).toLocaleString();
}

const status = computed(() => props.run?.status ?? "pending");
const reverting = computed(() => props.run?.kind === "revert");
const active = computed(() => Boolean(props.run?.active));
const stopping = computed(
  () => active.value && Boolean(props.run?.cancel_requested),
);
const pending = computed(() => active.value && status.value === "pending");
const warning = computed(() => WARNING_STATES.has(status.value));
const errorDetail = computed(() => String(props.run?.error ?? "").trim());

const total = computed(() => Number(props.run?.total) || 0);
const processed = computed(() => Number(props.run?.processed) || 0);
const applied = computed(() => Number(props.run?.applied) || 0);
const superseded = computed(() => Number(props.run?.superseded) || 0);
const refused = computed(
  () => (Number(props.run?.blocked) || 0) + (Number(props.run?.failed) || 0),
);

const counts = computed(() => ({
  total: figure(total.value),
  processed: figure(processed.value),
  applied: figure(applied.value),
  pictures: figure(props.run?.pictures),
  superseded: figure(superseded.value),
  refused: figure(refused.value),
}));

/** Progress as a whole number in 0 to 100, known only once a total is. */
const determinate = computed(
  () => active.value && !pending.value && total.value > 0,
);
const percent = computed(() => {
  if (!determinate.value) return 0;
  return Math.min(
    100,
    Math.max(0, Math.round((processed.value / total.value) * 100)),
  );
});

const progressLabel = computed(() =>
  reverting.value ? "Auto-stack undo progress" : "Auto-stack progress",
);
const progressText = computed(() =>
  reverting.value
    ? `${counts.value.processed} of ${counts.value.total} steps undone`
    : `${counts.value.processed} of ${counts.value.total} groups done`,
);

const metaText = computed(() => {
  if (stopping.value) return "Stopping";
  if (pending.value) return "Queued";
  if (determinate.value) return `${percent.value}%`;
  return active.value ? "Starting" : "";
});

/** Undo run's label: a revert that stopped short is finished, not begun. */
const revertLabel = computed(() =>
  reverting.value ? "Finish undo" : "Undo run",
);
const revertTip = computed(() =>
  reverting.value
    ? "Undo the steps this run has not reached yet. Steps already undone are skipped."
    : "Unstack every group this run stacked, newest step first.",
);

/** Stop while the run goes on (and has not been asked to already). */
const showStop = computed(() => active.value && !props.run?.cancel_requested);
const showRevert = computed(() => !active.value && props.canRevert);
const showDismiss = computed(() => !active.value);

/** While one request is on the wire, the other controls wait for it. */
const otherActionPending = (own) =>
  props.action != null && props.action !== own;
</script>

<template>
  <div
    class="as-banner"
    :class="{ 'as-banner--warning': warning }"
    role="group"
    :aria-label="reverting ? 'Auto-stack undo' : 'Auto-stack run'"
    data-testid="auto-stack-banner"
  >
    <v-icon class="as-banner__ico" size="18">{{
      reverting ? "mdi-undo-variant" : "mdi-flash-outline"
    }}</v-icon>

    <div class="as-banner__body">
      <!-- ── An apply run ─────────────────────────────────────────────── -->
      <p v-if="!reverting && stopping" class="as-banner__line">
        Stopping Auto-stack after the current step. <b>{{ counts.applied }}</b>
        groups stacked so far.
      </p>
      <p v-else-if="!reverting && pending" class="as-banner__line">
        Auto-stack is queued and starts when the server gets to it. You can keep
        working.
      </p>
      <p v-else-if="!reverting && active" class="as-banner__line">
        Auto-stacking exact matches. <b>{{ counts.processed }}</b> of
        <b>{{ counts.total }}</b> groups done, <b>{{ counts.applied }}</b>
        stacked. You can keep working.
      </p>
      <p
        v-else-if="!reverting && status === 'complete'"
        class="as-banner__line"
      >
        Auto-stack finished. <b>{{ counts.applied }}</b> groups stacked (<b>{{
          counts.pictures
        }}</b>
        pictures).<span v-if="superseded"
          >{{ " " }}<b>{{ counts.superseded }}</b> had already left the
          queue.</span
        ><span v-if="refused"
          >{{ " " }}<b>{{ counts.refused }}</b> could not be stacked and are
          still in the queue.</span
        >
      </p>
      <p
        v-else-if="!reverting && status === 'cancelled'"
        class="as-banner__line"
      >
        Auto-stack stopped. <b>{{ counts.applied }}</b> of
        <b>{{ counts.total }}</b> groups stacked; the rest are still in the
        queue.
      </p>
      <p
        v-else-if="!reverting && status === 'interrupted'"
        class="as-banner__line"
      >
        Auto-stack was interrupted when the server restarted.
        <b>{{ counts.applied }}</b> of <b>{{ counts.total }}</b> groups stacked;
        the rest are still in the queue.
      </p>
      <p v-else-if="!reverting" class="as-banner__line">
        Auto-stack stopped on an error. <b>{{ counts.applied }}</b> of
        <b>{{ counts.total }}</b> groups stacked; the rest are still in the
        queue.<span v-if="errorDetail">{{ " " }}{{ errorDetail }}</span>
      </p>

      <!-- ── A revert run ─────────────────────────────────────────────── -->
      <p v-else-if="stopping" class="as-banner__line">
        Stopping the undo after the current step. <b>{{ counts.applied }}</b>
        groups are back in the queue so far.
      </p>
      <p v-else-if="pending" class="as-banner__line">
        Undo of Auto-stack is queued and starts when the server gets to it.
      </p>
      <p v-else-if="active" class="as-banner__line">
        Undoing Auto-stack. <b>{{ counts.processed }}</b> of
        <b>{{ counts.total }}</b> steps undone, <b>{{ counts.applied }}</b>
        groups back in the queue.
      </p>
      <p v-else-if="status === 'complete'" class="as-banner__line">
        Auto-stack undone. <b>{{ counts.applied }}</b> groups are back in the
        queue.
      </p>
      <p v-else-if="status === 'blocked'" class="as-banner__line">
        Undo stopped at a step that later work was done on top of.
        <b>{{ counts.applied }}</b> groups are back in the queue. Undo the later
        work from History first, then choose {{ revertLabel }}.
      </p>
      <p v-else-if="status === 'cancelled'" class="as-banner__line">
        Undo stopped. <b>{{ counts.applied }}</b> groups are back in the queue;
        the rest are still stacked.
      </p>
      <p v-else-if="status === 'interrupted'" class="as-banner__line">
        Undo was interrupted when the server restarted.
        <b>{{ counts.applied }}</b> groups are back in the queue; the rest are
        still stacked.
      </p>
      <p v-else class="as-banner__line">
        Undo stopped on an error. <b>{{ counts.applied }}</b> groups are back in
        the queue; the rest are still stacked.<span v-if="errorDetail"
          >{{ " " }}{{ errorDetail }}</span
        >
      </p>

      <p
        v-if="unreachable"
        class="as-banner__note"
        data-testid="auto-stack-banner-unreachable"
      >
        No answer from the server, so these are the last figures read. It keeps
        asking.
      </p>
    </div>

    <span v-if="metaText" class="as-banner__meta">{{ metaText }}</span>

    <div
      v-if="showStop || showRevert || showDismiss"
      class="as-banner__actions"
    >
      <AppButton
        v-if="showStop"
        size="sm"
        :loading="action === 'cancel'"
        :disabled="otherActionPending('cancel')"
        data-testid="auto-stack-stop"
        @click="emit('stop')"
        >Stop</AppButton
      >
      <AppButton
        v-if="showRevert"
        variant="outline"
        size="sm"
        icon-left="undo-variant"
        :tooltip="revertTip"
        :loading="action === 'revert'"
        :disabled="otherActionPending('revert')"
        data-testid="auto-stack-revert"
        @click="emit('revert')"
        >{{ revertLabel }}</AppButton
      >
      <AppButton
        v-if="showDismiss"
        variant="ghost"
        size="sm"
        icon-left="close"
        icon-only
        tooltip="Dismiss"
        :disabled="action != null"
        data-testid="auto-stack-dismiss"
        @click="emit('dismiss')"
      />
    </div>

    <!-- The track's own clipping layer: the banner cannot clip, since its
         buttons' focus rings reach past its edge. -->
    <span v-if="active" class="as-banner__clip">
      <span
        class="as-banner__track"
        role="progressbar"
        :aria-label="progressLabel"
        :aria-valuenow="determinate ? percent : null"
        :aria-valuetext="determinate ? progressText : null"
        aria-valuemin="0"
        aria-valuemax="100"
      >
        <span class="as-banner__fill" :style="{ width: `${percent}%` }"></span>
      </span>
    </span>
  </div>
</template>

<style scoped>
/* The same panel surface as the scan banner it sits beside: ambient status
   about the list below, never part of the notice stack. */
.as-banner {
  position: relative;
  display: grid;
  grid-template-columns: auto 1fr auto auto;
  align-items: center;
  gap: var(--space-3);
  margin-inline: var(--space-5);
  padding: var(--space-3);
  border-radius: var(--radius-md);
  background: rgb(var(--v-theme-panel));
  color: rgb(var(--v-theme-on-panel));
  box-shadow: var(--elevation-1);
}

.as-banner__ico {
  color: rgb(var(--v-theme-on-panel));
  flex-shrink: 0;
}

.as-banner--warning .as-banner__ico {
  color: rgb(var(--v-theme-surface-warning));
}

.as-banner__body {
  min-width: 0;
}

.as-banner__line,
.as-banner__note {
  margin: 0;
  font-size: var(--text-sm);
  line-height: var(--leading-snug);
}

/* The counts are what the eye returns to, and tabular figures stop the
   sentence reflowing as they tick up. */
.as-banner__line b {
  font-weight: var(--weight-semibold);
  font-variant-numeric: tabular-nums;
}

.as-banner__note {
  margin-top: var(--space-2);
  font-size: var(--text-xs);
  color: rgb(var(--v-theme-surface-warning));
}

.as-banner__meta {
  justify-self: end;
  font-size: var(--text-xs);
  line-height: var(--leading-snug);
  color: rgba(var(--v-theme-on-panel), var(--opacity-text-secondary));
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.as-banner__actions {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.as-banner__clip {
  position: absolute;
  inset: 0;
  border-radius: inherit;
  overflow: hidden;
  pointer-events: none;
}

/* Pinned to the bottom edge and full width, so the banner itself is the meter. */
.as-banner__track {
  position: absolute;
  inset-inline: 0;
  bottom: 0;
  height: var(--countdown-h);
  background: rgba(var(--v-theme-on-panel), 0.14);
}

.as-banner__fill {
  display: block;
  height: 100%;
  background: rgb(var(--v-theme-accent));
  /* Progress arrives in polls, so the fill eases between them. The global
     reduced-motion rule collapses this. */
  transition: width var(--dur-2) var(--ease-standard);
}
</style>
