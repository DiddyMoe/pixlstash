<script setup>
/**
 * The one consent for bulk-stacking the exact-match tier.
 *
 * Usage:
 *   <DedupAutoStackDialog
 *     :open="autoStackOpen"
 *     :preview="dryRun"
 *     :loading="dryRunLoading"
 *     :preview-error="dryRunError"
 *     :busy="autoStackStarting"
 *     :start-error="autoStackStartError"
 *     @close="closeAutoStack"
 *     @confirm="confirmAutoStack"
 *   />
 *
 * Byte-identical files need no judgment, so this dialog replaces per-group
 * adjudication for the exact tier: one dry run, one confirmation, then a run in
 * the background that the queue's banner follows, can stop, and can undo whole.
 * The component is presentational: the parent owns the dry run, the start and
 * the API.
 */
import { computed } from "vue";
import AppDialog from "./AppDialog.vue";
import AppButton from "./AppButton.vue";

const props = defineProps({
  open: { type: Boolean, default: false },
  /**
   * The dry-run report from `POST /dedup/auto-stack`:
   * `{ dry_run, groups, pictures, scope, dry_run_summary, results, failures }`.
   * `results` is empty for a dry run, by design.
   *
   * Every number this dialog renders about the run comes out of
   * `dry_run_summary` - `{ groups, groups_by_tier, pictures,
   * covers_gaining_tags, covers_gaining_score, covers_gaining_metadata }` -
   * because the server derives the whole summary from one read of one group
   * list, so its rows cannot disagree with each other the way two separate
   * counts across a running scan can. The top-level `groups` / `pictures` are
   * only used as a fallback for a server that predates the summary.
   */
  preview: { type: Object, default: null },
  /**
   * Unresolved groups in every tier this bulk action does NOT touch, from the
   * counts endpoint's per-tier split.
   *
   * This one deliberately does not come from `dry_run_summary`. That summary's
   * `groups_by_tier` counts the groups **this run would act on** (exact-only,
   * zero-filled for the rest), so reading the queue's remainder out of it would
   * render a confident zero. The remainder is a property of the queue, not of
   * the run, and only `POST /dedup/counts` knows it.
   */
  queueRemaining: { type: Number, default: 0 },
  /** True while the dry run is in flight. */
  loading: { type: Boolean, default: false },
  /**
   * Why the dry run could not be read, or null when it was: the store's
   * `{ kind, status, detail }` (`useDedupStore.previewAutoStack`). Without this
   * a failed preview and a genuinely empty one are the same screen: a column of
   * zeroes and a disabled button, which says "there is nothing to stack" when
   * the truth is "nobody was able to ask". The kind picks the sentence, so a
   * refused request, a missing permission and a timeout each read as what they
   * are.
   */
  previewError: { type: Object, default: null },
  /**
   * True while the request that starts the run is on the wire. Only the start:
   * the run itself goes on in the background once the server has answered.
   */
  busy: { type: Boolean, default: false },
  /**
   * Why the run did not start, or null: the store's `{ kind, status, detail,
   * code }` (`useDedupStore.startAutoStack`). The dialog stays open on it, so
   * the user reads what happened where they asked, and can confirm again when
   * that helps.
   */
  startError: { type: Object, default: null },
});

const emit = defineEmits(["close", "confirm"]);

// The rows are declared once so the order is fixed and matches the design's,
// and so neither the "Covers gaining metadata" row nor the "Files deleted" row
// can be dropped by a later edit: the first is the whole reason the union is
// safe, and stating the second's zero out loud is the point of showing it.
const ROWS = [
  {
    key: "groups",
    icon: "mdi-layers-plus",
    label: "Stacks to create",
  },
  {
    key: "pictures",
    icon: "mdi-image-multiple-outline",
    label: "Pictures collapsed behind a cover",
  },
  {
    key: "coversGainingMetadata",
    icon: "mdi-tag-multiple-outline",
    label: "Covers gaining metadata from copies",
  },
  {
    key: "queueRemaining",
    icon: "mdi-blur",
    label: "Groups left in the queue to review",
  },
  {
    key: "filesDeleted",
    icon: "mdi-delete-off-outline",
    label: "Files deleted",
  },
];

/**
 * The dry run's own aggregates, or null for a server that predates them.
 *
 * One read of one group list on the server, so the rows below are internally
 * consistent even if a scan lands between this request and the next.
 */
const summary = computed(() => props.preview?.dry_run_summary ?? null);

/**
 * True while no figure from the dry run may be shown: in flight, failed, or
 * not asked yet. A count printed in any of those states is one nobody has
 * seen the server report.
 */
const figuresUnknown = computed(
  () => props.loading || Boolean(props.previewError) || !props.preview,
);

/**
 * The sentence for a failed dry run, or "" when there is none to say.
 *
 * A cancelled request says nothing: it was cancelled because the dialog moved
 * on. Retrying helps after a timeout, an unreachable server or a server fault,
 * so those say how; it does not help a refused request or a missing
 * permission, so those say why instead.
 */
const failureSentence = computed(() => {
  const error = props.previewError;
  if (!error) return "";
  const retry = "so these counts are unknown. Close this and try again.";
  switch (error.kind) {
    case "cancelled":
      return "";
    case "validation": {
      const reason = String(error.detail || "").trim();
      if (!reason) return "The server refused the preview request.";
      return `The server refused the preview request: ${reason}${/[.!?]$/.test(reason) ? "" : "."}`;
    }
    case "forbidden":
      return "You don't have permission to stack duplicates here.";
    case "timeout":
      return `The preview took too long to answer, ${retry}`;
    case "network":
      return `The server could not be reached, ${retry}`;
    default:
      return `The preview failed on the server, ${retry}`;
  }
});

/**
 * The sentence for a start that did not go through, or "" when there is none.
 *
 * Confirming again sends the same request, which the server answers with the
 * run the first attempt started if it did start one, so it is safe to offer
 * whenever the outcome is unknown or the fault was the server's. An answer
 * that came back as no is worded as that answer instead. A cancelled start
 * (the session ended) and a start still on the wire say nothing.
 */
const startSentence = computed(() => {
  const error = props.startError;
  if (!error) return "";
  const again =
    "Confirm again to retry: it sends the same request, which cannot start a second run.";
  switch (error.kind) {
    case "cancelled":
    case "busy":
      return "";
    case "timeout":
    case "network":
      return `No answer came from the server, so it is not known whether the run started. ${again}`;
    case "conflict":
      if (error.code === "auto_stack_run_active") {
        return "Another Auto-stack run is already going, and its progress shows above the queue. Start this one once it has finished.";
      }
      return refusedSentence(error.detail);
    case "validation":
      return refusedSentence(error.detail);
    case "forbidden":
      return "You don't have permission to stack duplicates here.";
    default:
      return `The run could not start on the server. ${again}`;
  }
});

/** "The server refused …", ending on the server's own reason when it gave one. */
function refusedSentence(detail) {
  const reason = String(detail || "").trim();
  if (!reason) return "The server refused to start the run.";
  return `The server refused to start the run: ${reason}${/[.!?]$/.test(reason) ? "" : "."}`;
}

/**
 * Groups this run would stack.
 *
 * Read from the summary's per-tier split, which is the same number the
 * summary's other rows were derived alongside. Auto-stack is exact-only, so the
 * split is the exact tier plus zeroes; summing it rather than naming `exact`
 * keeps the dialog correct if the run ever widens.
 */
const stacksToCreate = computed(() => {
  const byTier = summary.value?.groups_by_tier;
  if (byTier && typeof byTier === "object") {
    return Object.values(byTier).reduce(
      (sum, count) => sum + (Number(count) || 0),
      0,
    );
  }
  return Number(summary.value?.groups ?? props.preview?.groups) || 0;
});

const picturesCollapsed = computed(
  () => Number(summary.value?.pictures ?? props.preview?.pictures) || 0,
);

/**
 * Covers that would gain a tag or a better score from the union.
 *
 * The design's "covers gaining metadata from copies" row, and the answer to the
 * only real question a bulk stack raises: whether collapsing copies loses
 * anything. It cannot, because the union runs in the other direction, and this
 * number says how often it actually will.
 */
const coversGainingMetadata = computed(
  () => Number(summary.value?.covers_gaining_metadata) || 0,
);

/**
 * Whether the confirm button may fire. Not while the start is pending: that is
 * the button's `loading` state, which holds it without fading it. Not against
 * a run already going either, since the server would refuse it again.
 */
const canConfirm = computed(
  () =>
    !figuresUnknown.value &&
    stacksToCreate.value > 0 &&
    props.startError?.code !== "auto_stack_run_active",
);

/**
 * The confirm label. While the count is unknown it drops the number rather
 * than printing a zero nobody reported; the button is disabled then anyway.
 */
const confirmLabel = computed(() =>
  figuresUnknown.value
    ? "Create stacks"
    : `Create ${stacksToCreate.value.toLocaleString()} stacks`,
);

/**
 * The number a row shows.
 *
 * While the dry run is in flight every row shows an en dash rather than a
 * spinner, so the dialog keeps its height and nothing moves under the pointer
 * when the counts land. A failed dry run keeps the dashes.
 */
function rowValue(key) {
  // Deletion is a property of the feature, not a number the server reports:
  // there is no destructive route on this surface at all in 1.9. Stating the
  // zero out loud is the point of the row.
  if (key === "filesDeleted") return "0";
  // The queue's remainder comes from the counts endpoint, not the dry run, so
  // it stays readable while the dry run is still in flight.
  if (key === "queueRemaining") {
    return (Number(props.queueRemaining) || 0).toLocaleString();
  }
  if (figuresUnknown.value) return "–";
  if (key === "groups") return stacksToCreate.value.toLocaleString();
  if (key === "pictures") return picturesCollapsed.value.toLocaleString();
  if (key === "coversGainingMetadata") {
    return coversGainingMetadata.value.toLocaleString();
  }
  return "0";
}
</script>

<template>
  <AppDialog
    :open="open"
    title="Auto-stack exact matches"
    @close="emit('close')"
  >
    <p class="as-lede">
      Byte-identical files need no judgment. This groups every exact-match set
      behind one cover, the largest and best-tagged copy of each. Every other
      copy stays in your library, and every tag, character and set membership
      moves onto the stack. Near-duplicates stay in the queue for you to review.
    </p>

    <p v-if="failureSentence" class="as-failed" role="status">
      <v-icon size="16" class="as-icon">mdi-alert-outline</v-icon>
      {{ failureSentence }}
    </p>

    <dl class="as-rows">
      <div v-for="row in ROWS" :key="row.key" class="as-row">
        <dt class="as-term">
          <v-icon size="16" class="as-icon">{{ row.icon }}</v-icon>
          {{ row.label }}
        </dt>
        <dd class="as-value">{{ rowValue(row.key) }}</dd>
      </div>
    </dl>

    <p class="as-reversible">
      <v-icon size="16" class="as-icon">mdi-information-outline</v-icon>
      Runs in the background while you keep working. You can stop it at any
      time, and Undo run reverses every stack it made.
    </p>

    <p v-if="startSentence" class="as-failed as-start-failed" role="status">
      <v-icon size="16" class="as-icon">mdi-alert-outline</v-icon>
      {{ startSentence }}
    </p>

    <template #footer>
      <AppButton variant="ghost" @click="emit('close')">Cancel</AppButton>
      <AppButton
        variant="primary"
        icon-left="layers-plus"
        :disabled="!canConfirm"
        :loading="busy"
        @click="emit('confirm')"
      >
        {{ confirmLabel }}
      </AppButton>
    </template>
  </AppDialog>
</template>

<style scoped>
.as-lede {
  margin: 0;
  font-size: var(--text-base);
  line-height: var(--leading-body);
  color: rgba(var(--v-theme-on-surface), 0.8);
}

/* Same construction as the tier menu's in-place warning, so a failure and a
   caution read as one family rather than two components. */
.as-failed {
  display: flex;
  align-items: flex-start;
  gap: var(--space-2);
  margin: 0;
  padding: var(--space-3);
  border-radius: var(--radius-md);
  background: rgba(var(--v-theme-warning), 0.12);
  border: 1px solid rgba(var(--v-theme-surface-warning), 0.35);
  font-size: var(--text-xs);
  line-height: var(--leading-body);
  color: rgb(var(--v-theme-on-surface));
}

.as-rows {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
}

.as-row {
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: baseline;
  gap: var(--space-4);
  padding: var(--space-3) 0;
  border-bottom: 1px solid rgb(var(--v-theme-divider));
}

.as-row:last-child {
  border-bottom: none;
}

.as-term {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  font-size: var(--text-sm);
  color: rgba(var(--v-theme-on-surface), 0.8);
}

.as-icon {
  color: rgba(var(--v-theme-on-surface), 0.6);
  flex-shrink: 0;
}

.as-value {
  margin: 0;
  font-size: var(--text-md);
  font-weight: var(--weight-semibold);
  font-variant-numeric: tabular-nums;
  color: rgb(var(--v-theme-on-surface));
}

.as-reversible {
  display: flex;
  align-items: flex-start;
  gap: var(--space-2);
  margin: 0;
  font-size: var(--text-xs);
  line-height: var(--leading-body);
  color: rgba(var(--v-theme-on-surface), 0.6);
}
</style>
