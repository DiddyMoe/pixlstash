<script setup>
/**
 * The one consent for collapsing stacks to their covers.
 *
 * Usage:
 *   <KeepCoverOnlyDialog
 *     :open="keepCoverOpen"
 *     :preview="keepCoverPreview"
 *     :loading="keepCoverLoading"
 *     :preview-error="keepCoverPreviewError"
 *     :busy="keepCoverBusy"
 *     :progress="keepCoverProgress"
 *     @close="keepCoverOpen = false"
 *     @confirm="runKeepCoverOnly"
 *     @retry="loadKeepCoverPreview"
 *   />
 *
 * Presentational: the parent owns the preview, the real run and the API.
 *
 * Three properties of this dialog are load-bearing and are spelled out where
 * they are implemented (`docs/design/keep-cover-only.md`):
 *
 *   * **The title says what you keep; the button says what you lose.** The
 *     headline figure and the button label render from ONE computed value
 *     (`picturesMoving`), not merely from one endpoint, so they can never
 *     disagree. That is the failure the neighbouring auto-stack dialog shipped
 *     when it reported "62 stacks to create" for work that would create 3.
 *   * **Nothing is freed now.** The recovery copy explains when files can
 *     actually leave disk; a redundant `Originals deleted: 0` statistic does
 *     not compete with the consequence the user is confirming.
 *   * **No type-to-confirm.** That gate belongs to `DeleteForeverDialog`, where
 *     an on-disk original dies. This is a recoverable soft delete, one op-log
 *     batch and one Ctrl+Z; borrowing the heavier ceremony would flatten the
 *     distinction between "recoverable" and "gone".
 */
import { computed, nextTick, ref, watch } from "vue";
import AppDialog from "./AppDialog.vue";
import AppButton from "./AppButton.vue";
import {
  KEEP_COVER_ONLY_ICON_NAME,
  UNKNOWN_FIGURE,
  keepCoverOnlyBytesSentence,
  keepCoverOnlyConfirmLabel,
  keepCoverOnlyPreviewFailure,
  keepCoverOnlyProgressSentence,
  keepCoverOnlyRetentionSentence,
  keepCoverOnlySkipReasons,
  keepCoverOnlySkippedCount,
  keepCoverOnlyTitle,
} from "../../utils/keepCoverOnly";
import KeyCap from "./KeyCap.vue";

const props = defineProps({
  open: { type: Boolean, default: false },
  /**
   * The dry run from `POST /stacks/keep-cover-only/preview`.
   *
   * Every figure this dialog renders comes out of this one body, because the
   * server derives the whole report from one read over the same selection
   * through the same planner the mutation uses. Do not add a prop fed by a
   * second request, and do not derive a bucket by subtracting one of these
   * figures from another: the stack buckets are disjoint and already sum to
   * `stacks_selected`.
   */
  preview: { type: Object, default: null },
  /** True while the dry run is in flight. */
  loading: { type: Boolean, default: false },
  /**
   * Why the dry run could not be read, or null when it was: `requestFailure`'s
   * `{ kind, status, detail, code }` (`utils/apiError.js`). Without this a
   * failed preview and a genuinely empty one are the same screen: a zero over a
   * live confirm button, which says "there is nothing to collapse" when the
   * truth is "nobody was able to ask". The kind picks the sentence and whether
   * Try again is offered.
   */
  previewError: { type: Object, default: null },
  /** True while the real run is in flight. */
  busy: { type: Boolean, default: false },
  /**
   * `{ stacksDone, stacksTotal }` while a selection of more than one request
   * is being collapsed, or null. A large selection takes several seconds per
   * request, past the point where a spinner alone says enough.
   */
  progress: { type: Object, default: null },
});

const emit = defineEmits(["close", "confirm", "retry"]);

/**
 * True while no figure from the server may be shown.
 *
 * One flag for both states on purpose: a stale number from the previous
 * selection is exactly as wrong as an invented one.
 */
const figuresUnknown = computed(
  () => props.loading || Boolean(props.previewError) || !props.preview,
);

/** The failed preview's sentence, and whether Try again can help. */
const failure = computed(() => keepCoverOnlyPreviewFailure(props.previewError));

/** "500 of 1,201 stacks done…" while a long collapse runs, else "". */
const progressSentence = computed(() =>
  props.busy ? keepCoverOnlyProgressSentence(props.progress) : "",
);

/**
 * THE number: pictures that would move to the Scrapheap.
 *
 * `null` means "not known yet", which is a different thing from zero and has to
 * render differently. Both the headline block and the confirm button are
 * derived from this single ref below, which is the whole point: two readings of
 * the same endpoint can still drift; two readings of the same computed cannot.
 */
const picturesMoving = computed(() => {
  if (figuresUnknown.value) return null;
  return Number(props.preview.pictures_moving) || 0;
});

/** Stacks that would actually collapse: the figure the title names. */
const stacksEligible = computed(() => {
  if (figuresUnknown.value) return null;
  return Number(props.preview.stacks_eligible) || 0;
});

/** The headline numeral, or the en dash placeholder at the same size. */
const headlineFigure = computed(() =>
  picturesMoving.value === null
    ? UNKNOWN_FIGURE
    : picturesMoving.value.toLocaleString(),
);

const title = computed(() => keepCoverOnlyTitle(stacksEligible.value));

/** The confirm label, from the same figure as the headline. */
const confirmLabel = computed(() =>
  keepCoverOnlyConfirmLabel(picturesMoving.value),
);

/** Metadata transfer is useful only when it will actually happen. */
const coversGainingMetadata = computed(() =>
  figuresUnknown.value
    ? 0
    : Number(props.preview.covers_gaining_metadata) || 0,
);

const skipReasons = computed(() =>
  figuresUnknown.value ? [] : keepCoverOnlySkipReasons(props.preview),
);

const skippedCount = computed(() =>
  figuresUnknown.value ? 0 : keepCoverOnlySkippedCount(props.preview),
);

const skippedSummary = computed(() => {
  const count = skippedCount.value;
  return `${count.toLocaleString()} ${count === 1 ? "stack" : "stacks"} won't change`;
});

/**
 * What a screen reader is told, through the one status region that is mounted
 * with the dialog: why the preview failed, how far a long collapse has got, or
 * that some stacks won't change. Each of those appears after the dialog opens,
 * and a live region inserted together with its text is often not announced, so
 * they share this region rather than each carrying a role of its own.
 */
const announcement = computed(
  () =>
    failure.value.sentence ||
    progressSentence.value ||
    (skipReasons.value.length ? `${skippedSummary.value}.` : ""),
);

/**
 * The recovery window, read from the response.
 *
 * `scrapheap_retention_days` is `null` on a default install, which means the
 * Scrapheap never empties on its own. Hardcoding a window here would be the
 * same class of error the whole dialog exists to avoid, so this branches on the
 * live value and renders nothing until it has one.
 */
const retentionSentence = computed(() =>
  figuresUnknown.value
    ? ""
    : keepCoverOnlyRetentionSentence(props.preview.scrapheap_retention_days),
);

/**
 * What the copies hold on disk. A sentence, never a figure block: a figure is
 * for what changes now, and nothing is reclaimed until the Scrapheap is emptied.
 */
const bytesSentence = computed(() =>
  figuresUnknown.value
    ? ""
    : keepCoverOnlyBytesSentence(props.preview.bytes_held_by_copies),
);

const referenceFolderCount = computed(() =>
  figuresUnknown.value
    ? 0
    : Number(props.preview.reference_folder_pictures_moving) || 0,
);

const canConfirm = computed(
  () => !props.busy && !figuresUnknown.value && picturesMoving.value > 0,
);

// ── The keyboard, deliberately inverted ─────────────────────────────────────
// `AppDialog` implements the app's dialog contract: Escape dismisses, plain
// Enter accepts. This dialog opts OUT of the second half, and it must stay
// opted out.
//
// It does not listen for `accept`, so a bare Enter reaches nothing; and Cancel
// takes focus on open, which puts the key on a native button, where AppDialog's
// own ENTER_EXEMPT rule hands it to the button's default activation. So Enter
// cancels. That is the point: users arrive here straight from the duplicate
// queue with Enter under their finger from the verdict keys, and the very next
// press must not be consent for hundreds of deletions.
//
// If a later change "fixes" this by adding an @accept handler or by focusing
// the confirm button, it reintroduces exactly that hazard. Do not.
const cancelRef = ref(null);

function focusCancel() {
  nextTick(() => {
    cancelRef.value?.focus?.();
  });
}

watch(
  () => props.open,
  (isOpen) => {
    // After the dialog's own enter transition has mounted the footer.
    if (isOpen) focusCancel();
  },
);

/**
 * Ask again. Try again disappears with the failure it answers, so the keyboard
 * goes back to Cancel rather than to the page behind the dialog; Cancel, not
 * the confirm, for the same reason it starts there.
 */
function retry() {
  emit("retry");
  focusCancel();
}
</script>

<template>
  <AppDialog :open="open" :title="title" @close="emit('close')">
    <p class="kco-lede">
      Each stack keeps its cover. Every other picture in it moves to the
      Scrapheap, where you can restore it. A stack collapses whole, even if you
      only picked some of its pictures, and loose pictures are left alone.
    </p>

    <div v-if="failure.sentence" class="kco-failed">
      <v-icon size="16" class="kco-icon">mdi-alert-outline</v-icon>
      <div class="kco-failed-body">
        <!-- Heard through the dialog's status region, below. -->
        <p class="kco-failed-text" aria-hidden="true">{{ failure.sentence }}</p>
        <!-- In place, so the same selection is asked about again: closing
             the dialog to retry meant rebuilding a Ctrl+A over the grid. -->
        <AppButton
          v-if="failure.retryable"
          data-testid="keep-cover-retry"
          variant="secondary"
          size="sm"
          icon-left="refresh"
          @click="retry"
        >
          Try again
        </AppButton>
      </div>
    </div>

    <!-- The headline. One instance of the figure, and the confirm button below
         renders the same computed value. The numeral is `on-surface`, never
         `error`: a 22px red number would be the loudest object in the app. The
         hue rides the leading rail instead. -->
    <div class="kco-headline">
      <p class="kco-figure" data-testid="keep-cover-figure">
        {{ headlineFigure }}
      </p>
      <p class="kco-figure-label">pictures move to the Scrapheap</p>
    </div>

    <p v-if="coversGainingMetadata > 0" class="kco-metadata">
      <v-icon size="16" class="kco-icon">mdi-tag-multiple-outline</v-icon>
      <span>
        {{ coversGainingMetadata.toLocaleString() }}
        {{ coversGainingMetadata === 1 ? "cover gains" : "covers gain" }}
        metadata from copies before they move.
      </span>
    </p>

    <!-- Not a live region itself: it arrives with its text. The summary is
         announced through the status region below; the reasons stay here to
         be read. -->
    <section
      v-if="skipReasons.length"
      class="kco-skips"
      aria-labelledby="keep-cover-skips-title"
    >
      <p id="keep-cover-skips-title" class="kco-skips-title">
        <v-icon size="16" class="kco-icon">mdi-cancel</v-icon>
        {{ skippedSummary }}
      </p>
      <ul>
        <li v-for="reason in skipReasons" :key="reason.key">
          {{ reason.text }}
        </li>
      </ul>
    </section>

    <!-- Info-tinted, not error-tinted: recovery is the reassuring half, and
         DeleteForeverDialog's lock note already made this call for the same
         reason. -->
    <div class="kco-recovery">
      <v-icon size="18" class="kco-icon">mdi-backup-restore</v-icon>
      <div class="kco-recovery-body">
        <p v-if="retentionSentence" class="kco-sentence">
          {{ retentionSentence }}
        </p>
        <p v-if="bytesSentence" class="kco-sentence">{{ bytesSentence }}</p>
        <p v-if="referenceFolderCount > 0" class="kco-sentence">
          {{ referenceFolderCount.toLocaleString() }} of them live in a
          reference folder; those rows move but the files stay exactly where
          they are.
        </p>
      </div>
    </div>

    <p class="kco-reversible">
      <v-icon size="16" class="kco-icon">mdi-information-outline</v-icon>
      Reversible as one step with
      <!-- One flex item, so the paragraph's gap does not split the chord. -->
      <span><KeyCap k="Ctrl" /><kbd>Z</kbd></span>.
    </p>

    <!-- Seen beside the spinning confirm; heard through the status region. -->
    <p
      v-if="progressSentence"
      class="kco-progress"
      data-testid="keep-cover-progress"
      aria-hidden="true"
    >
      {{ progressSentence }}
    </p>
    <!-- The dialog's only live region. It is mounted with the dialog, so a
         screen reader is listening before the first sentence arrives. -->
    <p
      class="visually-hidden"
      role="status"
      data-testid="keep-cover-announcer"
    >
      {{ announcement }}
    </p>

    <template #footer>
      <AppButton ref="cancelRef" variant="ghost" @click="emit('close')">
        Cancel
      </AppButton>
      <AppButton
        variant="danger"
        :icon-left="KEEP_COVER_ONLY_ICON_NAME"
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
.kco-lede {
  margin: 0;
  font-size: var(--text-base);
  line-height: var(--leading-body);
  color: rgba(var(--v-theme-on-surface), 0.8);
}

/* Same construction as the auto-stack dialog's in-place warning, so a failure
   reads as one family across the two sibling confirms. */
.kco-failed {
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

.kco-failed-body {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: var(--space-2);
  min-width: 0;
}

.kco-failed-text {
  margin: 0;
}

/* The leading status rail carries the consequence's hue so the numeral does
   not have to. `--rail-w`, not a raw 3px. */
.kco-headline {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  margin: 0;
  padding-left: var(--space-4);
  border-left: var(--rail-w) solid rgb(var(--v-theme-surface-error));
}

.kco-figure {
  margin: 0;
  font-size: var(--text-xl);
  font-weight: var(--weight-semibold);
  line-height: var(--leading-tight);
  font-variant-numeric: tabular-nums;
  color: rgb(var(--v-theme-on-surface));
}

.kco-figure-label {
  margin: 0;
  font-size: var(--text-sm);
  line-height: var(--leading-snug);
  color: rgba(var(--v-theme-on-surface), 0.8);
}

.kco-metadata {
  display: flex;
  align-items: flex-start;
  gap: var(--space-2);
  margin: 0;
  font-size: var(--text-sm);
  line-height: var(--leading-body);
  color: rgba(var(--v-theme-on-surface), 0.8);
}

.kco-icon {
  color: rgba(var(--v-theme-on-surface), 0.6);
  flex-shrink: 0;
}

.kco-skips {
  margin: 0;
  padding: var(--space-3);
  border: 1px solid rgba(var(--v-theme-surface-warning), 0.35);
  border-radius: var(--radius-md);
  background: rgba(var(--v-theme-warning), 0.08);
  font-size: var(--text-xs);
  line-height: var(--leading-body);
  color: rgba(var(--v-theme-on-surface), 0.8);
}

.kco-skips-title {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin: 0;
  font-size: var(--text-sm);
  font-weight: var(--weight-semibold);
  color: rgb(var(--v-theme-on-surface));
}

.kco-skips ul {
  margin: var(--space-2) 0 0;
  padding-left: var(--space-5);
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.kco-recovery {
  display: flex;
  align-items: flex-start;
  gap: var(--space-3);
  margin: 0;
  padding: var(--space-4);
  border: 1px solid rgba(var(--v-theme-surface-info), 0.5);
  background: rgba(var(--v-theme-info), 0.08);
  border-radius: var(--radius-md);
  color: rgb(var(--v-theme-on-surface));
}

.kco-recovery-body {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  min-width: 0;
}

.kco-sentence {
  margin: 0;
  font-size: var(--text-sm);
  line-height: var(--leading-snug);
}

.kco-reversible {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin: 0;
  font-size: var(--text-xs);
  color: rgba(var(--v-theme-on-surface), 0.6);
}

/* The same quiet line as the undo hint, at full strength: it is the one
   thing changing while the run goes on. */
.kco-progress {
  margin: 0;
  font-size: var(--text-xs);
  line-height: var(--leading-snug);
  font-variant-numeric: tabular-nums;
  color: rgba(var(--v-theme-on-surface), 0.8);
}

.kco-reversible kbd {
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
  padding: var(--kbd-pad);
  border: 1px solid rgba(var(--v-theme-on-surface), 0.3);
  border-radius: var(--radius-sm);
  background: rgba(var(--v-theme-on-surface), 0.08);
}
</style>
