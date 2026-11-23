<template>
  <div class="s-chip">
    <span class="s-chip__label">{{ label }}</span>
    <span
      v-if="meta"
      class="s-chip__meta"
      :style="metaColor ? { color: metaColor } : null"
    >
      {{ meta }}
    </span>
    <slot />
    <button
      type="button"
      class="s-chip__remove"
      aria-label="Remove"
      :disabled="busy"
      :aria-busy="busy ? 'true' : undefined"
      @click="emit('remove')"
    >
      <Tooltip text="Remove" activator="parent" :describe="false" />
      <v-icon size="15" :class="{ 'mdi-spin': busy }">{{
        busy ? "mdi-loading" : "mdi-close"
      }}</v-icon>
    </button>
  </div>
</template>

<script setup>
import { VIcon } from "vuetify/components";
import Tooltip from "../widgets/Tooltip.vue";

defineProps({
  label: { type: String, default: "" },
  meta: { type: String, default: "" },
  metaColor: { type: String, default: "" },
  // The chip's removal is being saved. The X becomes the spinner and cannot be
  // pressed again until the save settles; the parent clears it either way.
  busy: { type: Boolean, default: false },
});

const emit = defineEmits(["remove"]);
</script>

<style scoped>
.s-chip {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-1) var(--space-1) var(--space-1) var(--space-3);
  border-radius: var(--radius-sm);
  background: rgb(var(--v-theme-input-background));
  border: 1px solid rgb(var(--v-theme-border));
}

.s-chip__label {
  flex: 1;
  min-width: 0;
  font-size: var(--text-sm);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.s-chip__meta {
  font-size: var(--text-xs);
  color: rgba(var(--v-theme-on-surface), 0.6);
  white-space: nowrap;
}

.s-chip__remove {
  width: 20px;
  height: 20px;
  flex-shrink: 0;
  color: rgba(var(--v-theme-on-surface), 0.6);
  border-radius: var(--radius-sm);
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.s-chip__remove:hover:not(:disabled) {
  color: rgb(var(--v-theme-surface-error));
  background: var(--hover-wash);
}

/* Pending, not disabled: the spinner is what tells the user the click landed,
   so the button does not fade, and `progress` says the rest of the pane is
   still live. */
.s-chip__remove:disabled {
  cursor: progress;
}

/* The spinner keeps turning under reduced motion: the global reset would
   freeze it into a broken ring. @mdi/font animates `::before`. */
@media (prefers-reduced-motion: reduce) {
  .s-chip__remove .mdi-spin::before {
    animation-duration: 2s !important;
    animation-iteration-count: infinite !important;
  }
}
</style>
