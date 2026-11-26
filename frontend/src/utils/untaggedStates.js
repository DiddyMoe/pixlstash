/**
 * Why an untagged picture is untagged: the values of the listing's
 * `tag_state` filter, and the one place they are spelled.
 *
 * The stats rail offers each as a row and the grid's strip shows the active
 * one as a chip, so both read the label from here.
 */
export const UNTAGGED_STATES = Object.freeze([
  {
    state: "pending",
    label: "Waiting for the tagger",
    chip: "waiting for the tagger",
    tooltip: "Show the pictures waiting for the tagger",
  },
  {
    state: "unprocessable",
    label: "Couldn't be read",
    chip: "couldn't be read",
    tooltip: "Show the pictures the tagger couldn't read",
  },
  {
    state: "none",
    label: "No tags apply",
    chip: "no tags apply",
    tooltip: "Show the tagged pictures that carry no tags",
  },
]);
