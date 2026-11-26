/**
 * Whether a key event belongs to an input method's composition (Japanese,
 * Chinese, Korean and other IMEs) rather than to the page.
 *
 * While a composition is open, Enter commits the candidate and Escape
 * cancels it: neither is a submit, an accept or a dismiss, so a handler
 * acting on either in a text-entry context returns early when this is true.
 * `isComposing` is the standard flag
 * (https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/isComposing);
 * Safari reports the keydown that ends a composition with `isComposing`
 * false but `keyCode` 229, the value every browser gives a key the IME
 * consumed (https://developer.mozilla.org/en-US/docs/Web/API/Element/keydown_event).
 *
 * An Escape handler that also keeps the key from an outer handler (a
 * Vuetify overlay closes on a window Escape listener) still stops it while
 * composing: the IME's Escape must close nothing.
 *
 * @param {KeyboardEvent} event
 * @returns {boolean}
 */
export function isImeComposing(event) {
  return event.isComposing === true || event.keyCode === 229;
}
