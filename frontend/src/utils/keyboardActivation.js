/**
 * Give a non-native composite row the same Enter/Space activation contract as
 * a button without stealing key presses from real controls nested inside it.
 *
 * Composite rows are used where the row contains its own edit/delete buttons,
 * so changing the outer element to <button> would create invalid nested
 * buttons. Callers still need role="button" and tabindex="0" in their markup.
 */
export function activateOnEnterOrSpace(event) {
  if (event?.target !== event?.currentTarget) return false;
  if (event.key !== "Enter" && event.key !== " ") return false;

  event.preventDefault();

  const target = event.currentTarget;
  const MouseEventCtor = event.view?.MouseEvent ?? globalThis.MouseEvent;
  if (typeof MouseEventCtor === "function") {
    target.dispatchEvent(
      new MouseEventCtor("click", {
        bubbles: true,
        cancelable: true,
        ctrlKey: event.ctrlKey,
        metaKey: event.metaKey,
        shiftKey: event.shiftKey,
      }),
    );
  } else {
    target.click();
  }
  return true;
}

/**
 * Whether a key event belongs to an input method's composition (Japanese,
 * Chinese, Korean). While one is open, Enter commits the candidate and Escape
 * cancels it: neither is a submit, an accept or a dismiss, so a text-field or
 * dialog handler returns early on it and does not prevent it. A handler that
 * stops Escape still stops this one, since a Vuetify overlay closes on a window
 * Escape listener. Safari sends the keydown that ends a composition with
 * `isComposing` false and `keyCode` 229.
 */
export function isImeComposing(event) {
  return event.isComposing === true || event.keyCode === 229;
}
