# SPDX-License-Identifier: Apache-2.0
"""The keyboard on the page: which keys are held, sent to Python as they change.

viser has no keyboard events, and its own camera flies on W A S D. So the page
gets the keys the way any web page does: a few lines of JavaScript, carried in
an HTML element on the panel, listen on the window before viser does, keep the
keys they claim from reaching its camera, and write the keys now held into a
text field of the panel, which viser sends to Python like any edit. The field
is hidden; clicking into another field of the panel types there as usual.
"""

from __future__ import annotations

from collections.abc import Callable

KEYS = {"KeyW": "W", "KeyA": "A", "KeyS": "S", "KeyD": "D", "Space": "_", "KeyQ": "Q", "KeyE": "E"}
"""The ``event.code``s claimed from the page, and the letter each is sent as."""

_TAG = "keys:"

_SCRIPT = """
const CODES = %(codes)s, TAG = "%(tag)s";
const held = new Set(), presses = {};
const typing = (t) => t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
const field = () => [...document.querySelectorAll("input")].find((e) => e.value.startsWith(TAG));
function hide(el) {
  // the field's whole row: up from the input to the box that holds nothing else
  let row = el;
  while (row.parentElement && row.parentElement.querySelectorAll("input").length === 1
         && !row.parentElement.querySelector("img[data-keys]")) row = row.parentElement;
  row.style.display = "none";
}
function send() {
  const el = field();
  if (!el) return;
  hide(el);
  // what is held, and how often each key has been pressed: a tap shorter than viser's 50 ms
  // send throttle is folded into the next message, but its count still goes up
  const counts = Object.entries(presses).map(([k, n]) => k + n).join(",");
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
  setter.call(el, TAG + [...held].sort().join("") + ":" + counts + ":" + Date.now());
  el.dispatchEvent(new Event("input", { bubbles: true }));
}
function key(event, down) {
  const k = CODES[event.code];
  if (!k || typing(event.target) || event.ctrlKey || event.metaKey || event.altKey) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  if (down ? held.has(k) : !held.has(k)) return;
  if (down) {
    held.add(k);
    presses[k] = (presses[k] || 0) + 1;
  } else held.delete(k);
  send();
}
window.addEventListener("keydown", (e) => key(e, true), true);
window.addEventListener("keyup", (e) => key(e, false), true);
window.addEventListener("blur", () => { if (held.size) { held.clear(); send(); } });
setTimeout(() => { const el = field(); if (el) hide(el); }, 500);
"""


def keyboard(gui, on_keys: Callable[[frozenset[str], frozenset[str]], None], *, legend: str = "") -> None:
    """Call ``on_keys(held, pressed)`` whenever the keys change.

    ``held`` is the letters of :data:`KEYS` held now (Space is ``"_"``), and ``pressed``
    the ones pressed since the last call, a tap included. ``legend`` is HTML shown on
    the panel.
    """
    import json

    script = _SCRIPT % {"codes": json.dumps(KEYS), "tag": _TAG}
    gui.add_html(
        f'<img data-keys src="data:," style="display:none" onerror="if (!window.__duckKeys) {{ window.__duckKeys = 1;'
        f' eval(this.nextElementSibling.textContent); }}"><script type="text/x-keys">{script}</script>'
        + (f"<div>{legend}</div>" if legend else "")
    )
    channel = gui.add_text("keys", initial_value=_TAG)
    seen: dict[str, int] = {}

    @channel.on_update
    def _(_event) -> None:
        value = channel.value
        if not value.startswith(_TAG):
            return
        letters, counts, _sent = (*value[len(_TAG) :].split(":"), "", "")[:3]
        pressed = set()
        for item in filter(None, counts.split(",")):
            k, n = item[0], int(item[1:])
            if n != seen.get(k, 0):  # more presses, or fewer: a page reloaded and counts again
                pressed.add(k)
            seen[k] = n
        on_keys(frozenset(letters), frozenset(pressed))
