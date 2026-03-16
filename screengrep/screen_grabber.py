"""
screen_grabber.py — GTK GUI for capturing image clips and target points.

Workflow
--------
1. A scaled screenshot of the primary monitor fills the window.
2. The user drags a rectangle to define the **clip region**.
3. The user clicks once inside (or near) the region to set the **target point**.
   The stored offset is relative to the clip's upper-left corner, so it can be
   negative or larger than the clip dimensions.
4. A dialog prompts for a collection name (defaults to the last used one) and a
   clip name.
5. The clip image and target offset are saved via ClipLibrary.

Usage
-----
    python screen_grabber.py
"""

from __future__ import annotations

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GdkPixbuf

from PIL import Image, ImageGrab
from screeninfo import get_monitors

from .clip_library import ClipLibrary


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class ScreenGrabberWindow(Gtk.Window):
    """Full-screen-ish window showing a scaled screenshot for clip selection.

    State machine
    ~~~~~~~~~~~~~
    IDLE        → user has not started dragging yet
    DRAGGING    → mouse button held, rubber-band rect being drawn
    AWAITING_TARGET → clip rect confirmed, waiting for a single click
    DONE        → target set; dialog shown automatically
    """

    _STATE_IDLE = "idle"
    _STATE_DRAGGING = "dragging"
    _STATE_AWAITING_TARGET = "awaiting_target"
    _STATE_DONE = "done"

    def __init__(self, library: ClipLibrary, last_collection: str = ""):
        super().__init__(title="Screen Grabber — drag clip, then click target")
        self._library = library
        self._last_collection = last_collection

        # Grab screenshot before the window appears
        monitor = get_monitors()[0]
        self._monitor_w = monitor.width
        self._monitor_h = monitor.height
        self._screenshot: Image.Image = ImageGrab.grab().crop( (0, 0, monitor.width, monitor.height))

        # Selection state
        self._state = self._STATE_IDLE
        self._drag_start: tuple[float, float] | None = None   # canvas coords
        self._clip_rect: tuple[int, int, int, int] | None = None  # canvas coords (x1,y1,x2,y2)
        self._target_canvas: tuple[float, float] | None = None    # canvas coords

        # Build UI
        self._drawing_area = Gtk.DrawingArea()
        self._drawing_area.set_events(
            Gdk.EventMask.BUTTON_PRESS_MASK
            | Gdk.EventMask.BUTTON_RELEASE_MASK
            | Gdk.EventMask.POINTER_MOTION_MASK
        )
        self._drawing_area.connect("draw", self._on_draw)
        self._drawing_area.connect("button-press-event", self._on_button_press)
        self._drawing_area.connect("button-release-event", self._on_button_release)
        self._drawing_area.connect("motion-notify-event", self._on_motion)

        self.add(self._drawing_area)
        self.connect("destroy", Gtk.main_quit)
        self.connect("key-press-event", self._on_key_press)

        # Scale screenshot to fit inside a reasonable window (max 90 % of monitor)
        max_w = int(self._monitor_w * 0.90)
        max_h = int(self._monitor_h * 0.90)
        scale = min(max_w / self._monitor_w, max_h / self._monitor_h)
        self._scale = scale
        self._canvas_w = int(self._monitor_w * scale)
        self._canvas_h = int(self._monitor_h * scale)

        self._pixbuf = self._pil_to_pixbuf(
            self._screenshot.resize((self._canvas_w, self._canvas_h), Image.LANCZOS)
        )

        self._drawing_area.set_size_request(self._canvas_w, self._canvas_h)
        self.set_resizable(False)
        self.show_all()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _canvas_to_screen(self, cx: float, cy: float) -> tuple[int, int]:
        """Convert canvas (scaled) coordinates to original screenshot pixels."""
        return (int(cx / self._scale), int(cy / self._scale))

    def _norm_rect(self, x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
        """Return a rectangle with x1≤x2 and y1≤y2, clamped to canvas bounds."""
        lx = max(0, int(min(x1, x2)))
        ly = max(0, int(min(y1, y2)))
        rx = min(self._canvas_w, int(max(x1, x2)))
        ry = min(self._canvas_h, int(max(y1, y2)))
        return lx, ly, rx, ry

    # ------------------------------------------------------------------
    # PIL ↔ GdkPixbuf
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_pixbuf(img: Image.Image) -> GdkPixbuf.Pixbuf:
        img_rgb = img.convert("RGB")
        data = img_rgb.tobytes()
        w, h = img_rgb.size
        return GdkPixbuf.Pixbuf.new_from_data(
            data, GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3
        )

    # ------------------------------------------------------------------
    # GTK event handlers
    # ------------------------------------------------------------------

    def _on_key_press(self, _widget, event):
        if event.keyval == Gdk.KEY_Escape:
            Gtk.main_quit()
        elif event.keyval == Gdk.KEY_r:
            # Reset — let user re-draw the clip
            self._state = self._STATE_IDLE
            self._clip_rect = None
            self._target_canvas = None
            self._drag_start = None
            self._drawing_area.queue_draw()
            self.set_title("Screen Grabber — drag clip, then click target")

    def _on_button_press(self, _widget, event):
        if event.button != 1:
            return

        if self._state == self._STATE_IDLE:
            self._drag_start = (event.x, event.y)
            self._state = self._STATE_DRAGGING

        elif self._state == self._STATE_AWAITING_TARGET:
            self._target_canvas = (event.x, event.y)
            self._state = self._STATE_DONE
            self._drawing_area.queue_draw()
            # Show save dialog on the next idle iteration so the draw completes
            GLib_idle_add_once(self._show_save_dialog)

    def _on_button_release(self, _widget, event):
        if event.button != 1 or self._state != self._STATE_DRAGGING:
            return
        assert self._drag_start is not None
        x1, y1 = self._drag_start
        x2, y2 = event.x, event.y
        self._clip_rect = self._norm_rect(x1, y1, x2, y2)
        self._drag_start = None

        # Require a non-trivial rectangle
        lx, ly, rx, ry = self._clip_rect
        if rx - lx < 2 or ry - ly < 2:
            self._clip_rect = None
            self._state = self._STATE_IDLE
            self._drawing_area.queue_draw()
            return

        self._state = self._STATE_AWAITING_TARGET
        self._drawing_area.queue_draw()
        self.set_title("Clip captured — click the target point")

    def _on_motion(self, _widget, event):
        if self._state == self._STATE_DRAGGING:
            self._drag_end = (event.x, event.y)
            self._drawing_area.queue_draw()

    def _on_draw(self, _widget, cr):
        # Draw scaled screenshot
        Gdk.cairo_set_source_pixbuf(cr, self._pixbuf, 0, 0)
        cr.paint()

        if self._state == self._STATE_DRAGGING and self._drag_start:
            x1, y1 = self._drag_start
            x2, y2 = getattr(self, "_drag_end", (x1, y1))
            lx, ly, rx, ry = self._norm_rect(x1, y1, x2, y2)
            _draw_selection_rect(cr, lx, ly, rx - lx, ry - ly)

        if self._clip_rect:
            lx, ly, rx, ry = self._clip_rect
            _draw_selection_rect(cr, lx, ly, rx - lx, ry - ly, confirmed=True)

        if self._target_canvas:
            tx, ty = self._target_canvas
            _draw_crosshair(cr, tx, ty)

    # ------------------------------------------------------------------
    # Save dialog
    # ------------------------------------------------------------------

    def _show_save_dialog(self):
        dialog = _SaveDialog(self, self._last_collection, list(self._library.collections.keys()))
        response = dialog.run()
        collection_name = dialog.collection_name
        clip_name = dialog.clip_name
        dialog.destroy()

        if response != Gtk.ResponseType.OK or not clip_name:
            # User cancelled — go back to awaiting target so they can retry
            self._state = self._STATE_AWAITING_TARGET
            self._target_canvas = None
            self.set_title("Clip captured — click the target point")
            return

        self._last_collection = collection_name
        self._save_clip(collection_name, clip_name)

    def _save_clip(self, collection_name: str, clip_name: str):
        """Crop the screenshot, save as a clip, and store the target offset."""
        assert self._clip_rect is not None
        lx, ly, rx, ry = self._clip_rect
        # Convert canvas rect to original screenshot coordinates
        sx1, sy1 = self._canvas_to_screen(lx, ly)
        sx2, sy2 = self._canvas_to_screen(rx, ry)
        clip_image = self._screenshot.crop((sx1, sy1, sx2, sy2))

        # Ensure collection exists
        if collection_name not in self._library.collections:
            self._library.create_collection(collection_name)

        try:
            self._library.add_item(collection_name, clip_name, clip_image)
        except ValueError as exc:
            _show_error(self, str(exc))
            self._state = self._STATE_AWAITING_TARGET
            self._target_canvas = None
            self.set_title("Clip captured — click the target point")
            return

        # Compute target offset relative to clip upper-left in screen pixels
        assert self._target_canvas is not None
        tx_canvas, ty_canvas = self._target_canvas
        tx_screen, ty_screen = self._canvas_to_screen(tx_canvas, ty_canvas)
        offset = (tx_screen - sx1, ty_screen - sy1)
        self._library.set_target_offset(collection_name, clip_name, offset)

        _show_info(
            self,
            f"Saved clip '{clip_name}' in collection '{collection_name}'.\n"
            f"Target offset: {offset[0]}px, {offset[1]}px"
        )

        # Reset for another capture
        self._state = self._STATE_IDLE
        self._clip_rect = None
        self._target_canvas = None
        self._drag_start = None
        self._drawing_area.queue_draw()
        self.set_title("Screen Grabber — drag clip, then click target")


# ---------------------------------------------------------------------------
# Save dialog
# ---------------------------------------------------------------------------

class _SaveDialog(Gtk.Dialog):
    """Prompts for collection name (with completion) and clip name."""

    def __init__(self, parent: Gtk.Window, last_collection: str, known_collections: list[str]):
        super().__init__(title="Save Clip", transient_for=parent, modal=True)
        self.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_SAVE, Gtk.ResponseType.OK,
        )
        self.set_default_response(Gtk.ResponseType.OK)

        grid = Gtk.Grid(row_spacing=8, column_spacing=8, margin=12)
        self.get_content_area().add(grid)

        # Collection entry with completion
        completion = Gtk.EntryCompletion()
        store = Gtk.ListStore(str)
        for name in known_collections:
            store.append([name])
        completion.set_model(store)
        completion.set_text_column(0)

        grid.attach(Gtk.Label(label="Collection:", xalign=0), 0, 0, 1, 1)
        self._collection_entry = Gtk.Entry()
        self._collection_entry.set_completion(completion)
        self._collection_entry.set_text(last_collection)
        self._collection_entry.set_activates_default(True)
        grid.attach(self._collection_entry, 1, 0, 1, 1)

        # Clip name entry
        grid.attach(Gtk.Label(label="Clip name:", xalign=0), 0, 1, 1, 1)
        self._clip_entry = Gtk.Entry()
        self._clip_entry.set_activates_default(True)
        grid.attach(self._clip_entry, 1, 1, 1, 1)

        self.show_all()

    @property
    def collection_name(self) -> str:
        return self._collection_entry.get_text().strip()

    @property
    def clip_name(self) -> str:
        return self._clip_entry.get_text().strip()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_selection_rect(cr, x: int, y: int, w: int, h: int, confirmed: bool = False):
    """Draw a rubber-band rectangle on a Cairo context."""
    if confirmed:
        cr.set_source_rgba(0.2, 0.8, 0.2, 0.5)   # green fill
    else:
        cr.set_source_rgba(0.2, 0.5, 1.0, 0.25)   # blue fill

    cr.rectangle(x, y, w, h)
    cr.fill_preserve()

    if confirmed:
        cr.set_source_rgba(0.1, 0.9, 0.1, 1.0)
    else:
        cr.set_source_rgba(0.2, 0.5, 1.0, 1.0)

    cr.set_line_width(2)
    cr.stroke()


def _draw_crosshair(cr, x: float, y: float, size: int = 12):
    """Draw a red crosshair at (x, y) on a Cairo context."""
    cr.set_source_rgba(1.0, 0.1, 0.1, 1.0)
    cr.set_line_width(2)
    cr.move_to(x - size, y)
    cr.line_to(x + size, y)
    cr.stroke()
    cr.move_to(x, y - size)
    cr.line_to(x, y + size)
    cr.stroke()
    cr.arc(x, y, 4, 0, 2 * 3.14159)
    cr.fill()


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def _show_error(parent: Gtk.Window, message: str):
    dlg = Gtk.MessageDialog(
        transient_for=parent, modal=True,
        message_type=Gtk.MessageType.ERROR,
        buttons=Gtk.ButtonsType.OK,
        text=message,
    )
    dlg.run()
    dlg.destroy()


def _show_info(parent: Gtk.Window, message: str):
    dlg = Gtk.MessageDialog(
        transient_for=parent, modal=True,
        message_type=Gtk.MessageType.INFO,
        buttons=Gtk.ButtonsType.OK,
        text=message,
    )
    dlg.run()
    dlg.destroy()


# ---------------------------------------------------------------------------
# GLib idle shim (avoids a top-level gi.repository.GLib import just for this)
# ---------------------------------------------------------------------------

def GLib_idle_add_once(fn):
    """Schedule ``fn`` to run once on the next GLib main-loop idle iteration."""
    from gi.repository import GLib
    def _wrapper():
        fn()
        return False  # don't repeat
    GLib.idle_add(_wrapper)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    library = ClipLibrary()

    win = ScreenGrabberWindow(library)
    win.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
