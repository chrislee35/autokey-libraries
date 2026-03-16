"""
screen_grep.py — Locate image clips on the live screen using template matching.

ScreenGrep wraps a single ClipLibrary collection and provides methods to find
clip images on the current screen, optionally offsetting the result to the
clip's stored target point.

Template matching uses OpenCV's TM_CCOEFF_NORMED algorithm on grayscale images
(3× less data than colour while retaining accuracy for UI elements).  All
multi-item methods share a single screenshot grab to avoid redundant I/O.

Typical usage::

    from clip_library import ClipLibrary
    from screen_grep import ScreenGrep, PatternNotFound

    lib = ClipLibrary(Path("~/.config/clip_library").expanduser())
    sg  = ScreenGrep(lib, "buttons")

    x, y = sg.find_target("ok_button")   # location of the target point
    if sg.is_present("spinner"):
        ...
"""

from __future__ import annotations

import numpy as np
import cv2
from PIL import ImageGrab, Image
from screeninfo import get_monitors

from .clip_library import ClipLibrary

# Normalised cross-correlation threshold in [0, 1].  A score ≥ this value is
# treated as a match.  Raise to reduce false positives; lower to tolerate
# slight rendering differences.
_MATCH_THRESHOLD = 0.85


class PatternNotFound(Exception):
    """Raised by find_item / find_target when the clip is not on screen."""


class ScreenGrep:
    """Locate clips from one ClipLibrary collection on the live screen.

    Args:
        library:    Populated ClipLibrary instance.
        collection: Name of the collection to search within.

    Raises:
        ValueError: If ``collection`` is not loaded in ``library``.
    """

    def __init__(self, collection: str, library: ClipLibrary | None = None):
        if library is None:
            library = ClipLibrary()
        if collection not in library.collections:
            raise ValueError(f"Collection '{collection}' is not loaded in library")
        self._library = library
        self._collection = collection
        self.last_screenshot: Image | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _grab_screen_gray(self) -> np.ndarray:
        """Capture the primary monitor and return a grayscale numpy array."""
        monitor = get_monitors()[0]
        screenshot = ImageGrab.grab().crop((0, 0, monitor.width, monitor.height))
        self.last_screenshot = screenshot
        return _pil_to_gray(screenshot)

    def _clip_gray(self, item_name: str) -> np.ndarray:
        """Return the grayscale numpy array for a clip, using the cache."""
        img = self._library.get_item(self._collection, item_name)
        return _pil_to_gray(img)

    def _match(self, screen_gray: np.ndarray, template_gray: np.ndarray) -> tuple[int, int] | None:
        """Run template matching on pre-converted grayscale arrays.

        Returns:
            ``(x, y)`` top-left pixel of the best match, or ``None`` if the
            best correlation score is below ``_MATCH_THRESHOLD``.
        """
        result = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= _MATCH_THRESHOLD:
            return (int(max_loc[0]), int(max_loc[1]))
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_item(self, item_name: str) -> tuple[int, int]:
        """Take a screenshot and return the top-left corner of ``item_name``.

        Args:
            item_name: Name of the clip to locate.

        Returns:
            ``(x, y)`` screen coordinates of the clip's top-left corner.

        Raises:
            PatternNotFound: If the clip is not found on the current screen.
            ValueError:      If ``item_name`` is not in the collection.
        """
        screen_gray = self._grab_screen_gray()
        template_gray = self._clip_gray(item_name)
        loc = self._match(screen_gray, template_gray)
        if loc is None:
            raise PatternNotFound(f"'{item_name}' not found on screen")
        return loc

    def find_target(self, item_name: str) -> tuple[int, int]:
        """Return the screen position of the target point for ``item_name``.

        Calls ``find_item`` then adds the clip's stored target offset.  If no
        offset has been saved the clip's top-left corner is returned as-is.

        Args:
            item_name: Name of the clip whose target point to locate.

        Returns:
            ``(x, y)`` screen coordinates of the target point.

        Raises:
            PatternNotFound: If the clip is not found on the current screen.
            ValueError:      If ``item_name`` is not in the collection.
        """
        x, y = self.find_item(item_name)
        offset = self._library.get_target_offset(self._collection, item_name)
        if offset is not None:
            x += offset[0]
            y += offset[1]
        return x, y

    def find_all_items(self) -> dict[str, tuple[int, int]]:
        """Find every clip in the collection on the current screen.

        Takes a **single screenshot** and matches all clips against it, making
        this significantly more efficient than calling ``find_item`` in a loop.

        Returns:
            Dict mapping each found clip name to its ``(x, y)`` top-left
            screen coordinates.  Clips not present on screen are omitted.
            Returns an empty dict if nothing is found.
        """
        screen_gray = self._grab_screen_gray()
        results: dict[str, tuple[int, int]] = {}
        for item_name in self._library.collections[self._collection].items:
            template_gray = self._clip_gray(item_name)
            loc = self._match(screen_gray, template_gray)
            if loc is not None:
                results[item_name] = loc
        return results

    def find_all_targets(self) -> dict[str, tuple[int, int]]:
        """Find every clip in the collection and return their target positions.

        Like ``find_all_items`` but offsets each result by the clip's stored
        target offset.  Uses a single screenshot grab for all clips.

        Returns:
            Dict mapping each found clip name to its target ``(x, y)`` screen
            coordinates.  Clips not present on screen are omitted.
        """
        items = self.find_all_items()
        results: dict[str, tuple[int, int]] = {}
        for item_name, (x, y) in items.items():
            offset = self._library.get_target_offset(self._collection, item_name)
            if offset is not None:
                results[item_name] = (x + offset[0], y + offset[1])
            else:
                results[item_name] = (x, y)
        return results

    def is_present(self, item_name: str) -> bool:
        """Return ``True`` if ``item_name`` is visible on the current screen.

        Optimisations vs. calling ``find_item`` and ignoring the location:

        * ``np.any`` short-circuits as soon as a cell exceeds the threshold,
          avoiding the full ``minMaxLoc`` scan over the correlation matrix.
        * The grayscale arrays are already cached by ``get_item``, so no disk
          I/O occurs on repeated calls for the same clip.

        Note:
            When checking several clips at once, prefer ``find_all_items()``
            which shares a single screenshot across all matches.

        Args:
            item_name: Name of the clip to check.

        Returns:
            ``True`` if a match above ``_MATCH_THRESHOLD`` exists, else
            ``False``.

        Raises:
            ValueError: If ``item_name`` is not in the collection.
        """
        screen_gray = self._grab_screen_gray()
        template_gray = self._clip_gray(item_name)
        result = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        # np.any with a boolean mask short-circuits — no need to scan for the
        # maximum location when all we need is existence.
        return bool(np.any(result >= _MATCH_THRESHOLD))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_gray(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an 8-bit grayscale numpy array for OpenCV."""
    return np.array(img.convert("L"))

if __name__ == "__main__":
    sg = ScreenGrep('test')
    print(sg.find_all_targets())