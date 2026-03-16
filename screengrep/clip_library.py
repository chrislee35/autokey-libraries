"""
clip_library.py — Persistent image clip collection manager.

A "clip" is a named PNG image stored on disk. Clips are organized into
named collections (one directory per collection under a shared root).
ClipLibrary provides create/read/delete operations for both collections
and individual clips, plus an in-memory LRU-style cache so frequently
used images avoid repeated disk I/O.

Each clip may have an associated *target offset* — an (x, y) pixel
coordinate measured relative to the clip's upper-left corner — stored in
a per-collection ``targets.json`` file.  The target can fall outside the
clip's bounding box (e.g. a click point near but not inside the matched
region).

Typical layout on disk:
    collections_root/
        buttons/
            targets.json        ← {"ok": [12, 8], "cancel": [6, 8]}
            ok.png
            cancel.png
        icons/
            targets.json
            warning.png
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Callable
from dataclasses import dataclass, field
from PIL import Image


def default_library_root() -> Path:
    """Return the platform-appropriate default root for the clip library.

    * **Linux / other**: ``$XDG_CONFIG_HOME/clip_library`` (falls back to
      ``~/.config/clip_library``)
    * **Windows**: ``%APPDATA%/clip_library``
    * **macOS**: ``~/Library/Application Support/clip_library``
    """
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("APPDATA", str(Path.home())))
    elif system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return base / "clip_library"

_TARGETS_FILE = "targets.json"


@dataclass(slots=True, frozen=True)
class ClipCollection:
    """A single named collection of image clips.

    Attributes:
        items:          Maps clip name → path on disk for every clip in this
                        collection.
        loaded:         Cache of already-opened PIL Images, keyed by clip name.
                        Populated lazily by ClipLibrary.get_item() and eagerly
                        by ClipLibrary.cache_collection().  Cleared by
                        clear_cache().
        target_offsets: Maps clip name → (x, y) pixel offset of the target
                        point relative to the clip's upper-left corner.
                        Loaded from ``targets.json`` on collection load and
                        kept in sync by set_target_offset / remove_item.

    Note:
        The dataclass is frozen so that the object itself cannot be replaced,
        but the dict fields are ordinary dicts and are mutated in place by
        ClipLibrary methods.
    """

    items: dict[str, Path]
    loaded: dict[str, Image.Image] = field(default_factory=dict)
    target_offsets: dict[str, tuple[int, int]] = field(default_factory=dict)


class ClipLibrary:
    """Manages a tree of image clip collections rooted at a single directory.

    On construction the library walks ``collections_root`` and calls
    ``load_collection`` for every immediate sub-directory it finds.
    Sub-directories that contain no PNGs are loaded as empty collections.

    Args:
        collections_root: Path to the directory that contains collection
                          sub-directories.  Created automatically if it does
                          not exist.  Defaults to ``default_library_root()``
                          when ``None``.
    """

    def __init__(self, collections_root: Path | None = None):
        self.collections: dict[str, ClipCollection] = {}
        self.root = collections_root if collections_root is not None else default_library_root()
        self.root.mkdir(parents=True, exist_ok=True)
        for root, dirs, files in self.root.walk():
            for dir in dirs:
                self.load_collection(root.joinpath(dir))

    def load_collection(self, directory: Path):
        """Read a collection directory from disk and register it in memory.

        Scans ``directory`` for ``*.png`` files and builds an ``items`` dict
        mapping each stem name (filename without ``.png``) to its full path.
        Also reads ``targets.json`` if present.
        The resulting ClipCollection is stored under the directory's bare name.

        Args:
            directory: Absolute path to the collection directory.

        Raises:
            FileNotFoundError: If ``directory`` does not exist.
            ValueError: If ``directory`` is not a directory.
        """
        if not directory.exists():
            raise FileNotFoundError(directory)
        if not directory.is_dir():
            raise ValueError("directory argument must be a directory")
        items: dict[str, Path] = {}
        for png in directory.glob("*.png"):
            name = png.name.removesuffix(".png")
            items[name] = png

        target_offsets: dict[str, tuple[int, int]] = {}
        targets_path = directory / _TARGETS_FILE
        if targets_path.exists():
            raw: dict[str, list[int]] = json.loads(targets_path.read_text())
            target_offsets = {k: (v[0], v[1]) for k, v in raw.items()}

        self.collections[directory.name] = ClipCollection(items, target_offsets=target_offsets)

    def create_collection(self, name: str):
        """Create a new, empty collection on disk and register it in memory.

        Args:
            name: Name for the new collection.  A sub-directory with this name
                  is created under ``self.root``.

        Raises:
            ValueError: If a collection with ``name`` already exists on disk.
        """
        directory: Path = self.root.joinpath(name)
        if directory.exists():
            raise ValueError(f"Collection {name} already exists")
        directory.mkdir(parents=True)
        self.collections[directory.name] = ClipCollection({})

    def add_item(self, collection: str, item_name: str, image: Image.Image):
        """Save a PIL Image as a new clip in an existing collection.

        The image is written to ``<collection>/<item_name>.png`` and also
        stored in the in-memory cache so subsequent ``get_item`` calls do not
        need a disk read.

        Args:
            collection: Name of the target collection (must already be loaded).
            item_name:  Stem name for the new clip (no ``.png`` extension).
            image:      PIL Image to save.

        Raises:
            ValueError: If the collection is not loaded, or if a clip with
                        ``item_name`` already exists in that collection.
        """
        if collection not in self.collections:
            raise ValueError(f"collection {collection} is not loaded")
        directory: Path = self.root.joinpath(collection)
        image_path: Path = directory.joinpath(item_name+".png")
        if image_path.exists():
            raise ValueError(f"item, {item_name} already exists")
        image.save(str(image_path))
        self.collections[collection].items[item_name] = image_path
        self.collections[collection].loaded[item_name] = image

    def get_item(self, collection: str, item_name: str) -> Image.Image:
        """Return the PIL Image for a clip, loading from disk if not cached.

        Args:
            collection: Name of the collection to look in.
            item_name:  Stem name of the clip to retrieve.

        Returns:
            The PIL Image for the requested clip.

        Raises:
            ValueError: If the collection is not loaded, the clip name is not
                        registered, or the PNG file is missing from disk.
        """
        if collection not in self.collections:
            raise ValueError(f"collection {collection} is not loaded")
        if item_name not in self.collections[collection].items:
            raise ValueError(f"item {item_name} not found in collection")

        if item_name not in self.collections[collection].loaded:
            directory: Path = self.root.joinpath(collection)
            image_path: Path = directory.joinpath(item_name+".png")
            if not image_path.exists():
                raise ValueError(f"item image file for {item_name} is missing.")
            self.collections[collection].loaded[item_name] = Image.open(str(image_path))

        return self.collections[collection].loaded[item_name]

    def set_target_offset(self, collection: str, item_name: str, offset: tuple[int, int]):
        """Record the target offset for a clip and persist it to ``targets.json``.

        The offset is measured in pixels from the clip image's upper-left
        corner and may be negative or larger than the clip's dimensions if
        the target lies outside the clip region.

        Args:
            collection: Name of the collection that owns the clip.
            item_name:  Stem name of the clip.
            offset:     ``(x, y)`` target offset in pixels.

        Raises:
            ValueError: If the collection or clip is not loaded.
        """
        if collection not in self.collections:
            raise ValueError(f"collection {collection} is not loaded")
        if item_name not in self.collections[collection].items:
            raise ValueError(f"item {item_name} not found in collection")
        self.collections[collection].target_offsets[item_name] = offset
        self._write_targets(collection)

    def get_target_offset(self, collection: str, item_name: str) -> tuple[int, int] | None:
        """Return the target offset for a clip, or ``None`` if none is set.

        Args:
            collection: Name of the collection that owns the clip.
            item_name:  Stem name of the clip.

        Returns:
            ``(x, y)`` offset tuple, or ``None`` if no target has been saved.

        Raises:
            ValueError: If the collection or clip is not loaded.
        """
        if collection not in self.collections:
            raise ValueError(f"collection {collection} is not loaded")
        if item_name not in self.collections[collection].items:
            raise ValueError(f"item {item_name} not found in collection")
        return self.collections[collection].target_offsets.get(item_name)

    def _write_targets(self, collection: str):
        """Serialise the in-memory target_offsets dict to ``targets.json``.

        Args:
            collection: Name of the collection to flush.
        """
        directory: Path = self.root.joinpath(collection)
        targets_path = directory / _TARGETS_FILE
        data = {k: list(v) for k, v in self.collections[collection].target_offsets.items()}
        targets_path.write_text(json.dumps(data, indent=2))

    def remove_collection(self, collection: str, confirmation_callback: Callable[[str], bool]):
        """Delete all clips and the collection directory from disk, then unregister it.

        Args:
            collection:             Name of the collection to remove.
            confirmation_callback:  Callable that receives the collection name
                                    and returns ``True`` to proceed.
                                    (Currently accepted but not called — reserved
                                    for future interactive confirmation prompts.)

        Raises:
            ValueError: If the collection is not loaded.
        """
        if collection not in self.collections:
            raise ValueError(f"collection {collection} is not loaded")
        directory: Path = self.root.joinpath(collection)
        for png in directory.glob("*.png"):
            png.unlink()
        targets_path = directory / _TARGETS_FILE
        if targets_path.exists():
            targets_path.unlink()
        directory.unlink()
        self.collections.pop(collection)

    def remove_item(self, collection: str, item_name: str):
        """Delete a single clip from disk and unregister it from the collection.

        Also evicts the clip from the in-memory cache and removes its target
        offset entry (updating ``targets.json`` on disk if needed).

        Args:
            collection: Name of the collection containing the clip.
            item_name:  Stem name of the clip to remove.

        Raises:
            ValueError: If the collection is not loaded or the clip is not found.
        """
        if collection not in self.collections:
            raise ValueError(f"collection {collection} is not loaded")
        if item_name not in self.collections[collection].items:
            raise ValueError(f"item {item_name} not found in collection")
        if item_name in self.collections[collection].loaded:
            self.collections[collection].loaded.pop(item_name)
        if item_name in self.collections[collection].target_offsets:
            self.collections[collection].target_offsets.pop(item_name)
            self._write_targets(collection)
        self.collections[collection].items.pop(item_name)
        directory: Path = self.root.joinpath(collection)
        image_path: Path = directory.joinpath(item_name+".png")
        image_path.unlink()

    def clear_cache(self, collection: str):
        """Evict all cached images for a single collection.

        Frees memory held by loaded PIL Images; they will be reloaded from
        disk on the next ``get_item`` call.

        Args:
            collection: Name of the collection whose cache should be cleared.
        """
        self.collections[collection].loaded.clear()

    def clear_all_caches(self):
        """Evict all cached images across every loaded collection."""
        for collection in self.collections.values():
            collection.loaded.clear()

    def cache_everything(self):
        """Pre-load every clip in every collection into memory.

        Useful before a batch of lookups where disk latency must be avoided.
        """
        for collection in self.collections.keys():
            self.cache_collection(collection)

    def cache_collection(self, collection: str):
        """Pre-load every clip in a single collection into memory.

        Args:
            collection: Name of the collection to warm.
        """
        for item in self.collections[collection].items.keys():
            self.get_item(collection, item)
