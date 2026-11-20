"""
write_guard.py
==============
The only code that writes a file. One identical copy of this file sits in
each code folder (the analysis scripts and the table and figure scripts).

The rule it enforces
--------------------
Running the scripts never alters, overwrites, moves or deletes an existing
file. Every output is a new file under one output root, OUTPUT_ROOT_DIR,
inside the data folder the scripts run from.

- The root is created by create_root with an exclusive mkdir and a marker
  file, ROOT_MARKER, made with an exclusive create. An existing folder is
  never adopted: if the root already exists, create_root stops. To run the
  scripts again, rename or move the earlier root yourself first.
- Every later write (write_new) needs the root to exist, to be a real folder
  (not a symlink) and to hold the marker.
- write_new(path, fn) refuses any path that already exists (a broken link
  included) or that resolves, symlinks followed, outside the root. It makes
  the missing parent folders inside the root and refuses a symlinked folder
  on the way. fn(tmp) writes to a temporary file that this module reserved
  with an exclusive create, in the target's folder and with the target's
  extension. The file is then published with os.link, which the operating
  system refuses if the target exists, so no path can be replaced even by a
  race. The temporary file is the only thing this module ever deletes.
- If the file system cannot make hard links, the write stops; it never falls
  back to a rename or replace.

OUTPUT_ROOT_DIR is blank here; the scripts refuse to write until it is set.
"""

import os

OUTPUT_ROOT_DIR = ""  # folder (relative to the data folder the scripts run from) that holds every file the scripts write; created by create_root, never adopted
ROOT_MARKER = ".output_root"
_TMP_COUNT = [0]


class WriteRefused(RuntimeError):
    """A write that would touch an existing file or leave the output root."""


def _root():
    if not OUTPUT_ROOT_DIR:
        raise WriteRefused("OUTPUT_ROOT_DIR is blank in write_guard.py; set the output root "
                           "before running.")
    return os.path.abspath(OUTPUT_ROOT_DIR)


def _inside(root_real, p_real):
    return os.path.commonpath([root_real, p_real]) == root_real


def create_root():
    """Create the output root and its marker (once, before the first write).
    Stops if anything exists at the root's path."""
    root = _root()
    if os.path.lexists(root):
        raise WriteRefused(
            f"{root} already exists. These scripts never write into an existing folder; "
            f"rename or move it yourself (it may hold an earlier run), then run again.")
    os.mkdir(root)
    fd = os.open(os.path.join(root, ROOT_MARKER), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, b"Output root of these scripts (write_guard.py). Every file here was "
                     b"written new; none replaced another.\n")
    finally:
        os.close(fd)
    print(f"  Output root created: {root}")
    return root


def require_root():
    """The real path of the output root; stops unless it exists as a real
    folder holding the marker."""
    root = _root()
    if os.path.islink(root):
        raise WriteRefused(f"{root} is a symlink; the output root must be a real folder.")
    if not os.path.isdir(root):
        raise WriteRefused(f"{root} not found: create the output root first "
                           f"(write_guard.create_root).")
    marker = os.path.join(root, ROOT_MARKER)
    if os.path.islink(marker) or not os.path.isfile(marker):
        raise WriteRefused(f"{root} has no {ROOT_MARKER} marker, so it was not created by "
                           f"create_root; nothing is written into it.")
    return os.path.realpath(root)


def _make_parents(root, folder):
    """Create the folders from the root down to `folder` (inside the root),
    refusing any existing component that is a symlink or not a folder."""
    rel = os.path.relpath(folder, root)
    if rel == ".":
        return
    cur = root
    for part in rel.split(os.sep):
        cur = os.path.join(cur, part)
        if os.path.islink(cur):
            raise WriteRefused(f"{cur} is a symlink; nothing is written through it.")
        if os.path.lexists(cur):
            if not os.path.isdir(cur):
                raise WriteRefused(f"{cur} exists and is not a folder.")
        else:
            os.mkdir(cur)


def write_new(path, fn):
    """Write a new file at `path` (inside the output root) with fn(tmp_path);
    returns the path written. Stops, writing nothing, if `path` exists or lies
    outside the root."""
    root = require_root()
    target = os.path.abspath(os.fspath(path))
    if not _inside(os.path.abspath(_root()), target) or target == os.path.abspath(_root()):
        raise WriteRefused(f"{target} is outside the output root {root}; nothing written.")
    if os.path.lexists(target):
        raise WriteRefused(f"{target} already exists; these scripts never replace a file. "
                           f"Nothing written.")
    parent = os.path.dirname(target)
    _make_parents(os.path.abspath(_root()), parent)
    real_parent = os.path.realpath(parent)
    if not _inside(root, real_parent):
        raise WriteRefused(f"{parent} resolves outside the output root; nothing written.")
    final = os.path.join(real_parent, os.path.basename(target))
    if os.path.lexists(final):
        raise WriteRefused(f"{final} already exists; nothing written.")
    _TMP_COUNT[0] += 1
    tmp = os.path.join(real_parent,
                       f".tmp_{os.getpid()}_{_TMP_COUNT[0]}_{os.path.basename(target)}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    os.close(fd)
    try:
        fn(tmp)
        try:
            os.link(tmp, final)
        except FileExistsError:
            raise WriteRefused(f"{final} appeared while writing; it was not replaced.") from None
        except OSError as e:
            raise WriteRefused(f"Cannot publish {final}: this file system does not support hard "
                               f"links ({e.strerror}). Nothing was replaced; move the data folder "
                               f"to a disk that does.") from None
    finally:
        if os.path.lexists(tmp) and not os.path.islink(tmp):
            os.unlink(tmp)
    return final
