"""Stage guard, release configuration, and safe file replacement.

Every transform goes through :class:`Stage`. It refuses a root that is the
configured BIDS root, is inside it, equals the source-data directory, or
contains either of them, and it never edits a file in place: writes are
queued, and :meth:`Stage.commit` replaces each file by writing a temporary
file in the same directory and renaming it over the old one. With a hardlink
copy of the tree, the rename gives the staged path a new inode, so the
working-tree file that shared the old inode is left untouched.
"""

from __future__ import annotations

import datetime as _dt
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EPOCH = "1900-01-01"
# BIDS recommends that shifted dates fall in 1925 or earlier; a date with a
# year at or below this is treated as already shifted.
SHIFTED_MAX_YEAR = 1925


class StageError(RuntimeError):
    """A path is unsafe to transform or to write a private output to."""


@dataclass(frozen=True)
class ReleaseConfig:
    bids_root: Path
    source_dir: Path
    epoch: _dt.date


def parse_epoch(value: str | _dt.date) -> _dt.date:
    """Parse and validate a date-shift epoch (``YYYY-MM-DD``, year <= 1925)."""
    if isinstance(value, _dt.date):
        epoch = value
    else:
        try:
            epoch = _dt.date.fromisoformat(str(value).strip())
        except ValueError as exc:
            raise ValueError(f"epoch must be YYYY-MM-DD, got {value!r}") from exc
    if epoch.year > SHIFTED_MAX_YEAR:
        raise ValueError(
            f"epoch {epoch} is after {SHIFTED_MAX_YEAR}; BIDS recommends shifted "
            f"dates in {SHIFTED_MAX_YEAR} or earlier, and already-shifted dates "
            f"are recognised by year <= {SHIFTED_MAX_YEAR}")
    return epoch


def load_release_config(config_dir: str | Path | None = None) -> ReleaseConfig:
    """Read the BIDS root, source dir and epoch from ``config/*.toml``."""
    from core.config import load_config

    cfg = load_config(config_dir=config_dir)
    paths = cfg["paths"]
    release = cfg.get("release", {})
    return ReleaseConfig(
        bids_root=Path(paths["bids_project_dir"]),
        source_dir=Path(paths["source_dir"]),
        epoch=parse_epoch(release.get("date_epoch", DEFAULT_EPOCH)),
    )


def _resolve(path: Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser())).resolve(strict=False)


def _within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def check_stage(stage: str | Path, cfg: ReleaseConfig) -> Path:
    """Return the resolved stage root, or raise :class:`StageError`.

    Refuses the configured BIDS root and anything inside it, the source-data
    directory, any directory that contains either of them, and a directory
    that is not a BIDS dataset root (no ``dataset_description.json``).
    """
    root = _resolve(stage)
    bids = _resolve(cfg.bids_root)
    source = _resolve(cfg.source_dir)
    if _within(root, bids):
        raise StageError(
            f"refusing to transform {root}: it is the configured BIDS root "
            f"({bids}) or inside it. Transforms run on a staged copy only.")
    if root == source:
        raise StageError(
            f"refusing to transform {root}: it is the configured source-data "
            f"directory. Transforms run on a staged copy only.")
    for name, protected in (("BIDS root", bids), ("source-data directory", source)):
        if protected.is_relative_to(root):
            raise StageError(
                f"refusing to transform {root}: it contains the configured "
                f"{name} ({protected}).")
    if not root.is_dir():
        raise StageError(f"stage root {root} is not a directory")
    if not (root / "dataset_description.json").is_file():
        raise StageError(
            f"stage root {root} has no dataset_description.json; point at the "
            f"top of the staged BIDS copy")
    return root


def check_private_output(path: str | Path, stage_root: Path | None,
                         cfg: ReleaseConfig) -> Path:
    """Validate a private output path (offsets, reports): outside the stage,
    outside the BIDS root, and not already present. Returns it resolved."""
    out = _resolve(path)
    if stage_root is not None and _within(out, _resolve(stage_root)):
        raise StageError(
            f"refusing to write private output {out} inside the staged copy "
            f"{stage_root}: it would ship with the export")
    if _within(out, _resolve(cfg.bids_root)):
        raise StageError(
            f"refusing to write private output {out} inside the BIDS root "
            f"{cfg.bids_root}")
    if out.exists():
        raise StageError(
            f"private output {out} already exists; refusing to overwrite it "
            f"(move it aside or choose another path)")
    return out


def check_private_input(path: str | Path, stage_root: Path) -> Path:
    """A private input (names list, redaction spec) must exist and must not
    be inside the tree that is going to ship."""
    p = _resolve(path)
    if _within(p, _resolve(stage_root)):
        raise StageError(
            f"private input {p} is inside {stage_root}; it would ship with "
            f"the export. Keep it outside the stage.")
    if not p.is_file():
        raise StageError(f"private input {p} does not exist")
    return p


def atomic_write_text(path: str | Path, text: str) -> None:
    """Replace ``path`` with ``text`` via temp file + rename, never in place.

    The new file inherits the old file's permission bits. Because the rename
    swaps the directory entry, any other hardlink to the old inode (e.g. the
    working-tree file of a hardlink-copied stage) keeps its content.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = None
    if path.exists():
        mode = path.stat().st_mode & 0o7777
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                               dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        else:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_private(path: str | Path, text: str, stage_root: Path | None,
                  cfg: ReleaseConfig) -> Path:
    """Write a private output after :func:`check_private_output`."""
    out = check_private_output(path, stage_root, cfg)
    atomic_write_text(out, text)
    return out


class Stage:
    """A dataset root that transforms read from and queue writes against.

    ``writable=True`` runs :func:`check_stage`; read-only users (``sweep``,
    ``flag``) may open any root, including the working tree, but cannot
    queue writes. Reads see queued writes, so steps chain in memory and a
    dry run can sweep the planned result.
    """

    def __init__(self, root: str | Path, cfg: ReleaseConfig, writable: bool = True):
        self.cfg = cfg
        self.writable = writable
        if writable:
            self.root = check_stage(root, cfg)
        else:
            self.root = _resolve(root)
            if not self.root.is_dir():
                raise StageError(f"{self.root} is not a directory")
        self._pending: dict[Path, str] = {}

    # paths ---------------------------------------------------------------
    def path(self, rel: str | Path) -> Path:
        """Lexical absolute path (symlinks not followed) inside the stage."""
        p = Path(rel)
        p = Path(os.path.normpath(p if p.is_absolute() else self.root / p))
        if not _within(p, self.root):
            raise StageError(f"{rel} is outside the stage {self.root}")
        return p

    def rel(self, path: str | Path) -> str:
        return self.path(path).relative_to(self.root).as_posix()

    # io --------------------------------------------------------------------
    def exists(self, rel: str | Path) -> bool:
        p = self.path(rel)
        return p in self._pending or p.is_file()

    def read_text(self, rel: str | Path) -> str:
        p = self.path(rel)
        if p in self._pending:
            return self._pending[p]
        with open(p, encoding="utf-8", errors="replace", newline="") as fh:
            return fh.read()

    def queue_write(self, rel: str | Path, text: str) -> None:
        if not self.writable:
            raise StageError("this stage was opened read-only")
        p = self.path(rel)
        # A symlinked directory could lead back into the working tree. (A
        # symlinked file is fine: the rename replaces the link, not its target.)
        real_parent = _resolve(p.parent)
        if not _within(real_parent, self.root):
            raise StageError(
                f"refusing to write {p}: its directory resolves to "
                f"{real_parent}, outside the stage {self.root}")
        self._pending[p] = text

    @property
    def pending(self) -> dict[Path, str]:
        return dict(self._pending)

    def commit(self) -> list[Path]:
        """Write every queued file (temp + rename). Returns the paths."""
        written = []
        for p, text in sorted(self._pending.items()):
            atomic_write_text(p, text)
            written.append(p)
        self._pending.clear()
        return written
