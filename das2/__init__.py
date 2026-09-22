"""
das2 -- water sensor anomaly intelligence.

`source_fingerprint()` exists because three separate debugging rounds were
spent on a container running code older than the repository. A rebuilt image
and a stale one are indistinguishable from the outside: the same command, the
same startup banner, and behaviour that contradicts the source in front of
you. The only reliable signal was whether a particular log line appeared,
which requires already knowing which commit introduced it.

So every run states what code it is. The fingerprint is a hash of the package
source as it exists INSIDE the container, which is the thing in question --
not a version constant someone has to remember to bump, and not a git SHA,
which is unavailable in an image built by COPY.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

__all__ = ["source_fingerprint", "build_stamp"]


def source_fingerprint() -> str:
    """
    A short hash of every .py and .yaml in this package, as installed.

    Paths are hashed alongside contents so that adding, moving or deleting a
    file changes the answer, and the file list is sorted so the result does
    not depend on filesystem ordering.
    """
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(
        p for pattern in ("*.py", "*.yaml", "*.yml")
        for p in root.rglob(pattern)
        if "__pycache__" not in p.parts
    ):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def build_stamp() -> str:
    """
    Fingerprint plus the newest source mtime -- when this code was last
    written into the image. A timestamp older than your last `git pull` means
    the image was never rebuilt, which is the failure this reports.
    """
    root = Path(__file__).resolve().parent
    newest = max(
        (p.stat().st_mtime for p in root.rglob("*.py")
         if "__pycache__" not in p.parts),
        default=0.0,
    )
    when = datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M") if newest \
        else "unknown"
    return f"{source_fingerprint()} (written {when})"
