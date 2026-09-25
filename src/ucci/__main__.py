"""``python -m ucci``: the ``ucci`` command line (see :mod:`ucci.cli`)."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
