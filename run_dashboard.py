"""Convenience launcher for the evaluation dashboard.

Usage::

    python run_dashboard.py                 # http://localhost:8501
    python run_dashboard.py --server.port 8600

Any extra arguments are forwarded verbatim to ``streamlit run``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Launch ``streamlit run src/dashboard.py`` from the project root.

    Args:
        argv: Extra Streamlit CLI arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code from the Streamlit CLI.
    """
    from streamlit.web import cli as stcli

    extra = list(sys.argv[1:] if argv is None else argv)
    app = Path(__file__).resolve().parent / "src" / "dashboard.py"
    sys.argv = ["streamlit", "run", str(app), *extra]
    return int(stcli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
