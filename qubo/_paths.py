"""Ensure this package dir and parent ``code-to-commit`` are on ``sys.path``.

Local dir first so ``import qubo`` resolves to ``qubo/qubo.py``.
Parent provides ``model`` and ``revised_dispatch``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_LOCAL = Path(__file__).resolve().parent
_PARENT = _LOCAL.parent
# Parent first would shadow local qubo.py; insert local at 0 after parent.
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
if str(_LOCAL) not in sys.path:
    sys.path.insert(0, str(_LOCAL))
