#!/usr/bin/env python3
"""
Backward-compatible entrypoint. Prefer:

  python scripts/evaluate_live.py --config configs/live_run.yaml --limit 1000
"""

from __future__ import annotations

import os
import runpy
import sys

if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv[1:1] = ["--config", "configs/live_run.yaml"]
    runpy.run_path(
        os.path.join(os.path.dirname(__file__), "evaluate_live.py"),
        run_name="__main__",
    )
