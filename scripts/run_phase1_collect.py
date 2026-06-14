"""Run Phase 1 data collection."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# if sys.version_info >= (3, 13):
#     raise RuntimeError("Phase 1 requires Python 3.10-3.12 because datasets/pyarrow may hang on Python 3.13.")

from aiguru.phase1.collect import main

if __name__ == "__main__":
    main()
