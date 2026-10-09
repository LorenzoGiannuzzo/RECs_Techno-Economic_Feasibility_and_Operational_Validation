"""Run the whole pipeline from the repository root without installing the package.

    python run_pipeline.py                      # metered data in data/confidential/
    python run_pipeline.py --data synthetic     # synthetic stand-in, no confidential data needed
    python run_pipeline.py --steps design,figures
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rec_pipeline.__main__ import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
