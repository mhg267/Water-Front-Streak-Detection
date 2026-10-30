"""Run the whole project: steps 1–10 of src/, final outputs in results/.

    python run.py                    # creates the masks (step 1) on the first run, then reuses them
    python run.py --recreate-masks   # rebuild the masks from data/original_video.mp4 first
    python run.py --help             # all parameters

To run a single step and see only its output, run its file instead, e.g.
    python src/step07_find_nodes.py  →  results/step07_find_nodes/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pipeline import build_parser, run  # noqa: E402

if __name__ == "__main__":
    run(build_parser().parse_args())
