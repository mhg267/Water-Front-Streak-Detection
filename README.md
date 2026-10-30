# Water-Front Streak Detection

## Structure

```
run.py                            runs the whole project
requirements.txt, environment.yml Python dependencies
data/
└── original_video.mp4            source video
src/
├── pipeline.py                   chains the steps, holds the parameters
├── step01_create_masks.py        video → binary masks
├── step02_origin_and_rays.py     origin and 681 rays
├── step03_front_radius.py        front radius r(θ)
├── step04_smooth_radius.py       smoothing r(θ)
├── step05_residual.py            residual (bump / dent per ray)
├── step06_angle_time_map.py      angle × time map
├── step07_find_nodes.py          local peaks → nodes
├── step08_link_nodes.py          linking nodes into chains
├── step09_remove_short_tracks.py removing short pieces → tracks
├── step10_results.py             images, videos, tables
├── structures.py                 data passed between steps
└── drawing.py                    drawing helpers
results/                          created when you run (not tracked by git)
```

## Setup

Requires Python 3.11 or newer.

With conda:

```bash
conda env create -f environment.yml
conda activate env_waterflow_detection
```

Without conda (virtual environment in `.venv/`). On Ubuntu/Debian, install the
venv module first if it is missing: `sudo apt install python3-venv`.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Activate the environment again (`conda activate …` or `source .venv/bin/activate`)
in every new terminal before running the project.

## Run

From the repository root:

```bash
python run.py                    # whole project → results/
python run.py --recreate-masks   # rebuild the masks from the video first
python run.py --help             # all parameters
```

The whole run writes to `results/`:

```
tracked_accumulation.png     water outline of every frame + every track
annotated_masks.mp4          mask video with the track points of each frame
annotated_frames/            the frames of annotated_masks.mp4 as PNG
tracked_masks.mp4            the same video with the track ids
tracked_feature_points.csv   one row per track point
track_summary.csv            one row per track
run_config.json              parameters of the run
step01_create_masks/         the masks (made on the first run, then reused)
```

Run a single step:

```bash
python src/step07_find_nodes.py  # → results/step07_find_nodes/
```

Each step file computes the steps before it and writes only its own output to
`results/<step file name>/`.
