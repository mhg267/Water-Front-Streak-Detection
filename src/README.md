# src — the pipeline, one file per step

```bash
python run.py                          # from the repository root: whole project → results/
python src/step07_find_nodes.py        # one step alone → results/step07_find_nodes/
```

Each step is a separate file; its docstring states the formulas it implements.
`pipeline.py` chains the steps: `run.py` uses it to run all of them, and a step
file run on its own uses it to compute the steps before it in memory.

| File | What it does | Run alone → `results/<file name>/` |
|---|---|---|
| `step01_create_masks.py` | Video → grayscale → threshold 15 → one binary mask per frame | the 304 masks (recreated); the later steps read them from here |
| `step02_origin_and_rays.py` | Origin at the image centre (640, 320); 681 rays from −85° to +85°, every 0.25° | `rays.csv`, `origin_and_rays.png` |
| `step03_front_radius.py` | Keep the largest white region; r(θ) = farthest water pixel on each ray; interpolate rays that hit no water | `front_radius.csv` (every ray, every frame), `front_frame_200.png` |
| `step04_smooth_radius.py` | 7-ray moving average → r_fine(θ) | `smoothed_radius.csv`, `front_frame_200.png` (raw red, smoothed green) |
| `step05_residual.py` | Residual e_w = r_fine − baseline for w = 25, 41, 61 rays | `residuals.csv`, `residual_frame_200.png` |
| `step06_angle_time_map.py` | Stack all frames into angle × time maps; normalised 5 × 5 Gaussian; strongest scale → bump map and dent map; start frame | `angle_time_maps.npz`, `bump_dent_map.png`, `start_frame.txt` |
| `step07_find_nodes.py` | Local peaks per frame (≥ 0.18 px, ≥ 5 rays apart) → nodes | `nodes.csv`, `nodes_on_map.png` |
| `step08_link_nodes.py` | Gate and cost; forward/backward choices; mutual edges; walk chains along mutual edges; cut at sudden turns | `chains.csv`, `chains_on_map.png` |
| `step09_remove_short_tracks.py` | Keep pieces with ≥ 12 nodes and ≥ 14 frames; number the tracks | `tracks.csv`, `removed_pieces.csv`, `removed_pieces.png`, `tracks_on_map.png` |
| `step10_results.py` | Result image, videos, CSV tables, run configuration | the same final outputs as `run.py` |

Other files:

| File | Role |
|---|---|
| `pipeline.py` | `compute(args, until)` runs steps 1 … until; parameters (`build_parser`) |
| `structures.py` | Data passed between steps: `FrameData`, `Node`, `TrackPoint`, `Track` |
| `drawing.py` | Colours and drawing helpers used by the steps that save images |

Step 1 writes the masks to `results/step01_create_masks/` the first time anything
is run, or with `--recreate-masks`; afterwards the later steps reuse them.

Step 9 also drops a track that reaches the last frame when its last 12 points
drift by more than 12 rays or have fewer than 55% mutual edges (no track in this
video).

## Main parameters

All of them work both with `run.py` and with a single step file.

| Parameter | Default | Step | Meaning |
|---|---|---|---|
| `--video` | `data/original_video.mp4` | 1 | Source video |
| `--mask-dir` | `results/step01_create_masks/` | 1, 3 | Masks |
| `--recreate-masks` | off | 1 | Rebuild the masks even if they exist |
| `--results-dir` | `results/` | all | Output folder |
| `--origin-x`, `--origin-y` | image centre | 2 | Origin of the rays (the nozzle) |
| `--number-of-angles`, `--angle-min`, `--angle-max` | 681, −85, 85 | 2 | Rays |
| `--fine-window` | 7 | 4 | Smoothing of r(θ), in rays |
| `--baseline-windows` | `25,41,61` | 5 | Baseline windows; `41` gives the one-scale variant |
| `--map-angle-window`, `--map-time-window` | 5, 5 | 6 | Gaussian smoothing of the map; `1` disables it |
| `--network-threshold` | 0.18 | 7 | Minimum node strength (px) |
| `--network-minimum-distance` | 5 | 7 | Minimum distance between nodes in one frame (rays) |
| `--maximum-angle-step`, `--maximum-radius-change`, `--maximum-spatial-change` | 4, 8, 12 | 8 | Gate per frame (rays, px, px) |
| `--maximum-frame-gap` | 2 | 8 | Link at most 2 frames ahead (skip at most 1) |
| `--maximum-local-angular-acceleration` | 3.5 | 8 | Sudden-turn threshold (rays/frame) |
| `--minimum-track-points`, `--minimum-track-span` | 12, 14 | 9 | Minimum track length (nodes, frames) |
