"""Water-front streak tracking: the steps chained together.

The whole project is run from the repository root with
    python run.py                 # steps 1–10, final outputs in results/
    python run.py --help          # all parameters

Each step file can also be run on its own; it computes the steps before it in
memory and writes only its own output to results/<step file name>/, e.g.
    python src/step07_find_nodes.py   →   results/step07_find_nodes/

Steps (one file each, named after the sections of the report):
    step01_create_masks.py        1   video → binary masks
    step02_origin_and_rays.py     2   origin and 681 rays
    step03_front_radius.py        3   front radius r(θ)
    step04_smooth_radius.py       4   smoothing r(θ)
    step05_residual.py            5   residual (bump / dent per ray)
    step06_angle_time_map.py      6   angle × time map
    step07_find_nodes.py          7   local peaks → nodes
    step08_link_nodes.py          8   linking nodes into chains
    step09_remove_short_tracks.py 9   removing short pieces → tracks
    step10_results.py             10  images, videos and tables
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Allow running this file from any working directory.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from structures import FrameData  # noqa: E402
from step01_create_masks import DEFAULT_VIDEO, STEP_NAME as MASK_STEP, create_masks  # noqa: E402
from step02_origin_and_rays import build_sampler, choose_origin  # noqa: E402
from step03_front_radius import extract_front, largest_component, read_mask  # noqa: E402
from step04_smooth_radius import moving_average  # noqa: E402
from step05_residual import compute_residuals, parse_windows  # noqa: E402
from step06_angle_time_map import build_multiscale_maps, find_active_start  # noqa: E402
from step07_find_nodes import build_nodes  # noqa: E402
from step08_link_nodes import build_chains, build_links  # noqa: E402
from step09_remove_short_tracks import filter_tracks  # noqa: E402
from step10_results import write_results  # noqa: E402

FEATURE_TYPES = ("protrusion", "indentation")    # bump, dent


def measure_frames(mask_paths, sampler, baseline_windows, args):
    """Steps 3–5 for every frame: r(θ), r_fine(θ) and the residuals."""
    frame_data = []
    number_of_angles = len(sampler[0])
    for row, mask_path in enumerate(mask_paths):
        digits = "".join(c for c in mask_path.stem if c.isdigit())
        frame_id = int(digits) if digits else row + 1
        front = extract_front(largest_component(read_mask(mask_path)), sampler)   # step 3
        if front is None:
            frame_data.append(
                FrameData(
                    row, frame_id, mask_path,
                    np.zeros(number_of_angles, np.float32),
                    np.zeros((number_of_angles, 2), np.int32),
                    np.zeros(number_of_angles, bool),
                    np.zeros((len(baseline_windows), number_of_angles),
                             np.float32),
                )
            )
            continue
        radius, points, detected = front
        fine = moving_average(radius, args.fine_window)                            # step 4
        residuals = compute_residuals(fine, detected, baseline_windows)           # step 5
        frame_data.append(
            FrameData(row, frame_id, mask_path, radius, points, detected, residuals)
        )
    return frame_data


def compute(args, until=10):
    """Run steps 1 … until in memory and return everything they produce."""
    ctx = SimpleNamespace(args=args)

    # Step 1: masks (only when missing or asked for).
    if args.mask_dir is None:
        args.mask_dir = Path(args.results_dir) / MASK_STEP
    mask_paths = sorted(args.mask_dir.glob("mask_*.png"))
    ctx.masks_created = args.recreate_masks or not mask_paths
    if ctx.masks_created:
        count = create_masks(args.video, args.mask_dir)
        print(f"step 1: {count} masks written to {args.mask_dir}")
        mask_paths = sorted(args.mask_dir.glob("mask_*.png"))
    if not mask_paths:
        raise RuntimeError(f"No masks found in {args.mask_dir}")
    ctx.mask_paths = mask_paths
    if until < 2:
        return ctx

    # Step 2: origin and rays.
    ctx.shape = read_mask(mask_paths[0]).shape
    ctx.origin = choose_origin(ctx.shape, args.origin_x, args.origin_y)
    ctx.sampler = build_sampler(ctx.shape, ctx.origin, args.angle_min,
                                args.angle_max, args.number_of_angles)
    ctx.angles = ctx.sampler[0]
    ctx.angle_step_rad = float(np.deg2rad(abs(ctx.angles[1] - ctx.angles[0])))
    if until < 3:
        return ctx

    # Steps 3–5: front radius, smoothing and residuals of every frame.
    ctx.baseline_windows = parse_windows(args.baseline_windows)
    ctx.frame_data = measure_frames(mask_paths, ctx.sampler, ctx.baseline_windows, args)
    if until < 6:
        return ctx

    # Step 6: angle × time maps.
    ctx.maps = build_multiscale_maps(ctx.frame_data, args)
    _, positive, negative, positive_scale, negative_scale, _ = ctx.maps
    ctx.active_start = find_active_start(positive, negative, args)
    if until < 7:
        return ctx

    # Step 7: nodes.
    ctx.nodes_by_type = {}
    next_id = 1
    for feature_type, score_map, scale_map in (
        ("protrusion", positive, positive_scale),
        ("indentation", negative, negative_scale),
    ):
        ctx.nodes_by_type[feature_type], next_id = build_nodes(
            ctx.frame_data, score_map, scale_map, feature_type, ctx.active_start,
            args, next_id
        )
    ctx.node_count = next_id - 1
    if until < 8:
        return ctx

    # Step 8: links and chains.
    ctx.links_by_type = {
        feature_type: build_links(ctx.nodes_by_type[feature_type],
                                  ctx.angle_step_rad, args)
        for feature_type in FEATURE_TYPES
    }
    ctx.pieces_by_type = {
        feature_type: build_chains(ctx.nodes_by_type[feature_type],
                                   ctx.links_by_type[feature_type],
                                   ctx.angle_step_rad, args)
        for feature_type in FEATURE_TYPES
    }
    if until < 9:
        return ctx

    # Step 9: keep the long pieces.
    ctx.tracks, ctx.rejected = filter_tracks(ctx.pieces_by_type, ctx.frame_data,
                                             ctx.angles, args)
    return ctx


def print_summary(ctx):
    bumps = sum(track.feature_type == "protrusion" for track in ctx.tracks)
    print(f"frames: {len(ctx.frame_data)} "
          f"(start frame {ctx.frame_data[ctx.active_start].frame_id})")
    print(f"nodes: {ctx.node_count}")
    print(f"tracks: {len(ctx.tracks)} ({bumps} bumps, {len(ctx.tracks) - bumps} dents), "
          f"{sum(len(track.points) for track in ctx.tracks)} points")
    print(f"removed pieces: {len(ctx.rejected)}")


def run(args):
    """Whole project: steps 1–10, final outputs in args.results_dir."""
    ctx = compute(args, until=9)
    write_results(ctx, args)                  # step 10
    print_summary(ctx)
    print(f"results: {args.results_dir}")


def run_step_alone(step_file, step_number, save_outputs, argv=None):
    """Run steps 1 … step_number and let that step save its own outputs.

    The outputs go to results/<name of the step file>/.
    """
    args = build_parser().parse_args(argv)
    out_dir = Path(args.results_dir) / Path(step_file).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = compute(args, until=min(step_number, 9))
    save_outputs(ctx, out_dir)
    print(f"step {step_number} outputs: {out_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Water-front streak tracking: ridge network on the angle × time "
            "map, mutual linking and splitting at sudden turns."
        )
    )
    results = PROJECT_DIR / "results"
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO,
                        help="source video, used by step 1")
    parser.add_argument("--recreate-masks", action="store_true",
                        help="run step 1 even if the masks already exist")
    parser.add_argument("--mask-dir", type=Path, default=None,
                        help="masks of step 1 (default: <results-dir>/step01_create_masks)")
    parser.add_argument("--results-dir", type=Path, default=results)
    parser.add_argument("--origin-x", type=int)
    parser.add_argument("--origin-y", type=int)
    parser.add_argument("--angle-min", type=float, default=-85.0)
    parser.add_argument("--angle-max", type=float, default=85.0)
    parser.add_argument("--number-of-angles", type=int, default=681)
    parser.add_argument("--fine-window", type=int, default=7)
    parser.add_argument("--baseline-windows", default="25,41,61")
    parser.add_argument("--map-angle-window", type=int, default=5)
    parser.add_argument("--map-time-window", type=int, default=5)
    parser.add_argument("--map-angle-sigma", type=float, default=1.0)
    parser.add_argument("--map-time-sigma", type=float, default=1.0)
    parser.add_argument("--active-score", type=float, default=0.25)
    parser.add_argument("--active-minimum-columns", type=int, default=50)
    parser.add_argument("--network-threshold", type=float, default=0.18)
    parser.add_argument("--network-minimum-distance", type=int, default=5)
    parser.add_argument("--maximum-frame-gap", type=int, default=2)
    parser.add_argument("--maximum-angle-step", type=int, default=4)
    parser.add_argument("--maximum-tangential-change", type=float, default=7.0)
    parser.add_argument("--maximum-radius-change", type=float, default=8.0)
    parser.add_argument("--maximum-spatial-change", type=float, default=12.0)
    parser.add_argument("--edge-angle-weight", type=float, default=0.45)
    parser.add_argument("--edge-radius-weight", type=float, default=0.15)
    parser.add_argument("--edge-spatial-weight", type=float, default=0.25)
    parser.add_argument("--edge-gap-penalty", type=float, default=0.2)
    parser.add_argument("--edge-score-weight", type=float, default=0.15)
    parser.add_argument("--score-cap", type=float, default=6.0)
    parser.add_argument("--minimum-track-points", type=int, default=12)
    parser.add_argument("--minimum-track-span", type=int, default=14)
    parser.add_argument("--spike-refinement-passes", type=int, default=2)
    parser.add_argument("--spike-refinement-trigger", type=float, default=2.5)
    parser.add_argument("--spike-search-radius", type=float, default=4.0)
    parser.add_argument("--refinement-scale-weight", type=float, default=0.35)
    parser.add_argument("--refinement-score-weight", type=float, default=0.08)
    parser.add_argument("--refinement-minimum-improvement", type=float, default=0.25)
    parser.add_argument("--maximum-local-angular-acceleration", type=float,
                        default=3.5)
    parser.add_argument("--terminal-window", type=int, default=12)
    parser.add_argument("--maximum-terminal-angle-displacement", type=int, default=12)
    parser.add_argument("--minimum-terminal-mutual-fraction", type=float, default=0.55)
    parser.add_argument("--map-color-clip", type=float, default=4.0)
    parser.add_argument("--map-render-scale-x", type=int, default=2)
    parser.add_argument("--map-render-scale-y", type=int, default=3)
    parser.add_argument("--fps", type=float, default=24.0)
    return parser
