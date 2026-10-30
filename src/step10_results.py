"""Step 10 — Results (report, Section 10).

Every track point already carries the (x, y) of the real front point on its ray
(step 3), so the tracks are drawn directly on the water edge.

Run with the whole project (python run.py), the outputs go to results/;
run alone (python src/step10_results.py), they go to results/step10_results/.

Outputs:
    tracked_accumulation.png     outline of every frame + every track (red = bump, blue = dent)
    annotated_masks.mp4          mask video with the track points of each frame and the origin
    annotated_frames/            the frames of annotated_masks.mp4 as PNG
    tracked_masks.mp4            mask video with the track points and track ids of each frame
    tracked_feature_points.csv   one row per track point
    track_summary.csv            one row per track
    run_config.json              parameters of the run and a hash of every source file
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from drawing import draw_component_contour, feature_color
from step03_front_radius import read_mask


def save_run_config(args):
    """Record the parameters of the run and the version of every source file."""
    source_dir = Path(__file__).resolve().parent
    source_hashes = {
        str(path.relative_to(source_dir.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_dir.glob("*.py"))
    }
    config = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
        "source_sha256": source_hashes,
    }
    (args.results_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def render_image_outputs(frame_data, tracks, origin, args):
    """tracked_accumulation.png, annotated_masks.mp4 (+ its frames) and tracked_masks.mp4."""
    first_mask = read_mask(frame_data[0].mask_path)
    height, width = first_mask.shape
    accumulation = np.full((height, width, 3), 255, np.uint8)
    points_by_frame = {}
    for track in tracks:
        for point in track.points:
            points_by_frame.setdefault(point.frame_id, []).append((track, point))

    frames_dir = args.results_dir / "annotated_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    annotated_writer = cv2.VideoWriter(
        str(args.results_dir / "annotated_masks.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height),
    )
    tracked_writer = cv2.VideoWriter(
        str(args.results_dir / "tracked_masks.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height),
    )
    if not annotated_writer.isOpened() or not tracked_writer.isOpened():
        raise RuntimeError("Cannot create the output videos")
    try:
        for data in frame_data:
            mask = read_mask(data.mask_path)
            draw_component_contour(accumulation, mask)
            annotated = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            tracked = annotated.copy()
            for track, point in points_by_frame.get(data.frame_id, []):
                color = feature_color(track.feature_type)
                cv2.circle(annotated, (point.x, point.y), 4, color, -1,
                           lineType=cv2.LINE_AA)
                cv2.circle(tracked, (point.x, point.y), 4, color, -1,
                           lineType=cv2.LINE_AA)
                cv2.putText(tracked, str(track.track_id),
                            (point.x + 4, point.y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1,
                            cv2.LINE_AA)
            cv2.drawMarker(annotated, origin, (0, 165, 255),
                           cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
            for image in (annotated, tracked):
                cv2.putText(image, f"frame {data.frame_id:06d}", (16, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0),
                            1, cv2.LINE_AA)
            cv2.imwrite(str(frames_dir / f"frame_{data.frame_id:06d}.png"), annotated)
            annotated_writer.write(annotated)
            tracked_writer.write(tracked)
    finally:
        annotated_writer.release()
        tracked_writer.release()

    for track in tracks:
        points = np.asarray([(p.x, p.y) for p in track.points], np.int32)
        cv2.polylines(accumulation, [points.reshape(-1, 1, 2)], False,
                      feature_color(track.feature_type), 1, cv2.LINE_AA)
    cv2.imwrite(str(args.results_dir / "tracked_accumulation.png"), accumulation)


def write_csvs(tracks, args):
    """tracked_feature_points.csv and track_summary.csv."""
    point_fields = ["track_id", "frame_id", "feature_type", "x", "y",
                    "angle_index", "angle_deg", "radius_px", "score_px",
                    "dominant_scale"]
    with (args.results_dir / "tracked_feature_points.csv").open(
            "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=point_fields)
        writer.writeheader()
        for track in tracks:
            for point in track.points:
                writer.writerow({"track_id": track.track_id,
                                 "frame_id": point.frame_id,
                                 "feature_type": track.feature_type,
                                 "x": point.x, "y": point.y,
                                 "angle_index": point.angle_index,
                                 "angle_deg": point.angle_deg,
                                 "radius_px": point.radius_px,
                                 "score_px": point.score_px,
                                 "dominant_scale": point.dominant_scale})

    summary_fields = ["track_id", "feature_type", "start_frame", "end_frame",
                      "frame_span", "observed_points", "coverage",
                      "mean_score_px", "terminal_reason"]
    with (args.results_dir / "track_summary.csv").open(
            "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        for track in tracks:
            span = track.last.frame_id - track.first.frame_id + 1
            writer.writerow({"track_id": track.track_id,
                             "feature_type": track.feature_type,
                             "start_frame": track.first.frame_id,
                             "end_frame": track.last.frame_id,
                             "frame_span": span,
                             "observed_points": len(track.points),
                             "coverage": len(track.points) / span,
                             "mean_score_px": np.mean([p.score_px for p in track.points]),
                             "terminal_reason": track.terminal_reason})


def write_results(ctx, args):
    """All final outputs of the project, in args.results_dir."""
    args.results_dir = Path(args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(args)
    write_csvs(ctx.tracks, args)
    render_image_outputs(ctx.frame_data, ctx.tracks, ctx.origin, args)


def save_outputs(ctx, out_dir):
    """Run alone: the same final outputs, in results/step10_results/."""
    ctx.args.results_dir = out_dir
    write_results(ctx, ctx.args)


if __name__ == "__main__":
    from pipeline import run_step_alone
    run_step_alone(__file__, 10, save_outputs)
