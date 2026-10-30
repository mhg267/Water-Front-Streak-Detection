"""Step 9 — Remove pieces that are too short (report, Section 9).

A chain piece is kept as a streak track only when
    n ≥ 12 nodes   and   t_last − t_first + 1 ≥ 14 frames.

A track that reaches the last frame is also dropped if its last 12 points drift
by more than 12 rays or have fewer than 55% mutual edges (none in this video).

Run alone → results/step09_remove_short_tracks/:
    tracks.csv           the kept tracks: id, type, first/last frame, number of nodes
    removed_pieces.csv   the removed pieces and why
    removed_pieces.png   the removed pieces (dashed orange) on the outlines of all frames
    tracks_on_map.png    the kept tracks on the bump/dent map
"""

import numpy as np

from structures import Track, TrackPoint


def nodes_to_track(nodes, edge_mutual, track_id, frame_data, angles):
    points = [
        TrackPoint(
            node.frame_id, node.row, node.feature_type, node.x, node.y,
            node.angle_index, float(angles[node.angle_index]), node.radius_px,
            node.score_px, node.dominant_scale,
        )
        for node in nodes
    ]
    terminal_reason = (
        "video_end_unverified" if points[-1].row == len(frame_data) - 1
        else "ridge_signal_ended"
    )
    return Track(track_id=track_id, feature_type=nodes[0].feature_type,
                 points=points, edge_mutual=list(edge_mutual),
                 terminal_reason=terminal_reason)


def terminal_is_stable(track, args):
    if track.terminal_reason != "video_end_unverified" or len(track.points) < 3:
        return True
    window = min(args.terminal_window, len(track.points))
    tail = track.points[-window:]
    displacement = abs(tail[-1].angle_index - tail[0].angle_index)
    tail_mutual = track.edge_mutual[-(window - 1):]
    mutual_fraction = float(np.mean(tail_mutual)) if tail_mutual else 0.0
    return (
        displacement <= args.maximum_terminal_angle_displacement
        and mutual_fraction >= args.minimum_terminal_mutual_fraction
    )


def filter_tracks(pieces_by_type, frame_data, angles, args):
    """Turn chain pieces into tracks, drop the short ones, number the kept ones."""
    accepted = []
    rejected = []
    next_id = 1
    for feature_type in ("protrusion", "indentation"):
        for segment_nodes, segment_edges in pieces_by_type[feature_type]:
            track = nodes_to_track(
                segment_nodes, segment_edges, next_id, frame_data, angles
            )
            span = track.last.frame_id - track.first.frame_id + 1
            if (
                len(track.points) < args.minimum_track_points
                or span < args.minimum_track_span
            ):
                track.terminal_reason = "rejected_short"
                rejected.append(track)
                continue
            if not terminal_is_stable(track, args):
                track.terminal_reason = "rejected_unstable_video_tail"
                rejected.append(track)
                continue
            accepted.append(track)
            next_id += 1

    accepted.sort(
        key=lambda track: (track.feature_type, track.first.angle_deg,
                           track.first.frame_id)
    )
    for track_id, track in enumerate(accepted, start=1):
        track.track_id = track_id
    return accepted, rejected


def save_outputs(ctx, out_dir):
    import cv2
    from drawing import (bump_dent_canvas, draw_dashed_polyline, draw_node_path,
                         map_feature_color, outline_canvas, write_csv)

    write_csv(out_dir / "tracks.csv",
              ["track_id", "feature_type", "start_frame", "end_frame", "nodes"],
              [(t.track_id, t.feature_type, t.first.frame_id, t.last.frame_id,
                len(t.points)) for t in ctx.tracks])
    write_csv(out_dir / "removed_pieces.csv",
              ["feature_type", "start_frame", "end_frame", "nodes", "reason"],
              [(t.feature_type, t.first.frame_id, t.last.frame_id, len(t.points),
                t.terminal_reason) for t in ctx.rejected])
    removed = outline_canvas(ctx.frame_data)
    for track in ctx.rejected:
        if len(track.points) >= 2:
            points = np.asarray([(p.x, p.y) for p in track.points], np.int32)
            draw_dashed_polyline(removed, points, (0, 165, 255))
    cv2.imwrite(str(out_dir / "removed_pieces.png"), removed)

    canvas, left, top, sx, sy = bump_dent_canvas(ctx)
    for track in ctx.tracks:
        draw_node_path(canvas, track.points, map_feature_color(track.feature_type),
                       left, top, sx, sy)
    cv2.imwrite(str(out_dir / "tracks_on_map.png"), canvas)
    print(f"{len(ctx.tracks)} tracks kept, {len(ctx.rejected)} pieces removed")


if __name__ == "__main__":
    from pipeline import run_step_alone
    run_step_alone(__file__, 9, save_outputs)
