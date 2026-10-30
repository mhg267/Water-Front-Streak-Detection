"""Step 7 — Local peaks: from regions to points (report, Section 7).

In each frame (one row S of P̂ or N̂), ray i becomes a node when
    1. S(i) > S(i−1) and S(i) ≥ S(i+1)          (local maximum)
    2. S(i) ≥ 0.18 px                            (strong enough)
    3. no stronger kept peak within 5 rays       (non-maximum suppression)
Interpolated rays are never nodes. Each node keeps the (x, y) of the real front point.

Run alone → results/step07_find_nodes/:
    nodes.csv           every node: frame, type, ray, strength, (x, y)
    nodes_on_map.png    the nodes on the bump/dent map (red = bump, blue = dent)
"""

import numpy as np

from structures import Node


def local_maxima(signal, threshold, minimum_distance):
    indices = np.flatnonzero(
        (signal[1:-1] > signal[:-2])
        & (signal[1:-1] >= signal[2:])
        & (signal[1:-1] >= threshold)
    ) + 1
    selected = []
    for index in indices[np.argsort(signal[indices])[::-1]]:
        index = int(index)
        if all(abs(index - kept) >= minimum_distance for kept in selected):
            selected.append(index)
    return sorted(selected)


def build_nodes(frame_data, score_map, scale_map, feature_type, active_start,
                args, first_node_id):
    """Nodes of one type (bump or dent), grouped by frame row."""
    nodes_by_row = [[] for _ in frame_data]
    next_id = first_node_id
    for row in range(active_start, len(frame_data)):
        for angle_index in local_maxima(
            score_map[row], args.network_threshold,
            args.network_minimum_distance,
        ):
            data = frame_data[row]
            if not data.detected[angle_index]:
                continue
            x, y = data.front_points[angle_index]
            nodes_by_row[row].append(
                Node(
                    next_id, row, data.frame_id, feature_type,
                    int(angle_index), float(score_map[row, angle_index]),
                    int(scale_map[row, angle_index]),
                    int(x), int(y), float(data.radius[angle_index]),
                )
            )
            next_id += 1
    return nodes_by_row, next_id


def save_outputs(ctx, out_dir):
    import cv2
    from drawing import bump_dent_canvas, map_feature_color, write_csv

    nodes = [node for feature_type in ("protrusion", "indentation")
             for row_nodes in ctx.nodes_by_type[feature_type] for node in row_nodes]
    write_csv(out_dir / "nodes.csv",
              ["node_id", "frame_id", "feature_type", "ray_index", "angle_deg",
               "strength_px", "x", "y"],
              [(n.node_id, n.frame_id, n.feature_type, n.angle_index,
                f"{ctx.angles[n.angle_index]:.2f}", f"{n.score_px:.3f}", n.x, n.y)
               for n in nodes])
    canvas, left, top, sx, sy = bump_dent_canvas(ctx)
    for n in nodes:
        cv2.circle(canvas, (left + n.angle_index * sx, top + n.row * sy), 1,
                   map_feature_color(n.feature_type), -1)
    cv2.imwrite(str(out_dir / "nodes_on_map.png"), canvas)
    print(f"{len(nodes)} nodes")


if __name__ == "__main__":
    from pipeline import run_step_alone
    run_step_alone(__file__, 7, save_outputs)
