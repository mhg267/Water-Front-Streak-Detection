"""Step 8 — Link nodes across frames (report, Section 8).

8.1  Gate: node B in frame t + Δt is a candidate for node A only if
         |i_B − i_A| ≤ Δt · min(4, 7 / (r̄ α)),  |r_B − r_A| ≤ 8 Δt,  d ≤ 12 Δt.
     Frame t + 2 is tried only when frame t + 1 has no candidate.
8.2  Cost (lower = more similar):
         0.45 Δθ/gate + 0.15 Δr/(8Δt) + 0.25 d/(12Δt) + 0.2 (Δt − 1) − 0.15 min(S_B, 6)/6
8.3  next(A) = cheapest candidate forward, prev(B) = cheapest candidate backward;
     edge A → B is kept only when next(A) = B and prev(B) = A (mutual).
     Chains are walked from every node, along mutual edges only.
8.4  Inside a chain, an edge pair is cut where the angular velocity changes by
     ≥ 3.5 rays/frame (sudden turn).

Run alone → results/step08_link_nodes/:
    chains.csv          every chain piece after the cuts, one row per node
    chains_on_map.png   the pieces on the bump/dent map (single nodes as dots)
"""

import numpy as np


def angular_gate_indices(first, second, angle_step_rad, args):
    radius = max(1.0, 0.5 * (first.radius_px + second.radius_px))
    radius_gate = args.maximum_tangential_change / (radius * angle_step_rad)
    return max(1.0, min(float(args.maximum_angle_step), radius_gate))


def edge_cost(first, second, angle_step_rad, args):
    """Cost of linking first → second, or None when second is outside the gate."""
    frame_gap = second.row - first.row
    if frame_gap < 1 or frame_gap > args.maximum_frame_gap:
        return None
    angle_delta = abs(second.angle_index - first.angle_index)
    angle_gate = angular_gate_indices(first, second, angle_step_rad, args)
    angle_gate *= frame_gap
    radius_delta = abs(second.radius_px - first.radius_px)
    spatial_delta = float(np.hypot(second.x - first.x, second.y - first.y))
    if (
        angle_delta > angle_gate
        or radius_delta > args.maximum_radius_change * frame_gap
        or spatial_delta > args.maximum_spatial_change * frame_gap
    ):
        return None
    score_reward = min(second.score_px, args.score_cap) / args.score_cap
    return float(
        args.edge_angle_weight * angle_delta / max(angle_gate, 1e-6)
        + args.edge_radius_weight * radius_delta
        / (args.maximum_radius_change * frame_gap)
        + args.edge_spatial_weight * spatial_delta
        / (args.maximum_spatial_change * frame_gap)
        + args.edge_gap_penalty * (frame_gap - 1)
        - args.edge_score_weight * score_reward
    )


def build_links(nodes_by_row, angle_step_rad, args):
    """Forward choice next(A), backward choice prev(B) and the set of mutual edges."""
    best_next = {}
    best_previous = {}
    feasible_edges = []
    all_nodes = [node for row_nodes in nodes_by_row for node in row_nodes]
    by_id = {node.node_id: node for node in all_nodes}

    for row, row_nodes in enumerate(nodes_by_row):
        for node in row_nodes:
            choices = []
            for gap in range(1, args.maximum_frame_gap + 1):
                target_row = row + gap
                if target_row >= len(nodes_by_row):
                    break
                for target in nodes_by_row[target_row]:
                    cost = edge_cost(node, target, angle_step_rad, args)
                    if cost is not None:
                        choices.append((cost, target.node_id))
                        feasible_edges.append((node.node_id, target.node_id, cost))
                if choices:
                    break
            if choices:
                best_next[node.node_id] = min(choices)[1]

    incoming = {}
    for source_id, target_id, cost in feasible_edges:
        incoming.setdefault(target_id, []).append((cost, source_id))
    for target_id, choices in incoming.items():
        best_previous[target_id] = min(choices)[1]

    mutual = {
        (source_id, target_id)
        for source_id, target_id in best_next.items()
        if best_previous.get(target_id) == source_id
    }
    return by_id, best_next, best_previous, mutual, feasible_edges


def follow_mutual_seed(seed, by_id, best_next, best_previous, mutual):
    """Walk backward and forward from seed along mutual edges only; stop at a non-mutual edge."""
    backward = []
    current = seed.node_id
    visited = {current}
    while current in best_previous:
        previous = best_previous[current]
        if previous in visited or (previous, current) not in mutual:
            break
        backward.append(by_id[previous])
        visited.add(previous)
        current = previous
    backward.reverse()

    forward = []
    current = seed.node_id
    while current in best_next:
        following = best_next[current]
        if following in visited or (current, following) not in mutual:
            break
        forward.append(by_id[following])
        visited.add(following)
        current = following
    nodes = backward + [seed] + forward
    return nodes, [True] * (len(nodes) - 1)


def angular_acceleration(first, middle, last):
    """Change of angular velocity, normalised by the frame gap."""
    first_gap = max(1, middle.row - first.row)
    second_gap = max(1, last.row - middle.row)
    first_velocity = (middle.angle_index - first.angle_index) / first_gap
    second_velocity = (last.angle_index - middle.angle_index) / second_gap
    return abs(second_velocity - first_velocity)


def refine_path_nodes(nodes, nodes_by_row, mutual, angle_step_rad, args):
    """Replace a single kink point by a valid maximum near the predicted trajectory."""
    nodes = list(nodes)
    for _ in range(args.spike_refinement_passes):
        changed = False
        for index in range(1, len(nodes) - 1):
            previous, current, following = nodes[index - 1:index + 2]
            acceleration = angular_acceleration(previous, current, following)
            if acceleration < args.spike_refinement_trigger:
                continue
            fraction = (
                (current.row - previous.row)
                / max(1, following.row - previous.row)
            )
            predicted = (
                previous.angle_index
                + fraction * (following.angle_index - previous.angle_index)
            )
            choices = []
            for candidate in nodes_by_row[current.row]:
                if candidate.node_id == current.node_id:
                    continue
                if abs(candidate.angle_index - predicted) > args.spike_search_radius:
                    continue
                if edge_cost(previous, candidate, angle_step_rad, args) is None:
                    continue
                if edge_cost(candidate, following, angle_step_rad, args) is None:
                    continue
                objective = (
                    abs(candidate.angle_index - predicted)
                    + args.refinement_scale_weight
                    * (
                        (candidate.dominant_scale != previous.dominant_scale)
                        + (candidate.dominant_scale != following.dominant_scale)
                    )
                    - args.refinement_score_weight
                    * min(candidate.score_px, args.score_cap)
                )
                choices.append((objective, candidate))
            if not choices:
                continue
            current_objective = (
                abs(current.angle_index - predicted)
                + args.refinement_scale_weight
                * (
                    (current.dominant_scale != previous.dominant_scale)
                    + (current.dominant_scale != following.dominant_scale)
                )
                - args.refinement_score_weight
                * min(current.score_px, args.score_cap)
            )
            best_objective, best_candidate = min(
                choices, key=lambda item: item[0]
            )
            if (
                best_objective + args.refinement_minimum_improvement
                < current_objective
            ):
                nodes[index] = best_candidate
                changed = True
        if not changed:
            break
    edge_mutual = [
        (nodes[index].node_id, nodes[index + 1].node_id) in mutual
        for index in range(len(nodes) - 1)
    ]
    return nodes, edge_mutual


def split_path(nodes, edge_mutual, args):
    """Cut a chain at non-mutual edges and at sudden turns; return the pieces."""
    split_edges = {
        index for index, is_mutual in enumerate(edge_mutual) if not is_mutual
    }
    for index in range(1, len(nodes) - 1):
        if angular_acceleration(
            nodes[index - 1], nodes[index], nodes[index + 1]
        ) >= args.maximum_local_angular_acceleration:
            split_edges.add(index - 1)
            split_edges.add(index)

    segments = []
    start = 0
    for edge_index in range(len(nodes) - 1):
        if edge_index not in split_edges:
            continue
        segment_nodes = nodes[start:edge_index + 1]
        segment_edges = edge_mutual[start:edge_index]
        if segment_nodes:
            segments.append((segment_nodes, segment_edges))
        start = edge_index + 1
    segment_nodes = nodes[start:]
    segment_edges = edge_mutual[start:]
    if segment_nodes:
        segments.append((segment_nodes, segment_edges))
    return segments


def build_chains(nodes_by_row, links, angle_step_rad, args):
    """All distinct chain pieces of one feature type, in seed order (strongest seed first)."""
    by_id, best_next, best_previous, mutual, _ = links
    seeds = sorted(by_id.values(), key=lambda node: node.score_px, reverse=True)
    seen_signatures = set()
    seen_segment_signatures = set()
    pieces = []
    for seed in seeds:
        nodes, edge_mutual = follow_mutual_seed(
            seed, by_id, best_next, best_previous, mutual
        )
        nodes, edge_mutual = refine_path_nodes(
            nodes, nodes_by_row, mutual, angle_step_rad, args
        )
        signature = tuple(node.node_id for node in nodes)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        for segment_nodes, segment_edges in split_path(nodes, edge_mutual, args):
            segment_signature = tuple(node.node_id for node in segment_nodes)
            if segment_signature in seen_segment_signatures:
                continue
            seen_segment_signatures.add(segment_signature)
            pieces.append((segment_nodes, segment_edges))
    return pieces


def save_outputs(ctx, out_dir):
    import cv2
    from drawing import bump_dent_canvas, draw_node_path, map_feature_color, write_csv

    rows = []
    canvas, left, top, sx, sy = bump_dent_canvas(ctx)
    piece_id = 0
    for feature_type in ("protrusion", "indentation"):
        color = map_feature_color(feature_type)
        for nodes, _ in ctx.pieces_by_type[feature_type]:
            piece_id += 1
            rows += [(piece_id, feature_type, order, n.node_id, n.frame_id,
                      n.angle_index, n.x, n.y) for order, n in enumerate(nodes)]
            if len(nodes) > 1:
                draw_node_path(canvas, nodes, color, left, top, sx, sy)
            else:
                cv2.circle(canvas, (left + nodes[0].angle_index * sx,
                                    top + nodes[0].row * sy), 1, color, -1)
    write_csv(out_dir / "chains.csv",
              ["piece_id", "feature_type", "order", "node_id", "frame_id",
               "ray_index", "x", "y"], rows)
    cv2.imwrite(str(out_dir / "chains_on_map.png"), canvas)
    print(f"{piece_id} chain pieces")


if __name__ == "__main__":
    from pipeline import run_step_alone
    run_step_alone(__file__, 8, save_outputs)
