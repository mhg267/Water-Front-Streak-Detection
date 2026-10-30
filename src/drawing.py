"""Drawing and saving helpers shared by the steps."""

import csv

import cv2
import numpy as np

from step03_front_radius import largest_component


def feature_color(feature_type):
    """BGR colour: red for bumps, blue for dents."""
    return (0, 0, 255) if feature_type == "protrusion" else (255, 0, 0)


def map_feature_color(feature_type):
    """Dark red / dark blue, readable on top of the red/blue bump-dent map."""
    return (0, 0, 120) if feature_type == "protrusion" else (120, 0, 0)


def draw_component_contour(canvas, mask):
    """Draw the outline of the largest water region in thin black."""
    contours, _ = cv2.findContours(
        largest_component(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if contours:
        cv2.drawContours(canvas, [max(contours, key=cv2.contourArea)], -1,
                         (0, 0, 0), 1, lineType=cv2.LINE_AA)


def draw_dashed_polyline(canvas, points, color):
    for index in range(len(points) - 1):
        if index % 4 < 2:
            cv2.line(canvas, tuple(points[index]), tuple(points[index + 1]),
                     color, 1, cv2.LINE_AA)


def signed_map_image(positive, negative, clip_value):
    red = np.clip(positive / clip_value, 0.0, 1.0)
    blue = np.clip(negative / clip_value, 0.0, 1.0)
    image = np.full((*positive.shape, 3), 255, np.uint8)
    image[..., 0] = np.clip(255 * (1.0 - red), 0, 255).astype(np.uint8)
    image[..., 1] = np.clip(255 * (1.0 - np.maximum(red, blue)), 0, 255).astype(np.uint8)
    image[..., 2] = np.clip(255 * (1.0 - blue), 0, 255).astype(np.uint8)
    return image


def map_canvas(body, angles, frame_data, args):
    sx, sy = args.map_render_scale_x, args.map_render_scale_y
    body = cv2.resize(
        body, (body.shape[1] * sx, body.shape[0] * sy),
        interpolation=cv2.INTER_NEAREST,
    )
    left, top, right, bottom = 90, 42, 30, 58
    canvas = np.full(
        (body.shape[0] + top + bottom, body.shape[1] + left + right, 3),
        255, np.uint8,
    )
    canvas[top:top + body.shape[0], left:left + body.shape[1]] = body
    cv2.rectangle(canvas, (left, top),
                  (left + body.shape[1] - 1, top + body.shape[0] - 1),
                  (0, 0, 0), 1)
    for angle in (-80, -40, 0, 40, 80):
        index = int(np.argmin(np.abs(angles - angle)))
        x = left + index * sx
        cv2.putText(canvas, str(angle), (x - 14, top + body.shape[0] + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                    cv2.LINE_AA)
    for row in np.linspace(0, len(frame_data) - 1, 5).astype(int):
        y = top + row * sy
        cv2.putText(canvas, str(frame_data[row].frame_id), (12, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                    cv2.LINE_AA)
    cv2.putText(canvas, "angle (deg)",
                (left + body.shape[1] // 2 - 40, canvas.shape[0] - 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "frame", (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas, left, top, sx, sy


def draw_node_path(canvas, nodes, color, left, top, sx, sy):
    """Draw a chain of nodes (anything with .angle_index and .row) on a map canvas."""
    points = np.asarray(
        [(left + node.angle_index * sx, top + node.row * sy) for node in nodes],
        np.int32,
    ).reshape(-1, 1, 2)
    cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------- helpers for the per-step outputs

EXAMPLE_FRAME = 200     # running example of the report


def example_row(frame_data):
    """Row of the example frame (frame 200), or the middle row if it does not exist."""
    for data in frame_data:
        if data.frame_id == EXAMPLE_FRAME:
            return data.row
    return len(frame_data) // 2


def example_mask_path(mask_paths):
    for path in mask_paths:
        if path.stem.endswith(f"{EXAMPLE_FRAME:06d}"):
            return path
    return mask_paths[len(mask_paths) // 2]


def mask_background(mask):
    """Faded mask for drawing on: water light gray, background dark gray."""
    image = np.where(mask > 0, 200, 70).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def bump_dent_canvas(ctx):
    """The bump/dent map of step 6 as an image, with its axes."""
    _, positive, negative, _, _, _ = ctx.maps
    body = signed_map_image(positive, negative, ctx.args.map_color_clip)
    return map_canvas(body, ctx.angles, ctx.frame_data, ctx.args)


def outline_canvas(frame_data):
    """White image with the water outline of every frame, as in the result image."""
    from step03_front_radius import read_mask
    first = read_mask(frame_data[0].mask_path)
    canvas = np.full((*first.shape, 3), 255, np.uint8)
    for data in frame_data:
        draw_component_contour(canvas, read_mask(data.mask_path))
    return canvas
