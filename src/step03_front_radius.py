"""Step 3 — Front radius r(θ) (report, Section 3).

3.1  Keep only the largest 8-connected white region (removes noise dots).
3.2  On each ray, r(θ) = the farthest r whose sample is still water.
     A ray that hits no water gets r by linear interpolation from its
     neighbours and is marked as an interpolated ray (detected = False).

Run alone → results/step03_front_radius/:
    front_radius.csv        r(θ) and the front point (x, y) of every ray in every frame
    front_frame_200.png     the 681 front points of frame 200 on its mask
"""

import cv2
import numpy as np


def read_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read {mask_path}")
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def largest_component(mask):
    number, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if number <= 1:
        return np.zeros_like(mask)
    label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == label, 255, 0).astype(np.uint8)


def extract_front(component, sampler):
    """Return (radius, front points, detected) or None if fewer than 3 rays hit water."""
    angles, sample_x, sample_y, valid = sampler
    foreground = np.zeros_like(valid)
    foreground[valid] = component[sample_y[valid], sample_x[valid]] > 0
    radii = np.arange(sample_x.shape[1], dtype=np.int32)
    farthest = np.where(foreground, radii[None, :], -1).max(axis=1)
    detected = farthest >= 0
    if np.count_nonzero(detected) < 3:
        return None
    indices = np.arange(len(angles))
    radius = np.interp(indices, indices[detected], farthest[detected]).astype(
        np.float32
    )
    radius_index = np.clip(
        np.rint(radius).astype(np.int32), 0, sample_x.shape[1] - 1
    )
    points = np.column_stack(
        (sample_x[indices, radius_index], sample_y[indices, radius_index])
    ).astype(np.int32)
    return radius, points, detected


def save_outputs(ctx, out_dir):
    from drawing import EXAMPLE_FRAME, example_row, mask_background, write_csv

    write_csv(out_dir / "front_radius.csv",
              ["frame_id", "ray_index", "angle_deg", "radius_px", "x", "y", "real_ray"],
              [(data.frame_id, i, f"{ctx.angles[i]:.2f}", f"{data.radius[i]:.1f}",
                *data.front_points[i], int(data.detected[i]))
               for data in ctx.frame_data for i in range(len(ctx.angles))])
    data = ctx.frame_data[example_row(ctx.frame_data)]
    image = mask_background(read_mask(data.mask_path))
    cv2.polylines(image, [data.front_points.reshape(-1, 1, 2)], False,
                  (0, 220, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"front_frame_{EXAMPLE_FRAME}.png"), image)


if __name__ == "__main__":
    from pipeline import run_step_alone
    run_step_alone(__file__, 3, save_outputs)
