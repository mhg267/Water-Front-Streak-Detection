"""Step 2 — Origin and polar coordinates (report, Section 2).

The origin is the image centre (640, 320), at the nozzle. Rays are cast at
    θ_i = −85° + 0.25° · i,   i = 0 … 680,
and each ray is sampled every pixel along r:
    x = o_x + r cos θ,   y = o_y + r sin θ   (rounded to the nearest pixel).

Run alone → results/step02_origin_and_rays/:
    rays.csv              index and angle of every ray, and the origin
    origin_and_rays.png   the origin and one ray every 5° on the mask of frame 200
"""

import cv2
import numpy as np


def choose_origin(shape, origin_x=None, origin_y=None):
    """Image centre unless the user gives the origin."""
    height, width = shape
    return (origin_x if origin_x is not None else width // 2,
            origin_y if origin_y is not None else height // 2)


def build_sampler(shape, origin, angle_min, angle_max, number_of_angles,
                  dtype=np.float32):
    """Return the ray angles (degree) and the pixel samples of every ray."""
    height, width = shape
    angles_deg = np.linspace(
        angle_min, angle_max, number_of_angles, dtype=dtype
    )
    angles_rad = np.deg2rad(angles_deg)
    max_radius = int(np.ceil(np.hypot(width, height)))
    radii = np.arange(max_radius + 1, dtype=np.int32)
    sample_x = np.rint(
        origin[0] + np.cos(angles_rad)[:, None] * radii
    ).astype(np.int32)
    sample_y = np.rint(
        origin[1] + np.sin(angles_rad)[:, None] * radii
    ).astype(np.int32)
    valid = (
        (sample_x >= 0) & (sample_x < width)
        & (sample_y >= 0) & (sample_y < height)
    )
    return angles_deg, sample_x, sample_y, valid


def save_outputs(ctx, out_dir):
    from drawing import example_mask_path, mask_background, write_csv
    from step03_front_radius import read_mask

    angles, sample_x, sample_y, valid = ctx.sampler
    write_csv(out_dir / "rays.csv", ["ray_index", "angle_deg", "origin_x", "origin_y"],
              [(i, f"{a:.2f}", *ctx.origin) for i, a in enumerate(angles)])
    image = mask_background(read_mask(example_mask_path(ctx.mask_paths)))
    for index in range(0, len(angles), 20):
        last = np.flatnonzero(valid[index])[-1]
        cv2.line(image, ctx.origin, (int(sample_x[index, last]), int(sample_y[index, last])),
                 (0, 200, 255), 1, cv2.LINE_AA)
    cv2.drawMarker(image, ctx.origin, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    cv2.imwrite(str(out_dir / "origin_and_rays.png"), image)


if __name__ == "__main__":
    from pipeline import run_step_alone
    run_step_alone(__file__, 2, save_outputs)
