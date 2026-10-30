"""Step 4 — Smooth r(θ) (report, Section 4).

    r_fine(θ_i) = (1/7) Σ_{k=−3..3} r(θ_{i+k})

At the two ends the data is mirrored inward to borrow the missing neighbours.

Run alone → results/step04_smooth_radius/:
    smoothed_radius.csv     r(θ) and r_fine(θ) of every ray in every frame
    front_frame_200.png     frame 200: raw front (red) and smoothed front (green)
"""

import numpy as np


def odd_window(value, maximum):
    value = max(1, min(int(value), int(maximum)))
    if value % 2 == 0:
        value -= 1
    return max(1, value)


def moving_average(values, window):
    window = odd_window(window, len(values))
    if window == 1:
        return values.astype(np.float32, copy=True)
    padding = window // 2
    padded = np.pad(values, padding, mode="reflect")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(padded, kernel, mode="valid")


def save_outputs(ctx, out_dir):
    import cv2
    from drawing import EXAMPLE_FRAME, example_row, mask_background, write_csv
    from step03_front_radius import read_mask

    window = ctx.args.fine_window
    rows = []
    for data in ctx.frame_data:
        fine = moving_average(data.radius, window)
        rows += [(data.frame_id, i, f"{ctx.angles[i]:.2f}", f"{data.radius[i]:.1f}",
                  f"{fine[i]:.3f}") for i in range(len(ctx.angles))]
    write_csv(out_dir / "smoothed_radius.csv",
              ["frame_id", "ray_index", "angle_deg", "radius_px", "fine_px"], rows)

    data = ctx.frame_data[example_row(ctx.frame_data)]
    fine = moving_average(data.radius, window)
    theta = np.deg2rad(ctx.angles.astype(np.float64))
    smooth_points = np.column_stack((ctx.origin[0] + fine * np.cos(theta),
                                     ctx.origin[1] + fine * np.sin(theta)))
    image = mask_background(read_mask(data.mask_path))
    cv2.polylines(image, [data.front_points.reshape(-1, 1, 2)], False,
                  (0, 0, 255), 1, cv2.LINE_AA)
    cv2.polylines(image, [np.rint(smooth_points).astype(np.int32).reshape(-1, 1, 2)],
                  False, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"front_frame_{EXAMPLE_FRAME}.png"), image)


if __name__ == "__main__":
    from pipeline import run_step_alone
    run_step_alone(__file__, 4, save_outputs)
