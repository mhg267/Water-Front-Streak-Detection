"""Step 5 — Residual: how much each ray bulges or caves in (report, Section 5).

For each baseline window w ∈ {25, 41, 61}:
    b_w = moving average of r_fine over w rays
    e_w = r_fine − b_w        (positive = bump, negative = dent)
e_w is set to 0 on the (w − 1)/2 rays at each end and on interpolated rays.
The maximum over the three scales is taken later, after the map is smoothed (step 6).

Run alone → results/step05_residual/:
    residuals.csv           e_w of every ray in every frame, one column per window
    residual_frame_200.png  frame 200: front points coloured by e_41 (red = bump, blue = dent)
"""

import numpy as np

from step04_smooth_radius import moving_average


def parse_windows(text):
    windows = sorted({int(value.strip()) for value in text.split(",")})
    if not windows or any(value < 3 for value in windows):
        raise ValueError("baseline-windows must contain odd values >= 3")
    return [value if value % 2 else value - 1 for value in windows]


def compute_residuals(fine, detected, baseline_windows):
    """One residual curve per window, stacked into an array (windows × rays)."""
    residuals = []
    for window in baseline_windows:
        residual = (fine - moving_average(fine, window)).astype(np.float32)
        margin = window // 2
        residual[:margin] = 0.0
        residual[-margin:] = 0.0
        residual[~detected] = 0.0
        residuals.append(residual)
    return np.stack(residuals)


def save_outputs(ctx, out_dir):
    import cv2
    from drawing import EXAMPLE_FRAME, example_row, mask_background, write_csv
    from step03_front_radius import read_mask

    windows = ctx.baseline_windows
    write_csv(out_dir / "residuals.csv",
              ["frame_id", "ray_index", "angle_deg"] + [f"e_{w}" for w in windows],
              [(data.frame_id, i, f"{ctx.angles[i]:.2f}",
                *(f"{value:.3f}" for value in data.residuals[:, i]))
               for data in ctx.frame_data for i in range(len(ctx.angles))])

    data = ctx.frame_data[example_row(ctx.frame_data)]
    middle = len(windows) // 2
    image = mask_background(read_mask(data.mask_path))
    for (x, y), value in zip(data.front_points, data.residuals[middle]):
        strength = min(abs(float(value)) / 4.0, 1.0)
        shade = int(255 * (1.0 - strength))
        color = (shade, shade, 255) if value > 0 else (255, shade, shade)
        cv2.circle(image, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"residual_frame_{EXAMPLE_FRAME}.png"), image)


if __name__ == "__main__":
    from pipeline import run_step_alone
    run_step_alone(__file__, 5, save_outputs)
