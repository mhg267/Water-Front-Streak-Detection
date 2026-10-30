"""Step 6 — The angle × time map (report, Section 6).

6.1  Stack the residuals of all frames: E_w(t, i) = e_w of ray i in frame t.
6.2  Smooth max(E_w, 0) and max(−E_w, 0) with a 5 × 5 Gaussian (σ = 1), first along
     the angle then along time, averaging only over real data (normalized
     convolution, Knutsson & Westin 1993).
6.3  Keep the strongest scale in every cell → bump map P̂ and dent map N̂.

The start frame is the first frame where at least 50 rays have a signal ≥ 0.25 px.

Run alone → results/step06_angle_time_map/:
    angle_time_maps.npz   raw residuals, bump map P̂, dent map N̂, winning scale of each cell
    bump_dent_map.png     P̂ in red and N̂ in blue (rows = frames, columns = angles)
    start_frame.txt       the start frame
"""

import cv2
import numpy as np

from step04_smooth_radius import odd_window


def normalized_gaussian(values, valid, args):
    """Blur only over existing samples; no temporal reflection is created."""
    angle_window = odd_window(args.map_angle_window, values.shape[1])
    time_window = odd_window(args.map_time_window, values.shape[0])
    weighted = values.astype(np.float32) * valid.astype(np.float32)
    weights = valid.astype(np.float32)

    weighted = cv2.GaussianBlur(
        weighted, (angle_window, 1), sigmaX=args.map_angle_sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    weights = cv2.GaussianBlur(
        weights, (angle_window, 1), sigmaX=args.map_angle_sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    angle_smoothed = np.divide(
        weighted, weights, out=np.zeros_like(weighted), where=weights > 1e-6
    )

    temporal_weight = (weights > 1e-6).astype(np.float32)
    numerator = cv2.GaussianBlur(
        angle_smoothed * temporal_weight,
        (1, time_window), sigmaX=0.0, sigmaY=args.map_time_sigma,
        borderType=cv2.BORDER_CONSTANT,
    )
    denominator = cv2.GaussianBlur(
        temporal_weight,
        (1, time_window), sigmaX=0.0, sigmaY=args.map_time_sigma,
        borderType=cv2.BORDER_CONSTANT,
    )
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator),
        where=denominator > 1e-6,
    )


def build_multiscale_maps(frame_data, args):
    """Return the raw residuals, the bump/dent maps, the winning scale of each cell and the valid mask."""
    residual_stack = np.stack([data.residuals for data in frame_data])
    valid = np.stack([data.detected for data in frame_data])
    positive_scales = []
    negative_scales = []
    for scale in range(residual_stack.shape[1]):
        residual = residual_stack[:, scale]
        positive_scales.append(
            normalized_gaussian(np.maximum(residual, 0.0), valid, args)
        )
        negative_scales.append(
            normalized_gaussian(np.maximum(-residual, 0.0), valid, args)
        )
    positive_stack = np.stack(positive_scales)
    negative_stack = np.stack(negative_scales)
    positive = np.max(positive_stack, axis=0)
    negative = np.max(negative_stack, axis=0)
    positive_scale = np.argmax(positive_stack, axis=0).astype(np.uint8)
    negative_scale = np.argmax(negative_stack, axis=0).astype(np.uint8)
    positive[~valid] = 0.0
    negative[~valid] = 0.0
    return (
        residual_stack, positive, negative, positive_scale, negative_scale,
        valid,
    )


def find_active_start(positive, negative, args):
    support = np.count_nonzero(
        np.maximum(positive, negative) >= args.active_score, axis=1
    )
    rows = np.flatnonzero(support >= args.active_minimum_columns)
    return int(rows[0]) if len(rows) else 0


def save_outputs(ctx, out_dir):
    from drawing import bump_dent_canvas

    residuals, positive, negative, positive_scale, negative_scale, valid = ctx.maps
    np.savez_compressed(out_dir / "angle_time_maps.npz",
                        frame_ids=np.asarray([d.frame_id for d in ctx.frame_data]),
                        angles_deg=ctx.angles, residuals=residuals,
                        bump_map=positive, dent_map=negative,
                        bump_scale=positive_scale, dent_scale=negative_scale,
                        real_ray=valid)
    canvas, *_ = bump_dent_canvas(ctx)
    cv2.imwrite(str(out_dir / "bump_dent_map.png"), canvas)
    (out_dir / "start_frame.txt").write_text(
        f"{ctx.frame_data[ctx.active_start].frame_id}\n", encoding="utf-8")


if __name__ == "__main__":
    from pipeline import run_step_alone
    run_step_alone(__file__, 6, save_outputs)
