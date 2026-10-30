"""Step 1 — Create masks from the video (report, Section 1).

Each colour frame is converted to grayscale,
    I = 0.299 R + 0.587 G + 0.114 B,
and every pixel brighter than the threshold becomes water:
    mask = 255 (water) if I > 15, else 0 (background).
One mask per frame is saved as mask_000001.png, mask_000002.png, …

The masks are written to results/step01_create_masks/, where the later steps read
them. Running the whole project (python run.py) or any later step creates them when
that folder has no masks yet, or again with --recreate-masks. Running this file
alone always recreates them:
    python src/step01_create_masks.py
"""

from pathlib import Path

import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_DIR / "data" / "original_video.mp4"
STEP_NAME = Path(__file__).stem                  # step01_create_masks
THRESHOLD = 15


def create_masks(video_path, mask_dir, threshold=THRESHOLD):
    """Write one binary mask per frame of the video; return the number of masks."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open the video {video_path}")
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            mask_path = mask_dir / f"mask_{frame_count:06d}.png"
            if not cv2.imwrite(str(mask_path), mask):
                raise RuntimeError(f"Cannot save the mask {mask_path}")
    finally:
        capture.release()
    return frame_count


def save_outputs(ctx, out_dir):
    """Run alone: (re)create the masks in results/step01_create_masks/."""
    if not ctx.masks_created:
        count = create_masks(ctx.args.video, ctx.args.mask_dir)
        print(f"{count} masks")


if __name__ == "__main__":
    from pipeline import run_step_alone
    run_step_alone(__file__, 1, save_outputs)
