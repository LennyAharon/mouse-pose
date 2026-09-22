"""
Extract a fixed-length clip containing the most movement from a longer video.

Movement ("motion energy") is measured either from pose predictions (mean keypoint
displacement between frames, for keypoints above a likelihood threshold) or, when no
predictions are available, from raw pixel differences on a downsampled version of the
video. The output clip is always re-encoded to h264/yuv420p mp4, regardless of the
source codec/container, so downstream tools have one predictable target format.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def make_video_snippet(
    video_file: Path,
    out_dir: Path,
    preds_file: Path | None = None,
    clip_length: int = 30,
    likelihood_thresh: float = 0.9,
    crf: int = 23,
    preset: str = "medium",
) -> tuple[Path, int, float]:
    """Extract a `clip_length`-second clip containing the most movement.

    Parameters
    ----------
    video_file: path to the source video
    out_dir: directory to save the clip in (created if it doesn't exist)
    preds_file: csv of pose predictions; if given, movement is measured from keypoint
        displacement instead of raw pixels
    clip_length: length of the clip in seconds
    likelihood_thresh: when using preds_file, only count keypoints with a likelihood
        above this threshold (0-1) toward the movement measure
    crf: h264 constant rate factor (lower = higher quality/larger file)
    preset: ffmpeg x264 preset (encode speed vs. compression efficiency tradeoff)

    Returns
    -------
    - path to the output clip
    - clip start: frame index
    - clip start: seconds

    """
    video = cv2.VideoCapture(str(video_file))
    fps = video.get(cv2.CAP_PROP_FPS)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    win_len = int(fps * clip_length)

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{video_file.stem}.short.mp4"

    encode_flags = [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", preset,
    ]

    if win_len >= n_frames:
        # video is already shorter than the requested clip length -- keep it all, just
        # re-encode so the output format is still guaranteed h264/yuv420p/mp4
        clip_start_idx = 0
        clip_start_sec = 0.0
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(video_file), *encode_flags, str(dst)]
    else:
        # find the `clip_length`-second window with the highest average motion energy
        if preds_file is None:
            me = compute_video_motion_energy(video_file)
        else:
            df = pd.read_csv(preds_file, header=[0, 1, 2], index_col=0)
            me = compute_motion_energy_from_prediction_df(df, likelihood_thresh)
        me_win = pd.Series(me).rolling(window=win_len, center=False).mean()
        # rolling() places each result at the window's right edge; shift back to the start
        clip_start_idx = int(me_win.argmax() - win_len)
        clip_start_sec = clip_start_idx / fps
        if np.isnan(clip_start_sec) or clip_start_sec < 0:
            # all predictions were below likelihood_thresh -- fall back to the start
            clip_start_idx, clip_start_sec = 0, 0.0
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-ss", str(clip_start_sec),
            "-i", str(video_file),
            "-t", str(clip_length),
            *encode_flags,
            str(dst),
        ]

    if not dst.exists():
        subprocess.run(ffmpeg_cmd, check=True)

    return dst, clip_start_idx, float(clip_start_sec)


def compute_video_motion_energy(
    video_file: Path,
    resize_dims: int = 32,
) -> np.ndarray:
    """Per-frame motion energy: summed absolute pixel difference from the prior frame."""
    frames = read_nth_frames(video_file=video_file, n=1, resize_dims=resize_dims)
    batches = frames.reshape(frames.shape[0], -1)

    diffs = np.concatenate([np.zeros((1, batches.shape[1])), np.diff(batches, axis=0)])
    me = np.sum(np.abs(diffs), axis=1)

    return me


def compute_motion_energy_from_prediction_df(
    df: pd.DataFrame,
    likelihood_thresh: float,
) -> np.ndarray:
    """Per-frame motion energy: mean keypoint displacement from the prior frame."""
    kps_and_conf = df.to_numpy().reshape(df.shape[0], -1, 3)
    kps = kps_and_conf[:, :, :2]
    conf = kps_and_conf[:, :, -1]
    conf2 = np.concatenate([conf[:, :, None], conf[:, :, None]], axis=2)

    kps[conf2 < likelihood_thresh] = np.nan

    me = np.nanmean(np.linalg.norm(kps[1:] - kps[:-1], axis=2), axis=-1)
    me = np.concatenate([[0], me])
    return me


def read_nth_frames(
    video_file: Path,
    n: int = 1,
    resize_dims: int = 64,
) -> np.ndarray:
    """Read every nth frame of a video, resized to (resize_dims, resize_dims), as RGB."""
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise OSError(f"Error opening video file {video_file}")

    frames = []
    frame_counter = 0
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=frame_total, desc=f"scanning {video_file.name}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_counter % n == 0:
                frame_resize = cv2.resize(frame, (resize_dims, resize_dims))
                frame_rgb = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb.astype(np.float16))
            frame_counter += 1
            pbar.update(1)

    cap.release()

    return np.array(frames)
