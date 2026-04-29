#!/usr/bin/env python3
"""
Resize video files in a LeRobot v2 dataset to reduce vision token count during training.

Applies Qwen3-VL's smart_resize logic (with configurable max_pixels) to determine target
resolution, then re-encodes all MP4 videos via ffmpeg. Non-video files (parquet, metadata)
are copied as-is, and meta/info.json is updated with the new resolution.

Usage:
    python run_scripts/data/resize_dataset_videos.py \
        --dataset-path /path/to/real_robot/swap_cubes

    # Output: /path/to/real_robot/swap_cubes_resized
"""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── Qwen3-VL smart_resize logic ──────────────────────────────────────────────

PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
FACTOR = PATCH_SIZE * SPATIAL_MERGE_SIZE  # 28


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = FACTOR,
    min_pixels: int = 4 * FACTOR ** 2,
    max_pixels: int = 256 * 256,
) -> tuple[int, int]:
    """Qwen3-VL smart_resize: rescale while keeping dims divisible by factor."""
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


# ── Video probing ─────────────────────────────────────────────────────────────

def probe_video(path: Path) -> tuple[int, int, float]:
    """Return (width, height, fps) of a video file using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    for stream in info["streams"]:
        if stream["codec_type"] == "video":
            w = int(stream["width"])
            h = int(stream["height"])
            # Parse fps from r_frame_rate (e.g. "10/1")
            num, den = stream["r_frame_rate"].split("/")
            fps = float(num) / float(den)
            return w, h, fps
    raise ValueError(f"No video stream found in {path}")


# ── Video re-encoding ─────────────────────────────────────────────────────────

def resize_video(src: Path, dst: Path, target_w: int, target_h: int) -> None:
    """Re-encode a video to target_w x target_h using ffmpeg."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", f"scale={target_w}:{target_h}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",  # drop audio
        str(dst),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Resize LeRobot v2 dataset videos")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the source dataset")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Path for resized dataset (default: {dataset_path}_resized)")
    parser.add_argument("--max-pixels", type=int, default=256 * 256,
                        help="max_pixels for smart_resize (default: 65536 = 256*256)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel ffmpeg workers")
    args = parser.parse_args()

    src = Path(args.dataset_path)
    dst = Path(args.output_path) if args.output_path else Path(f"{args.dataset_path}_resized")

    if dst.exists():
        print(f"[!] Output path already exists: {dst}")
        print("    Remove it first or specify a different --output-path.")
        return

    print(f"Source:  {src}")
    print(f"Output:  {dst}")
    print(f"max_pixels: {args.max_pixels}")

    # ── 1. Copy non-video directories ────────────────────────────────────────
    for subdir in ["data", "meta"]:
        src_sub = src / subdir
        dst_sub = dst / subdir
        if src_sub.exists():
            print(f"Copying {subdir}/ ...")
            shutil.copytree(src_sub, dst_sub)

    # ── 2. Discover video files and compute target resolution ────────────────
    videos_dir = src / "videos"
    if not videos_dir.exists():
        print("[!] No videos/ directory found. Nothing to resize.")
        return

    # Probe one video to get source resolution
    sample_mp4 = next(videos_dir.rglob("*.mp4"), None)
    if sample_mp4 is None:
        print("[!] No MP4 files found in videos/")
        return

    src_w, src_h, fps = probe_video(sample_mp4)
    target_h, target_w = smart_resize(src_h, src_w, max_pixels=args.max_pixels)
    tokens_per_image = (target_h // FACTOR) * (target_w // FACTOR)

    print(f"\nSource resolution:  {src_w}×{src_h}")
    print(f"Target resolution:  {target_w}×{target_h}")
    print(f"Tokens per image:   {tokens_per_image}")
    print(f"FPS:                {fps}")

    # ── 3. Re-encode all videos ──────────────────────────────────────────────
    all_mp4s = sorted(videos_dir.rglob("*.mp4"))
    print(f"\nResizing {len(all_mp4s)} video files with {args.num_workers} workers...")

    def process_one(mp4_path: Path) -> str:
        rel = mp4_path.relative_to(src)
        dst_path = dst / rel
        resize_video(mp4_path, dst_path, target_w, target_h)
        return str(rel)

    done = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(process_one, p): p for p in all_mp4s}
        for future in as_completed(futures):
            done += 1
            rel = future.result()
            if done % 20 == 0 or done == len(all_mp4s):
                print(f"  [{done}/{len(all_mp4s)}] {rel}")

    # ── 4. Update meta/info.json ─────────────────────────────────────────────
    info_path = dst / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)

        for key, feat in info.get("features", {}).items():
            if feat.get("dtype") == "video":
                # Update shape: [C, H, W] or [H, W, C]
                shape = feat["shape"]
                if shape[0] == 3:  # [C, H, W]
                    feat["shape"] = [3, target_h, target_w]
                else:  # [H, W, C]
                    feat["shape"] = [target_h, target_w, 3]

                # Update info sub-fields
                if "info" in feat:
                    feat["info"]["video.height"] = target_h
                    feat["info"]["video.width"] = target_w
                    feat["info"]["video.codec"] = "h264"

        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

        print(f"\nUpdated {info_path}")

    print(f"\nDone! Resized dataset at: {dst}")


if __name__ == "__main__":
    main()
