# coding=utf-8
from __future__ import annotations

import argparse
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tools.motion import MOTION_TYPE_SPECS
from tools.motion.extractor import FEATURE_EXTRACTOR_REGISTRY, ExtractPipeline
from tools.motion.parser import MOTION_TYPE_ALIASES, add_types_args


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v", ".mpg", ".mpeg"}


def parse_device(value: str):
    if value.lower() == "cpu":
        return "cpu"
    if "," in value:
        return [int(x) for x in value.split(",") if x.strip()]
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Run motion YOLO detection on one video or a directory of videos.")
    parser.add_argument("--model", required=True, type=Path, help="Path to motion YOLO .pt model.")
    parser.add_argument("--source", "--input", dest="source", required=True, type=Path, help="Video file or video directory.")
    parser.add_argument("--output-dir", "--save-dir", dest="output_dir", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--motion",
        nargs="+",
        required=True,
        help=(
            "Motion feature names or motion types. Examples: "
            "gray_diff_enhanced_short gray_diff_enhanced_long, or gde."
        ),
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--device", default="0", help="Device, e.g. 0, 0,1, or cpu.")
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=[0],
        help="Class IDs to keep. Defaults to 0. Use '--classes' with no values to keep all classes.",
    )
    parser.add_argument("--suffix", default="_motion_det", help="Output filename suffix before .mp4.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output videos.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--verbose", action="store_true", help="Enable Ultralytics per-frame verbose output.")
    parser.add_argument("--show-conf", action=argparse.BooleanOptionalAction, default=True, help="Draw confidence scores.")
    parser.add_argument("--show-labels", action=argparse.BooleanOptionalAction, default=True, help="Draw class labels.")
    parser.add_argument("--show-boxes", action=argparse.BooleanOptionalAction, default=True, help="Draw boxes.")
    parser.add_argument("--line-width", type=int, default=None, help="Annotation line width.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N output frames per video.")
    add_types_args(parser, MOTION_TYPE_SPECS)
    return parser.parse_args()


def collect_videos(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in VIDEO_EXTS:
        return [source]
    if not source.exists():
        raise FileNotFoundError(f"input path not found: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"input path is not a video file or directory: {source}")
    return [p for p in sorted(source.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]


def make_output_path(video_path: Path, output_dir: Path, suffix: str) -> Path:
    return output_dir / f"{video_path.stem}{suffix}.mp4"


def resolve_motion_selection(items: list[str]) -> tuple[list[str], list[str]]:
    feature_names = []
    motion_types = []
    seen_features = set()
    seen_types = set()

    for item in items:
        motion_type = MOTION_TYPE_ALIASES.get(item, item)
        if motion_type in MOTION_TYPE_SPECS:
            if motion_type not in seen_types:
                motion_types.append(motion_type)
                seen_types.add(motion_type)
            for feature in MOTION_TYPE_SPECS[motion_type]["feature_names"]:
                if feature not in seen_features:
                    feature_names.append(feature)
                    seen_features.add(feature)
            continue

        extractor_cls = FEATURE_EXTRACTOR_REGISTRY.get(item)
        if extractor_cls is None:
            supported = sorted(set(MOTION_TYPE_ALIASES) | set(MOTION_TYPE_SPECS) | set(FEATURE_EXTRACTOR_REGISTRY))
            raise ValueError(f"Unsupported --motion item '{item}'. Supported values: {supported}")
        motion_type = extractor_cls.motion_type
        if motion_type not in seen_types:
            motion_types.append(motion_type)
            seen_types.add(motion_type)
        if item not in seen_features:
            feature_names.append(item)
            seen_features.add(item)

    return feature_names, motion_types


def collect_motion_kwargs(args, motion_types: list[str]) -> dict:
    kwargs = {}
    seen_dests = set()
    for motion_type in motion_types:
        for arg_spec in MOTION_TYPE_SPECS[motion_type]["args"]:
            dest = arg_spec["dest"]
            if dest in seen_dests:
                continue
            kwargs[dest] = getattr(args, dest)
            seen_dests.add(dest)
    return kwargs


def model_input_channels(model) -> int:
    yaml = getattr(getattr(model, "model", None), "yaml", {}) or {}
    return int(yaml.get("channels", 3))


def check_motion_channels(model, feature_names: list[str]):
    expected = model_input_channels(model)
    actual = 3 + len(feature_names)
    if expected != actual:
        raise ValueError(
            f"Model expects {expected} input channels, but selected motion features produce {actual} "
            f"channels (RGB + {feature_names})."
        )


def open_writer(video_path: Path, output_path: Path, cap):
    import cv2

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25.0
    if width <= 0 or height <= 0:
        raise RuntimeError(f"failed to read video size: {video_path}")
    if total_frames <= 0:
        total_frames = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to create output video: {output_path}")

    return writer, total_frames


def iter_motion_frames(video_path: Path, feature_names: list[str], motion_kwargs: dict, max_outputs: int | None = None):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    pipeline = ExtractPipeline(feature_names, **motion_kwargs)
    frame_cache = OrderedDict()
    pending = {}
    frame_idx = -1
    yielded = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            frame_cache[frame_idx] = frame
            while len(frame_cache) > pipeline.max_frame_lag + 16:
                frame_cache.popitem(last=False)

            for target_idx, features in pipeline.update(frame, frame_idx):
                bucket = pending.setdefault(target_idx, {})
                bucket.update(features)

            ready_indices = sorted(
                idx for idx, bucket in pending.items() if all(name in bucket for name in feature_names)
            )
            for target_idx in ready_indices:
                rgb = frame_cache.pop(target_idx, None)
                bucket = pending.pop(target_idx)
                if rgb is None:
                    continue
                merged = np.concatenate([rgb] + [bucket[name][..., None] for name in feature_names], axis=2)
                yield target_idx, merged
                yielded += 1
                if max_outputs is not None and yielded >= max_outputs:
                    return
    finally:
        cap.release()


def predict_one_video(model, video_path: Path, output_path: Path, args, feature_names: list[str], motion_kwargs: dict):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[SKIP] open failed: {video_path}")
        return 0, 0.0

    writer = None
    frame_count = 0
    infer_seconds = 0.0
    classes = args.classes if args.classes else None

    try:
        writer, total_frames = open_writer(video_path, output_path, cap)
    finally:
        cap.release()

    print(f"[START] {video_path}")
    print(f"[SAVE ] {output_path}")

    try:
        with tqdm(
            total=total_frames,
            desc=video_path.name,
            unit="frame",
            dynamic_ncols=True,
            disable=args.no_progress,
        ) as pbar:
            previous_target_idx = -1
            for target_idx, merged in iter_motion_frames(video_path, feature_names, motion_kwargs, args.max_frames):
                t0 = time.perf_counter()
                results = model.predict(
                    source=merged,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    classes=classes,
                    device=parse_device(args.device),
                    verbose=args.verbose,
                    save=False,
                )
                infer_seconds += time.perf_counter() - t0

                annotated = results[0].plot(
                    conf=args.show_conf,
                    labels=args.show_labels,
                    boxes=args.show_boxes,
                    line_width=args.line_width,
                )
                writer.write(annotated)
                frame_count += 1

                pbar.update(max(target_idx - previous_target_idx, 1))
                previous_target_idx = target_idx
    finally:
        writer.release()

    if frame_count:
        avg_ms = infer_seconds / frame_count * 1000.0
        infer_fps = frame_count / infer_seconds if infer_seconds > 0 else 0.0
        print(f"[DONE ] {video_path.name}: {frame_count} frames, {avg_ms:.2f} ms/frame, {infer_fps:.2f} infer FPS")

    return frame_count, infer_seconds


def main():
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"model not found: {args.model}")

    videos = collect_videos(args.source)
    if not videos:
        raise RuntimeError(f"no video files found in: {args.source}")

    from ultralytics import YOLO

    feature_names, motion_types = resolve_motion_selection(args.motion)
    motion_kwargs = collect_motion_kwargs(args, motion_types)
    model = YOLO(str(args.model), task="motion")
    check_motion_channels(model, feature_names)

    print(f"model   : {args.model}")
    print(f"source  : {args.source}")
    print(f"output  : {args.output_dir}")
    print(f"videos  : {len(videos)}")
    print(f"features: {feature_names}")
    print(f"classes : {'all' if args.classes == [] else args.classes}")

    total_frames = 0
    total_infer_seconds = 0.0
    for video_path in videos:
        output_path = make_output_path(video_path, args.output_dir, args.suffix)
        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] output exists: {output_path}")
            continue

        frame_count, infer_seconds = predict_one_video(model, video_path, output_path, args, feature_names, motion_kwargs)
        total_frames += frame_count
        total_infer_seconds += infer_seconds

    if total_frames:
        avg_ms = total_infer_seconds / total_frames * 1000.0
        infer_fps = total_frames / total_infer_seconds if total_infer_seconds > 0 else 0.0
        print(f"[ALL  ] {total_frames} frames, {avg_ms:.2f} ms/frame, {infer_fps:.2f} infer FPS")


if __name__ == "__main__":
    main()
