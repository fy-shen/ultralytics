# coding=utf-8
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v", ".mpg", ".mpeg"}


def parse_device(value: str):
    if value.lower() == "cpu":
        return "cpu"
    if "," in value:
        return [int(x) for x in value.split(",") if x.strip()]
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ordinary YOLO detection on one video or a directory of videos.")
    parser.add_argument("--model", required=True, type=Path, help="Path to YOLO .pt model.")
    parser.add_argument("--source", "--input", dest="source", required=True, type=Path, help="Video file or video directory.")
    parser.add_argument("--output-dir", "--save-dir", dest="output_dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
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
    parser.add_argument("--suffix", default="_det", help="Output filename suffix before .mp4.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output videos.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--verbose", action="store_true", help="Enable Ultralytics per-frame verbose output.")
    parser.add_argument("--show-conf", action=argparse.BooleanOptionalAction, default=True, help="Draw confidence scores.")
    parser.add_argument("--show-labels", action=argparse.BooleanOptionalAction, default=True, help="Draw class labels.")
    parser.add_argument("--show-boxes", action=argparse.BooleanOptionalAction, default=True, help="Draw boxes.")
    parser.add_argument("--line-width", type=int, default=None, help="Annotation line width.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames per video, useful for smoke tests.")
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


def predict_one_video(model, video_path: Path, output_path: Path, args) -> tuple[int, float]:
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
        print(f"[START] {video_path}")
        print(f"[SAVE ] {output_path}")

        with tqdm(
            total=total_frames,
            desc=video_path.name,
            unit="frame",
            dynamic_ncols=True,
            disable=args.no_progress,
        ) as pbar:
            while True:
                if args.max_frames is not None and frame_count >= args.max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break

                t0 = time.perf_counter()
                results = model.predict(
                    source=frame,
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
                pbar.update(1)

    finally:
        cap.release()
        if writer is not None:
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

    model = YOLO(str(args.model), task="detect")
    print(f"model : {args.model}")
    print(f"source: {args.source}")
    print(f"output: {args.output_dir}")
    print(f"videos: {len(videos)}")
    print(f"classes: {'all' if args.classes == [] else args.classes}")

    total_frames = 0
    total_infer_seconds = 0.0
    for video_path in videos:
        output_path = make_output_path(video_path, args.output_dir, args.suffix)
        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] output exists: {output_path}")
            continue

        frame_count, infer_seconds = predict_one_video(model, video_path, output_path, args)
        total_frames += frame_count
        total_infer_seconds += infer_seconds

    if total_frames:
        avg_ms = total_infer_seconds / total_frames * 1000.0
        infer_fps = total_frames / total_infer_seconds if total_infer_seconds > 0 else 0.0
        print(f"[ALL  ] {total_frames} frames, {avg_ms:.2f} ms/frame, {infer_fps:.2f} infer FPS")


if __name__ == "__main__":
    main()
