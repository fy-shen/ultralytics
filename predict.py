# coding=utf-8
from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from tools.motion import MOTION_TYPE_SPECS, VIDEO_SUFFIXES
from tools.motion.extractor import ExtractPipeline
from tools.motion.parser import add_types_args, format_motion_args, get_motion_kwargs_from_args

FRAME_NAME_PATTERN = re.compile(r"^(?P<video_name>.+)_(?P<frame_idx>\d+)$")
MOTION_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(description="Motion model inference for image/video sources")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")

    source_group = parser.add_argument_group("source args")
    source_group.add_argument("--source", type=str, default=None, help="Image/video file or directory")
    source_group.add_argument("--yaml", type=str, default=None, help="YOLO dataset yaml path")
    source_group.add_argument("--split", choices=("train", "val", "test", "minival"), default="val",
                              help="Dataset split used with --yaml", )

    output_group = parser.add_argument_group("output args")
    output_group.add_argument("--save-dir", type=str, default="runs/motion_predict",
                              help="Directory to save results")
    output_group.add_argument("--image-output", choices=("image", "video"), default="video",
                              help="Output format for image/image-folder mode", )
    output_group.add_argument("--image-video-fps", type=int, default=30, help="FPS when image mode output is video")

    infer_group = parser.add_argument_group("inference args")
    infer_group.add_argument("--imgsz", type=int, default=1280)
    infer_group.add_argument("--conf", type=float, default=0.25)
    infer_group.add_argument("--iou", type=float, default=0.7)
    infer_group.add_argument("--device", type=str, default=None)
    infer_group.add_argument("--line-width", type=int, default=None)
    infer_group.add_argument("--show-conf", action=argparse.BooleanOptionalAction, default=True)
    infer_group.add_argument("--show-labels", action=argparse.BooleanOptionalAction, default=True)
    infer_group.add_argument("--show-boxes", action=argparse.BooleanOptionalAction, default=True)
    infer_group.add_argument("--no-progress", action="store_true")

    motion_group = parser.add_argument_group("motion args")
    motion_group.add_argument("--motion-types", nargs="+", choices=tuple(MOTION_TYPE_SPECS), default=None,
                              help="Motion extractor types used for video mode; auto-select feature names/params", )
    motion_group.add_argument("--motion", nargs="+", default=None,
                              help="Explicit motion feature names. Overrides --motion-types selected features.", )
    add_types_args(parser, MOTION_TYPE_SPECS)

    return parser.parse_args()


def _is_video(path: Path):
    return path.suffix.lower() in VIDEO_SUFFIXES


def _is_image(path: Path):
    return path.suffix.lower().lstrip(".") in IMG_FORMATS


def _find_motion_image_path(rgb_path: Path, motion_name: str):
    candidates = [
        Path(str(rgb_path).replace("images", motion_name)),
        Path(str(rgb_path).replace("videos", motion_name)),
        rgb_path.parent.parent / motion_name / rgb_path.name,
        rgb_path.parent / motion_name / rgb_path.name,
    ]

    unique = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)

    for candidate in unique:
        if candidate.exists():
            return candidate
        for ext in MOTION_IMAGE_EXTS:
            alt = candidate.with_suffix(ext)
            if alt.exists():
                return alt
    tried = ", ".join(str(x) for x in unique)
    raise FileNotFoundError(f"Motion image '{motion_name}' not found for {rgb_path}. Tried: {tried}")


def _concat_with_motion(rgb: np.ndarray, rgb_path: Path, motion_names: list[str]):
    if not motion_names:
        return rgb
    motion_channels = []
    for motion_name in motion_names:
        motion_path = _find_motion_image_path(rgb_path, motion_name)
        m = cv2.imread(str(motion_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Failed to read motion image: {motion_path}")
        if m.shape[:2] != rgb.shape[:2]:
            m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        motion_channels.append(m[..., None])
    return np.concatenate([rgb] + motion_channels, axis=2)


def _predict_plot(model, merged: np.ndarray, args):
    results = model.predict(
        source=merged,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=False,
        verbose=False,
    )
    result = results[0]
    return result.plot(
        line_width=args.line_width,
        conf=args.show_conf,
        labels=args.show_labels,
        boxes=args.show_boxes,
    )


def _iter_ready_video_frames(video_path: Path, motion_names: list[str], motion_kwargs: dict):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_idx = -1
    if not motion_names:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                yield frame_idx, frame, fps, total_frames
        finally:
            cap.release()
        return

    pipeline = ExtractPipeline(motion_names, **motion_kwargs)
    pending = {}
    frame_cache = OrderedDict()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            frame_cache[frame_idx] = frame
            while len(frame_cache) > pipeline.max_frame_lag + 8:
                frame_cache.popitem(last=False)

            for target_idx, features in pipeline.update(frame, frame_idx):
                bucket = pending.setdefault(target_idx, {})
                bucket.update(features)

            ready_indices = sorted(idx for idx, b in pending.items() if all(name in b for name in motion_names))
            for target_idx in ready_indices:
                bucket = pending.pop(target_idx)
                rgb = frame_cache.pop(target_idx, None)
                if rgb is None:
                    continue
                merged = np.concatenate([rgb] + [bucket[name][..., None] for name in motion_names], axis=2)
                yield target_idx, merged, fps, total_frames
    finally:
        cap.release()


def _group_image_paths(image_paths: list[Path]):
    grouped = {}
    fallback_idx = 0
    for path in sorted(image_paths):
        m = FRAME_NAME_PATTERN.match(path.stem)
        if m:
            video_name = m.group("video_name")
            frame_idx = int(m.group("frame_idx"))
        else:
            video_name = path.stem
            frame_idx = fallback_idx
            fallback_idx += 1
        grouped.setdefault(video_name, []).append((frame_idx, path))
    return [(name, [p for _, p in sorted(items, key=lambda x: x[0])]) for name, items in sorted(grouped.items())]


def process_video(model, video_path: Path, motion_names: list[str], args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{video_path.stem}.mp4"

    writer = None
    motion_kwargs, _ = get_motion_kwargs_from_args(args, list(args.motion_types or []))

    cap_tmp = cv2.VideoCapture(str(video_path))
    total = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    cap_tmp.release()
    pbar = tqdm(total=total, desc=f"Predict {video_path.name}", unit="frame", disable=args.no_progress)
    written = 0

    for _, merged, src_fps, _ in _iter_ready_video_frames(video_path, motion_names, motion_kwargs):
        plotted = _predict_plot(model, merged, args)
        if writer is None:
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(src_fps),
                (plotted.shape[1], plotted.shape[0]),
            )
        writer.write(plotted)
        written += 1
        pbar.update(1)
        if written % 30 == 0:
            pbar.set_postfix(saved=written)

    pbar.close()
    if writer is not None:
        writer.release()
    return out_path


def process_image_group(model, group_name: str, image_paths: list[Path], motion_names: list[str], args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    disable_bar = args.no_progress

    if args.image_output == "video":
        out_path = save_dir / f"{group_name}.mp4"
        writer = None
        pbar = tqdm(image_paths, desc=f"Predict {group_name}", unit="frame", disable=disable_bar)
        for image_path in pbar:
            rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if rgb is None:
                raise FileNotFoundError(f"Failed to read image: {image_path}")
            merged = _concat_with_motion(rgb, image_path, motion_names)
            plotted = _predict_plot(model, merged, args)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(args.image_video_fps),
                    (plotted.shape[1], plotted.shape[0]),
                )
            writer.write(plotted)
        if writer is not None:
            writer.release()
        return [out_path]

    out_paths = []
    pbar = tqdm(image_paths, desc=f"Predict {group_name}", unit="image", disable=disable_bar)
    for image_path in pbar:
        rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        merged = _concat_with_motion(rgb, image_path, motion_names)
        plotted = _predict_plot(model, merged, args)
        out_path = save_dir / image_path.name
        cv2.imwrite(str(out_path), plotted)
        out_paths.append(out_path)
    return out_paths


def resolve_source(source: str):
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")

    if p.is_file():
        if _is_video(p):
            return [p], []
        if _is_image(p):
            return [], [(p.stem, [p])]
        raise ValueError(f"Unsupported source file: {p}")

    files = sorted(x for x in p.iterdir() if x.is_file())
    videos = [x for x in files if _is_video(x)]
    images = [x for x in files if _is_image(x)]
    image_groups = _group_image_paths(images) if images else []
    if not videos and not image_groups:
        raise ValueError(f"No supported image/video found in directory: {p}")
    return videos, image_groups


def _expand_list_file(list_path: Path):
    if list_path.suffix.lower() == ".txt":
        items = []
        for line in list_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = (list_path.parent / p).resolve()
            items.append(str(p))
        return items
    if list_path.suffix.lower() == ".csv":
        content = list_path.read_text(encoding="utf-8")
        items = []
        for raw in content.split(","):
            line = raw.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = (list_path.parent / p).resolve()
            items.append(str(p))
        return items
    return [str(list_path)]


def _dedup_paths(paths):
    out = []
    seen = set()
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        out.append(p)
        seen.add(key)
    return out


def _collect_media_from_entry(entry: Path):
    videos = []
    images = []

    if not entry.exists():
        raise FileNotFoundError(f"Source path does not exist: {entry}")

    if entry.is_file():
        if entry.suffix.lower() in {".txt", ".csv"}:
            nested_videos = []
            nested_images = []
            for x in _expand_list_file(entry):
                v, i = _collect_media_from_entry(Path(x))
                nested_videos.extend(v)
                nested_images.extend(i)
            return nested_videos, nested_images
        if _is_video(entry):
            return [entry], []
        if _is_image(entry):
            return [], [entry]
        return [], []

    # Directory: collect recursively to support nested dataset structures from yaml
    for f in entry.rglob("*"):
        if not f.is_file():
            continue
        if _is_video(f):
            videos.append(f)
        elif _is_image(f):
            images.append(f)
    return videos, images


def _collect_media_from_entries(entries):
    videos = []
    images = []
    for entry in entries:
        v, i = _collect_media_from_entry(Path(entry))
        videos.extend(v)
        images.extend(i)
    return _dedup_paths(sorted(videos)), _dedup_paths(sorted(images))


def resolve_sources(args):
    if bool(args.source) == bool(args.yaml):
        raise ValueError("Specify exactly one of --source or --yaml.")

    if args.source:
        videos, image_groups = resolve_source(args.source)
        return videos, image_groups, None

    data = check_det_dataset(args.yaml)
    split_value = data.get(args.split)
    if split_value is None and args.split == "val":
        split_value = data.get("test")
    if split_value is None:
        raise ValueError(f"Split '{args.split}' not found in dataset yaml: {args.yaml}")

    split_items = split_value if isinstance(split_value, list) else [split_value]
    source_items = []
    for item in split_items:
        source_items.extend(_expand_list_file(Path(item)))

    videos, images = _collect_media_from_entries(source_items)
    image_groups = _group_image_paths(images) if images else []
    return videos, image_groups, data


def main():
    args = parse_args()
    model = YOLO(args.model, task="motion")

    model_channels = getattr(getattr(model, "model", None), "yaml", {}).get("channels", 3)
    expected_motion_channels = int(model_channels) - 3
    auto_motion_names = []
    if args.motion_types:
        _, auto_motion_names = get_motion_kwargs_from_args(args, args.motion_types)
    motion_names = list(args.motion or auto_motion_names)

    if expected_motion_channels > 0:
        if not motion_names:
            raise ValueError(
                f"Model expects {expected_motion_channels} motion channels but --motion is empty."
            )
        if len(motion_names) != expected_motion_channels:
            raise ValueError(
                f"--motion={motion_names} count mismatch. Expected {expected_motion_channels} names for model channels={model_channels}."
            )
    elif motion_names:
        raise ValueError(
            f"Model channels={model_channels} expects pure RGB input, but motion features were provided: {motion_names}"
        )

    videos, image_groups, data = resolve_sources(args)
    if data is not None:
        print(f"[INFO] Dataset yaml: {args.yaml}")
        print(f"[INFO] Split: {args.split}, videos={len(videos)}, image_groups={len(image_groups)}")
    if args.motion_types:
        print(format_motion_args(args))
    elif motion_names:
        print(f"[motion args]\nmotion={motion_names}")

    outputs = []
    for video_path in videos:
        outputs.append(process_video(model, video_path, motion_names, args))
    for group_name, image_paths in image_groups:
        outputs.extend(process_image_group(model, group_name, image_paths, motion_names, args))

    print("[DONE] Saved outputs:")
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
