# coding=utf-8
from __future__ import annotations

import argparse
import re
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from tools.motion import MOTION_TYPE_SPECS, VIDEO_SUFFIXES
from tools.motion.extractor import ExtractPipeline

FRAME_NAME_PATTERN = re.compile(r"^(?P<video_name>.+)_(?P<frame_idx>\d+)$")
MOTION_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _add_motion_param_args(parser: argparse.ArgumentParser):
    seen_dests = set()
    for motion_type, spec in MOTION_TYPE_SPECS.items():
        group = parser.add_argument_group(f"{motion_type} params", spec["description"])
        for arg in spec["args"]:
            if arg["dest"] in seen_dests:
                continue
            group.add_argument(*arg["flags"], dest=arg["dest"], **arg["kwargs"])
            seen_dests.add(arg["dest"])


def parse_args():
    parser = argparse.ArgumentParser(description="Motion model inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input path: image/video file, image/video folder, or yolo dataset yaml",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
        help="Dataset split used when --source is a yaml",
    )
    parser.add_argument("--save-dir", type=str, default="runs/motion_predict", help="Directory to save outputs")
    parser.add_argument(
        "--motion",
        nargs="+",
        default=None,
        help=("Motion selection list. Item can be a motion type (e.g. gray_diff) or a feature name "
              "(e.g. gray_diff_short). If omitted, predictor runs in pure RGB detect mode (task=detect)."),
    )
    _add_motion_param_args(parser)

    parser.add_argument("--image-output", choices=("image", "video"), default="video")
    parser.add_argument("--image-video-fps", type=int, default=25)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs. Default behavior is to skip existing results.",
    )
    parser.add_argument("--video-workers", type=int, default=4, help="Parallel workers for multi-video/group input")

    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--line-width", type=int, default=None)
    parser.add_argument("--show-conf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-boxes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.video_workers < 1:
        raise ValueError("--video-workers must be >= 1")
    return args


class MotionPredictRunner:
    def __init__(self, args):
        self.args = args
        self.input_mode = "empty"
        self.videos = []
        self.image_groups = []
        self.data_info = None

        # 确定模型模式
        self.rgb_detect_mode = self.args.motion is None
        self.model_task = "detect" if self.rgb_detect_mode else "motion"
        self.model = YOLO(self.args.model, task=self.model_task)

        if self.rgb_detect_mode:
            self.motion_names = []
            self.used_motion_types = []
            self.motion_kwargs = {}
        else:
            model_channels = int(getattr(getattr(self.model, "model", None), "yaml", {}).get("channels", 3))
            self.motion_names, self.used_motion_types = self._resolve_motion_selection(model_channels - 3)
            self.motion_kwargs = self._collect_motion_kwargs()

    @staticmethod
    def _is_video(path: Path):
        return path.suffix.lower() in VIDEO_SUFFIXES

    @staticmethod
    def _is_image(path: Path):
        return path.suffix.lower().lstrip(".") in IMG_FORMATS

    @staticmethod
    def _parse_frame_name(path: Path):
        m = FRAME_NAME_PATTERN.match(path.stem)
        if m:
            return m.group("video_name"), int(m.group("frame_idx"))
        return path.stem, 0

    def _log_line(self, msg: str):
        if self.args.no_progress:
            print(msg)
        else:
            tqdm.write(msg)

    def _resolve_motion_selection(self, expected_motion_channels: int):
        feature_to_type = {}
        all_type_names = list(MOTION_TYPE_SPECS)
        for motion_type, spec in MOTION_TYPE_SPECS.items():
            for f in spec["feature_names"]:
                feature_to_type[f] = motion_type
        supported = sorted(all_type_names + list(feature_to_type))

        selected = list(self.args.motion or [])
        feature_names = []
        used_types = []
        seen_feature = set()
        seen_type = set()

        # 确认使用的 motion_types 或 feature_names
        for item in selected:
            if item in MOTION_TYPE_SPECS:
                if item not in seen_type:
                    used_types.append(item)
                    seen_type.add(item)
                for feature in MOTION_TYPE_SPECS[item]["feature_names"]:
                    if feature not in seen_feature:
                        feature_names.append(feature)
                        seen_feature.add(feature)
                continue

            if item in feature_to_type:
                motion_type = feature_to_type[item]
                if motion_type not in seen_type:
                    used_types.append(motion_type)
                    seen_type.add(motion_type)
                if item not in seen_feature:
                    feature_names.append(item)
                    seen_feature.add(item)
                continue

            raise ValueError(f"Unsupported --motion item '{item}'. Supported values: {supported}")

        # 确认使用特征与模型通道是否匹配
        if expected_motion_channels > 0:
            if not feature_names:
                raise ValueError(f"Model expects {expected_motion_channels} motion channels. Please set --motion.")
            if len(feature_names) != expected_motion_channels:
                raise ValueError(
                    f"Selected motion features={feature_names}, but model expects {expected_motion_channels} channels."
                )
        elif feature_names:
            raise ValueError(f"RGB model does not accept motion features: {feature_names}")

        return feature_names, used_types

    def _collect_motion_kwargs(self):
        kwargs = {}
        for motion_type in self.used_motion_types:
            for arg_spec in MOTION_TYPE_SPECS[motion_type]["args"]:
                kwargs[arg_spec["dest"]] = getattr(self.args, arg_spec["dest"])
        return kwargs

    # 阶段 3：输入源解析（视频 / 图像组 / yaml）
    def _read_list_file(self, path: Path):
        if path.suffix.lower() == ".txt":
            lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        elif path.suffix.lower() == ".csv":
            lines = [x.strip() for x in path.read_text(encoding="utf-8").split(",") if x.strip()]
        else:
            return [path]
        out = []
        for line in lines:
            p = Path(line)
            if not p.is_absolute():
                p = (path.parent / p).resolve()
            out.append(p)
        return out

    def _collect_dataset_images_from_yaml(self, yaml_path: str):
        data = check_det_dataset(yaml_path)
        split_value = data.get(self.args.split)
        if split_value is None and self.args.split == "val":
            split_value = data.get("test")
        if split_value is None:
            raise ValueError(f"Split '{self.args.split}' not found in dataset yaml: {yaml_path}")

        items = split_value if isinstance(split_value, list) else [split_value]
        raw_paths = []
        for item in items:
            raw_paths.extend(self._read_list_file(Path(item)))

        images = []
        for p in raw_paths:
            if not p.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {p}")
            if p.is_file():
                if self._is_image(p):
                    images.append(p)
                elif p.suffix.lower() in {".txt", ".csv"}:
                    for nested in self._read_list_file(p):
                        if self._is_image(nested):
                            images.append(nested)
                continue
            for f in p.rglob("*"):
                if f.is_file() and self._is_image(f):
                    images.append(f)
        return data, sorted(images)

    def _build_image_groups(self, image_paths: list[Path], strict_duplicate_prefix: bool):
        grouped = defaultdict(list)
        prefix_dirs = defaultdict(set)
        prefix_images = defaultdict(list)

        for path in sorted(image_paths):
            video_name, frame_idx = self._parse_frame_name(path)
            grouped[video_name].append((frame_idx, path))
            prefix_dirs[video_name].add(str(path.parent.resolve()))
            prefix_images[video_name].append(str(path))

        if strict_duplicate_prefix:
            collisions = {k: sorted(v) for k, v in prefix_dirs.items() if len(v) > 1}
            if collisions:
                lines = ["Detected duplicated video prefixes across different directories:"]
                for name, dirs in sorted(collisions.items()):
                    lines.append(f"- {name}:")
                    for d in dirs:
                        lines.append(f"  {d}")
                    lines.append("  duplicated images:")
                    for p in prefix_images[name]:
                        lines.append(f"    {p}")
                raise ValueError("\n".join(lines))

        groups = []
        for name, items in sorted(grouped.items()):
            items_sorted = sorted(items, key=lambda x: x[0])
            idx_set = set()
            duplicates = []
            for frame_idx, path in items_sorted:
                if frame_idx in idx_set:
                    duplicates.append(str(path))
                idx_set.add(frame_idx)
            if duplicates:
                raise ValueError(f"Duplicated frame index in group '{name}': {duplicates}")
            groups.append((name, [p for _, p in items_sorted]))
        return groups

    def resolve_input(self):
        src = Path(self.args.source)
        if not src.exists():
            raise FileNotFoundError(f"Source does not exist: {self.args.source}")

        # YOLO 格式 yaml 数据
        if src.is_file() and src.suffix.lower() in {".yaml", ".yml"}:
            data, images = self._collect_dataset_images_from_yaml(str(src))
            self.input_mode = "image"
            self.videos = []
            self.image_groups = self._build_image_groups(images, strict_duplicate_prefix=True)
            self.data_info = data
            return

        # 图像或视频文件
        if src.is_file():
            if self._is_video(src):
                self.input_mode = "video"
                self.videos = [src]
                self.image_groups = []
                self.data_info = None
                return
            if self._is_image(src):
                self.input_mode = "image"
                self.videos = []
                self.image_groups = self._build_image_groups([src], strict_duplicate_prefix=False)
                self.data_info = None
                return
            raise ValueError(f"Unsupported source file: {src}")

        # 图像或视频文件夹
        files = sorted(x for x in src.iterdir() if x.is_file())
        videos = [x for x in files if self._is_video(x)]
        images = [x for x in files if self._is_image(x)]

        if videos and images:
            raise ValueError(f"Mixed image and video files found in {src}. Please keep one source type per run.")
        if videos:
            self.input_mode = "video"
            self.videos = videos
            self.image_groups = []
            self.data_info = None
            return
        if images:
            self.input_mode = "image"
            self.videos = []
            self.image_groups = self._build_image_groups(images, strict_duplicate_prefix=False)
            self.data_info = None
            return

        print(f"[WARN] No image/video files found in directory: {src}")
        self.input_mode = "empty"
        self.videos = []
        self.image_groups = []
        self.data_info = None

    def print_plan(self):
        print(f"[INFO] input_mode={self.input_mode}")
        print(f"[INFO] selected_motion_features={self.motion_names}")

        if self.data_info is not None:
            print(f"[INFO] Dataset yaml: {self.args.source}")
            print(f"[INFO] Split: {self.args.split}, image_groups={len(self.image_groups)}")
        if self.rgb_detect_mode:
            print("[INFO] run_mode=rgb_detect (task=detect)")

        if self.input_mode == "video" and self.motion_names:
            print("[INFO] extraction params:")
            for motion_type in self.used_motion_types:
                print(f"  [{motion_type}]")
                for arg_spec in MOTION_TYPE_SPECS[motion_type]["args"]:
                    dest = arg_spec["dest"]
                    print(f"    {dest}={getattr(self.args, dest)}")

    # 阶段 4：单帧融合和预测
    def _find_motion_image_path(self, rgb_path: Path, motion_name: str):
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

    def _concat_with_motion(self, rgb: np.ndarray, rgb_path: Path):
        if not self.motion_names:
            return rgb
        motion_channels = []
        for motion_name in self.motion_names:
            motion_path = self._find_motion_image_path(rgb_path, motion_name)
            motion_img = cv2.imread(str(motion_path), cv2.IMREAD_GRAYSCALE)
            if motion_img is None:
                raise FileNotFoundError(f"Failed to read motion image: {motion_path}")
            if motion_img.shape[:2] != rgb.shape[:2]:
                motion_img = cv2.resize(motion_img, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            motion_channels.append(motion_img[..., None])
        return np.concatenate([rgb] + motion_channels, axis=2)

    def _predict_plot(self, model, merged: np.ndarray):
        results = model.predict(
            source=merged,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            iou=self.args.iou,
            device=self.args.device,
            save=False,
            verbose=False,
        )
        return results[0].plot(
            line_width=self.args.line_width,
            conf=self.args.show_conf,
            labels=self.args.show_labels,
            boxes=self.args.show_boxes,
        )

    # 阶段 5：视频帧流读取（含在线 motion 特征生成）
    def _iter_ready_video_frames(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_idx = -1

        if not self.motion_names:
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

        pipeline = ExtractPipeline(self.motion_names, **self.motion_kwargs)
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

                ready_indices = sorted(
                    idx for idx, bucket in pending.items() if all(name in bucket for name in self.motion_names)
                )
                for target_idx in ready_indices:
                    bucket = pending.pop(target_idx)
                    rgb = frame_cache.pop(target_idx, None)
                    if rgb is None:
                        continue
                    merged = np.concatenate([rgb] + [bucket[name][..., None] for name in self.motion_names], axis=2)
                    yield target_idx, merged, fps, total_frames
        finally:
            cap.release()

    # 阶段 6：保存输出（视频 / 图像）
    def _process_video(self, model, video_path: Path, position: int | None = None, leave: bool = True):
        save_dir = Path(self.args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{video_path.stem}.mp4"
        if out_path.exists() and not self.args.overwrite:
            self._log_line(f"[SKIP] output exists: {out_path}")
            return out_path

        writer = None
        cap_tmp = cv2.VideoCapture(str(video_path))
        total = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        cap_tmp.release()
        pbar = tqdm(
            total=total,
            desc=f"Predict {video_path.name[:40]:<40}",
            unit="frame",
            position=position,
            leave=leave,
            disable=self.args.no_progress,
        )
        try:
            for _, merged, src_fps, _ in self._iter_ready_video_frames(video_path):
                plotted = self._predict_plot(model, merged)
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(out_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(src_fps),
                        (plotted.shape[1], plotted.shape[0]),
                    )
                writer.write(plotted)
                pbar.update(1)
        finally:
            pbar.close()
            if writer is not None:
                writer.release()
        return out_path

    def _process_image_group(self, model, group_name: str, image_paths: list[Path], position=None, leave=True):
        save_dir = Path(self.args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.args.image_output == "video":
            out_path = save_dir / f"{group_name}.mp4"
            if out_path.exists() and not self.args.overwrite:
                self._log_line(f"[SKIP] output exists: {out_path}")
                return [out_path]

            writer = None
            pbar = tqdm(
                image_paths,
                desc=f"Predict {group_name[:40]:<40}",
                unit="frame",
                position=position,
                leave=leave,
                disable=self.args.no_progress,
            )
            try:
                for image_path in pbar:
                    rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if rgb is None:
                        raise FileNotFoundError(f"Failed to read image: {image_path}")
                    merged = self._concat_with_motion(rgb, image_path)
                    plotted = self._predict_plot(model, merged)
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(out_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            float(self.args.image_video_fps),
                            (plotted.shape[1], plotted.shape[0]),
                        )
                    writer.write(plotted)
            finally:
                pbar.close()
                if writer is not None:
                    writer.release()
            return [out_path]

        out_paths = []
        out_group_dir = save_dir / group_name
        out_group_dir.mkdir(parents=True, exist_ok=True)
        pbar = tqdm(
            image_paths,
            desc=f"Predict {group_name[:40]:<40}",
            unit="image",
            position=position,
            leave=leave,
            disable=self.args.no_progress,
        )
        try:
            for image_path in pbar:
                out_path = out_group_dir / image_path.name
                if out_path.exists() and not self.args.overwrite:
                    continue
                rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if rgb is None:
                    raise FileNotFoundError(f"Failed to read image: {image_path}")
                merged = self._concat_with_motion(rgb, image_path)
                plotted = self._predict_plot(model, merged)
                cv2.imwrite(str(out_path), plotted)
                out_paths.append(out_path)
        finally:
            pbar.close()
        return out_paths

    # 并行调度（多视频 / 多图像组）
    def _run_parallel(self, tasks, task_name: str, worker_fn):
        workers = max(1, min(int(self.args.video_workers), len(tasks)))
        if workers == 1:
            outputs = []
            for item in tasks:
                outputs.extend(worker_fn(self.model, item, None, True))
            return outputs

        print(f"[INFO] {task_name}_parallel_workers={workers}")
        thread_local = threading.local()  # 线程局部存储
        position_pool = Queue()
        for i in range(workers):
            position_pool.put(i + 1)  # 0 留给 overall 条

        outputs = []
        errors = []
        overall = tqdm(
            total=len(tasks),
            desc=f"Predict {task_name}",
            unit="task",
            position=0,
            disable=self.args.no_progress,
        )

        def _task(item):
            position = position_pool.get()  # 获取进度条位置
            try:
                # model 线程内复用，线程间隔离
                model = getattr(thread_local, "model", None)
                if model is None:
                    model = YOLO(self.args.model, task=self.model_task)
                    thread_local.model = model
                return worker_fn(model, item, position, False)
            finally:
                position_pool.put(position)  # 归还进度条位置

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 执行 _task(item) 返回一个 Future 对象，表示这个异步任务未来会有结果
            # 通过 future 对应到原始任务
            future_map = {executor.submit(_task, item): item for item in tasks}
            # as_completed: 哪个任务先执行完，先返回该 future
            for future in as_completed(future_map):
                item = future_map[future]
                try:
                    outputs.extend(future.result())
                except Exception as exc:
                    errors.append((item, exc))
                    item_name = item.name if isinstance(item, Path) else item[0]
                    self._log_line(f"[ERROR] {item_name}: {type(exc).__name__}: {exc}")
                overall.update(1)  # 进度更新，某个任务失败不会停掉整个线程池，其他任务继续，最后汇总报错
        overall.close()

        if errors:
            details = []
            for item, exc in errors[:5]:
                item_name = item.name if isinstance(item, Path) else item[0]
                details.append(f"{item_name}: {type(exc).__name__}: {exc}")
            more = f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""
            raise RuntimeError(f"{len(errors)} task(s) failed: {'; '.join(details)}{more}")
        return outputs

    def _run_videos(self):
        if len(self.videos) > 1 and self.args.video_workers > 1:
            return self._run_parallel(
                tasks=self.videos,
                task_name="videos",
                worker_fn=lambda model, item, pos, leave: [self._process_video(model, item, pos, leave)],
            )
        return [self._process_video(self.model, video_path) for video_path in self.videos]

    def _run_image_groups(self):
        if len(self.image_groups) > 1 and self.args.video_workers > 1:
            return self._run_parallel(
                tasks=self.image_groups,
                task_name="image_groups",
                worker_fn=lambda model, item, pos, leave: self._process_image_group(
                    model,
                    group_name=item[0],
                    image_paths=item[1],
                    position=pos,
                    leave=leave,
                ),
            )
        outputs = []
        for group_name, image_paths in self.image_groups:
            outputs.extend(self._process_image_group(self.model, group_name, image_paths))
        return outputs

    def run(self):
        self.resolve_input()
        if self.input_mode == "empty":
            return

        self.print_plan()
        outputs = self._run_videos() if self.input_mode == "video" else self._run_image_groups()

        print("[DONE] Saved outputs:")
        for out in outputs:
            print(out)


def main():
    args = parse_args()
    MotionPredictRunner(args).run()


if __name__ == "__main__":
    main()
