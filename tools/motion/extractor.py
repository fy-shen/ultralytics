import cv2
import numpy as np
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from tqdm import tqdm


from tools.motion.utils import get_arg, split_path, iter_video_paths


class BaseExtractor:
    motion_type = None
    description = ""
    cli_args = ()
    output_names = ()
    frame_lag = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.reset()

    def _reset(self):
        raise NotImplementedError

    def _update(self, frame):
        raise NotImplementedError

    def reset(self):
        self._reset()

    def update(self, frame):
        return self._update(frame)


class GrayDiffExtractor(BaseExtractor):
    motion_type = "gray_diff"
    description = "Extract gray-diff motion feature maps from video"
    cli_args = (
        {
            "flags": ("--gray-diff-alpha", "--alpha"),
            "kwargs": {"type": float, "default": 0.5, "help": "Temporal decay factor for accumulated motion"},
            "dest": "gray_diff_alpha",
        },
    )
    output_names = ("gray_diff_short", "gray_diff_long")
    frame_lag = 1

    def __init__(self, **kwargs):
        self.alpha = get_arg(kwargs, "gray_diff_alpha", "alpha", type_func=float, default=0.5)
        super().__init__(**kwargs)

    def _reset(self):
        self.gray1 = None
        self.gray2 = None
        self.diff21 = None
        self.accumulated_diff = None

    def _update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.gray1 is None:
            self.gray1 = gray
            return None
        if self.gray2 is None:
            self.gray2 = gray
            self.diff21 = cv2.absdiff(self.gray2, self.gray1)
            self.accumulated_diff = np.zeros_like(self.gray2, dtype=np.float32)
            return None
        """
        1 - diff21 - 2 - diff32 - 3
        diff21 + diff32 -> diff_short
        diff21 累加 -> diff_long
        """
        diff32 = cv2.absdiff(gray, self.gray2)
        # 两次帧差取 AND，抑制噪声，只保留连续变化区域
        diff = cv2.bitwise_and(self.diff21, diff32)
        # 自适应阈值，增强弱/慢运动
        diff = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
        # 反色：白色=运动，黑色=背景
        diff = cv2.bitwise_not(diff)

        # 指数衰减累积历史运动
        self.accumulated_diff = cv2.addWeighted(
            self.accumulated_diff,
            self.alpha,
            self.diff21.astype(np.float32),
            1.0 - self.alpha,
            0,
        )
        accumulated_norm = cv2.normalize(self.accumulated_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 更新状态
        self.gray1 = self.gray2
        self.gray2 = gray
        self.diff21 = diff32
        return {"gray_diff_short": diff.astype(np.uint8), "gray_diff_long": accumulated_norm}


class GrayDiffEnhancedExtractor(BaseExtractor):
    motion_type = "gde"
    description = "Extract enhanced gray-diff motion maps for tiny/far/night targets"
    """
    默认策略偏向“保留目标运动”，不过度抑制噪声。
    建议优先固定一套参数在多数据集复用，仅在极端场景微调。
    """
    cli_args = (
        {
            "flags": ("--gde-alpha", "--gray-diff-enhanced-alpha"),
            "kwargs": {"type": float, "default": 0.6, "help": "Temporal decay factor for accumulated motion"},
            "dest": "gde_alpha",
        },
        {
            "flags": ("--gde-clip", "--gray-diff-enhanced-clahe-clip"),
            "kwargs": {"type": float, "default": 2.0, "help": "CLAHE clip limit (<=0 disables CLAHE)"},
            "dest": "gde_clip",
        },
        {
            "flags": ("--gde-grid", "--gray-diff-enhanced-clahe-grid"),
            "kwargs": {"type": int, "default": 7, "help": "CLAHE tile grid size"},
            "dest": "gde_grid",
        },
        {
            "flags": ("--gde-th", "--gray-diff-enhanced-threshold-mode"),
            "kwargs": {"type": str, "choices": ("adaptive", "otsu"), "default": "adaptive", "help": "Threshold mode"},
            "dest": "gde_th",
        },
        {
            "flags": ("--gde-blk", "--gray-diff-enhanced-adapt-block-size"),
            "kwargs": {"type": int, "default": 5, "help": "Adaptive threshold block size (odd)"},
            "dest": "gde_blk",
        },
        {
            "flags": ("--gde-c", "--gray-diff-enhanced-adapt-c"),
            "kwargs": {"type": float, "default": 4, "help": "Adaptive threshold C"},
            "dest": "gde_c",
        },
        {
            "flags": ("--gde-fuse", "--gray-diff-enhanced-short-mode"),
            "kwargs": {"type": str, "choices": ("and", "or", "hybrid"), "default": "hybrid", "help": "Short-term fusion mode"},
            "dest": "gde_fuse",
        },
        {
            "flags": ("--gde-lam", "--gray-diff-enhanced-hybrid-lambda"),
            "kwargs": {"type": float, "default": 0.4, "help": "OR branch weight in hybrid mode [0,1]"},
            "dest": "gde_lam",
        },
        {
            "flags": ("--gde-norm", "--gray-diff-enhanced-long-norm"),
            "kwargs": {"type": str, "choices": ("percentile", "minmax"), "default": "percentile", "help": "Long-term map normalization"},
            "dest": "gde_norm",
        },
        {
            "flags": ("--gde-pl", "--gray-diff-enhanced-long-p-low"),
            "kwargs": {"type": float, "default": 0.1, "help": "Low percentile for long-term normalization"},
            "dest": "gde_pl",
        },
        {
            "flags": ("--gde-ph", "--gray-diff-enhanced-long-p-high"),
            "kwargs": {"type": float, "default": 99.9, "help": "High percentile for long-term normalization"},
            "dest": "gde_ph",
        },
    )
    output_names = ("gray_diff_enhanced_short", "gray_diff_enhanced_long")
    frame_lag = 1

    def __init__(self, **kwargs):
        # 参数短名优先，同时兼容旧参数名，避免历史脚本失效
        self.alpha = get_arg(kwargs, "gde_alpha", "gray_diff_enhanced_alpha", type_func=float, default=0.6)
        self.clahe_clip = get_arg(kwargs, "gde_clip", "gray_diff_enhanced_clahe_clip", type_func=float, default=2.0)
        self.clahe_grid = get_arg(kwargs, "gde_grid", "gray_diff_enhanced_clahe_grid", type_func=int, default=8)
        self.threshold_mode = get_arg(kwargs, "gde_th", "gray_diff_enhanced_threshold_mode", type_func=str, default="adaptive")
        self.adapt_block_size = get_arg(kwargs, "gde_blk", "gray_diff_enhanced_adapt_block_size", type_func=int, default=5)
        self.adapt_c = get_arg(kwargs, "gde_c", "gray_diff_enhanced_adapt_c", type_func=float, default=2.0)
        self.short_mode = get_arg(kwargs, "gde_fuse", "gray_diff_enhanced_short_mode", type_func=str, default="hybrid")
        self.hybrid_lambda = get_arg(kwargs, "gde_lam", "gray_diff_enhanced_hybrid_lambda", type_func=float, default=0.4)
        self.long_norm = get_arg(kwargs, "gde_norm", "gray_diff_enhanced_long_norm", type_func=str, default="percentile")
        self.long_p_low = get_arg(kwargs, "gde_pl", "gray_diff_enhanced_long_p_low", type_func=float, default=1.0)
        self.long_p_high = get_arg(kwargs, "gde_ph", "gray_diff_enhanced_long_p_high", type_func=float, default=99.0)

        self.adapt_block_size = max(3, self.adapt_block_size)
        if self.adapt_block_size % 2 == 0:
            self.adapt_block_size += 1
        self.hybrid_lambda = float(np.clip(self.hybrid_lambda, 0.0, 1.0))
        self.long_p_low = float(np.clip(self.long_p_low, 0.0, 100.0))
        self.long_p_high = float(np.clip(self.long_p_high, 0.0, 100.0))
        super().__init__(**kwargs)

    def _reset(self):
        self.gray1 = None
        self.gray2 = None
        self.diff21 = None
        self.accumulated_diff = None
        self.clahe = None
        if self.clahe_clip > 0:
            grid = max(1, self.clahe_grid)
            self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(grid, grid))

    def _preprocess_gray(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # CLAHE 局部对比度增强，增强夜间/远距离弱变化
        # 可调参数: clip(增强强度), grid(局部块大小)。
        if self.clahe is not None:
            gray = self.clahe.apply(gray)
        return gray

    def _fuse_short(self, diff21, diff32):
        and_map = cv2.bitwise_and(diff21, diff32)
        if self.short_mode == "and":
            return and_map

        or_map = cv2.bitwise_or(diff21, diff32)
        if self.short_mode == "or":
            return or_map

        # hybrid 为 AND 与 OR 的线性加权，保留 AND 降噪能力，同时注入 OR 分支提高微小目标召回
        return cv2.addWeighted(and_map, 1.0 - self.hybrid_lambda, or_map, self.hybrid_lambda, 0.0)

    def _threshold_short(self, diff):
        if self.threshold_mode == "otsu":
            # otsu：自动搜索类间方差最大的全局阈值
            # 优点：无需人工设定阈值；整体对比度还可以时很稳
            # 缺点：全局共享一个阈值；光照不均、局部噪声重时容易失效
            _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary

        # adaptive：对每个像素 x,y 在其邻域窗口(blockSize) 通过 Mean 或 Gaussian 统计数值并减去偏置(C) 得到一个计算值，
        # 若原始像素值高于计算值输出255，否则为0。窗口越大越平滑，越小对细节更敏感，噪声越多。C 越大更容易判断成运动目标
        # 优点：应对局部亮度变化更好；比全局阈值更适合夜间/阴影/非均匀照明
        # 缺点：参数敏感；窗口太小会雪花噪声，太大又抹小目标
        return cv2.adaptiveThreshold(
            diff,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.adapt_block_size,
            self.adapt_c,
        )

    def _normalize_long(self, long_map):
        """
        功能: 归一化长时响应，避免少量强运动压制整体动态范围。
        实现: percentile 或 minmax 归一化。
        可调参数: norm(percentile/minmax), pl/ph(分位点)。
        """
        if self.long_norm == "minmax":
            # y = (x - xmin) / (xmax - xmin) * 255
            # 对极端值非常敏感，只要少量超强运动（云边、树梢、抖动）出现，其他区域会被压得很暗
            return cv2.normalize(long_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # percentile 分位数归一化
        # 优点：忽略最极端的低/高尾部，抗异常值更强，长时图更稳
        # 缺点：如果 pl/ph 设得不合适，可能压掉弱响应或放大噪声
        # np.percentile 找到一个数 v，使得数组里约 99% 的元素 <= v，从而不被极端值带偏
        low = np.percentile(long_map, self.long_p_low)
        high = np.percentile(long_map, self.long_p_high)
        if high <= low:
            return np.zeros_like(long_map, dtype=np.uint8)
        long_map = np.clip((long_map - low) / (high - low) * 255.0, 0, 255)
        return long_map.astype(np.uint8)

    def _update(self, frame):
        gray = self._preprocess_gray(frame)
        if self.gray1 is None:
            self.gray1 = gray
            return None
        if self.gray2 is None:
            self.gray2 = gray
            self.diff21 = cv2.absdiff(self.gray2, self.gray1)
            self.accumulated_diff = np.zeros_like(self.gray2, dtype=np.float32)
            return None

        diff32 = cv2.absdiff(gray, self.gray2)
        short_raw = self._fuse_short(self.diff21, diff32)
        short_map = self._threshold_short(short_raw)
        short_map = cv2.bitwise_not(short_map)

        self.accumulated_diff = cv2.addWeighted(
            self.accumulated_diff,
            self.alpha,
            short_map.astype(np.float32),
            1.0 - self.alpha,
            0,
        )
        long_map = self._normalize_long(self.accumulated_diff)

        self.gray1 = self.gray2
        self.gray2 = gray
        self.diff21 = diff32
        return {
            "gray_diff_enhanced_short": short_map.astype(np.uint8),
            "gray_diff_enhanced_long": long_map,
        }


class FgMaskExtractor(BaseExtractor):
    motion_type = "fgmask"
    description = "Extract foreground masks with OpenCV MOG2"
    cli_args = (
        {
            "flags": ("--fgmask-history", "--history"),
            "kwargs": {"type": int, "default": 500, "help": "MOG2 history length"},
            "dest": "fgmask_history",
        },
        {
            "flags": ("--fgmask-var-threshold", "--var-threshold"),
            "kwargs": {"type": int, "default": 50, "help": "MOG2 variance threshold"},
            "dest": "fgmask_var_threshold",
        },
        {
            "flags": ("--fgmask-kernel-size", "--kernel-size"),
            "kwargs": {"type": int, "default": 3, "help": "Morphology kernel size"},
            "dest": "fgmask_kernel_size",
        },
        {
            "flags": ("--fgmask-min-area", "--min-area"),
            "kwargs": {"type": int, "default": 0, "help": "Minimum contour area to keep"},
            "dest": "fgmask_min_area",
        },
    )
    output_names = ("fgmask",)
    frame_lag = 0

    def __init__(self, **kwargs):
        self.history = get_arg(kwargs, "fgmask_history", "history", type_func=int, default=500)
        self.var_threshold = get_arg(kwargs, "fgmask_var_threshold", "var_threshold", type_func=int, default=50)
        kernel_size = get_arg(kwargs, "fgmask_kernel_size", "kernel_size", type_func=int, default=3)
        self.min_area = get_arg(kwargs, "fgmask_min_area", "min_area", type_func=int, default=0)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        super().__init__(**kwargs)

    def _reset(self):
        self.frame_index = 0
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=False,
        )

    def _update(self, frame):
        # 前景提取
        fgmask = self.fgbg.apply(frame)
        # 去噪
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        # 面积过滤
        if self.min_area > 0:
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean_mask = np.zeros_like(fgmask)
            for cnt in contours:
                if cv2.contourArea(cnt) >= self.min_area:
                    cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
            fgmask = clean_mask

        self.frame_index += 1
        if self.frame_index == 1:
            return None
        return {"fgmask": fgmask}


class FlowExtractor(BaseExtractor):
    motion_type = "flow"
    description = "Extract two-stream optical flow as grayscale images"
    cli_args = (
        {
            "flags": ("--flow-bound", "--bound"),
            "kwargs": {"type": int, "default": 15, "help": "Flow value bound for normalization"},
            "dest": "flow_bound",
        },
    )
    output_names = ("flow_x", "flow_y")
    frame_lag = 0

    def __init__(self, **kwargs):
        self.bound = get_arg(kwargs, "flow_bound", "bound", type_func=int, default=15)
        super().__init__(**kwargs)

    def _reset(self):
        self.prev_gray = None

    def _update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            pyr_scale=0.5,      # 金字塔缩放比例，数值越大对细节越敏感
            levels=4,           # 金字塔层数，数值越大更关注大运动，通常3~5
            winsize=21,         # 核心参数，数值越大越平滑抗噪
            iterations=3,       # 每层金字塔的迭代次数，数值越大越精确速度越慢，通常3~5
            poly_n=7,           # 拟合局部信号的像素邻域大小，数值越大噪声越小，通常5~7
            poly_sigma=1.5,     # 高斯平滑标准差，poly_n=5，poly_sigma=1.1~1.2；poly_n=7，poly_sigma=1.5
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        self.prev_gray = gray
        return {
            "flow_x": self.flow_to_gray(flow[..., 0], self.bound),
            "flow_y": self.flow_to_gray(flow[..., 1], self.bound),
        }

    @staticmethod
    def flow_to_gray(flow, bound=15):
        flow = np.clip(flow, -bound, bound)
        flow = (flow + bound) / (2 * bound) * 255.0
        return flow.astype(np.uint8)


FEATURE_EXTRACTOR_REGISTRY = OrderedDict(
    {
        "gray_diff_short": GrayDiffExtractor,
        "gray_diff_long": GrayDiffExtractor,
        "gray_diff_enhanced_short": GrayDiffEnhancedExtractor,
        "gray_diff_enhanced_long": GrayDiffEnhancedExtractor,
        "fgmask": FgMaskExtractor,
        "flow_x": FlowExtractor,
        "flow_y": FlowExtractor,
    }
)


def build_motion_type_specs():
    specs = OrderedDict()
    seen_motion_types = set()
    for extractor_cls in FEATURE_EXTRACTOR_REGISTRY.values():
        motion_type = getattr(extractor_cls, "motion_type", None)
        if not motion_type:
            raise ValueError(f"{extractor_cls.__name__} missing required class attribute 'motion_type'.")
        if motion_type in seen_motion_types:
            continue
        seen_motion_types.add(motion_type)
        specs[motion_type] = {
            "description": getattr(extractor_cls, "description", motion_type),
            "feature_names": list(getattr(extractor_cls, "output_names", ())),
            "args": [deepcopy(x) for x in getattr(extractor_cls, "cli_args", ())],
        }
    return specs


class ExtractPipeline:
    def __init__(self, feature_names, **kwargs):
        if not feature_names:
            raise ValueError("Motion feature pipeline requires at least one feature name.")

        self.feature_names = list(feature_names)
        self.extractors = []
        self.max_frame_lag = 0
        extractor_types = []
        for name in self.feature_names:
            extractor_cls = FEATURE_EXTRACTOR_REGISTRY.get(name)
            if extractor_cls is None:
                supported = ", ".join(FEATURE_EXTRACTOR_REGISTRY)
                raise ValueError(f"Unsupported motion feature '{name}'. Supported features: {supported}")
            if extractor_cls not in extractor_types:
                extractor_types.append(extractor_cls)
                self.extractors.append(extractor_cls(**kwargs))
        self.max_frame_lag = max((extractor.frame_lag for extractor in self.extractors), default=0)

    def reset(self):
        for extractor in self.extractors:
            extractor.reset()

    def update(self, frame, frame_idx):
        aligned_outputs = []
        for extractor in self.extractors:
            features = extractor.update(frame)
            if features is None:
                continue
            # 目标索引，extractor 接收到 idx 时输出的特征所对应的帧索引为 idx-lag
            target_idx = frame_idx - extractor.frame_lag
            if target_idx < 0:
                continue
            filtered = {name: features[name] for name in self.feature_names if name in features}
            if filtered:
                aligned_outputs.append((target_idx, filtered))
        return aligned_outputs


def save_motion_features(
    video_path,
    feature_names,
    save_dirs=None,
    source_token="videos",
    progress=True,
    progress_position=None,
    progress_leave=True,
    **kwargs,
):
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_dir, video_name, _ = split_path(video_path)

    pipeline = ExtractPipeline(feature_names, **kwargs)
    frame_idx = 0
    saved_count = 0
    save_dirs = save_dirs or {}
    feature_dirs = {}
    pending = {}
    for name in feature_names:
        out_dir = Path(save_dirs.get(name) or Path(str(video_dir).replace(source_token, name)))
        out_dir.mkdir(parents=True, exist_ok=True)
        feature_dirs[name] = out_dir

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    with tqdm(
        total=total_frames,
        desc=f"Extract {video_name[:40]:<40}",
        unit="frame",
        disable=not progress,
        position=progress_position,
        leave=progress_leave,
    ) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # target_idx 可能不一致，先存放到 bucket
            for target_idx, features in pipeline.update(frame, frame_idx):
                bucket = pending.setdefault(target_idx, {})
                bucket.update(features)
            # 所有需要的特征就位，并且 target_idx 对齐
            ready_indices = sorted(idx for idx, bucket in pending.items() if all(name in bucket for name in feature_names))
            for target_idx in ready_indices:
                file_name = f"{video_name}_{target_idx:06d}.jpg"
                bucket = pending.pop(target_idx)
                for name in feature_names:
                    cv2.imwrite(str(feature_dirs[name] / file_name), bucket[name])
                saved_count += 1
            frame_idx += 1
            pbar.update(1)
            if progress and frame_idx % 30 == 0:
                pbar.set_postfix(saved=saved_count, pending=len(pending))
    cap.release()
    return saved_count


def run_motion_extraction(args, motion_types):
    from tools.motion.parser import get_motion_kwargs_from_args

    motion_kwargs, feature_names = get_motion_kwargs_from_args(args, motion_types)
    video_paths = iter_video_paths(video=args.video, video_dir=args.video_dir)
    if not video_paths:
        print("[WARN] No videos found.")
        return

    workers = max(1, min(int(getattr(args, "workers", 1)), len(video_paths)))
    no_progress = bool(getattr(args, "no_progress", False))

    def _task(video_path, position):
        saved = save_motion_features(
            video_path,
            feature_names=feature_names,
            progress=not no_progress,
            progress_position=position,
            progress_leave=False if workers > 1 else True,
            **motion_kwargs,
        )
        return video_path, saved

    if workers == 1:
        for video_path in video_paths:
            print(f"[INFO] Start extract {video_path.name} motion feature maps ...")
            _, saved = _task(video_path, 0)
            print(f"[DONE] {video_path.name}: saved={saved}")
        return

    print(f"[INFO] Parallel extraction enabled: workers={workers}")
    position_pool = Queue()
    for i in range(workers):
        position_pool.put(i + 1)  # 0 留给 overall 进度条

    results = []
    errors = []
    overall = tqdm(
        total=len(video_paths),
        desc="Extract videos",
        unit="video",
        position=0,
        disable=no_progress,
    )

    def _wrapped(video_path):
        position = position_pool.get()
        try:
            return _task(video_path, position)
        finally:
            position_pool.put(position)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_wrapped, video_path): video_path for video_path in video_paths}
        for future in as_completed(futures):
            video_path = futures[future]
            try:
                item = future.result()
                results.append(item)
                msg = f"[DONE] {video_path.name}: saved={item[1]}"
            except Exception as exc:  # noqa: BLE001
                errors.append((video_path, exc))
                msg = f"[ERROR] {video_path.name}: {type(exc).__name__}: {exc}"
            if no_progress:
                print(msg)
            else:
                tqdm.write(msg)
            overall.update(1)
    overall.close()

    if errors:
        details = "; ".join(f"{p.name}: {type(e).__name__}: {e}" for p, e in errors[:5])
        more = f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""
        raise RuntimeError(f"{len(errors)} extraction task(s) failed: {details}{more}")
