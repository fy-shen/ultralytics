import cv2
import numpy as np
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
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


def save_motion_features(video_path, feature_names, save_dirs=None, source_token="videos", **kwargs):
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

    with tqdm(total=total_frames, desc=f"Extract {video_name}", unit="frame") as pbar:
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
            if frame_idx % 30 == 0:
                pbar.set_postfix(saved=saved_count, pending=len(pending))
    cap.release()
    return saved_count


def run_motion_extraction(args, motion_types):
    from tools.motion.parser import get_motion_kwargs_from_args

    motion_kwargs, feature_names = get_motion_kwargs_from_args(args, motion_types)
    for video_path in iter_video_paths(video=args.video, video_dir=args.video_dir):
        print(f"[INFO] Start extract {video_path.name} motion feature maps ...")
        save_motion_features(
            video_path,
            feature_names=feature_names,
            **motion_kwargs,
        )
