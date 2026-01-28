# coding=utf-8
import cv2
import numpy as np
import argparse
import os


def adaptive_threshold(diff_image, max_val=255):
    """
    自适应阈值：用于增强慢速/微弱运动
    """
    return cv2.adaptiveThreshold(
        diff_image,
        max_val,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        5,
        5
    )


def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext


def process_video(video_path, alpha):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_dir, video_name, ext = splitfn(video_path)

    # ---------- 输出目录 ----------
    short_dir = video_dir.replace("videos", "gray_diff_short")
    long_dir = video_dir.replace("videos", "gray_diff_long")
    os.makedirs(short_dir, exist_ok=True)
    os.makedirs(long_dir, exist_ok=True)

    # ---------- 初始化 ----------
    ret, frame1 = cap.read()
    while not ret:
        ret, frame1 = cap.read()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    ret, frame2 = cap.read()
    while not ret:
        ret, frame2 = cap.read()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 第一帧差（t-1 -> t）
    diff21 = cv2.absdiff(gray2, gray1)

    # 长时运动累积缓冲
    accumulated_diff = np.zeros_like(gray2, dtype=np.float32)

    frame_id = 1

    # ---------- 主循环 ----------
    print(f"Processing {video_name}{ext} ...")
    while True:
        ret, frame3 = cap.read()
        if not ret:
            break

        gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

        # 当前帧差（t -> t+1）
        diff32 = cv2.absdiff(gray3, gray2)

        # ---------- 短时运动 ----------
        # 两次帧差取 AND，抑制噪声，只保留连续变化区域
        diff = cv2.bitwise_and(diff21, diff32)

        # 自适应阈值，增强弱/慢运动
        diff = adaptive_threshold(diff)

        # 反色：白色=运动，黑色=背景
        diff = cv2.bitwise_not(diff)

        # ---------- 长时运动 ----------
        # 指数衰减累积历史运动
        accumulated_diff = cv2.addWeighted(
            accumulated_diff,
            alpha,
            diff21.astype(np.float32),
            1.0 - alpha,
            0
        )

        # 归一化到 0~255，方便保存和可视化
        accumulated_norm = cv2.normalize(
            accumulated_diff, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # ---------- 保存 ----------
        name = f"{video_name}_{frame_id:06d}.jpg"

        cv2.imwrite(
            os.path.join(short_dir, name),
            diff.astype(np.uint8)
        )

        cv2.imwrite(
            os.path.join(long_dir, name),
            accumulated_norm
        )

        # ---------- 更新状态 ----------
        gray2 = gray3
        diff21 = diff32
        frame_id += 1

    cap.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract 3-channel motion feature maps from video"
    )

    # 二选一输入
    parser.add_argument("--video", type=str, default=None, help="Path to a single video file")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory containing multiple videos")

    # parser.add_argument("--save-dir", type=str, required=True, help="Directory to save feature maps")
    parser.add_argument("--alpha", type=float, default=0.5, help="Temporal decay factor for accumulated motion")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ---------- 输入合法性检查 ----------
    if args.video is None and args.video_dir is None:
        raise ValueError("You must specify either --video or --video-dir")

    # ---------- 单视频模式（优先） ----------
    if args.video is not None:
        if not os.path.isfile(args.video):
            raise FileNotFoundError(f"Video not found: {args.video}")

        process_video(args.video, args.alpha)

    # ---------- 目录模式 ----------
    else:
        if not os.path.isdir(args.video_dir):
            raise NotADirectoryError(f"Video directory not found: {args.video_dir}")

        for video in sorted(os.listdir(args.video_dir)):
            video_path = os.path.join(args.video_dir, video)

            if not video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue

            process_video(video_path, args.alpha)
