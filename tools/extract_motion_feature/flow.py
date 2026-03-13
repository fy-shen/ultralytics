# coding=utf-8
import cv2
import numpy as np
import argparse
import os


def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext


def flow_to_gray(flow, bound=15):
    """
    将光流分量映射到 0~255 灰度图（双流网络标准做法）
    """
    flow = np.clip(flow, -bound, bound)
    flow = (flow + bound) * (255.0 / (2 * bound))
    return flow.astype(np.uint8)


def process_video(video_path, bound):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_dir, video_name, ext = splitfn(video_path)

    # ---------- 输出目录 ----------
    flow_x_dir = video_dir.replace("videos", "flow_x")
    flow_y_dir = video_dir.replace("videos", "flow_y")
    os.makedirs(flow_x_dir, exist_ok=True)
    os.makedirs(flow_y_dir, exist_ok=True)

    # ---------- 初始化 ----------
    ret, frame1 = cap.read()
    while not ret:
        ret, frame1 = cap.read()

    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    frame_id = 1
    print(f"Processing {video_name}{ext} ...")

    # ---------- 主循环 ----------
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # ---------- 计算稠密光流 ----------
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,      # 金字塔缩放比例，数值越大对细节越敏感
            levels=4,           # 金字塔层数，数值越大更关注大运动，通常3~5
            winsize=21,         # 核心参数，数值越大越平滑抗噪
            iterations=3,       # 每层金字塔的迭代次数，数值越大越精确速度越慢，通常3~5
            poly_n=7,           # 拟合局部信号的像素邻域大小，数值越大噪声越小，通常5~7
            poly_sigma=1.5,     # 高斯平滑标准差，poly_n=5，poly_sigma=1.1~1.2；poly_n=7，poly_sigma=1.5
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )

        flow_x = flow_to_gray(flow[..., 0], bound)
        flow_y = flow_to_gray(flow[..., 1], bound)

        # ---------- 保存 ----------
        name = f"{video_name}_{frame_id:06d}.jpg"

        cv2.imwrite(
            os.path.join(flow_x_dir, name),
            flow_x
        )

        cv2.imwrite(
            os.path.join(flow_y_dir, name),
            flow_y
        )

        # ---------- 更新 ----------
        prev_gray = gray
        frame_id += 1

    cap.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract two-stream optical flow (x / y) as grayscale images"
    )

    parser.add_argument("--video", type=str, default=None, help="Path to a single video file")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory containing multiple videos")
    parser.add_argument("--bound", type=int, default=15, help="Flow value bound for normalization")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.video is None and args.video_dir is None:
        raise ValueError("You must specify either --video or --video-dir")

    # ---------- 单视频 ----------
    if args.video is not None:
        if not os.path.isfile(args.video):
            raise FileNotFoundError(f"Video not found: {args.video}")
        process_video(args.video, args.bound)

    # ---------- 目录模式 ----------
    else:
        if not os.path.isdir(args.video_dir):
            raise NotADirectoryError(f"Video directory not found: {args.video_dir}")

        for video in sorted(os.listdir(args.video_dir)):
            video_path = os.path.join(args.video_dir, video)

            if not video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue

            process_video(video_path, args.bound)
