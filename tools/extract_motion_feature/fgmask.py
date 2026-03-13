# coding=utf-8
import cv2
import os
import time
import argparse


def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext


def process_video(video_path, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_dir, video_name, ext = splitfn(video_path)

    save_dir = video_dir.replace("videos", "fgmask")
    os.makedirs(save_dir, exist_ok=True)

    # ---------- Background Subtractor ----------
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=args.history,
        varThreshold=args.var_threshold,
        detectShadows=False
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (args.kernel_size, args.kernel_size)
    )

    frame_id = 0
    print(f"Processing {video_name}{ext} ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        # ---------- 前景提取 ----------
        fgmask = fgbg.apply(frame)

        # 去噪（开运算）
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # ---------- 可选：面积过滤 ----------
        if args.min_area > 0:
            contours, _ = cv2.findContours(
                fgmask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            clean_mask = fgmask.copy()
            clean_mask[:] = 0

            for cnt in contours:
                if cv2.contourArea(cnt) >= args.min_area:
                    cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

            fgmask = clean_mask

        t1 = time.time()
        if args.profile:
            print(f"\r[{video_name}] Frame {frame_id:06d} | CPU time: {(t1 - t0) * 1000:6.2f} ms", end="", flush=True)

        # ---------- 显示 ----------
        if args.show:
            cv2.imshow("Foreground Mask", fgmask)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # ---------- 保存 ----------
        if args.save and frame_id > 0:
            save_name = f"{video_name}_{frame_id:06d}.jpg"
            cv2.imwrite(os.path.join(save_dir, save_name), fgmask)

        frame_id += 1

    cap.release()
    if args.profile:
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPU-based Background Subtraction with OpenCV (MOG2)"
    )

    # 二选一输入
    parser.add_argument("--video", type=str, default=None, help="Path to a single video")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory containing videos")

    # MOG2 参数
    parser.add_argument("--history", type=int, default=500)
    parser.add_argument("--var-threshold", type=int, default=50)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=0)

    # Flags
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--profile", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # ---------- 输入合法性检查 ----------
    if args.video is None and args.video_dir is None:
        raise ValueError("You must specify either --video or --video-dir")

    # ---------- 单视频优先 ----------
    if args.video is not None:
        if not os.path.isfile(args.video):
            raise FileNotFoundError(args.video)
        process_video(args.video, args)

    # ---------- 目录模式 ----------
    else:
        if not os.path.isdir(args.video_dir):
            raise NotADirectoryError(args.video_dir)

        for video in sorted(os.listdir(args.video_dir)):
            video_path = os.path.join(args.video_dir, video)
            if not video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue
            process_video(video_path, args)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
