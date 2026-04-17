import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

import cv2
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".flv", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser(description="Split videos into frame images (non-recursive).")
    parser.add_argument("--video-dir", required=True, type=str, help="Directory containing video files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output image directory (default: replace 'videos' in video-dir with 'images')",
    )
    parser.add_argument("--workers", type=int, default=4, help="Thread workers for parallel video decoding")
    parser.add_argument(
        "--opencv-threads",
        type=int,
        default=1,
        help="OpenCV internal threads. Use 1 to avoid over-subscription with multi-worker mode",
    )
    parser.add_argument("--jpg-quality", type=int, default=100, help="JPEG quality for saved frames (1-100)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frame images")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat decode mismatch (saved < reported frame count) as failure",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.opencv_threads < 0:
        parser.error("--opencv-threads must be >= 0")
    if not (1 <= args.jpg_quality <= 100):
        parser.error("--jpg-quality must be in [1, 100]")
    return args


def list_videos(video_dir: Path):
    if not video_dir.is_dir():
        raise NotADirectoryError(f"video-dir not found: {video_dir}")
    videos = [p for p in sorted(video_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return videos


def build_prefix_map(videos):
    stem_counts = Counter(p.stem for p in videos)
    prefix_map = {}
    for p in videos:
        if stem_counts[p.stem] == 1:
            prefix_map[p] = p.stem
        else:
            prefix_map[p] = f"{p.stem}_{p.suffix.lower().lstrip('.')}"
    return prefix_map


def split_single_video(
    video_path: Path,
    out_image_path: Path,
    prefix: str,
    jpg_quality: int,
    overwrite: bool,
    strict: bool,
    position: int,
    disable_progress: bool,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "video": str(video_path),
            "saved": 0,
            "expected": 0,
            "status": "error",
            "message": "cannot open video",
        }

    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    pbar = tqdm(
        total=expected if expected > 0 else None,
        desc=f"{video_path.name[:40]:<40}",
        unit="frame",
        position=position,
        leave=False,
        disable=disable_progress,
    )

    saved = 0
    write_fail = 0
    try:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_name_path = out_image_path / f"{prefix}_{idx:06}.jpg"
            if overwrite or not frame_name_path.exists():
                ok = cv2.imwrite(str(frame_name_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
                if ok:
                    saved += 1
                else:
                    write_fail += 1
            else:
                saved += 1
            idx += 1
            pbar.update(1)
    finally:
        cap.release()
        pbar.close()

    mismatch = expected > 0 and saved < expected
    drop = max(expected - saved, 0) if expected > 0 else 0
    if strict and mismatch:
        status = "error"
        message = f"decode mismatch: expected={expected}, saved={saved}, dropped={drop}"
    elif mismatch:
        status = "warn"
        message = f"possible decode loss: expected={expected}, saved={saved}, dropped={drop}"
    elif write_fail:
        status = "warn"
        message = f"write_fail={write_fail}"
    else:
        status = "ok"
        message = ""

    return {
        "video": str(video_path),
        "saved": saved,
        "expected": expected,
        "status": status,
        "message": message,
    }


def split_video_parallel(in_video_path: Path, out_image_path: Path, args):
    cv2.setNumThreads(int(args.opencv_threads))
    videos = list_videos(in_video_path)
    if not videos:
        print(f"[WARN] no video files found in: {in_video_path}")
        return []

    out_image_path.mkdir(parents=True, exist_ok=True)
    prefix_map = build_prefix_map(videos)

    workers = max(1, min(int(args.workers), len(videos)))
    position_pool = Queue()
    for i in range(workers):
        position_pool.put(i)

    def task(video_path: Path):
        position = position_pool.get()
        try:
            try:
                return split_single_video(
                    video_path=video_path,
                    out_image_path=out_image_path,
                    prefix=prefix_map[video_path],
                    jpg_quality=args.jpg_quality,
                    overwrite=args.overwrite,
                    strict=args.strict,
                    position=position,
                    disable_progress=args.no_progress,
                )
            except Exception as exc:  # noqa: BLE001
                return {
                    "video": str(video_path),
                    "saved": 0,
                    "expected": 0,
                    "status": "error",
                    "message": f"exception: {type(exc).__name__}: {exc}",
                }
        finally:
            position_pool.put(position)

    results = []
    overall = tqdm(total=len(videos), desc="videos", unit="video", disable=args.no_progress)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(task, v): v for v in videos}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            overall.update(1)
            status = result["status"].upper()
            msg = f"[{status}] {Path(result['video']).name}: saved={result['saved']}"
            if result["expected"]:
                msg += f", expected={result['expected']}"
            if result["message"]:
                msg += f", {result['message']}"
            tqdm.write(msg)
    overall.close()
    return results


def main():
    args = parse_args()
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(str(video_dir).replace("videos", "images"))
    results = split_video_parallel(video_dir, output_dir, args)

    if not results:
        return

    ok = sum(1 for r in results if r["status"] == "ok")
    warn = sum(1 for r in results if r["status"] == "warn")
    err = sum(1 for r in results if r["status"] == "error")
    print(f"[SUMMARY] output={output_dir}, ok={ok}, warn={warn}, error={err}")
    if err > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
