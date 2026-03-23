from pathlib import Path
import argparse
import sys
import random

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def valid_image(img_path, dataset_root, split, feature_dirs):
    """检查特征文件是否都存在"""
    if split is None:
        return all(
            (dataset_root / feat / img_path.name).exists()
            for feat in feature_dirs
        )
    else:
        return all(
            (dataset_root / feat / split / img_path.name).exists()
            for feat in feature_dirs
        )


def write_txt(dataset_root, split, image_paths):
    out_txt = dataset_root / f"{split}.txt"
    with out_txt.open("w") as f:
        f.write("\n".join(str(p.resolve()) for p in image_paths))

    print(f"[{split}] {len(image_paths)} images written to {out_txt}")


def write_list(file_path, items):
    with open(file_path, "w") as f:
        for x in items:
            f.write(f"{x}\n")

    print(f"[INFO] saved {len(items)} items to {file_path}")


# ===============================
# 自动划分逻辑
# ===============================
def split_by_video(images, val_ratio, dataset_root):
    video_dict = {}

    for img in images:
        video = img.stem.rsplit("_", 1)[0]
        video_dict.setdefault(video, []).append(img)

    for v in video_dict:
        video_dict[v] = sorted(video_dict[v])

    videos = sorted(video_dict.keys())
    random.shuffle(videos)

    val_video_num = int(len(videos) * val_ratio)
    val_videos = set(videos[:val_video_num])

    train, val = [], []

    for v, imgs in video_dict.items():
        if v in val_videos:
            val.extend(imgs)
        else:
            train.extend(imgs)

    val_video_txt = dataset_root / "val_videos.txt"
    write_list(val_video_txt, sorted(val_videos))
    return train, val


def split_random(images, val_ratio):
    images = images.copy()
    random.shuffle(images)

    val_num = int(len(images) * val_ratio)

    val = images[:val_num]
    train = images[val_num:]

    return train, val


# ===============================
# 生成 txt
# ===============================

def build_txt(dataset_root, split, feature_dirs):
    image_dir = dataset_root / "images" / split
    valid_images = []

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        if valid_image(img_path, dataset_root, split, feature_dirs):
            valid_images.append(img_path)

    write_txt(dataset_root, split, valid_images)


def auto_split(dataset_root, feature_dirs, val_ratio, video_pattern):
    image_dir = dataset_root / "images"

    images = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and valid_image(p, dataset_root, None, feature_dirs)
    ])

    if len(images) == 0:
        print("[ERROR] no images found")
        return

    if video_pattern:
        print("[INFO] detected video frame naming pattern")
        train, val = split_by_video(images, val_ratio, dataset_root)
    else:
        print("[INFO] detected numeric naming pattern")
        train, val = split_random(images, val_ratio)

    write_txt(dataset_root, "train", train)
    write_txt(dataset_root, "val", val)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="dataset root")
    parser.add_argument("--feature-dirs", nargs="+", default=["gray_diff_short", "gray_diff_long"],
                        help="feature directories")
    parser.add_argument("--splits", nargs="*", default=["train", "val"], help="splits")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="validation ratio")
    parser.add_argument("--video-pattern", type=bool, default=True, help="split by video name")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    dataset_root = args.data

    if not dataset_root.exists():
        print(f"[ERROR] dataset root not found: {dataset_root}")
        sys.exit(1)

    if len(args.splits):
        if all((dataset_root / "images" / split).exists() for split in args.splits):
            for split in args.splits:
                build_txt(dataset_root, split, args.feature_dirs)

    else:
        print("[INFO] no split detected, auto splitting dataset")
        auto_split(dataset_root, args.feature_dirs, args.val_ratio, args.video_pattern)


if __name__ == "__main__":
    main()
