from pathlib import Path
import argparse
import sys


FEATURE_DIRS = (
    "gray_diff_short",
    "gray_diff_long",
    "fgmask",
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def build_txt(dataset_root: Path, split: str) -> None:
    """
    根据 images/{split} 下的图片生成 {split}.txt
    只有当所有特征目录下存在同名图片时才写入
    """
    image_dir = dataset_root / "images" / split
    if not image_dir.exists():
        print(f"[WARN] {image_dir} not found, skip.")
        return

    out_txt = dataset_root / f"{split}.txt"
    valid_images = []

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        # 检查所有特征是否存在
        if all(
            (dataset_root / feat / split / img_path.name).exists()
            for feat in FEATURE_DIRS
        ):
            valid_images.append(str(img_path.resolve()))

    # 写入 txt
    with out_txt.open("w") as f:
        f.write("\n".join(valid_images))

    print(f"[{split}] {len(valid_images)} images written to {out_txt}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate txt files for dataset splits"
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to dataset root directory"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to process (default: train val)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_root

    if not dataset_root.exists():
        print(f"[ERROR] Dataset root not found: {dataset_root}")
        sys.exit(1)

    for split in args.splits:
        build_txt(dataset_root, split)


if __name__ == "__main__":
    main()
