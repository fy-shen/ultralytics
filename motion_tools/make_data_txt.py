import os
from pathlib import Path


FEATURE_DIRS = [
    "gray_diff_short",
    "gray_diff_long",
    "fgmask",
]


def build_txt(
    dataset_root,
    split="train",
    image_exts=(".jpg", ".png", ".jpeg")
):
    """
    根据 images/{split} 下的图片生成 {split}.txt
    只有当对应特征目录下的同名图片都存在时才写入
    """
    dataset_root = Path(dataset_root)

    image_dir = dataset_root / "images" / split
    assert image_dir.exists(), f"{image_dir} not found"

    out_txt = dataset_root / f"{split}.txt"

    valid_images = []

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in image_exts:
            continue

        ok = True
        for feat in FEATURE_DIRS:
            feat_img = dataset_root / feat / split / img_path.name
            if not feat_img.exists():
                ok = False
                break

        if ok:
            valid_images.append(str(img_path.resolve()))

    # 写入 txt
    with open(out_txt, "w") as f:
        for p in valid_images:
            f.write(p + "\n")

    print(f"[{split}] {len(valid_images)} images written to {out_txt}")


if __name__ == "__main__":
    dataset_root = "/home/sfy/SFY/disk1/data/FBD-SV-2024"

    build_txt(dataset_root, split="train")
    build_txt(dataset_root, split="val")
