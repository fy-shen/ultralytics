import argparse
import os.path
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from ultralytics.data.utils import check_det_dataset


# ===============================
# 配置（尺寸分桶，自行修改）
# ===============================
SIZE_BINS = [
    (0, 5),
    (5, 10),
    (10, 15),
    (15, 20),
    (20, 25),
    (25, 30),
    (30, 50),
    (50, 100),
    (100, float("inf")),
]


def get_size_bucket(size):
    for low, high in SIZE_BINS:
        if low <= size < high:
            if high == float("inf"):
                return f">{int(low)}"
            return f"{int(low)}-{int(high)}"
    return "unknown"


def get_count_bucket(n):
    if n == 0:
        return "0"
    elif n == 1:
        return "1"
    elif 2 <= n <= 5:
        return "2-5"
    else:
        return ">5"


def img_to_label_path(img_path: Path):
    return Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")


def load_labels(label_path):
    if not label_path.exists():
        return []
    lines = label_path.read_text().strip().splitlines()
    targets = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = map(float, parts)
        targets.append((int(cls), w, h))
    return targets


# ===============================
# 主函数
# ===============================

def analyze(data, split):
    # 获取类别名
    class_names = data["names"]

    txt_path = data[split]
    with open(txt_path, "r") as f:
        image_paths = [Path(x.strip()) for x in f if x.strip()]

    print(f"[INFO] total images: {len(image_paths)}")

    # 1. 类别统计
    class_count = defaultdict(int)

    # 2. 每图每类目标数量
    per_image_class = {cls: defaultdict(int) for cls in class_names.keys()}

    # 3. 尺寸统计
    size_hist = {cls: defaultdict(int) for cls in class_names.keys()}

    for img_path in image_paths:
        label_path = img_to_label_path(img_path)
        targets = load_labels(label_path)

        # 每图统计
        class_counter = defaultdict(int)

        for cls, w, h in targets:
            class_count[cls] += 1
            class_counter[cls] += 1

            # 尺寸（按模型输入要求）
            size = (w + h) / 2 * 1280
            bucket = get_size_bucket(size)
            size_hist[cls][bucket] += 1

        # 每类计数分桶
        for cls in class_names.keys():
            n = class_counter.get(cls, 0)
            bucket = get_count_bucket(n)
            per_image_class[cls][bucket] += 1

    # =========================
    # 输出 1：类别统计
    # =========================
    print("\n===== 类别统计 =====")
    total = sum(class_count.values())

    for cls, name in class_names.items():
        cnt = class_count[cls]
        ratio = cnt / total * 100 if total > 0 else 0
        print(f"{name}: {cnt} ({ratio:.2f}%)")

    # =========================
    # 绘图 2：每图目标数
    # =========================
    count_bins = ["0", "1", "2-5", ">5"]

    fig, axes = plt.subplots(1, len(class_names), figsize=(6 * len(class_names), 5))

    if len(class_names) == 1:
        axes = [axes]

    for i, (cls, name) in enumerate(class_names.items()):
        ax = axes[i]
        values = [per_image_class[cls].get(b, 0) for b in count_bins]

        bars = ax.bar(count_bins, values)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                    str(int(height)), ha='center', va='bottom')

        ax.set_title(f"{name} - objects per image")
        ax.set_xlabel("count bucket")
        ax.set_ylabel("image count")

    plt.tight_layout()
    hist_path = os.path.join(data["path"], f"{split}_objects_count.png")
    plt.savefig(hist_path)
    print(f"[Saved] {hist_path}")

    # =========================
    # 绘图 3：尺寸分布
    # =========================
    size_bin_labels = []
    for low, high in SIZE_BINS:
        if high == float("inf"):
            size_bin_labels.append(f">{int(low)}")
        else:
            size_bin_labels.append(f"{int(low)}-{int(high)}")

    fig, axes = plt.subplots(1, len(class_names), figsize=(6 * len(class_names), 5))

    if len(class_names) == 1:
        axes = [axes]

    for i, (cls, name) in enumerate(class_names.items()):
        ax = axes[i]
        values = [size_hist[cls].get(b, 0) for b in size_bin_labels]

        bars = ax.bar(size_bin_labels, values)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                    str(int(height)), ha='center', va='bottom')

        ax.set_title(f"{name} - size distribution")
        ax.set_xlabel("size (pixel)")
        ax.set_ylabel("count")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    hist_path = os.path.join(data["path"], f"{split}_objects_size.png")
    plt.savefig(hist_path)
    print(f"[Saved] {hist_path}")


# ===============================
# main
# ===============================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    return parser.parse_args()


def main():
    args = parse_args()
    data = check_det_dataset(args.yaml)
    analyze(data, "train")
    analyze(data, "val")


if __name__ == "__main__":
    main()
