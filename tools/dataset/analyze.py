import argparse
import os.path
import math
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

from tools.dataset import get_image_size
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

COUNT_BINS = [
    0, 1, (2, 5), (5, float("inf"))
]


def get_bucket(n, bins):
    for b in bins:
        if isinstance(b, int):
            if n == b:
                return f'{b}'
        else:
            low, high = b
            if low <= n < high:
                if high == float("inf"):
                    return f">{int(low)}"
                return f"{int(low)}-{int(high)}"
    print(f"[WARN] '{n}' can not get bucket")
    return "unknown"


def get_bin_labels(bins):
    bin_labels = []
    for b in bins:
        if isinstance(b, int):
            bin_labels.append(f'{b}')
        else:
            low, high = b
            if high == float("inf"):
                bin_labels.append(f">{int(low)}")
            else:
                bin_labels.append(f"{int(low)}-{int(high)}")
    return bin_labels


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


def compute_box_size(w, h, img_w, img_h, imgsz):
    """
    w,h: normalized
    img_w,img_h: 原图尺寸
    imgsz: 模型输入尺寸
    """
    # 转为原图像素
    bw = w * img_w
    bh = h * img_h

    if imgsz > 0:
        scale = min(imgsz / img_w, imgsz / img_h)
        bw *= scale
        bh *= scale

    return (bw + bh) / 2


class Plotter:
    def __init__(self, save_dir):
        self.save_dir = save_dir

    @staticmethod
    def _auto_layout(n):
        cols = min(4, math.ceil(math.sqrt(n)))
        rows = math.ceil(n / cols)
        return rows, cols

    def creat_mutil_axes(self, class_names):
        n = len(class_names)
        rows, cols = self._auto_layout(n)
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes = axes.flatten() if n > 1 else [axes]
        return fig, axes

    def save_fig(self, fig, axes, class_names, save_name):
        for j in range(len(class_names), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path)
        print(f"[Saved] {save_path}")

    def plot_bar_grid(self, data_dict, class_names, x_labels,
                      title_prefix, xlabel, ylabel, save_name):

        fig, axes = self.creat_mutil_axes(class_names)

        for i, (cls, name) in enumerate(class_names.items()):
            ax = axes[i]
            values = [data_dict[cls].get(b, 0) for b in x_labels]

            bars = ax.bar(x_labels, values)

            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h,
                        str(int(h)), ha='center', va='bottom', fontsize=8)

            ax.set_title(f"{name} - {title_prefix}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        self.save_fig(fig, axes, class_names, save_name)

    def plot_histogram(self, data_dict, class_names, x_labels,
                       title_prefix, xlabel, ylabel, save_name):
        fig, axes = self.creat_mutil_axes(class_names)

        # 等宽坐标：每个bin宽度=1
        x_edges = list(range(len(x_labels)))  # 左边界

        for i, (cls, name) in enumerate(class_names.items()):
            ax = axes[i]
            values = [data_dict[cls].get(b, 0) for b in x_labels]

            # width=1 + align='edge' -> 无缝直方图
            bars = ax.bar(x_edges, values, width=1.0, align='edge', edgecolor='black')

            # 数值标注
            for xi, v in zip(x_edges, values):
                if v > 0:
                    ax.text(xi + 0.5, v, str(int(v)), ha='center', va='bottom', fontsize=8)

            # x轴刻度放在边界位置
            ax.set_xticks(x_edges)
            ax.set_xticklabels([b.split('-')[0] if '-' in b else b for b in x_labels])

            # 限制范围（保证紧贴）
            ax.set_xlim(0, len(x_labels))

            # y轴线性
            ax.set_yscale('linear')

            ax.set_title(f"{name} - {title_prefix}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # 美化
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        self.save_fig(fig, axes, class_names, save_name)


# ===============================
# 主函数
# ===============================
def load_process(args):
    img_path, imgsz = args

    label_path = img_to_label_path(img_path)
    targets = load_labels(label_path)

    img_w, img_h = get_image_size(img_path)

    size_set = []
    class_counter = defaultdict(int)

    for cls, w, h in targets:
        size = compute_box_size(w, h, img_w, img_h, imgsz)
        size_set.append((cls, size))
        class_counter[cls] += 1

    return size_set, class_counter


def analyze(data, split, args):
    plotter = Plotter(data["path"])
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

    with Pool(cpu_count()) as p:
        results = list(tqdm(
            p.imap(load_process, [(p, args.imgsz) for p in image_paths]),
            total=len(image_paths)
        ))

    for size_set, class_counter in results:
        for cls, size in size_set:
            class_count[cls] += 1
            bucket = get_bucket(size, SIZE_BINS)
            size_hist[cls][bucket] += 1

        for cls in class_names.keys():
            n = class_counter.get(cls, 0)
            bucket = get_bucket(n, COUNT_BINS)
            per_image_class[cls][bucket] += 1

    # 1. 类别统计
    print("\n===== 类别统计 =====")
    total = sum(class_count.values())

    for cls, name in class_names.items():
        cnt = class_count[cls]
        ratio = cnt / total * 100 if total > 0 else 0
        print(f"{name}: {cnt} ({ratio:.2f}%)")

    # 2. 每图目标数
    plotter.plot_bar_grid(
        per_image_class,
        class_names,
        x_labels=get_bin_labels(COUNT_BINS),
        title_prefix="objects per image",
        xlabel="count bucket",
        ylabel="image count",
        save_name=f"{split}_objects_count.png",
    )

    # 3. 目标尺寸统计
    plotter.plot_histogram(
        size_hist,
        class_names,
        x_labels=get_bin_labels(SIZE_BINS),
        title_prefix="size distribution",
        xlabel="size (pixel)",
        ylabel="obj count",
        save_name=f"{split}_objects_size.png"
    )


# ===============================
# main
# ===============================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=-1, help="模型输入尺寸缩放后统计，小于零时按原图分辨率统计")
    return parser.parse_args()


def main():
    args = parse_args()
    data = check_det_dataset(args.yaml)
    analyze(data, "train", args)
    analyze(data, "val", args)


if __name__ == "__main__":
    main()
