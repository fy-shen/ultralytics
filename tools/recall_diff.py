import os
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from ultralytics.utils.metrics import box_iou
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.plotting import Colors
from tools.dataset import get_image_size, img_to_label_path

matplotlib.use("Agg")
colors = Colors()


# ==============================
# 工具函数
# ==============================
def find_model_name(p, key='val'):
    find_key = False
    for part in p.parts[::-1]:
        if find_key:
            return part
        if key in part:
            find_key = True
    raise KeyError(f"Can not find model name from path {p} use key {key}")


def load_yolo_labels(path, img_w, img_h):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.array([]), np.zeros((0, 4))

    try:
        data = np.loadtxt(path).reshape(-1, 5)
    except Exception:
        return np.array([]), np.zeros((0, 4))

    cls = data[:, 0].astype(int)

    x = data[:, 1] * img_w
    y = data[:, 2] * img_h
    w = data[:, 3] * img_w
    h = data[:, 4] * img_h
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    return cls, boxes


def match_gt(gt_cls, gt_boxes, pred_cls, pred_boxes, iou_thr):
    if len(gt_boxes) == 0:
        return []

    matched = [False] * len(gt_boxes)
    used_pred = set()

    if len(pred_boxes) == 0:
        return matched

    ious = box_iou(
        torch.from_numpy(gt_boxes).float(),
        torch.from_numpy(pred_boxes).float(),
    )

    for i in range(len(gt_boxes)):
        best_j = -1
        best_iou = iou_thr

        for j in range(len(pred_boxes)):
            if j in used_pred:
                continue
            if gt_cls[i] != pred_cls[j]:
                continue

            if ious[i, j] > best_iou:
                best_iou = ious[i, j]
                best_j = j

        if best_j >= 0:
            matched[i] = True
            used_pred.add(best_j)

    return matched


def draw_worker(args):
    img_path, gt_boxes, gt_cls, matched_a, matched_b, save_path, cls_names = args

    img = cv2.imread(str(img_path))
    ih, iw = img.shape[:2]
    for i, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = map(int, box)
        cls = int(gt_cls[i])
        color = colors(cls, True)

        A = matched_a[i]
        B = matched_b[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)

        if A and not B:
            tag = "A"
        elif B and not A:
            tag = "B"
        elif A and B:
            tag = "AB"
        else:
            tag = "MISS"
        label = f"{cls_names[cls]} {tag}"

        # 背景框
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        if x1 + tw < iw:
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(img, label, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (x2 - tw - 2, y1 - th - 4), (x2, y1), color, -1)
            cv2.putText(img, label, (x2 - tw - 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(save_path), img)


def compute_bins_and_ticks(max_size):
    # 控制 tick 间隔
    if max_size <= 100:
        step = 5
    elif max_size <= 300:
        step = 10
    else:
        step = 20

    bin_step = 5
    bins = np.arange(0, max_size + bin_step, bin_step)
    ticks = np.arange(0, max_size + step, step)

    return bins, ticks, step


def draw_hist_per_class(
        diff_sizes_ab, diff_sizes_ba, total_gt_per_class,
        model_name1, model_name2, cls_names, save_dir
):
    all_classes = set(total_gt_per_class.keys()) | set(diff_sizes_ab.keys()) | set(diff_sizes_ba.keys())

    if len(all_classes) == 0:
        print("No difference to plot.")
        return

    for cls in sorted(all_classes):
        sizes_ab = diff_sizes_ab.get(cls, [])
        sizes_ba = diff_sizes_ba.get(cls, [])
        total_gt = total_gt_per_class.get(cls, 0)

        if total_gt == 0:
            continue

        # 统计信息
        ab_num = len(sizes_ab)
        ba_num = len(sizes_ba)
        ab_ratio = ab_num / total_gt if total_gt > 0 else 0
        ba_ratio = ba_num / total_gt if total_gt > 0 else 0

        # bin 统一
        max_size = 0
        if sizes_ab:
            max_size = max(max_size, max(sizes_ab))
        if sizes_ba:
            max_size = max(max_size, max(sizes_ba))

        max_size = max(int(max_size) + 5, 80)
        bins, ticks, _ = compute_bins_and_ticks(max_size)

        # 画图
        width = max(8, min(20, max_size // 20))
        fig, axes = plt.subplots(2, 1, figsize=(width, 8), sharex=True)

        def plot_ax(ax, data, title):
            counts, edges, _ = ax.hist(data, bins=bins)
            ax.set_title(title)
            ax.set_ylabel("Count")
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

            for count, edge in zip(counts, edges):
                if count > 0:
                    ax.text(edge + 2.5, count, str(int(count)), ha='center', va='bottom', fontsize=8)

        plot_ax(axes[0], sizes_ab, f"{model_name1} detect but {model_name2} miss")
        plot_ax(axes[1], sizes_ba, f"{model_name2} detect but {model_name1} miss")
        axes[1].set_xticks(ticks)
        axes[1].set_xlabel("Object Size")

        fig.suptitle(f"{cls_names[cls]} Recall Difference", fontsize=14, y=0.98)

        # 统计信息
        name_w = max(len(model_name1), len(model_name2), len('ground truth'), 12)

        header = f"{'Model':<{name_w}} | {'Count':^8} | {'Ratio':^8}"
        sep = "-" * (name_w + 3 + 8 + 3 + 8)
        row1 = f"{model_name1:<{name_w}} | {ab_num:>8d} | {ab_ratio:>8.2%}"
        row2 = f"{model_name2:<{name_w}} | {ba_num:>8d} | {ba_ratio:>8.2%}"
        row3 = f"{'ground truth':<{name_w}} | {total_gt:>8d} | {'-':^8}"
        info_text = "\n".join([header, sep, row1, row2, row3])

        fig.text(
            0.5, 0.93,
            info_text,
            ha='center',
            va='top',
            fontsize=11,
            family='monospace'
        )
        plt.tight_layout(rect=(0., 0., 1., 0.85))
        save_path = save_dir / f"{cls_names[cls]}_recall_hist.png"
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"[Class {cls_names[cls]}] GT={total_gt}, A better={ab_num}, B better={ba_num}")


def process_one_image(args):
    img_path, pred1_dir, pred2_dir, iou = args

    gt_path = img_to_label_path(img_path)
    if not gt_path.exists():
        return None

    img_w, img_h = get_image_size(img_path)

    gt_cls, gt_boxes = load_yolo_labels(gt_path, img_w, img_h)
    pred1_cls, pred1_boxes = load_yolo_labels(pred1_dir / gt_path.name, img_w, img_h)
    pred2_cls, pred2_boxes = load_yolo_labels(pred2_dir / gt_path.name, img_w, img_h)

    matched_a = match_gt(gt_cls, gt_boxes, pred1_cls, pred1_boxes, iou)
    matched_b = match_gt(gt_cls, gt_boxes, pred2_cls, pred2_boxes, iou)

    diff_ab = defaultdict(list)
    diff_ba = defaultdict(list)
    total_gt = defaultdict(int)

    # 提前判断是否需要画图
    need_draw = False

    for i in range(len(gt_boxes)):
        cls = gt_cls[i]
        total_gt[cls] += 1

        w = gt_boxes[i][2] - gt_boxes[i][0]
        h = gt_boxes[i][3] - gt_boxes[i][1]
        mean_size = (w + h) / 2

        A = matched_a[i]
        B = matched_b[i]

        if A and not B:
            diff_ab[cls].append(mean_size)
            need_draw = True
        if B and not A:
            diff_ba[cls].append(mean_size)
            need_draw = True

    return diff_ab, diff_ba, total_gt, (img_path, gt_boxes, gt_cls, matched_a, matched_b, need_draw)


# ==============================
def main():
    args = parse_args()
    data = check_det_dataset(args.yaml)

    model_name1 = find_model_name(args.pred1)
    model_name2 = find_model_name(args.pred2)
    save_dir = data['path'] / 'compare' / f'{model_name1}_Diff_{model_name2}'
    img_save_dir = save_dir / 'images'
    img_save_dir.mkdir(parents=True, exist_ok=True)
    hist_save_dir = save_dir / 'fig'
    hist_save_dir.mkdir(parents=True, exist_ok=True)

    txt_path = data[args.split]
    with open(txt_path, "r") as f:
        img_paths = [Path(x.strip()) for x in f if x.strip()]

    diff_sizes_ab = defaultdict(list)
    diff_sizes_ba = defaultdict(list)
    total_gt_per_class = defaultdict(int)

    tasks = [(img_path, args.pred1, args.pred2, args.iou) for img_path in img_paths]
    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(
            process_one_image, tasks, chunksize=max(8, len(tasks) // (os.cpu_count() * 4))), total=len(tasks)
        ))

    draw_tasks = []
    for res in results:
        if res is None:
            continue

        diff_ab, diff_ba, total_gt, draw_info = res
        for k, v in diff_ab.items():
            diff_sizes_ab[k].extend(v)
        for k, v in diff_ba.items():
            diff_sizes_ba[k].extend(v)
        for k, v in total_gt.items():
            total_gt_per_class[k] += v
        # 收集绘图任务
        if args.save_img:
            img_path, gt_boxes, gt_cls, matched_a, matched_b, need_draw = draw_info
            if need_draw:
                draw_tasks.append(
                    (img_path, gt_boxes, gt_cls, matched_a, matched_b, img_save_dir / img_path.name, data['names']))

    if args.save_img and len(draw_tasks) > 0:
        print(f"Saving {len(draw_tasks)} images with threads...")
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
            list(tqdm(executor.map(draw_worker, draw_tasks), total=len(draw_tasks)))

    # 画直方图
    draw_hist_per_class(diff_sizes_ab, diff_sizes_ba, total_gt_per_class,
                        model_name1, model_name2, data['names'], hist_save_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred1", type=Path, required=True, help="model A pred txt dir")
    parser.add_argument("--pred2", type=Path, required=True, help="model B pred txt dir")
    parser.add_argument("--yaml", type=Path, required=True, help="yolo dataset yaml file path")
    parser.add_argument("--split", type=str, default='val', help="train / val / test")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--save-img", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
