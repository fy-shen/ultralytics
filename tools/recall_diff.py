import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from ultralytics.utils.metrics import box_iou
import matplotlib

matplotlib.use("Agg")

# ==============================
# 参数区
# ==============================
GT_DIR = "/home/sfy/SFY/disk1/data/FBD-SV-2024/labels/val"
PRED_A_DIR = "/home/sfy/SFY/disk1/data/FBD-SV-2024/runs/detect/yolo26s/yolo26s-1280/val/labels"
PRED_B_DIR = "/home/sfy/SFY/disk1/data/FBD-SV-2024/runs/detect/yolo26s/yolo26s-640/val/labels"
A, B = "yolo26s-1280", "yolo26s-640"
IMG_DIR = "/home/sfy/SFY/disk1/data/FBD-SV-2024/images/val"
SAVE_DIR = f"/home/sfy/SFY/disk1/data/FBD-SV-2024/compare/{A}_Diff_{B}/images"
IOU_THRES = 0.5
SAVE_IMG = True

os.makedirs(SAVE_DIR, exist_ok=True)


# ==============================
# 工具函数
# ==============================
def load_yolo_labels(path, img_w, img_h, is_pred=False):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.array([]), np.zeros((0, 4))

    data = np.loadtxt(path)
    data = data.reshape(-1, 5)

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


def match_gt(gt_cls, gt_boxes, pred_cls, pred_boxes):
    if len(gt_boxes) == 0:
        return []

    matched = [False] * len(gt_boxes)

    if len(pred_boxes) == 0:
        return matched

    ious = box_iou(
        torch.tensor(gt_boxes, dtype=torch.float32),
        torch.tensor(pred_boxes, dtype=torch.float32)
    )

    for i in range(len(gt_boxes)):
        for j in range(len(pred_boxes)):
            if gt_cls[i] == pred_cls[j] and ious[i, j] > IOU_THRES:
                matched[i] = True
                break

    return matched


def draw_and_save_image(img_path, gt_boxes, matched_a, matched_b, save_path):
    img = cv2.imread(img_path)

    all_blue = True

    for i, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = map(int, box)

        A = matched_a[i]
        B = matched_b[i]

        if A and not B:
            color = (0, 0, 255)  # 红
            all_blue = False
        elif not A and B:
            color = (0, 255, 0)  # 绿
            all_blue = False
        elif not A and not B:
            color = (255, 0, 255)  # 紫
            all_blue = False
        else:
            color = (255, 0, 0)  # 蓝

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    if not all_blue:
        cv2.imwrite(save_path, img)


def draw_hist(diff_sizes_ab, diff_sizes_ba, total_gt):
    if len(diff_sizes_ab) == 0 and len(diff_sizes_ba) == 0:
        print("No difference to plot.")
        return

    # ==============================
    # 统一最大尺寸
    # ==============================
    max_size = 0
    if len(diff_sizes_ab) > 0:
        max_size = max(max_size, max(diff_sizes_ab))
    if len(diff_sizes_ba) > 0:
        max_size = max(max_size, max(diff_sizes_ba))

    max_size = int(max_size) + 5
    bins = np.arange(0, max_size + 5, 5)

    # ==============================
    # 创建上下子图
    # ==============================
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    def raw_sub_ax(ax, diff_sizes, title):
        counts, edges, _ = ax.hist(diff_sizes, bins=bins)
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        # 标注柱子数量
        for count, edge in zip(counts, edges):
            if count > 0:
                ax.text(edge + 2.5, count, str(int(count)), ha='center', va='bottom', fontsize=8)
        # 标注统计信息
        ax.text(
            0.98, 0.95,
            f"Total GT: {total_gt}\n"
            f"Diff Num: {len(diff_sizes)}",
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.6)
        )
        # 统一 X 轴刻度（10 像素间隔）
        ax.set_xticks(np.arange(0, max_size + 10, 10))

    raw_sub_ax(axes[0], diff_sizes_ab, f"{A} detect but {B} miss")
    raw_sub_ax(axes[1], diff_sizes_ba, f"{B} detect but {A} miss")

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(SAVE_DIR), "recall_hist.png"), dpi=150)
    plt.close()


# ==============================
# 主逻辑
# ==============================
def main():
    gt_files = glob.glob(os.path.join(GT_DIR, "*.txt"))

    diff_sizes_ab = []
    diff_sizes_ba = []

    total_gt = 0

    for gt_path in tqdm(gt_files):
        name = os.path.basename(gt_path)

        pred_a_path = os.path.join(PRED_A_DIR, name)
        pred_b_path = os.path.join(PRED_B_DIR, name)

        img_name = name.replace(".txt", ".jpg")
        img_path = os.path.join(IMG_DIR, img_name)

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        gt_cls, gt_boxes = load_yolo_labels(gt_path, img_w, img_h)
        pred_a_cls, pred_a_boxes = load_yolo_labels(pred_a_path, img_w, img_h, True)
        pred_b_cls, pred_b_boxes = load_yolo_labels(pred_b_path, img_w, img_h, True)

        matched_a = match_gt(gt_cls, gt_boxes, pred_a_cls, pred_a_boxes)
        matched_b = match_gt(gt_cls, gt_boxes, pred_b_cls, pred_b_boxes)
        if SAVE_IMG:
            draw_and_save_image(
                img_path,
                gt_boxes,
                matched_a,
                matched_b,
                os.path.join(SAVE_DIR, img_name)
            )

        total_gt += len(gt_boxes)
        for i in range(len(gt_boxes)):
            w = gt_boxes[i][2] - gt_boxes[i][0]
            h = gt_boxes[i][3] - gt_boxes[i][1]
            mean_size = (w + h) / 2
            if matched_a[i] and not matched_b[i]:
                diff_sizes_ab.append(mean_size)
            if matched_b[i] and not matched_a[i]:
                diff_sizes_ba.append(mean_size)

    print("Total GT:", total_gt)
    print("A detect but B miss:", len(diff_sizes_ab))
    print("B detect but A miss:", len(diff_sizes_ba))

    if len(diff_sizes_ab) == 0:
        print("No difference found.")
        return

    # 画直方图
    draw_hist(diff_sizes_ab, diff_sizes_ba, total_gt)


if __name__ == "__main__":
    main()
