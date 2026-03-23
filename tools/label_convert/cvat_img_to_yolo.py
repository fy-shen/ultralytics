import os
import argparse
from pathlib import Path
from tqdm import tqdm

from utils import CVATLoader, CVATImg, CVATBox
from ultralytics.data.utils import check_det_dataset


# ===============================
# IoU计算
# ===============================
def xywh_to_xyxy(x, y, w, h):
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return x1, y1, x2, y2


def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = xywh_to_xyxy(*box1)
    x2_min, y2_min, x2_max, y2_max = xywh_to_xyxy(*box2)

    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union = area1 + area2 - inter_area
    return inter_area / union


def iou_filter(boxes, iou_thres):
    """
    boxes: [(cls, x, y, w, h)]
    return: filtered boxes
    """
    keep = []

    # 按类别分组
    cls_dict = {}
    for b in boxes:
        cls_dict.setdefault(b[0], []).append(b)

    for cls, cls_boxes in cls_dict.items():
        # 按面积排序（大框优先）
        cls_boxes = sorted(cls_boxes, key=lambda x: x[3] * x[4], reverse=True)

        selected = []
        for b in cls_boxes:
            keep_flag = True
            for s in selected:
                if abs(b[1] - s[1]) > (b[3] + s[3]) / 2:
                    continue
                if abs(b[2] - s[2]) > (b[4] + s[4]) / 2:
                    continue
                iou = compute_iou(b[1:], s[1:])
                if iou > iou_thres:
                    keep_flag = False
                    break
            if keep_flag:
                selected.append(b)

        keep.extend(selected)

    return keep


# ===============================
# 主流程
# ===============================
def main():
    args = parse_args()
    data = check_det_dataset(args.yaml)
    CLASS = {v: k for k, v in data["names"].items()}

    label_dir = data['path'] / 'labels'
    print(f'[INFO] labels save dir: {label_dir}')
    os.makedirs(str(label_dir), exist_ok=True)

    loader = CVATLoader(args.xml)
    images = loader.root.findall('image')

    # ===== 全局统计 =====
    total_before = 0
    total_after = 0
    affected_images = 0

    task_id, drop_idx = None, 0

    for image in tqdm(images):
        img = CVATImg(image)

        # ===== frame idx 修正 =====
        # CVAT 中 image.name 按 frame_<idx> 无限累计
        new_task = (task_id != img.task_id)
        task_id = img.task_id
        if new_task:
            drop_idx = img.idx
        frame_idx = img.idx - drop_idx

        # ===== 标签路径 =====
        label_fn = f"{loader.tasks[task_id]['source'].split('.')[0]}_{frame_idx:06}.txt"
        label_path = os.path.join(label_dir, label_fn)

        # ===== 收集bbox =====
        boxes = []
        for box in img.boxes:
            b = CVATBox(box, img.w, img.h)

            if b.label not in CLASS:
                print(f"[WARN] unknown label: {b.label}")
                continue
            if b.w <= 0 or b.h <= 0:
                continue

            # cls = 0
            cls = CLASS[b.label]
            boxes.append((cls, b.x, b.y, b.w, b.h))

        before = len(boxes)
        total_before += before

        # ===== IoU过滤 =====
        if not args.disable_iou_filter and before > 0:
            boxes = iou_filter(boxes, args.iou_thres)

        after = len(boxes)
        total_after += after

        if before != after:
            affected_images += 1
            if args.verbose:
                print(f"[IMG] {label_fn}: {before} -> {after}")

        # ===== 写文件 =====
        with open(label_path, mode='w') as fp:
            for cls, x, y, w, h in boxes:
                fp.write(f"{cls} {x/img.w:.6f} {y/img.h:.6f} {w/img.w:.6f} {h/img.h:.6f}\n")

    # ===============================
    # 输出统计
    # ===============================
    removed = total_before - total_after
    remove_ratio = (removed / total_before * 100) if total_before > 0 else 0
    affected_ratio = (affected_images / len(images) * 100) if len(images) > 0 else 0

    print("\n===== IoU Filter Summary =====")
    print(f"Total images: {len(images)}")
    print(f"Total boxes before: {total_before}")
    print(f"Total boxes after:  {total_after}")
    print(f"Removed boxes:      {removed} ({remove_ratio:.2f}%)")
    print(f"Images affected:    {affected_images} ({affected_ratio:.2f}%)")


# ===============================
# 参数
# ===============================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, required=True, help="CVAT xml file path")
    parser.add_argument("--yaml", type=Path, required=True, help="ultralytics dataset yaml file path")
    parser.add_argument("--iou-thres", type=float, default=0.7, help="IoU threshold for duplicate removal")
    parser.add_argument("--disable-iou-filter", action="store_true", help="disable IoU filtering")
    parser.add_argument("--verbose", action="store_true", help="print per-image filtering info")

    return parser.parse_args()


if __name__ == '__main__':
    main()
