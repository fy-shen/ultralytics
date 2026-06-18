# coding: utf-8
"""
将 CVAT interpolation XML 标注转换为 YOLO 检测标签

输入:
    annotations.xml

输出:
    labels/
        300m楼_000000.txt
        300m楼_000001.txt
        ...

图像命名格式:
    <视频名称>_<左侧补零6位数字索引>.jpg

例如:
    300m楼_000123.jpg
    对应:
    300m楼_000123.txt
"""

import os
import xml.etree.ElementTree as ET
from collections import defaultdict


# =========================
# 配置部分
# =========================
XML_PATH = "/home/sfy/SFY/disk1/data/300m天/annotations.xml"

# 输出标签目录
OUTPUT_DIR = "/home/sfy/SFY/disk1/data/drone_bird/labels/0519"

# 类别映射
CLASS_MAP = {
    "drone": 0,
    "bird": 1,
}

# 是否为 frame 创建空 txt
# True: 没目标也生成空文件
# False: 仅生成有目标的文件
CREATE_EMPTY_LABEL = True


# =========================
# 解析 XML
# =========================
tree = ET.parse(XML_PATH)
root = tree.getroot()

# 图像尺寸
meta = root.find("meta")
task = meta.find("task")

width = int(task.find("original_size/width").text)
height = int(task.find("original_size/height").text)

# 视频名称
video_name = task.find("name").text
video_name = os.path.splitext(video_name)[0]

# 总帧数
start_frame = int(task.find("start_frame").text)
stop_frame = int(task.find("stop_frame").text)

print(f"Video Name : {video_name}")
print(f"Image Size : {width} x {height}")
print(f"Frame Range: {start_frame} ~ {stop_frame}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# frame_id -> labels
frame_labels = defaultdict(list)


# =========================
# 处理 track
# =========================
for track in root.findall("track"):

    label_name = track.attrib["label"]

    if label_name not in CLASS_MAP:
        print(f"Skip unknown label: {label_name}")
        continue

    cls_id = CLASS_MAP[label_name]

    for box in track.findall("box"):

        # outside=1 表示目标离开画面
        outside = int(box.attrib.get("outside", 0))
        if outside == 1:
            continue

        frame_id = int(box.attrib["frame"])

        xtl = float(box.attrib["xtl"])
        ytl = float(box.attrib["ytl"])
        xbr = float(box.attrib["xbr"])
        ybr = float(box.attrib["ybr"])

        # =========================
        # 转 YOLO 格式
        # =========================
        x_center = (xtl + xbr) / 2.0
        y_center = (ytl + ybr) / 2.0

        box_w = xbr - xtl
        box_h = ybr - ytl

        # 归一化
        x_center /= width
        y_center /= height
        box_w /= width
        box_h /= height

        yolo_line = (
            f"{cls_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_w:.6f} "
            f"{box_h:.6f}"
        )

        frame_labels[frame_id].append(yolo_line)


# =========================
# 写出 txt
# =========================
for frame_id in range(start_frame, stop_frame + 1):

    label_path = os.path.join(
        OUTPUT_DIR,
        f"{video_name}_{frame_id:06d}.txt"
    )

    labels = frame_labels.get(frame_id, [])

    if len(labels) == 0 and not CREATE_EMPTY_LABEL:
        continue

    with open(label_path, "w", encoding="utf-8") as f:
        if labels:
            f.write("\n".join(labels))

print(f"Done! Labels saved to: {OUTPUT_DIR}")