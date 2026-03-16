import os
import argparse
from tqdm import tqdm

from utils import CVATLoader, CVATImg, CVATBox


CLASS = {
    'bird': 0,
    'drone': 1,
}


def main():
    args = parse_args()
    label_dir = os.path.join(args.save_dir, 'labels')
    os.makedirs(label_dir, exist_ok=True)

    loader = CVATLoader(args.xml)
    images = loader.root.findall('image')
    task_id, drop_idx = None, 0
    for image in tqdm(images):
        img = CVATImg(image)
        info = ''
        # CVAT 中 image.name 按 frame_<idx> 无限累计，需重新定位帧索引
        new_task = (task_id != img.task_id)
        task_id = img.task_id
        if new_task:
            drop_idx = img.idx
        frame_idx = img.idx - drop_idx
        # 标签路径
        label_fn = f"{loader.tasks[task_id]['source'].split('.')[0]}_{frame_idx:06}.txt"
        label_path = os.path.join(label_dir, label_fn)

        for box in img.boxes:
            b = CVATBox(box, img.w, img.h)
            cls = CLASS[b.label]
            info += f"{cls} {b.x} {b.y} {b.w} {b.h}\n"

        with open(label_path, mode='w') as fp:
            fp.writelines(info)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="/home/sfy/SFY/disk1/data/drone_bird/labels_raw/annotations.xml")
    parser.add_argument("--save-dir", type=str, default="/home/sfy/SFY/disk1/data/drone_bird")
    return parser.parse_args()


if __name__ == '__main__':
    main()
