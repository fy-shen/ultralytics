from PIL import Image
from pathlib import Path


def get_image_size(img_path):
    with Image.open(img_path) as img:
        return img.size


def img_to_label_path(img_path: Path):
    return Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
