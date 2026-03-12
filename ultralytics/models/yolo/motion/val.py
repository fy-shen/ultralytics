import torch
from typing import Optional

from ..detect.val import DetectionValidator
from ultralytics.data import build_motion_dataset


class MotionValidator(DetectionValidator):
    def build_dataset(self, img_path: str, mode: str = "val", batch: Optional[int] = None) -> torch.utils.data.Dataset:
        return build_motion_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

