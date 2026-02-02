import torch
from typing import Optional

from ..detect.val import DetectionValidator
from ultralytics.data import build_motion_dataset


class MotionValidator(DetectionValidator):
    def build_dataset(self, img_path: str, mode: str = "val", batch: Optional[int] = None) -> torch.utils.data.Dataset:
        """Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset.
        """
        return build_motion_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

