from __future__ import annotations

import numpy as np
import torch

from ..detect.predict import DetectionPredictor


class MotionPredictor(DetectionPredictor):
    """Predictor for motion models that accept RGB + motion feature channels."""

    def preprocess(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            if im.shape[-1] >= 3:
                rgb = im[..., :3][..., ::-1]  # BGR -> RGB for first 3 channels only
                im = np.concatenate([rgb, im[..., 3:]], axis=-1) if im.shape[-1] > 3 else rgb
            im = im.transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        if not_tensor:
            im /= 255
        return im
