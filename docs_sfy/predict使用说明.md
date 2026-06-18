# predict.py 使用说明

## 1. 功能概述

`predict.py` 用于目标检测推理，支持两种模式：

- `RGB检测模式`：不传 `--motion`，按普通检测模型推理（`task=detect`）。
- `运动特征模式`：传入 `--motion`，按多通道运动模型推理（`task=motion`）。

支持输入类型：

- 单个图片 / 单个视频
- 图片目录 / 视频目录（目录下不要混放图片和视频）
- YOLO 数据集 `yaml`

---

## 2. 基本命令

### 2.1 RGB检测（不使用运动特征）

```bash
python predict.py \
  --model runs/train/weights/best.pt \
  --source /path/to/video_or_images \
  --save-dir runs/motion_predict
```

### 2.2 运动特征推理（示例：gray_diff 两通道）

```bash
python predict.py \
  --model runs/train_motion/weights/best.pt \
  --source /path/to/images_or_video \
  --motion gray_diff \
  --save-dir runs/motion_predict
```

也可按特征名指定：

```bash
--motion gray_diff_short gray_diff_long
```

---

## 3. 输入与输出规则

### 3.1 `--source` 为视频

- 输出为 `save-dir/视频名.mp4`
- 多视频目录可并行处理（`--video-workers`）

### 3.2 `--source` 为图片

- `--image-output video`（默认）：每个分组输出一个 `mp4`
- `--image-output image`：每组输出到 `save-dir/组名/*.jpg`

图片分组默认按文件名规则：`视频名_帧号.jpg`（例如 `demo_000123.jpg`）。

### 3.3 `--source` 为数据集 yaml

- 从 `--split`（`train/val/test`）读取图像
- 会严格检查同名前缀跨目录冲突，避免不同目录同组混合

---

## 4. 常用参数

- `--imgsz`：推理尺寸，默认 `1280`
- `--conf`：置信度阈值，默认 `0.25`
- `--iou`：NMS IoU 阈值，默认 `0.7`
- `--device`：设备，如 `0`、`cpu`
- `--overwrite`：覆盖已有输出
- `--video-workers`：并行 worker 数，默认 `4`
- `--no-progress`：关闭进度条
- `--show-conf/--no-show-conf`：是否显示置信度
- `--show-labels/--no-show-labels`：是否显示标签
- `--show-boxes/--no-show-boxes`：是否显示框

运动参数会随 `--motion` 自动生效，例如：

- `--gray-diff-alpha`
- `--fgmask-history --fgmask-var-threshold --fgmask-kernel-size --fgmask-min-area`
- `--flow-bound`

---

## 5. 常见问题

- 报错 `Model expects X motion channels`：
  `--motion` 选择的特征数与模型输入通道不一致，需与训练时一致。
- 报错 `Motion image 'xxx' not found`：
  图片模式下未找到对应运动图，请检查目录结构和命名。
- 报错 `Mixed image and video files found`：
  一个目录中同时存在图片和视频，需拆分后分别推理。
- 输出存在但被跳过：
  默认不覆盖，添加 `--overwrite`。

---

## 6. 建议实践

- 首次跑建议先设置 `--video-workers 1`，确认流程和输入结构正确后再提升并行。
- 运动模型推理时，`--motion` 参数建议直接复用训练配置，避免通道不匹配。
- 批量任务建议固定 `--save-dir` 并启用 `--overwrite`（按需要）控制重跑行为。
