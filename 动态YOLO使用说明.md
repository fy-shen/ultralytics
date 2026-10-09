# 动态 YOLO 使用说明

本文档记录本仓库中 `task=motion` 动态 YOLO 的数据处理、训练和验证流程。

## 1. 环境安装

```bash
cd ./ultralytics

pip install -e .
pip install "opencv-python>=4.6.0" "tqdm"
```

## 2. 基本概念

动态 YOLO 在 RGB 图像外额外拼接运动特征图作为输入通道，数据集目录结构如下：

```text
video:   /path/to/dataset/videos/train/xxx.mp4
RGB:     /path/to/dataset/images/train/xxx_000123.jpg
motion:  /path/to/dataset/<motion_name>/train/xxx_000123.jpg
label:   /path/to/dataset/labels/train/xxx_000123.txt
```

训练时会根据参数 `--motion <motion_name>` 来选择加入训练的运动特征，
模型输入通道数必须等于 `3 + motion 特征数量`，例如 2 个运动特征时，输入应为 5 通道。
`yaml_model_load()` 会从模型 yaml 文件名后缀读取通道数，例如 `yolo11s-5.yaml` 表示 5 通道；若不使用运动特征，使用普通 YOLO 训练，就去掉 `-5` 这类后缀。

## 3. 提取运动特征

```bash
python extract_feature.py -h
```

`extract_feature.py` 现在支持两类输出：

- `--frames-only`：只把视频拆成 RGB 图像帧，输出到 `images` 目录，不提取运动特征。
- `--motion ...`：提取运动特征，同时默认补齐完整 RGB 图像帧到 `images` 目录。

可用运动类型：

```text
gd     -> gd_short, gd_long
gde    -> gde_short, gde_long
fgmask -> fgmask
flow   -> flow_x, flow_y
```

### 3.1 只抽取视频帧

单个视频：

```bash
python extract_feature.py \
  --video /path/to/dataset/videos/train/demo.mp4 \
  --frames-only
```

视频目录：

```bash
python extract_feature.py \
  --video-dir /path/to/dataset/videos/train \
  --frames-only \
  --workers 4
```

每 3 帧保存 1 帧：

```bash
python extract_feature.py \
  --video-dir /path/to/dataset/videos/train \
  --frames-only \
  --frame-stride 3 \
  --workers 4
```

`--frame-stride` 只在 `--frames-only` 仅抽 RGB 帧时生效；提取运动特征时不会跳帧。

### 3.2 抽取运动特征

```bash
python extract_feature.py \
  --video-dir /path/to/dataset/videos \
  --motion gde \
  --workers 4
```

提取运动特征时默认会保存完整 RGB 帧，若 RGB 帧已经存在则会跳过，若需强制覆盖使用 `--overwrite-rgb`，
若不需要保存 RGB 帧使用 `--no-save-rgb`。

> 目前实验表明 gde(gray diff enhanced) 效果最好，参数可调，优先使用默认参数

## 4. YOLO

### 4.1 传统yolo

- 训练

```bash
python train_motion.py \
  --task detect \
  --model configs/models/yolov11s.yaml \
  --data configs/dataset/drone_bird.yaml \
  --weights weights/yolo11/yolo11s.pt \
  --epochs 80 --imgsz 1280 --batch 8 --device 0
```

- 视频推理

```shell
python predict.py \
  --model path/to/best.pt \
  --source path/to/video_or_video_dir \
  --output-dir runs/predict \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.7 \
  --device 0
```

### 4.2 融合运动特征

- 训练
```bash
python train_motion.py \
  --task detect \
  --model configs/models/yolo26s-5.yaml \
  --data configs/dataset/drone_bird.yaml \
  --motion gray_diff_enhanced_short gray_diff_enhanced_long \
  --weights weights/motion/yolo26s-5-1280-single-add.pt \
  --single-cls \
  --epochs 80 --imgsz 1280 --batch 8 --device 0
```

- 视频推理
```shell
python predict_motion.py \
  --model path/to/best.pt \
  --source path/to/video_or_video_dir \
  --output-dir runs/predict \
  --motion gde \
  --imgsz 1280 \
  --conf 0.25 \
  --iou 0.7 \
  --device 0
```

