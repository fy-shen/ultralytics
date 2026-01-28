# 1. 构建模型
 - [创建或读取模型入口](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/engine/model.py#L141-L144)

 - 使用 yaml 创建模型主要通过 [`smart_load`(入口)](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/engine/model.py#L250)

   - [`smart_load`(主体)](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/engine/model.py#L1055)
核心是通过 `task` 和 `key` 
在[`task_map`](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/models/yolo/model.py#L85)中获取到所需要的类，
`task` 可在最开始用`ultralytics.YOLO`创建模型时自定义

   - [`parse_model`](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/nn/tasks.py#L1518) 是加载模型的核心代码


# 2. 训练

 - [加载数据集参数](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/engine/trainer.py#L652-L662)

 - [`trainer.train()`](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/engine/trainer.py#L219)
   - 非 DDP 模式 [`setup_train` 入口](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/engine/trainer.py#L365)
   - [`setup_train` 主函数](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/engine/trainer.py#L266)，
   训练配置的核心函数，涉及读取权重、freeze layers、AMP、Batchsize、Dataloader、Optimizer、Scheduler


- [读取模型权重 `setup_model`](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/engine/trainer.py#L671)

## 2.1 数据集构建
- [`DetectionTrainer.get_dataloader`](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py#L93)
- [`build_dataset`](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py#L77)
- [`build_yolo_dataset`](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/data/build.py#L235)，此处选择 `YOLODataset` 为例
- [`YOLODataset`](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/data/dataset.py#L74) 
初始化主要依赖基类 [`BaseDataset`](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/data/base.py#L108)，
主要是设置一些数据集相关参数，构造图像、标签以及 [`build_transforms`](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/data/dataset.py#L215)
  - transforms 常规使用 [`v8_transforms`](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/data/augment.py#L2417)
  - transforms 包含以下类型
    - pre_transform(`mosaic`, `CopyPaste`, `RandomPerspective`)
    - `MixUp`
    - `CutMix`
    - `Albumentations`
    - `RandomHSV`
    - `RandomFlip`
    - `RandomFlip`
    - `Format`
- [数据集迭代](https://github.com/fy-shen/ultralytics/blob/main/ultralytics/data/base.py#L376)