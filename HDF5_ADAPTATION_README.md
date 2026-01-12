# HDF5 Dataset Adaptation for MDT Training

## 快速开始

### 1. 数据准备

确保你的HDF5数据按以下结构组织：

```
your_dataset/
├── train/
│   ├── task1/
│   │   ├── episode_0.hdf5
│   │   ├── episode_1.hdf5
│   │   └── instr.txt (可选，用于语言标注)
│   └── task2/
│       └── ...
└── val/
    └── (同样的结构)
```

### 2. 测试数据集

运行测试脚本验证数据格式：

```bash
cd /group/ycyang/jfwu/aloha/mdt_policy
python test_hdf5_datamodule.py --hdf5_dir /path/to/your/dataset --batch_size 4
```

### 3. 配置训练

修改 `conf/config.yaml` 或创建新配置：

```yaml
datamodule:
  _target_: mdt.datasets.hdf5_data_module.HDF5DataModule
  hdf5_dir: /path/to/your/hdf5/dataset
  sequence_length: 10    # 动作序列长度
  chunk_size: 3          # 原始数据的chunk大小
  skip_frame: 1          # 帧跳跃
  batch_size: 32
  num_workers: 8
  rgb_shape: [224, 224]
  modalities: ["vis"]    # 或 ["vis", "lang"]
  
  # 高级选项
  action_mode: "first"           # 从chunk提取动作的方式
  gen_frame_source: "last"       # 图像生成损失使用哪一帧
  max_skip_frame: 5              # 随机帧跳跃范围（数据增强）
  use_left_camera: false         # 使用左侧相机还是右侧
```

### 4. 开始训练

```bash
python mdt/training.py
```

## 数据格式转换说明

### 输入格式（你的HDF5Dataset）

```python
{
    "rgb_initial": [1, 3, H, W],              # 初始帧
    "rgb_future": [seq_len, 3, H, W],         # 未来帧序列
    "rgb_initial_gripper": [1, 3, H, W],      # 抓手初始帧
    "rgb_future_gripper": [seq_len, 3, H, W], # 抓手未来帧
    "actions": [seq_len, chunk_size, 7],      # 动作chunks
    "mask": [seq_len, chunk_size],            # 动作有效性mask
    "lang": str,                               # 语言指令
    "idx": int,
    "delta_t": int,                            # 帧间隔
}
```

### 输出格式（MDT期望）

```python
{
    "rgb_obs": {
        "rgb_static": [B, T+1, 3, H, W],      # 静态相机（拼接后）
        "rgb_gripper": [B, T+1, 3, H, W],     # 抓手相机（拼接后）
        "gen_static": [B, 3, H, W],           # 用于生成损失
        "gen_gripper": [B, 3, H, W],          # 用于生成损失
    },
    "actions": [B, T, 7],                     # 从chunks提取的动作
    "lang_text": List[str],                   # 语言指令列表
    "future_frame_diff": int,                 # 帧间隔
    "idx": [B],                               # 样本索引
}
```

## 关键参数说明

### action_mode
控制如何从 `[seq_len, chunk_size, 7]` 提取为 `[seq_len, 7]`：
- `"first"`: 取每个chunk的第一个动作（默认，推荐）
- `"mean"`: 对chunk内的动作求平均
- `"last"`: 取每个chunk的最后一个动作

### gen_frame_source
选择哪一帧用于图像生成损失（masked autoencoder）：
- `"last"`: 使用最后一个有效的未来帧（默认）
- `"middle"`: 使用中间的未来帧
- `int`: 使用指定索引的未来帧

### max_skip_frame
帧跳跃的范围，用于时间增强：
- `null`: 固定使用 `skip_frame`
- `int`: 在 `[skip_frame, max_skip_frame]` 范围内随机采样

## 代码结构

```
mdt_policy/mdt/datasets/
├── hdf5_datasets.py           # 你的原始HDF5数据集
├── hdf5_wrapper.py            # 格式转换包装类
├── hdf5_data_module.py        # PyTorch Lightning DataModule
└── ...

mdt_policy/
├── test_hdf5_datamodule.py    # 测试脚本
└── conf/datamodule/
    └── hdf5.yaml              # 配置文件模板
```

## 工作流程

1. **HDF5Dataset** 读取原始数据
2. **MDT_HDF5_Wrapper** 转换单个样本格式
3. **mdt_collate_fn** 将样本batch化为MDT格式
4. **HDF5DataModule** 管理train/val dataloaders
5. **MDTAgent.training_step** 使用转换后的数据训练

## 故障排查

### 问题：形状不匹配
- 检查 `sequence_length` 和 `chunk_size` 是否与数据一致
- 运行测试脚本查看实际形状

### 问题：找不到数据
- 确保目录结构正确（train/val子目录）
- 检查HDF5文件命名格式：`episode_*.hdf5`

### 问题：语言指令为空
- 确保有 `instr.txt` 文件
- 格式：每行 `<episode_id> <instruction>`
- 或在训练时只使用视觉模态 `modalities: ["vis"]`

## 扩展支持

### 添加更多相机视角
修改 `hdf5_wrapper.py` 中的 `__getitem__` 方法，在 `rgb_obs` 中添加新的键。

### 自定义动作维度
修改代码中的硬编码维度 `7` 为你的动作维度。

### 添加本体感觉状态
在 wrapper 中添加对 robot state/proprio 的处理。

## 性能优化建议

1. **增加 num_workers**: 根据CPU核心数调整（建议8-16）
2. **调整 batch_size**: 根据GPU内存调整
3. **使用 prefetch_factor**: 默认为2，可以增加
4. **max_skip_frame**: 提供时间数据增强，提高泛化性

## 联系

如有问题，请检查：
1. 运行测试脚本的输出
2. 数据加载的形状是否正确
3. MDT模型的输入期望是否匹配
