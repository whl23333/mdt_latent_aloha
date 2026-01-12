"""
在新 dataset 中引入 MDT 预处理的完整示例

使用方法：
python examples/use_mdt_preprocessing.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torchvision.transforms as T

# 假设你的新 dataset 类
class MyCustomDataset(torch.utils.data.Dataset):
    """
    自定义 dataset，集成 MDT 预处理
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        sequence_length: int = 16,
        transforms_config_path: str = None,
        use_relative_actions: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.use_relative_actions = use_relative_actions
        
        # ============ 步骤1: 加载 transforms 配置 ============
        if transforms_config_path is None:
            # 使用默认的 ALOHA transforms
            transforms_config_path = Path(__file__).parent.parent / \
                'conf/datamodule/transforms/aloha_transforms.yaml'
        
        transforms_cfg = OmegaConf.load(transforms_config_path)
        
        # ============ 步骤2: 实例化 transforms ============
        self.transforms = self._build_transforms(transforms_cfg, split)
        
        # 加载你的数据索引
        self.data_files = list(self.data_dir.glob('*.npz'))
        print(f"Loaded {len(self.data_files)} files from {data_dir}")
    
    def _build_transforms(self, config, split):
        """从配置构建 transforms"""
        transforms = {}
        for modality, transform_list in config[split].items():
            composed = []
            for t_cfg in transform_list:
                composed.append(instantiate(t_cfg))
            transforms[modality] = T.Compose(composed)
        return transforms
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # ============ 步骤3: 加载原始数据 ============
        data_file = self.data_files[idx]
        data = np.load(data_file)
        
        # 假设数据格式
        rgb_static = data['rgb_static']      # (H, W, 3), uint8
        rgb_gripper = data['rgb_gripper']    # (H, W, 3), uint8
        robot_obs = data['robot_obs']        # (seq_len, 14), float
        actions = data['actions']            # (seq_len, 7), float
        
        # ============ 步骤4: 应用预处理 ============
        # 4.1 RGB 预处理
        # 转为 (C, H, W) byte tensor
        rgb_static_t = torch.from_numpy(rgb_static).byte().permute(2, 0, 1)
        rgb_gripper_t = torch.from_numpy(rgb_gripper).byte().permute(2, 0, 1)
        
        # 应用 transforms (Resize -> RandomShift -> Scale -> Normalize)
        if 'rgb_static' in self.transforms:
            rgb_static_t = self.transforms['rgb_static'](rgb_static_t)
        if 'rgb_gripper' in self.transforms:
            rgb_gripper_t = self.transforms['rgb_gripper'](rgb_gripper_t)
        
        # 4.2 状态观测预处理
        robot_obs_t = torch.from_numpy(robot_obs).float()
        if 'robot_obs' in self.transforms:
            robot_obs_t = self.transforms['robot_obs'](robot_obs_t)
        
        # 4.3 动作预处理（相对动作）
        actions_t = torch.from_numpy(actions).float()
        if self.use_relative_actions and 'actions' in self.transforms:
            # RelativeActions expects numpy arrays
            actions_processed = self.transforms['actions']((actions, robot_obs))
            actions_t = torch.from_numpy(actions_processed).float()
        
        # ============ 步骤5: 序列填充 ============
        current_len = robot_obs_t.shape[0]
        if current_len < self.sequence_length:
            pad_size = self.sequence_length - current_len
            
            # 重复最后一帧
            robot_obs_t = torch.cat([
                robot_obs_t,
                robot_obs_t[-1:].repeat(pad_size, 1)
            ], dim=0)
            
            # 动作：相对动作零填充（除夹爪），绝对动作重复
            if self.use_relative_actions:
                actions_pad = torch.zeros(pad_size, actions_t.shape[1])
                actions_pad[:, -1] = actions_t[-1, -1]  # 保持夹爪状态
                actions_t = torch.cat([actions_t, actions_pad], dim=0)
            else:
                actions_t = torch.cat([
                    actions_t,
                    actions_t[-1:].repeat(pad_size, 1)
                ], dim=0)
        
        # ============ 步骤6: 返回格式化数据 ============
        return {
            'rgb_obs': {
                'rgb_static': rgb_static_t,      # (3, 224, 224), float, normalized
                'rgb_gripper': rgb_gripper_t,    # (3, 84, 84), float, normalized
            },
            'robot_obs': robot_obs_t,            # (seq_len, 14), float, normalized
            'actions': actions_t,                # (seq_len, 7), float
            'idx': idx,
        }


# ============ 使用示例 ============
if __name__ == '__main__':
    import numpy as np
    
    # 1. 创建示例数据（模拟你的真实数据）
    example_data_dir = Path('/tmp/example_dataset')
    example_data_dir.mkdir(exist_ok=True)
    
    for i in range(5):
        np.savez(
            example_data_dir / f'episode_{i:03d}.npz',
            rgb_static=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            rgb_gripper=np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
            robot_obs=np.random.randn(20, 14).astype(np.float32),
            actions=np.random.randn(20, 7).astype(np.float32),
        )
    
    # 2. 创建 dataset
    dataset = MyCustomDataset(
        data_dir=str(example_data_dir),
        split='train',
        sequence_length=16,
        use_relative_actions=False,
    )
    
    # 3. 测试数据加载
    sample = dataset[0]
    
    print("\n========== Dataset Output ==========")
    print(f"RGB static shape: {sample['rgb_obs']['rgb_static'].shape}")
    print(f"RGB static dtype: {sample['rgb_obs']['rgb_static'].dtype}")
    print(f"RGB static range: [{sample['rgb_obs']['rgb_static'].min():.3f}, {sample['rgb_obs']['rgb_static'].max():.3f}]")
    
    print(f"\nRGB gripper shape: {sample['rgb_obs']['rgb_gripper'].shape}")
    print(f"Robot obs shape: {sample['robot_obs'].shape}")
    print(f"Actions shape: {sample['actions'].shape}")
    
    # 4. 创建 DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )
    
    batch = next(iter(dataloader))
    print(f"\n========== Batched Output ==========")
    print(f"Batch RGB static: {batch['rgb_obs']['rgb_static'].shape}")
    print(f"Batch robot obs: {batch['robot_obs'].shape}")
    print(f"Batch actions: {batch['actions'].shape}")
    
    print("\n✓ 预处理集成成功！")
