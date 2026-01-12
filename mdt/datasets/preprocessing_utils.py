"""
预处理工具函数，可直接用于新 dataset
"""
import torch
from typing import Dict
from omegaconf import DictConfig
from hydra.utils import instantiate
import torchvision.transforms as transforms


def build_transforms(transforms_config: DictConfig, split: str = 'train') -> Dict:
    """
    从配置构建 transforms
    
    Args:
        transforms_config: Transform 配置 (from yaml)
        split: 'train' or 'val'
    
    Returns:
        Dict of composed transforms for each modality
    """
    transform_dict = {}
    
    for modality, transform_list in transforms_config[split].items():
        composed_transforms = []
        for transform_cfg in transform_list:
            composed_transforms.append(instantiate(transform_cfg))
        transform_dict[modality] = transforms.Compose(composed_transforms)
    
    return transform_dict


def apply_rgb_transforms(
    rgb_data: torch.Tensor, 
    transforms_dict: Dict, 
    modality_key: str
) -> torch.Tensor:
    """
    应用 RGB transforms
    
    Args:
        rgb_data: (H, W, C) numpy array or (C, H, W) tensor, uint8
        transforms_dict: Transform 字典
        modality_key: 如 'rgb_static', 'rgb_gripper'
    
    Returns:
        Transformed tensor (C, H, W), float32, normalized
    """
    # 确保格式正确
    if isinstance(rgb_data, torch.Tensor):
        if rgb_data.dim() == 3 and rgb_data.shape[-1] in [1, 3]:
            # (H, W, C) -> (C, H, W)
            rgb_data = rgb_data.permute(2, 0, 1)
    else:
        # numpy array
        rgb_data = torch.from_numpy(rgb_data).byte()
        if rgb_data.dim() == 3 and rgb_data.shape[-1] in [1, 3]:
            rgb_data = rgb_data.permute(2, 0, 1)
    
    # 应用 transforms
    if modality_key in transforms_dict:
        rgb_data = transforms_dict[modality_key](rgb_data)
    else:
        # 默认预处理：scale to [0,1] and normalize
        rgb_data = rgb_data.float() / 255.0
    
    return rgb_data


def apply_state_transforms(
    state_data: torch.Tensor,
    transforms_dict: Dict,
    modality_key: str = 'robot_obs'
) -> torch.Tensor:
    """
    应用状态观测 transforms
    
    Args:
        state_data: (seq_len, state_dim) tensor, float
        transforms_dict: Transform 字典
        modality_key: 如 'robot_obs', 'scene_obs'
    
    Returns:
        Transformed tensor
    """
    if not isinstance(state_data, torch.Tensor):
        state_data = torch.from_numpy(state_data).float()
    
    if modality_key in transforms_dict:
        state_data = transforms_dict[modality_key](state_data)
    
    return state_data


def apply_action_transforms(
    actions: torch.Tensor,
    robot_obs: torch.Tensor,
    transforms_dict: Dict,
    use_relative: bool = False
) -> torch.Tensor:
    """
    应用动作 transforms（可选相对动作）
    
    Args:
        actions: (seq_len, action_dim) tensor
        robot_obs: (seq_len, state_dim) tensor for relative actions
        transforms_dict: Transform 字典
        use_relative: 是否使用相对动作
    
    Returns:
        Transformed actions
    """
    if not isinstance(actions, torch.Tensor):
        actions = torch.from_numpy(actions).float()
    
    if use_relative and 'actions' in transforms_dict:
        # RelativeActions transform expects tuple (actions, robot_obs)
        actions = transforms_dict['actions']((
            actions.numpy(), 
            robot_obs.numpy()
        ))
        actions = torch.from_numpy(actions).float()
    
    return actions


def pad_sequence_to_length(
    sequence_dict: Dict[str, torch.Tensor],
    target_length: int,
    pad_mode: str = 'replicate'
) -> Dict[str, torch.Tensor]:
    """
    填充序列到目标长度
    
    Args:
        sequence_dict: 包含各模态数据的字典
        target_length: 目标序列长度
        pad_mode: 'replicate' (重复最后帧) 或 'zero' (零填充)
    
    Returns:
        填充后的序列字典
    """
    padded_dict = {}
    
    for key, value in sequence_dict.items():
        if not isinstance(value, torch.Tensor):
            padded_dict[key] = value
            continue
        
        current_length = value.shape[0]
        if current_length >= target_length:
            padded_dict[key] = value[:target_length]
            continue
        
        pad_size = target_length - current_length
        
        if pad_mode == 'replicate':
            # 重复最后一帧
            last_frame = value[-1:].expand(pad_size, *value.shape[1:])
            padded_dict[key] = torch.cat([value, last_frame], dim=0)
        else:  # zero padding
            pad_shape = (pad_size,) + value.shape[1:]
            zeros = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
            padded_dict[key] = torch.cat([value, zeros], dim=0)
    
    return padded_dict


# 示例：在新 dataset 中使用
class ExampleNewDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transforms_config: DictConfig,
        split: str = 'train',
        sequence_length: int = 16,
        use_relative_actions: bool = False,
    ):
        self.data_dir = data_dir
        self.split = split
        self.sequence_length = sequence_length
        self.use_relative_actions = use_relative_actions
        
        # 构建 transforms
        self.transforms = build_transforms(transforms_config, split)
        
        # 加载数据索引等...
    
    def __getitem__(self, idx):
        # 1. 加载原始数据
        rgb_static = self.load_rgb_static(idx)      # (H, W, 3)
        rgb_gripper = self.load_rgb_gripper(idx)    # (H, W, 3)
        robot_obs = self.load_robot_obs(idx)        # (seq_len, state_dim)
        actions = self.load_actions(idx)            # (seq_len, action_dim)
        
        # 2. 应用 transforms
        rgb_static = apply_rgb_transforms(
            rgb_static, self.transforms, 'rgb_static'
        )
        rgb_gripper = apply_rgb_transforms(
            rgb_gripper, self.transforms, 'rgb_gripper'
        )
        robot_obs = apply_state_transforms(
            robot_obs, self.transforms, 'robot_obs'
        )
        actions = apply_action_transforms(
            actions, robot_obs, self.transforms, self.use_relative_actions
        )
        
        # 3. 填充到固定长度
        sequence = {
            'robot_obs': robot_obs,
            'actions': actions,
        }
        sequence = pad_sequence_to_length(sequence, self.sequence_length)
        
        # 4. 返回格式化数据
        return {
            'rgb_obs': {
                'rgb_static': rgb_static,
                'rgb_gripper': rgb_gripper,
            },
            'robot_obs': sequence['robot_obs'],
            'actions': sequence['actions'],
            'idx': idx,
        }
    
    def __len__(self):
        return self.num_samples
