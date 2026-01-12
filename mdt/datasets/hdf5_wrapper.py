"""
Wrapper for HDF5Dataset_for_MotoGPT_CALVINLike to adapt to MDT training format
"""
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig, OmegaConf
import torchvision
import hydra
from mdt.models.latent_motion_tokenizer.src.models.latent_motion_tokenizer import LatentMotionTokenizer3D_Simple_Sngl_Dcdr
import os
# in latent_motion_tokenizer.yaml, _target_: latent_motion_tokenizer.src.models.latent_motion_tokenizer.LatentMotionTokenizer
# add search path
import sys
LATENT_MOTION_TOKENIZER_PATH = "/group/ycyang/jfwu/aloha/mdt_policy/mdt/models/"
sys.path.insert(0, LATENT_MOTION_TOKENIZER_PATH)

class MDT_HDF5_Wrapper(Dataset):
    """
    Wrapper to convert HDF5Dataset_for_MotoGPT_CALVINLike output format to MDT format
    
    Args:
        base_dataset: Instance of HDF5Dataset_for_MotoGPT_CALVINLike
        use_left_camera: Whether to use left camera instead of/in addition to gripper camera
        action_mode: How to extract actions from chunk. Options:
            - 'first': Take first action in chunk [default]
            - 'mean': Average actions in chunk
            - 'last': Take last action in chunk
        gen_frame_source: Which future frame to use for image generation loss.
            - 'last': Use last available future frame
            - 'middle': Use middle future frame
            - int: Use specific index
    """
    
    def __init__(
        self,
        base_dataset,
        use_left_camera=False,
        action_mode='first',
        gen_frame_source='last',
        key='vis',  # 'vis' or 'lang'
        batch_size=32,
        num_workers=8,
        transforms_config_path: str = None,
        latent_motion_tokenizer_cfg_path: str = None,
        latent_motion_tokenizer_ckpt_path: str = None,
    ):
        self.base_dataset = base_dataset
        self.use_left_camera = use_left_camera
        self.action_mode = action_mode
        self.gen_frame_source = gen_frame_source
        self.key = key
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = {}
        split = self.base_dataset.split  # 'train' or 'val'
        if transforms_config_path is not None:
            transforms_cfg = OmegaConf.load(transforms_config_path)
            for modality, transforms_list in transforms_cfg[split].items():
                composed = []
                for t_cfg in transforms_list:
                    composed.append(hydra.utils.instantiate(t_cfg))
                self.transforms[modality] = torchvision.transforms.Compose(composed)
        
        # Store tokenizer config paths (will be instantiated in mdt_agent instead)
        self.latent_motion_tokenizer_cfg_path = latent_motion_tokenizer_cfg_path
        self.latent_motion_tokenizer_ckpt_path = latent_motion_tokenizer_ckpt_path
        
        assert action_mode in ['first', 'mean', 'last'], \
            f"action_mode must be 'first', 'mean', or 'last', got {action_mode}"
    
    def __len__(self):
        return len(self.base_dataset)
    
    def _extract_actions(self, actions_chunk, mask_chunk):
        """
        Extract actions from [seq_len, chunk_size, 7] to [seq_len, 7]
        
        Args:
            actions_chunk: [seq_len, chunk_size, 7]
            mask_chunk: [seq_len, chunk_size]
        
        Returns:
            actions: [seq_len, 7]
        """
        if self.action_mode == 'first':
            # Take first valid action in each chunk
            return actions_chunk[:, 0, :]
        elif self.action_mode == 'mean':
            # Average valid actions in each chunk
            actions_sum = (actions_chunk * mask_chunk.unsqueeze(-1)).sum(dim=1)
            actions_count = mask_chunk.sum(dim=1, keepdim=True).clamp(min=1)
            return actions_sum / actions_count
        elif self.action_mode == 'last':
            # Take last valid action in each chunk
            seq_len, chunk_size = mask_chunk.shape
            result = torch.zeros(seq_len, 7)
            for i in range(seq_len):
                valid_indices = mask_chunk[i].nonzero(as_tuple=True)[0]
                if len(valid_indices) > 0:
                    last_valid = valid_indices[-1]
                    result[i] = actions_chunk[i, last_valid]
                else:
                    result[i] = actions_chunk[i, 0]  # fallback to first
            return result
    
    def _get_gen_frame(self, rgb_future, latent_mask):
        """
        Select which future frame to use for image generation loss
        
        Args:
            rgb_future: [seq_len, 3, H, W]
            latent_mask: [seq_len]
        
        Returns:
            gen_frame: [3, H, W]
        """
        valid_indices = latent_mask.nonzero(as_tuple=True)[0]
        
        if len(valid_indices) == 0:
            # No valid frames, return first frame
            return rgb_future[0]
        
        if self.gen_frame_source == 'last':
            return rgb_future[valid_indices[-1]]
        elif self.gen_frame_source == 'middle':
            mid_idx = valid_indices[len(valid_indices) // 2]
            return rgb_future[mid_idx]
        elif isinstance(self.gen_frame_source, int):
            idx = min(self.gen_frame_source, len(valid_indices) - 1)
            return rgb_future[valid_indices[idx]]
        else:
            return rgb_future[valid_indices[-1]]
    
    def __getitem__(self, idx):
        """
        Convert from HDF5Dataset format to MDT format
        
        Input format (from base_dataset):
        {
            "rgb_initial": [1, 3, H, W],
            "rgb_future": [seq_len, 3, H, W],
            "rgb_initial_gripper": [1, 3, H, W],
            "rgb_future_gripper": [seq_len, 3, H, W],
            "rgb_initial_left": [1, 3, H, W],
            "rgb_future_left": [seq_len, 3, H, W],
            "actions": [seq_len, chunk_size, 7],
            "mask": [seq_len, chunk_size],
            "latent_mask": [seq_len],
            "lang": str,
            "idx": int,
            "delta_t": int,
        }
        
        Output format (MDT expected):
        {
            "rgb_obs": {
                "rgb_static": [T+1, 3, H, W],  # concat initial + future
                "rgb_gripper": [T, 3, H, W],
                "gen_static": [3, H, W],  # frame for generation loss
                "gen_gripper": [3, H, W]
            },
            "actions": [T, 7],
            "lang_text": str,
            "future_frame_diff": int,
            "idx": int
        }
        """
        # Get data from base dataset
        data = self.base_dataset[idx]
        
        # Extract actions from chunks
        actions = self._extract_actions(data['actions'], data['mask'])  # [seq_len, 7]
        
        # Concatenate initial + future frames for rgb_static
        rgb_static = torch.cat([
            data['rgb_initial'],      # [1, 3, H, W]
            data['rgb_future']        # [seq_len, 3, H, W]
        ], dim=0)  # [seq_len+1, 3, H, W]
        
        # Choose which camera to use for gripper view
        if self.use_left_camera:
            rgb_gripper_initial = data['rgb_initial_left']
            rgb_gripper_future = data['rgb_future_left']
        else:
            rgb_gripper_initial = data['rgb_initial_gripper']
            rgb_gripper_future = data['rgb_future_gripper']
        
        rgb_gripper = torch.cat([
            rgb_gripper_initial,      # [1, 3, H, W]
            rgb_gripper_future        # [seq_len, 3, H, W]
        ], dim=0)  # [seq_len+1, 3, H, W]
        
        # Select frames for image generation loss
        gen_static = self._get_gen_frame(data['rgb_future'], data['latent_mask'])
        gen_gripper = self._get_gen_frame(rgb_gripper_future, data['latent_mask'])

        if 'rgb_static' in self.transforms:
            rgb_static = self.transforms['rgb_static'](rgb_static)
        if 'gen_static' in self.transforms:
            gen_static = self.transforms['gen_static'](gen_static)
        if 'rgb_gripper' in self.transforms:
            rgb_gripper = self.transforms['rgb_gripper'](rgb_gripper)
        if 'gen_gripper' in self.transforms:
            gen_gripper = self.transforms['gen_gripper'](gen_gripper)
        
        # Prepare images for motion tokenizer (will be computed in mdt_agent)
        rgb_initial_1 = None
        rgb_future_1 = None
        rgb_initial_2 = None
        rgb_future_2 = None
        
        if "rgb_for_latent_motion" in self.transforms:
            rgb_initial_1 = self.transforms['rgb_for_latent_motion'](data['rgb_initial']) # [1, 3, H, W]
            rgb_future_1 = self.transforms['rgb_for_latent_motion'](data['rgb_future'][-1:]) # [1, 3, H, W]
            rgb_initial_2 = self.transforms['rgb_for_latent_motion'](data['rgb_initial_left']) # [1, 3, H, W]
            rgb_future_2 = self.transforms['rgb_for_latent_motion'](data['rgb_future_left'][-1:]) # [1, 3, H, W]

        
        
        # Build output in MDT format
        output = {
            "rgb_obs": {
                "rgb_static": rgb_static,       # [T+1, 3, H, W]
                "rgb_gripper": rgb_gripper,     # [T+1, 3, H, W]
                "gen_static": gen_static,       # [3, H, W]
                "gen_gripper": gen_gripper,     # [3, H, W]
            },
            "actions": actions,                  # [T, 7]
            "lang_text": data['lang'],          # str
            "future_frame_diff": data['delta_t'],  # int
            "idx": data['idx'],                 # int
            # Images for motion tokenizer (to be processed in mdt_agent)
            "rgb_initial_1": rgb_initial_1,     # [1, 3, H, W] or None
            "rgb_future_1": rgb_future_1,       # [1, 3, H, W] or None
            "rgb_initial_2": rgb_initial_2,     # [1, 3, H, W] or None
            "rgb_future_2": rgb_future_2,       # [1, 3, H, W] or None
        }
        
        return output


def mdt_collate_fn(batch):
    """
    Custom collate function to handle MDT data format
    
    Args:
        batch: List of samples from MDT_HDF5_Wrapper
    
    Returns:
        Batched data in MDT format with proper shapes:
        {
            "rgb_obs": {
                "rgb_static": [B, T+1, 3, H, W],
                "rgb_gripper": [B, T+1, 3, H, W],
                "gen_static": [B, 3, H, W],
                "gen_gripper": [B, 3, H, W]
            },
            "actions": [B, T, 7],
            "lang_text": List[str],
            "future_frame_diff": int (avg),
            "idx": [B]
        }
    """
    # Stack rgb observations
    rgb_static = torch.stack([item['rgb_obs']['rgb_static'] for item in batch])
    rgb_gripper = torch.stack([item['rgb_obs']['rgb_gripper'] for item in batch])
    gen_static = torch.stack([item['rgb_obs']['gen_static'] for item in batch])
    gen_gripper = torch.stack([item['rgb_obs']['gen_gripper'] for item in batch])
    
    # Stack actions
    actions = torch.stack([item['actions'] for item in batch])
    
    # Collect language texts
    lang_texts = [item['lang_text'] for item in batch]
    
    # Average future_frame_diff (or take first, they should be similar in a batch)
    future_frame_diff = int(sum(item['future_frame_diff'] for item in batch) / len(batch))
    
    # Stack indices
    indices = torch.tensor([item['idx'] for item in batch])

    # Stack motion images if available
    rgb_initial_1 = None
    rgb_future_1 = None
    rgb_initial_2 = None
    rgb_future_2 = None
    if batch[0].get("rgb_initial_1", None) is not None:
        rgb_initial_1 = torch.stack([item['rgb_initial_1'] for item in batch])  # [B, 1, 3, H, W]
        rgb_future_1 = torch.stack([item['rgb_future_1'] for item in batch])    # [B, 1, 3, H, W]
        rgb_initial_2 = torch.stack([item['rgb_initial_2'] for item in batch])  # [B, 1, 3, H, W]
        rgb_future_2 = torch.stack([item['rgb_future_2'] for item in batch])    # [B, 1, 3, H, W]
    
    return {
        "rgb_obs": {
            "rgb_static": rgb_static,
            "rgb_gripper": rgb_gripper,
            "gen_static": gen_static,
            "gen_gripper": gen_gripper,
        },
        "actions": actions,
        "lang_text": lang_texts,
        "future_frame_diff": future_frame_diff,
        "idx": indices,
        # Motion images for tokenizer (to be computed in mdt_agent)
        "rgb_initial_1": rgb_initial_1,
        "rgb_future_1": rgb_future_1,
        "rgb_initial_2": rgb_initial_2,
        "rgb_future_2": rgb_future_2,
    }
