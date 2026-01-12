"""
DataModule for HDF5 datasets adapted to MDT training format
"""
import logging
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from mdt.datasets.hdf5_datasets import HDF5Dataset_for_MotoGPT_CALVINLike
from mdt.datasets.hdf5_wrapper import MDT_HDF5_Wrapper, mdt_collate_fn

logger = logging.getLogger(__name__)


class HDF5DataModule(pl.LightningDataModule):
    """
    DataModule for HDF5 datasets with MDT-compatible format
    
    Args:
        hdf5_dir: Root directory containing train/val subdirectories with HDF5 files
        sequence_length: Length of action sequence window
        chunk_size: Number of actions per chunk in raw data
        skip_frame: Frame skip for temporal subsampling
        batch_size: Batch size for training and validation
        num_workers: Number of dataloader workers
        rgb_shape: Tuple of (height, width) for RGB images
        use_left_camera: Whether to use left camera view
        action_mode: How to extract actions from chunks ('first', 'mean', 'last')
        gen_frame_source: Which frame to use for generation loss ('last', 'middle', or int)
        max_skip_frame: Maximum frame skip (for random augmentation)
        camera_key: Key for main camera in HDF5
        qpos_key: Key for qpos in HDF5
        camera_gripper_key: Key for gripper camera in HDF5
        camera_left_key: Key for left camera in HDF5
        modalities: List of modality keys (e.g., ['vis'] or ['vis', 'lang'])
    """
    
    def __init__(
        self,
        hdf5_dir: str,
        sequence_length: int = 10,
        chunk_size: int = 3,
        skip_frame: int = 1,
        batch_size: int = 32,
        num_workers: int = 8,
        rgb_shape: tuple = (224, 224),
        use_left_camera: bool = False,
        action_mode: str = 'first',
        gen_frame_source: str = 'last',
        max_skip_frame: int = None,
        camera_key: str = "observations/images/cam_high",
        qpos_key: str = "observations/qpos",
        camera_gripper_key: str = "observations/images/cam_right_wrist",
        camera_left_key: str = "observations/images/cam_left_wrist",
        modalities: list = None,
        rgb_preprocessor: DictConfig = None,
        transforms_config_path: str = None,
        latent_motion_tokenizer_cfg_path: str = None,
        latent_motion_tokenizer_ckpt_path: str = None,
        **kwargs
    ):
        super().__init__()
        
        self.hdf5_dir = Path(hdf5_dir)
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.skip_frame = skip_frame
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rgb_shape = rgb_shape
        self.use_left_camera = use_left_camera
        self.action_mode = action_mode
        self.gen_frame_source = gen_frame_source
        self.max_skip_frame = max_skip_frame
        
        self.camera_key = camera_key
        self.qpos_key = qpos_key
        self.camera_gripper_key = camera_gripper_key
        self.camera_left_key = camera_left_key
        self.transforms_config_path = transforms_config_path

        self.latent_motion_tokenizer_cfg_path = latent_motion_tokenizer_cfg_path
        self.latent_motion_tokenizer_ckpt_path = latent_motion_tokenizer_ckpt_path
        
        # Set modalities - default to vision only
        self.modalities = modalities if modalities is not None else ['vis']
        
        self.rgb_preprocessor = rgb_preprocessor
        self.train_datasets = None
        self.val_datasets = None
        
        # Verify directory structure
        if not self.hdf5_dir.exists():
            raise ValueError(f"HDF5 directory does not exist: {self.hdf5_dir}")
        
        self.train_dir = self.hdf5_dir / "train"
        self.val_dir = self.hdf5_dir / "val"
        
        if not self.train_dir.exists() or not self.val_dir.exists():
            raise ValueError(
                f"HDF5 directory must contain 'train' and 'val' subdirectories. "
                f"Found: {list(self.hdf5_dir.iterdir())}"
            )
    
    def setup(self, stage=None):
        """
        Setup datasets for training and validation
        """
        logger.info(f"Setting up HDF5 datasets from {self.hdf5_dir}")
        
        # Create base datasets
        train_base_dataset = HDF5Dataset_for_MotoGPT_CALVINLike(
            hdf5_dir=self.hdf5_dir.as_posix(),
            split='train',
            skip_frame=self.skip_frame,
            sequence_length=self.sequence_length,
            chunk_size=self.chunk_size,
            rgb_shape=self.rgb_shape,
            rgb_preprocessor=self.rgb_preprocessor,
            max_skip_frame=self.max_skip_frame,
            camera_key=self.camera_key,
            qpos_key=self.qpos_key,
            camera_gripper_key=self.camera_gripper_key,
            camera_left_key=self.camera_left_key,
        )
        
        val_base_dataset = HDF5Dataset_for_MotoGPT_CALVINLike(
            hdf5_dir=self.hdf5_dir.as_posix(),
            split='val',
            skip_frame=self.skip_frame,
            sequence_length=self.sequence_length,
            chunk_size=self.chunk_size,
            rgb_shape=self.rgb_shape,
            rgb_preprocessor=self.rgb_preprocessor,
            max_skip_frame=None,  # No random skip in validation
            camera_key=self.camera_key,
            qpos_key=self.qpos_key,
            camera_gripper_key=self.camera_gripper_key,
            camera_left_key=self.camera_left_key,
        )
        
        # Wrap datasets for MDT format
        self.train_datasets = {}
        self.val_datasets = {}
        
        for modality in self.modalities:
            train_wrapped = MDT_HDF5_Wrapper(
                base_dataset=train_base_dataset,
                use_left_camera=self.use_left_camera,
                action_mode=self.action_mode,
                gen_frame_source=self.gen_frame_source,
                key=modality,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                transforms_config_path=self.transforms_config_path,
                latent_motion_tokenizer_cfg_path=self.latent_motion_tokenizer_cfg_path,
                latent_motion_tokenizer_ckpt_path=self.latent_motion_tokenizer_ckpt_path,
            )
            
            val_wrapped = MDT_HDF5_Wrapper(
                base_dataset=val_base_dataset,
                use_left_camera=self.use_left_camera,
                action_mode=self.action_mode,
                gen_frame_source=self.gen_frame_source,
                key=modality,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                transforms_config_path=self.transforms_config_path,
                latent_motion_tokenizer_cfg_path=self.latent_motion_tokenizer_cfg_path,
                latent_motion_tokenizer_ckpt_path=self.latent_motion_tokenizer_ckpt_path,
            )
            
            self.train_datasets[modality] = train_wrapped
            self.val_datasets[modality] = val_wrapped
        
        logger.info(f"Setup complete. Train size: {len(train_base_dataset)}, Val size: {len(val_base_dataset)}")
        logger.info(f"Modalities: {self.modalities}")
    
    def train_dataloader(self):
        """
        Create training dataloaders for each modality
        """
        return {
            key: DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=True,
                collate_fn=mdt_collate_fn,
                prefetch_factor=2,
                persistent_workers=True,
            )
            for key, dataset in self.train_datasets.items()
        }
    
    def val_dataloader(self):
        """
        Create validation dataloaders for each modality
        """
        from pytorch_lightning.trainer.supporters import CombinedLoader
        
        val_dataloaders = {
            key: DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False,
                collate_fn=mdt_collate_fn,
            )
            for key, dataset in self.val_datasets.items()
        }
        
        # Use CombinedLoader to handle multiple modalities
        combined_val_loaders = CombinedLoader(val_dataloaders, "max_size_cycle")
        return combined_val_loaders
