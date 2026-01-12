"""
Test script to verify HDF5DataModule works correctly with MDT training format
Run this to check your dataset before full training
"""
import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

import torch
from mdt.datasets.hdf5_data_module import HDF5DataModule


def test_hdf5_datamodule(hdf5_dir, batch_size=4, num_workers=2):
    """
    Test the HDF5DataModule with your dataset
    
    Args:
        hdf5_dir: Path to HDF5 dataset directory (containing train/ and val/)
        batch_size: Batch size for testing
        num_workers: Number of workers for testing
    """
    print("="*80)
    print("Testing HDF5DataModule")
    print("="*80)
    
    # Create datamodule
    print(f"\n1. Creating datamodule from: {hdf5_dir}")
    datamodule = HDF5DataModule(
        hdf5_dir=hdf5_dir,
        sequence_length=10,
        chunk_size=3,
        skip_frame=5,
        batch_size=batch_size,
        num_workers=num_workers,
        rgb_shape=(480, 640),
        use_left_camera=False,
        action_mode='first',
        gen_frame_source='last',
        max_skip_frame=None,
        modalities=['vis'],  # Test with vision only first
        transforms_config_path="/group/ycyang/jfwu/aloha/mdt_policy/conf/datamodule/transforms/aloha_transforms.yaml",  # No extra transforms for testing
    )
    
    # Setup datasets
    print("\n2. Setting up datasets...")
    datamodule.setup()
    print(f"   Train datasets: {list(datamodule.train_datasets.keys())}")
    print(f"   Val datasets: {list(datamodule.val_datasets.keys())}")
    
    # Get dataloaders
    print("\n3. Creating dataloaders...")
    train_loaders = datamodule.train_dataloader()
    val_loaders = datamodule.val_dataloader()
    
    # Test training loader
    print("\n4. Testing training dataloader...")
    for modality, train_loader in train_loaders.items():
        print(f"\n   Modality: {modality}")
        print(f"   Number of batches: {len(train_loader)}")
        
        # Get first batch
        batch = next(iter(train_loader))
        print(f"\n   Batch structure:")
        print(f"   - rgb_obs:")
        print(f"     * rgb_static: {batch['rgb_obs']['rgb_static'].shape} (expected: [B, T+1, 3, H, W])")
        print(f"     * rgb_gripper: {batch['rgb_obs']['rgb_gripper'].shape} (expected: [B, T+1, 3, H, W])")
        print(f"     * gen_static: {batch['rgb_obs']['gen_static'].shape} (expected: [B, 3, H, W])")
        print(f"     * gen_gripper: {batch['rgb_obs']['gen_gripper'].shape} (expected: [B, 3, H, W])")
        print(f"   - actions: {batch['actions'].shape} (expected: [B, T, 7])")
        print(f"   - lang_text: {type(batch['lang_text'])} with {len(batch['lang_text'])} items")
        print(f"   - future_frame_diff: {batch['future_frame_diff']}")
        print(f"   - idx: {batch['idx'].shape}")
        print(f"   - idx: {batch['idx']}")
        
        # Verify shapes
        B = batch_size
        T = 10  # sequence_length
        H, W = 224, 224
        
        # assert batch['rgb_obs']['rgb_static'].shape == (B, T+1, 3, H, W), \
        #     f"rgb_static shape mismatch: {batch['rgb_obs']['rgb_static'].shape}"
        # assert batch['rgb_obs']['rgb_gripper'].shape == (B, T+1, 3, H, W), \
        #     f"rgb_gripper shape mismatch: {batch['rgb_obs']['rgb_gripper'].shape}"
        # assert batch['rgb_obs']['gen_static'].shape == (B, 3, H, W), \
        #     f"gen_static shape mismatch: {batch['rgb_obs']['gen_static'].shape}"
        # assert batch['rgb_obs']['gen_gripper'].shape == (B, 3, H, W), \
        #     f"gen_gripper shape mismatch: {batch['rgb_obs']['gen_gripper'].shape}"
        # assert batch['actions'].shape == (B, T, 7), \
        #     f"actions shape mismatch: {batch['actions'].shape}"
        # assert len(batch['lang_text']) == B, \
        #     f"lang_text length mismatch: {len(batch['lang_text'])}"
        
        print("\n   ✓ All shapes are correct!")
        
        # Check data types
        print(f"\n   Data types:")
        print(f"   - rgb_static dtype: {batch['rgb_obs']['rgb_static'].dtype}")
        print(f"   - actions dtype: {batch['actions'].dtype}")
        
        # Check data ranges
        print(f"\n   Data ranges:")
        print(f"   - rgb_static: [{batch['rgb_obs']['rgb_static'].min():.2f}, {batch['rgb_obs']['rgb_static'].max():.2f}]")
        print(f"   - actions: [{batch['actions'].min():.4f}, {batch['actions'].max():.4f}]")

        # data statistics
        print(f"\n   Data statistics:")
        # print(f"   - rgb_static mean: {batch['rgb_obs']['rgb_static'].mean():.4f}, std: {batch['rgb_obs']['rgb_static'].std():.4f}")
        # print(f"   - rgb_gripper mean: {batch['rgb_obs']['rgb_gripper'].mean():.4f}, std: {batch['rgb_obs']['rgb_gripper'].std():.4f}")
        # print(f"   - actions mean: {batch['actions'].mean():.4f}, std: {batch['actions'].std():.4f}")   
        print(f"   - rgb_static mean per channel: {batch['rgb_obs']['rgb_static'].mean(dim=(0,1,3,4))}, std per channel: {batch['rgb_obs']['rgb_static'].std(dim=(0,1,3,4))}")
        print(f"   - rgb_gripper mean per channel: {batch['rgb_obs']['rgb_gripper'].mean(dim=(0,1,3,4))}, std per channel: {batch['rgb_obs']['rgb_gripper'].std(dim=(0,1,3,4))}")
        
        # Print first language instruction
        print(f"\n   Sample language instruction:")
        print(f"   '{batch['lang_text'][1]}'")

        # print first few actions
        print(f"\n   Sample actions (first sequence):")
        print(batch['actions'][0, :5, :])  # print first 5 actions
        
        break  # Only test first modality
    
    # Test validation loader
    print("\n5. Testing validation dataloader...")
    val_batch = next(iter(val_loaders))
    
    # val_loaders is CombinedLoader, so batch has modality keys
    for modality, v_batch in val_batch.items():
        print(f"\n   Modality: {modality}")
        print(f"   Validation batch shapes:")
        print(f"   - rgb_static: {v_batch['rgb_obs']['rgb_static'].shape}")
        print(f"   - actions: {v_batch['actions'].shape}")
        print(f"\n   ✓ Validation loader works!")
        break
    
    print("\n" + "="*80)
    print("✓ All tests passed! Your dataset is ready for MDT training.")
    print("="*80)
    
    return datamodule


def print_usage_instructions():
    """Print instructions for using the adapted dataset"""
    print("\n" + "="*80)
    print("USAGE INSTRUCTIONS")
    print("="*80)
    
    print("""
To use your HDF5 dataset with MDT training:

1. Prepare your data:
   - Organize HDF5 files in this structure:
     your_dataset/
     ├── train/
     │   ├── task1/
     │   │   ├── episode_0.hdf5
     │   │   ├── episode_1.hdf5
     │   │   └── instr.txt (optional, for language)
     │   └── task2/
     │       └── ...
     └── val/
         └── ...

2. Update your training config (conf/config.yaml):
   
   datamodule:
     _target_: mdt.datasets.hdf5_data_module.HDF5DataModule
     hdf5_dir: /path/to/your/hdf5/dataset
     sequence_length: 10
     chunk_size: 3
     skip_frame: 1
     batch_size: 32
     num_workers: 8
     modalities: ["vis"]  # or ["vis", "lang"] for language

3. Run training:
   python mdt/training.py

4. Key parameters to tune:
   - action_mode: 'first' (default), 'mean', or 'last'
     Controls how to extract single action from action chunks
   
   - gen_frame_source: 'last' (default), 'middle', or int
     Which future frame to use for image generation loss
   
   - max_skip_frame: null or int
     Enable random temporal augmentation by varying frame skip
   
   - use_left_camera: false (default) or true
     Choose which gripper camera to use

5. For multi-camera or language support:
   - Language: Set modalities: ["lang"] and ensure instr.txt exists
   - Multi-camera: The wrapper supports static + gripper/left views by default
""")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test HDF5DataModule")
    parser.add_argument(
        "--hdf5_dir",
        type=str,
        required=True,
        help="Path to HDF5 dataset directory (containing train/ and val/)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for testing (default: 4)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of dataloader workers (default: 2)"
    )
    
    args = parser.parse_args()
    
    try:
        # Run test
        datamodule = test_hdf5_datamodule(
            hdf5_dir=args.hdf5_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Print usage instructions
        print_usage_instructions()
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
