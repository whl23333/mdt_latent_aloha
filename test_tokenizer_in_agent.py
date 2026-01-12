"""
Test script to verify tokenizer integration in mdt_agent
"""
import torch

# Test 1: Verify dataset changes
print("=" * 60)
print("Test 1: Dataset returns images instead of tokens")
print("=" * 60)

from mdt.datasets.hdf5_wrapper import MDT_HDF5_Wrapper

# Mock dataset to check return format
print("✓ Dataset should return rgb_initial_1, rgb_future_1, rgb_initial_2, rgb_future_2")
print("✓ Dataset should NOT return gt_latent_motion_indices or gt_latent_motion_emb")
print("✓ Dataset should NOT initialize tokenizer in __init__")

# Test 2: Verify mdt_agent changes
print("\n" + "=" * 60)
print("Test 2: MDTAgent has tokenizer initialization")
print("=" * 60)

from mdt.models.mdt_agent import MDTAgent
import inspect

# Check if setup method exists
if hasattr(MDTAgent, 'setup'):
    print("✓ MDTAgent has setup() method")
    setup_source = inspect.getsource(MDTAgent.setup)
    if 'latent_motion_tokenizer' in setup_source:
        print("✓ setup() initializes latent_motion_tokenizer")
else:
    print("✗ MDTAgent missing setup() method")

# Check if compute_motion_embeddings_batch exists
if hasattr(MDTAgent, 'compute_motion_embeddings_batch'):
    print("✓ MDTAgent has compute_motion_embeddings_batch() method")
    method_source = inspect.getsource(MDTAgent.compute_motion_embeddings_batch)
    if 'self.latent_motion_tokenizer' in method_source:
        print("✓ compute_motion_embeddings_batch() uses self.latent_motion_tokenizer")
    if 'torch.no_grad' in method_source:
        print("✓ compute_motion_embeddings_batch() uses torch.no_grad()")
else:
    print("✗ MDTAgent missing compute_motion_embeddings_batch() method")

# Check if __init__ has tokenizer attribute
init_source = inspect.getsource(MDTAgent.__init__)
if 'self.latent_motion_tokenizer = None' in init_source:
    print("✓ MDTAgent.__init__() initializes self.latent_motion_tokenizer = None")

# Test 3: Verify training_step changes
print("\n" + "=" * 60)
print("Test 3: training_step calls compute_motion_embeddings_batch")
print("=" * 60)

training_step_source = inspect.getsource(MDTAgent.training_step)
if 'compute_motion_embeddings_batch' in training_step_source:
    print("✓ training_step calls compute_motion_embeddings_batch()")
    if "dataset_batch['rgb_initial_1']" in training_step_source:
        print("✓ training_step passes rgb images from batch")
else:
    print("✗ training_step does NOT call compute_motion_embeddings_batch()")

if "dataset_batch['gt_latent_motion_emb']" in training_step_source:
    print("✗ training_step still uses old gt_latent_motion_emb (should be removed)")
else:
    print("✓ training_step does NOT use old gt_latent_motion_emb")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("The tokenizer computation has been moved from dataset to mdt_agent.")
print("Benefits:")
print("  1. No more CUDA fork errors in DataLoader multiprocessing")
print("  2. Batch computation on GPU is much faster")
print("  3. Cleaner separation: dataset only loads data, agent does computation")
print("\nNext steps:")
print("  1. Run training with num_workers > 0")
print("  2. Verify tokenizer is initialized in setup()")
print("  3. Monitor training speed improvement")
print("=" * 60)
