# Tokenizer计算移至MDTAgent的改动说明

## 改动概览

将latent motion tokenizer的计算从`dataset.__getitem__`移至`mdt_agent.training_step`中进行批量计算。

## 改动原因

1. **避免CUDA多进程问题**：在DataLoader的worker进程中使用CUDA会导致"Cannot re-initialize CUDA in forked subprocess"错误
2. **提升计算效率**：批量计算比逐个样本计算更快
3. **更清晰的架构**：dataset只负责加载数据，agent负责模型计算

## 具体修改

### 1. `mdt/datasets/hdf5_wrapper.py`

**修改内容：**
- 移除了tokenizer的初始化和调用
- 只保存tokenizer的配置路径：`latent_motion_tokenizer_cfg_path`和`latent_motion_tokenizer_ckpt_path`
- `__getitem__`返回原始图像数据而非token：
  ```python
  "rgb_initial_1": rgb_initial_1,  # [1, 3, H, W]
  "rgb_future_1": rgb_future_1,    # [1, 3, H, W]
  "rgb_initial_2": rgb_initial_2,  # [1, 3, H, W]
  "rgb_future_2": rgb_future_2,    # [1, 3, H, W]
  ```
- 更新`mdt_collate_fn`以stack这些图像数据

**移除的代码：**
```python
# 不再在dataset中初始化tokenizer
self.latent_motion_tokenizer = hydra.utils.instantiate(...)
self.latent_motion_tokenizer.cuda()

# 不再在__getitem__中计算tokens
gt_latent_motion_indices = self.latent_motion_tokenizer(...)
gt_latent_motion_emb = self.latent_motion_tokenizer.vector_quantizer.get_codebook_entry(...)
```

### 2. `mdt/models/mdt_agent.py`

**新增内容：**

#### a) 添加tokenizer属性初始化（`__init__`）
```python
self.latent_motion_tokenizer = None  # Will be loaded from datamodule config
```

#### b) 添加setup方法
```python
def setup(self, stage: str):
    """Called at the beginning of fit, initialize tokenizer from datamodule."""
    if stage == 'fit' and self.pred_latent_motion_embeddings:
        # Load tokenizer config from datamodule
        train_ds = self.trainer.datamodule.train_dataset
        latent_motion_tokenzier_cfg = OmegaConf.load(train_ds.latent_motion_tokenizer_cfg_path)
        self.latent_motion_tokenizer = hydra.utils.instantiate(latent_motion_tokenzier_cfg)
        
        # Load checkpoint
        state_dict = torch.load(train_ds.latent_motion_tokenizer_ckpt_path, map_location='cpu')
        self.latent_motion_tokenizer.load_state_dict(state_dict['model'], strict=False)
        
        # Freeze and move to GPU
        for param in self.latent_motion_tokenizer.parameters():
            param.requires_grad = False
        self.latent_motion_tokenizer.eval()
        self.latent_motion_tokenizer = self.latent_motion_tokenizer.to(self.device)
```

#### c) 添加批量计算motion embeddings的方法
```python
def compute_motion_embeddings_batch(self, rgb_initial_1, rgb_future_1, rgb_initial_2, rgb_future_2):
    """
    Batch compute motion embeddings using the latent motion tokenizer.
    
    Args:
        rgb_initial_1: [B, 1, 3, H, W]
        rgb_future_1: [B, 1, 3, H, W]
        rgb_initial_2: [B, 1, 3, H, W]
        rgb_future_2: [B, 1, 3, H, W]
        
    Returns:
        motion_embeds: [B, per_latent_motion_len, latent_motion_dim]
    """
    device = next(self.latent_motion_tokenizer.parameters()).device
    
    with torch.no_grad():
        motion_token_indices = self.latent_motion_tokenizer(
            cond_pixel_values1=rgb_initial_1.to(device),
            target_pixel_values1=rgb_future_1.to(device),
            cond_pixel_values2=rgb_initial_2.to(device),
            target_pixel_values2=rgb_future_2.to(device),
            return_motion_token_ids_only=True
        )
        
        batch_size = motion_token_indices.shape[0]
        motion_embeds = torch.stack([
            self.latent_motion_tokenizer.vector_quantizer.get_codebook_entry(
                motion_token_indices[i]
            )
            for i in range(batch_size)
        ], dim=0)
    
    return motion_embeds
```

#### d) 修改training_step
```python
if self.pred_latent_motion_embeddings:
    # Batch compute motion tokens using tokenizer on GPU
    motion_embeds_gt = self.compute_motion_embeddings_batch(
        dataset_batch['rgb_initial_1'],
        dataset_batch['rgb_future_1'],
        dataset_batch['rgb_initial_2'],
        dataset_batch['rgb_future_2']
    )  # (B, per_latent_motion_len, latent_motion_dim)
    
    act_loss, sigmas, noise = self.motion_diffusion_loss(
        perceptual_emb,
        latent_goal,
        motion_embeds_gt
    )
```

## 数据流程

### 修改前：
```
Dataset.__getitem__
  ├─ 加载图像数据
  ├─ tokenizer(images) [单个样本, CPU→GPU→CPU]
  └─ 返回: gt_latent_motion_emb [num_tokens, e_dim]
       ↓
DataLoader collate_fn
  └─ stack: [B, num_tokens, e_dim]
       ↓
training_step
  └─ 直接使用 dataset_batch['gt_latent_motion_emb']
```

**问题：**
- tokenizer在worker进程中使用CUDA导致fork错误
- 逐样本计算效率低
- 需要CPU↔GPU数据传输

### 修改后：
```
Dataset.__getitem__
  ├─ 加载图像数据
  └─ 返回: rgb_initial_1, rgb_future_1, rgb_initial_2, rgb_future_2 [各1, 3, H, W]
       ↓
DataLoader collate_fn
  └─ stack: [B, 1, 3, H, W]
       ↓
training_step
  ├─ compute_motion_embeddings_batch(images) [批量, GPU]
  │    └─ tokenizer(batch_images) [with torch.no_grad()]
  └─ motion_diffusion_loss(motion_embeds_gt)
```

**优势：**
- 无CUDA多进程问题
- 批量计算更快
- 数据已在GPU上，无需额外传输

## 兼容性说明

1. **checkpoint加载**：tokenizer不在checkpoint中，从单独的配置文件加载
2. **DataLoader配置**：现在可以使用`num_workers > 0`而不会出错
3. **训练脚本**：无需修改，PyTorch Lightning会自动调用`setup()`方法

## 验证方法

运行测试脚本：
```bash
cd /group/ycyang/jfwu/aloha/mdt_policy
python test_tokenizer_in_agent.py
```

所有测试应该通过：
- ✓ Dataset返回图像数据
- ✓ MDTAgent有setup()方法
- ✓ MDTAgent有compute_motion_embeddings_batch()方法
- ✓ training_step调用批量计算

## 预期效果

1. **速度提升**：批量GPU计算比逐样本CPU/GPU混合计算快很多
2. **稳定性提升**：不再有CUDA fork错误
3. **代码质量提升**：职责分离更清晰

## 注意事项

- tokenizer在`setup(stage='fit')`时初始化，因此只有训练时才会加载
- 如果需要在validation/test时使用tokenizer，需要相应修改`setup()`方法
- tokenizer始终是frozen的（requires_grad=False）
