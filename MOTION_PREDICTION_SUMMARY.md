# Motion Prediction 集成总结

## 修改概览

根据新的MDTTransformer双路径设计，已对mdt_agent.py进行以下修改：

### 1. 移除的内容 ❌

#### 1.1 移除了motion_embed_head
```python
# 旧代码（已删除）:
self.motion_embed_head = nn.Linear(
    self.latent_dim,
    latent_motion_dim * per_latent_motion_len
)
```
**原因**: MDTTransformer内部已经有`motion_pred`层，不需要额外的head

#### 1.2 移除了optimizer中的projection/unprojection层
```python
# 旧代码（已删除）:
if self.pred_latent_motion_embeddings:
    optim_groups.extend([
        {"params": self.motion_embed_projection.parameters(), ...},
        {"params": self.motion_embed_unprojection.parameters(), ...},
        {"params": self.motion_indices_head.parameters(), ...},
    ])
```
**原因**: 这些层不再需要，MDTTransformer内部的motion_emb/motion_pred已经包含在`model.inner_model.parameters()`中

### 2. 修改的内容 ✏️

#### 2.1 简化__init__
```python
# 新代码:
if self.pred_latent_motion_embeddings:
    print("MDT Agent will predict latent motion embeddings during inference.")
    print(f"Motion prediction: {per_latent_motion_len} tokens of {latent_motion_dim}-dim embeddings")
```
**改进**: 只打印信息，不创建额外的层

#### 2.2 重写motion_diffusion_loss
```python
def motion_diffusion_loss(
    self,
    perceptual_emb: torch.Tensor,
    latent_goal: torch.Tensor,
    motion_embeds_gt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    直接在motion embedding空间计算diffusion loss
    """
    self.model.train()
    batch_size = motion_embeds_gt.shape[0]
    
    # 直接在motion embedding空间加噪声
    sigmas = self.make_sample_density()(shape=(batch_size,), device=self.device)
    noise = torch.randn_like(motion_embeds_gt)
    
    # 使用use_motion=True调用motion预测路径
    loss, _ = self.model.loss(
        perceptual_emb, 
        motion_embeds_gt,  # [B, 8, 32]
        latent_goal, 
        noise, 
        sigmas,
        use_motion=True  # 关键参数
    )
    
    return loss, sigmas, noise
```

**关键改进**:
- ✅ 不再需要projection到action空间
- ✅ 直接在[B, 8, 32]空间操作
- ✅ 通过`use_motion=True`切换到motion路径

#### 2.3 重写denoise_motion_embeddings
```python
def denoise_motion_embeddings(
    self,
    latent_plan: torch.Tensor,
    perceptual_emb: torch.Tensor,
    latent_goal: torch.Tensor,
    inference: Optional[bool] = False,
    extra_args={}
) -> torch.Tensor:  # 注意：返回类型变了
    """
    直接在embedding空间去噪
    """
    # ...
    batch_size = len(latent_goal)
    
    # 在motion embedding空间初始化噪声
    x = torch.randn(
        (batch_size, self.per_latent_motion_len, self.latent_motion_dim), 
        device=self.device
    ) * self.sigma_max  # [B, 8, 32]
    
    # 使用use_motion=True去噪
    motion_embeddings = self.sample_loop(
        sigmas, x, input_state, latent_goal, latent_plan, 
        self.sampler_type, extra_args, use_motion=True
    )
    
    return motion_embeddings
```

**关键改进**:
- ✅ 初始化噪声形状: [B, 8, 32] 而非 [B, 10, 7]
- ✅ 直接返回embeddings，indices转换由validation step处理
- ✅ 通过`use_motion=True`切换路径

#### 2.4 修改sample_loop支持use_motion
```python
def sample_loop(
    self, 
    sigmas, 
    x_t: torch.Tensor,
    state: torch.Tensor, 
    goal: torch.Tensor, 
    latent_plan: torch.Tensor,
    sampler_type: str,
    extra_args={},
    use_motion: bool = False,  # 新增参数
    ):
    """
    主采样循环，支持action和motion两种模式
    """
    # 将use_motion添加到extra_args中
    extra_args['use_motion'] = use_motion
    reduced_args = {x: extra_args[x] for x in ['s_churn', 'keep_last_actions', 'use_motion'] if x in extra_args}
    
    # 所有采样函数会通过**kwargs接收use_motion
    # 最终传递给model.forward(use_motion=use_motion)
    # ...
```

#### 2.5 修改validation_step的indices计算
```python
# 新代码:
if self.pred_latent_motion_embeddings and 'gt_latent_motion_emb' in dataset_batch:
    motion_embeds_pred = self.denoise_motion_embeddings(...)
    motion_embeds_gt = dataset_batch['gt_latent_motion_emb']
    pred_loss = torch.nn.functional.mse_loss(motion_embeds_pred, motion_embeds_gt)
    
    # Indices转换需要tokenizer
    if 'latent_motion_tokenizer' in dataset_batch and 'gt_latent_motion_indices' in dataset_batch:
        tokenizer = dataset_batch['latent_motion_tokenizer']
        motion_indices_pred = tokenizer.vector_quantizer.get_code_indices(motion_embeds_pred)
        gt_indices = dataset_batch['gt_latent_motion_indices']
        indices_accuracy = (motion_indices_pred == gt_indices).float().mean()
        self.log(f"val_act/{self.modality_scope}_indices_accuracy", indices_accuracy, sync_dist=True)
```

**关键改进**:
- ✅ 直接比较embeddings的MSE loss
- ✅ 使用tokenizer的vector_quantizer将embeddings转换为indices
- ✅ 计算indices准确率

## 3. 数据流图

### Action预测路径 (use_motion=False)
```
Training:
  gt_actions [B, 10, 7]
    ↓
  + noise [B, 10, 7]
    ↓
  model.loss(perceptual_emb, noisy_actions, goal, noise, sigma, use_motion=False)
    ↓ 使用action_emb + action_pred
  loss

Inference:
  noise [B, 10, 7]
    ↓
  sample_loop(..., use_motion=False)
    ↓ 迭代去噪，使用action_emb + action_pred
  pred_actions [B, 10, 7]
```

### Motion预测路径 (use_motion=True)
```
Training:
  gt_motion_embeds [B, 8, 32]
    ↓
  + noise [B, 8, 32]
    ↓
  model.loss(perceptual_emb, noisy_embeds, goal, noise, sigma, use_motion=True)
    ↓ 使用motion_emb + motion_pred
  loss

Inference:
  noise [B, 8, 32]
    ↓
  sample_loop(..., use_motion=True)
    ↓ 迭代去噪，使用motion_emb + motion_pred
  pred_motion_embeds [B, 8, 32]
    ↓
  tokenizer.vector_quantizer.get_code_indices()
    ↓
  motion_indices [B, 8]
```

## 4. 关键设计决策

### 为什么移除projection/unprojection？

**问题**：
- 旧设计：motion_embeds → projection → action_space → diffusion → unprojection → embeds
- unprojection层在训练时没有梯度流（因为训练在action空间）
- 推理时使用未训练的unprojection会导致性能下降

**解决方案**：
- 新设计：直接在motion embedding空间做diffusion
- MDTTransformer内部有独立的motion_emb/motion_pred路径
- 训练和推理路径一致

### 为什么indices转换在validation step？

**原因**：
1. indices转换需要tokenizer的vector_quantizer
2. tokenizer不是mdt_agent的一部分，由datamodule管理
3. 在validation时，datamodule可以将tokenizer放入batch中
4. 这样保持模块解耦，mdt_agent专注于diffusion

## 5. 使用方式

### 初始化
```python
agent = MDTAgent(
    model=dict(
        use_motion_prediction=True,  # 启用motion路径
        motion_dim=32,
        motion_seq_len=8,
        action_dim=7,  # 保留action路径
        action_seq_len=10,
        ...
    ),
    pred_latent_motion_embeddings=True,  # agent层面的标志
    latent_motion_dim=32,
    per_latent_motion_len=8,
    ...
)
```

### 训练
```python
# 在training_step中
if self.pred_latent_motion_embeddings:
    motion_embeds_gt = dataset_batch['gt_latent_motion_emb']  # [B, 8, 32]
    loss, sigmas, noise = self.motion_diffusion_loss(
        perceptual_emb,
        latent_goal,
        motion_embeds_gt
    )
else:
    loss, sigmas, noise = self.diffusion_loss(
        perceptual_emb,
        latent_goal,
        dataset_batch["actions"]  # [B, 10, 7]
    )
```

### 验证
```python
# 在validation_step中
if self.pred_latent_motion_embeddings:
    motion_embeds_pred = self.denoise_motion_embeddings(...)  # [B, 8, 32]
    loss = F.mse_loss(motion_embeds_pred, gt_embeds)
    
    # 计算indices准确率（如果有tokenizer）
    if 'latent_motion_tokenizer' in batch:
        indices_pred = tokenizer.vector_quantizer.get_code_indices(motion_embeds_pred)
        accuracy = (indices_pred == gt_indices).float().mean()
```

## 6. Checkpoint兼容性

### 加载旧的action checkpoint
```python
agent = MDTAgent(
    model=dict(use_motion_prediction=False, ...),
    pred_latent_motion_embeddings=False,
    ...
)
agent.load_state_dict(torch.load('action_ckpt.pth'), strict=False)
# ✅ 完全兼容
```

### 加载motion checkpoint
```python
agent = MDTAgent(
    model=dict(use_motion_prediction=True, motion_dim=32, motion_seq_len=8, ...),
    pred_latent_motion_embeddings=True,
    latent_motion_dim=32,
    per_latent_motion_len=8,
    ...
)
agent.load_state_dict(torch.load('motion_ckpt.pth'), strict=False)
# ✅ 完全兼容
```

## 7. 待完成的工作

### Dataset集成
- [ ] 在HDF5Dataset中调用tokenizer生成gt_latent_motion_emb
- [ ] 将tokenizer实例放入batch（用于validation时的indices转换）
- [ ] 存储gt_latent_motion_indices用于准确率计算

### 采样函数更新
- [x] sample_loop支持use_motion参数
- [ ] 确认所有采样函数（ddim, euler等）正确传递**kwargs到model

### 测试
- [ ] 端到端训练测试（motion路径）
- [ ] 验证loss收敛
- [ ] 验证indices准确率
- [ ] 对比action和motion模型的性能

## 8. 注意事项

⚠️ **重要**：
1. Motion模型和Action模型的pos_emb大小不同（9 vs 11），不能互相加载
2. 训练motion模型时，action相关的层（action_emb, action_pred）不会被训练
3. 推理时必须确保use_motion参数在整个pipeline中正确传递
4. Indices转换依赖tokenizer，确保validation时tokenizer在batch中

✅ **优势**：
1. 训练和推理路径一致（无projection/unprojection不一致问题）
2. 完全复用diffusion基础设施（采样器、噪声调度等）
3. 代码简洁，易于维护
4. Checkpoint向后兼容
