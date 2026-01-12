"""
ALOHA 真机 QPo训练预处理配置指南

基于你的数据特点：
- qpos 维度: 7 (6个关节 + 1个夹爪)
- action 范围: [-1.8201, 1.7307]
- 值域差异大（如 Joint1: ~1.3-1.5, Joint3: ~-1.5 到 -1.6）
"""

# ==============================================================================
# 推荐方案：相对 QPOs (Delta) + 归一化
# ==============================================================================

"""
为什么选择相对 qpos？

1. ✓ 学习难度低
   - 相邻帧的 qpos 变化通常很小且平滑
   - 扩散模型更容易学习小范围分布
   
2. ✓ 泛化性好
   - 不依赖特定的起始位置
   - 可以从不同初始状态执行相同轨迹
   
3. ✓ 减少累积误差
   - 每步基于真实观测的当前状态
   - 不会因为预测误差而逐渐偏离
   
4. ✓ 符合 MDT 原始设计
   - CALVIN 数据集使用相对动作
   - 代码中已有完整支持
"""

# ==============================================================================
# 步骤 1: 准备预处理器
# ==============================================================================

"""
首次训练前，需要从训练数据中统计归一化参数：

```bash
cd /group/ycyang/jfwu/aloha/mdt_policy

python -c "
from examples.qpos_preprocessing_integration import prepare_qpos_preprocessors

prepare_qpos_preprocessors(
    hdf5_dir='/group/ycyang/jfwu/aloha/data/whl',
    output_dir='/group/ycyang/jfwu/aloha/data/whl/preprocessors',
    use_relative=True,
)
"
```

这将生成：
- preprocessors/qpos_normalizer.npz        # 绝对 qpos 归一化器
- preprocessors/relative_qpos_transform.npz # 相对 qpos 变换器
"""

# ==============================================================================
# 步骤 2: 修改 Dataset 集成预处理
# ==============================================================================

"""
在 HDF5Dataset_for_MotoGPT_CALVINLike 或 wrapper 中添加预处理：

```python
# 在 hdf5_datasets.py 或 hdf5_wrapper.py 中

from mdt.utils.qpos_preprocessing import QPosNormalizer, RelativeQPosTransform

class HDF5Dataset_with_QPosPreprocessing:
    def __init__(
        self,
        hdf5_dir,
        use_relative_qpos=True,
        qpos_normalizer_path=None,
        relative_transform_path=None,
        ...
    ):
        # 加载预处理器
        if qpos_normalizer_path:
            self.qpos_normalizer = QPosNormalizer.load(qpos_normalizer_path)
        
        if use_relative_qpos and relative_transform_path:
            self.rel_transform = RelativeQPosTransform.load(relative_transform_path)
        
        self.use_relative_qpos = use_relative_qpos
    
    def __getitem__(self, idx):
        # ... 加载原始数据 ...
        
        qpos_sequence = data['qpos']  # (seq_len, 7)
        
        if self.use_relative_qpos:
            # 方案 B: 相对 qpos
            qpos_current = qpos_sequence[0]  # 第一帧作为当前状态
            
            # 构造每帧的"当前"状态
            qpos_current_seq = np.concatenate([
                qpos_current[None, :],
                qpos_sequence[:-1]
            ], axis=0)
            
            # 计算归一化的 delta
            delta_qpos = self.rel_transform.to_relative(
                qpos_sequence, qpos_current_seq
            )
            
            # 可选：同时归一化初始状态
            if self.qpos_normalizer:
                qpos_current = self.qpos_normalizer.normalize(qpos_current)
            
            data['actions'] = torch.from_numpy(delta_qpos).float()
            data['robot_obs'] = torch.from_numpy(qpos_current).float()
        
        else:
            # 方案 A: 绝对 qpos（备选）
            if self.qpos_normalizer:
                qpos_sequence = self.qpos_normalizer.normalize(qpos_sequence)
            
            data['actions'] = torch.from_numpy(qpos_sequence).float()
        
        return data
```
"""

# ==============================================================================
# 步骤 3: 更新配置文件
# ==============================================================================

"""
在 conf/datamodule/aloha_qpos.yaml 中：

```yaml
_target_: mdt.datasets.hdf5_data_module.HDF5DataModule
root_data_dir: /group/ycyang/jfwu/aloha/data/whl

# 动作空间配置
action_space: 7
action_max: [1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.0]  # 稍微放宽边界
action_min: [-1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.0]

# qpos 预处理配置
use_relative_qpos: True
qpos_normalizer_path: ${root_data_dir}/preprocessors/qpos_normalizer.npz
relative_transform_path: ${root_data_dir}/preprocessors/relative_qpos_transform.npz

observation_space:
  rgb_obs: ['rgb_static', 'rgb_gripper']
  state_obs: ['robot_obs']  # 包含当前 qpos（归一化后）
  actions: ['rel_actions']  # 相对 qpos

proprioception_dims:
  n_state_obs: 7  # qpos 维度
  keep_indices: [[0, 7]]  # 保留所有维度
  normalize: False  # 已经在 dataset 中归一化
```
"""

# ==============================================================================
# 步骤 4: 推理时反归一化
# ==============================================================================

"""
在推理/部署时，需要将预测的归一化 delta 转回真实 qpos：

```python
# 在 rollout 或 evaluation 脚本中

from mdt.utils.qpos_preprocessing import RelativeQPosTransform

# 加载预处理器
rel_transform = RelativeQPosTransform.load(
    'preprocessors/relative_qpos_transform.npz'
)

# 推理循环
qpos_current = env.get_qpos()  # 从环境获取当前 qpos

for step in range(max_steps):
    # 1. 编码当前观测
    obs = get_observation(env)
    
    # 2. 模型预测（输出归一化的 delta）
    delta_qpos_normalized = model.predict(obs)  # (7,) in [-1, 1]
    
    # 3. 反归一化 + 转回绝对 qpos
    qpos_target = rel_transform.to_absolute(
        delta_qpos_normalized, 
        qpos_current
    )
    
    # 4. 执行动作
    env.step(qpos_target)
    
    # 5. 更新当前状态（重要！）
    qpos_current = env.get_qpos()  # 获取真实执行后的 qpos
```
"""

# ==============================================================================
# 额外建议
# ==============================================================================

"""
1. 夹爪处理
   - 从数据看，夹爪维度全为 0
   - 建议检查夹爪数据是否正确记录
   - 如果夹爪是二值（开/关），可以不归一化这一维

2. 数据增强（可选）
   - 训练时可添加小噪声到 qpos：
     qpos += np.random.randn(7) * 0.01
   - 增强模型鲁棒性

3. Action Chunking
   - 保持 act_window_size=10, multistep=3 的设置
   - 预测10步动作序列，每3步重新预测一次

4. 值域检查
   - 运行 prepare_qpos_preprocessors 后
   - 检查打印的统计信息，确认范围合理
   - 特别注意各关节的 delta_max 是否异常

5. 调试技巧
   - 可视化预测的 qpos 轨迹
   - 检查 delta 的分布是否集中在 [-1, 1]
   - 监控反归一化后的 qpos 是否在物理限制内
"""

# ==============================================================================
# 完整的训练命令示例
# ==============================================================================

"""
# 1. 准备预处理器（首次）
python -c "
from examples.qpos_preprocessing_integration import prepare_qpos_preprocessors
prepare_qpos_preprocessors(
    hdf5_dir='/group/ycyang/jfwu/aloha/data/whl',
    use_relative=True,
)
"

# 2. 训练模型
python train.py \
    datamodule=aloha_qpos \
    model=mdt_transformer \
    root_data_dir=/group/ycyang/jfwu/aloha/data/whl \
    use_relative_qpos=True \
    act_dim=7 \
    seed=42

# 3. 评估/推理
python eval.py \
    checkpoint_path=outputs/xxx/checkpoints/last.ckpt \
    datamodule=aloha_qpos \
    num_episodes=10
"""

if __name__ == '__main__':
    print(__doc__)
