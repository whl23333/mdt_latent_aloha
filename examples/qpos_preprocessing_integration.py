"""
将 qpos 预处理集成到现有 Dataset 的示例
"""
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from mdt.utils.qpos_preprocessing import (
    QPosNormalizer,
    RelativeQPosTransform,
    compute_qpos_statistics,
    print_qpos_statistics,
)


class QPosDatasetWrapper:
    """
    对现有 Dataset 添加 qpos 预处理
    
    使用场景：
    1. 方案A：预测绝对 qpos（归一化）
    2. 方案B：预测相对 qpos（delta）
    """
    
    def __init__(
        self,
        base_dataset,
        use_relative: bool = True,  # 推荐使用相对动作
        qpos_normalizer_path: str = None,
        relative_transform_path: str = None,
    ):
        self.base_dataset = base_dataset
        self.use_relative = use_relative
        
        # 加载或创建预处理器
        if qpos_normalizer_path and Path(qpos_normalizer_path).exists():
            self.qpos_normalizer = QPosNormalizer.load(qpos_normalizer_path)
        else:
            self.qpos_normalizer = None
        
        if use_relative:
            if relative_transform_path and Path(relative_transform_path).exists():
                self.rel_transform = RelativeQPosTransform.load(relative_transform_path)
            else:
                self.rel_transform = None
        else:
            self.rel_transform = None
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # 获取原始数据
        data = self.base_dataset[idx]
        
        # 提取 qpos 序列
        qpos_sequence = data['actions']  # (seq_len, 7)
        
        if isinstance(qpos_sequence, torch.Tensor):
            qpos_sequence = qpos_sequence.numpy()
        
        # 方案A: 绝对 qpos 归一化
        if not self.use_relative:
            if self.qpos_normalizer is not None:
                qpos_sequence = self.qpos_normalizer.normalize(qpos_sequence)
        
        # 方案B: 相对 qpos (delta)
        else:
            # 当前状态是第一帧的 qpos
            qpos_current = qpos_sequence[0]
            qpos_targets = qpos_sequence  # (seq_len, 7)
            
            # 计算相对增量
            if self.rel_transform is not None:
                # 构造 qpos_current 序列（每帧的当前状态）
                qpos_current_seq = np.concatenate([
                    qpos_current[None, :],  # t=0 的当前状态
                    qpos_sequence[:-1]       # t=1...n-1 的当前状态
                ], axis=0)
                
                # 计算 delta
                delta_qpos = self.rel_transform.to_relative(
                    qpos_targets, qpos_current_seq
                )
                
                # 可选：同时归一化绝对位置
                if self.qpos_normalizer is not None:
                    qpos_current = self.qpos_normalizer.normalize(qpos_current)
                
                # 替换为 delta
                qpos_sequence = delta_qpos
        
        # 转回 tensor
        data['actions'] = torch.from_numpy(qpos_sequence).float()
        
        # 可选：添加当前状态（用于条件化）
        if self.use_relative:
            data['robot_obs'] = torch.from_numpy(qpos_current).float()
        
        return data


def prepare_qpos_preprocessors(
    hdf5_dir: str,
    output_dir: str = None,
    use_relative: bool = True,
):
    """
    从训练数据中准备 qpos 预处理器
    
    Args:
        hdf5_dir: HDF5 数据目录
        output_dir: 保存预处理器的目录
        use_relative: 是否使用相对动作
    """
    from mdt.datasets.hdf5_datasets import HDF5Dataset_for_MotoGPT_CALVINLike
    
    if output_dir is None:
        output_dir = Path(hdf5_dir) / 'preprocessors'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("准备 QPOs 预处理器")
    print("="*80)
    
    # 1. 加载训练数据
    print("\n步骤 1: 加载训练数据...")
    dataset = HDF5Dataset_for_MotoGPT_CALVINLike(
        hdf5_dir=hdf5_dir,
        split='train',
        skip_frame=1,
        sequence_length=10,
        chunk_size=1,
        act_dim=7,
        do_extract_action=True,
    )
    
    # 2. 收集所有 qpos 数据
    print("步骤 2: 收集 qpos 数据...")
    all_qpos = []
    
    num_samples = min(len(dataset), 1000)  # 采样部分数据即可
    for i in range(0, num_samples, 10):
        try:
            sample = dataset[i]
            qpos = sample['actions']  # (seq_len, 7)
            if isinstance(qpos, torch.Tensor):
                qpos = qpos.numpy()
            all_qpos.append(qpos)
        except Exception as e:
            print(f"警告: 样本 {i} 加载失败: {e}")
            continue
    
    all_qpos = np.stack(all_qpos, axis=0)  # (N, seq_len, 7)
    print(f"收集了 {len(all_qpos)} 个序列")
    
    # 3. 计算统计信息
    print("\n步骤 3: 计算统计信息...")
    stats = compute_qpos_statistics(all_qpos)
    
    joint_names = [
        'Joint_0', 'Joint_1', 'Joint_2', 'Joint_3',
        'Joint_4', 'Joint_5', 'Gripper'
    ]
    print_qpos_statistics(stats, joint_names)
    
    # 4. 创建并保存归一化器
    print("\n步骤 4: 创建 QPosNormalizer...")
    normalizer = QPosNormalizer(normalize_gripper=False, gripper_dim=6)
    normalizer.fit(all_qpos, margin=0.05)
    
    normalizer_path = output_dir / 'qpos_normalizer.npz'
    normalizer.save(str(normalizer_path))
    
    # 5. 如果使用相对动作，创建相对变换器
    if use_relative:
        print("\n步骤 5: 创建 RelativeQPosTransform...")
        rel_transform = RelativeQPosTransform(normalize=True, clip=True)
        rel_transform.fit(all_qpos, percentile=99.0)
        
        transform_path = output_dir / 'relative_qpos_transform.npz'
        rel_transform.save(str(transform_path))
    
    print("\n" + "="*80)
    print("预处理器准备完成！")
    print(f"保存位置: {output_dir}")
    print("="*80)
    
    return {
        'normalizer_path': str(normalizer_path),
        'transform_path': str(transform_path) if use_relative else None,
        'stats': stats,
    }


def test_qpos_preprocessing():
    """
    测试 qpos 预处理流程
    """
    print("="*80)
    print("测试 QPOs 预处理")
    print("="*80)
    
    # 1. 准备测试数据
    qpos_example = np.array([
        [ 0.4181,  1.3532, -1.4948, -0.1664,  1.0980,  0.1267,  0.0000],
        [ 0.4329,  1.3868, -1.5244, -0.1664,  1.1082,  0.1267,  0.0000],
        [ 0.4520,  1.4363, -1.5675, -0.1664,  1.1170,  0.1267,  0.0000],
        [ 0.4623,  1.4682, -1.5957, -0.1646,  1.1237,  0.1270,  0.0000],
        [ 0.4729,  1.5072, -1.6297, -0.1648,  1.1297,  0.1310,  0.0000]
    ])
    
    print("\n原始 qpos 数据:")
    print(qpos_example)
    
    # 2. 方案A: 绝对 qpos 归一化
    print("\n" + "="*80)
    print("方案 A: 预测归一化的绝对 qpos")
    print("="*80)
    
    # 模拟从数据中统计得到的范围
    qpos_min = np.array([0.3, 1.2, -1.7, -0.2, 1.0, 0.1, 0.0])
    qpos_max = np.array([0.6, 1.6, -1.3, -0.1, 1.2, 0.2, 0.0])
    
    normalizer = QPosNormalizer(qpos_min, qpos_max, normalize_gripper=False)
    
    normalized = normalizer.normalize(qpos_example)
    print("\n归一化后 ([-1, 1] 范围):")
    print(normalized)
    
    denormalized = normalizer.denormalize(normalized)
    print("\n反归一化后:")
    print(denormalized)
    print(f"最大误差: {np.abs(qpos_example - denormalized).max():.8f}")
    
    # 3. 方案B: 相对 qpos (delta)
    print("\n" + "="*80)
    print("方案 B: 预测归一化的相对 qpos (delta)")
    print("="*80)
    
    # 模拟从数据中统计得到的最大增量
    max_delta = np.array([0.05, 0.1, 0.1, 0.05, 0.05, 0.01, 0.0])
    
    rel_transform = RelativeQPosTransform(max_delta, normalize=True, clip=True)
    
    # 计算序列的 delta
    qpos_current_seq = np.concatenate([
        qpos_example[:1],      # t=0
        qpos_example[:-1]       # t=1...n-1
    ], axis=0)
    
    delta_qpos = rel_transform.to_relative(qpos_example, qpos_current_seq)
    print("\n相对增量 (归一化到 [-1, 1]):")
    print(delta_qpos)
    
    # 从 delta 恢复
    recovered = rel_transform.to_absolute(delta_qpos, qpos_current_seq)
    print("\n从 delta 恢复的绝对 qpos:")
    print(recovered)
    print(f"最大误差: {np.abs(qpos_example - recovered).max():.8f}")
    
    # 4. 推荐配置
    print("\n" + "="*80)
    print("推荐配置")
    print("="*80)
    
    print("""
建议使用方案 B (相对 qpos)，原因：

1. ✓ 更容易学习
   - 大部分时候变化是平滑且小幅的
   - 扩散模型更容易捕捉小范围的增量分布

2. ✓ 更好的泛化性
   - 相对动作不依赖绝对起始位置
   - 可以从不同初始状态执行相同的动作轨迹

3. ✓ 避免累积误差
   - 每步都基于真实观测的当前状态
   - 不会因为预测误差累积导致偏离

4. ✓ 符合 MDT 原始设计
   - 原 CALVIN 数据集就是使用相对动作
   - 代码中已有相关处理逻辑

配置示例：
```yaml
# 在 datamodule config 中
observation_space:
  actions: ['rel_actions']  # 使用相对动作

# 或者在 dataset 初始化时
use_relative_actions: True
qpos_normalizer_path: 'preprocessors/qpos_normalizer.npz'
relative_transform_path: 'preprocessors/relative_qpos_transform.npz'
```
    """)


if __name__ == '__main__':
    # 运行测试
    test_qpos_preprocessing()
    
    # 如果有真实数据，可以运行：
    # prepare_qpos_preprocessors(
    #     hdf5_dir='/group/ycyang/jfwu/aloha/data/whl',
    #     output_dir='/group/ycyang/jfwu/aloha/data/whl/preprocessors',
    #     use_relative=True,
    # )
