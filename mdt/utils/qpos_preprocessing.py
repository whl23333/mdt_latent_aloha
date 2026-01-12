"""
QPos (关节空间) 数据预处理工具
用于 ALOHA 真机训练
"""
import numpy as np
import torch
from typing import Dict, Tuple, Optional


class QPosNormalizer:
    """
    关节空间位置归一化器
    
    关键设计决策：
    1. 每个关节独立归一化（因为值域差异大）
    2. 使用 min-max 归一化到 [-1, 1] 便于扩散模型训练
    3. 夹爪维度可选单独处理
    """
    
    def __init__(
        self,
        qpos_min: np.ndarray = None,
        qpos_max: np.ndarray = None,
        normalize_gripper: bool = False,
        gripper_dim: int = 6,  # 最后一维是夹爪
    ):
        """
        Args:
            qpos_min: 每个关节的最小值 (7,)
            qpos_max: 每个关节的最大值 (7,)
            normalize_gripper: 是否归一化夹爪维度
            gripper_dim: 夹爪所在的维度索引
        """
        self.qpos_min = qpos_min
        self.qpos_max = qpos_max
        self.normalize_gripper = normalize_gripper
        self.gripper_dim = gripper_dim
        
        if qpos_min is not None and qpos_max is not None:
            self.qpos_range = qpos_max - qpos_min
            # 避免除零
            self.qpos_range[self.qpos_range < 1e-6] = 1.0
    
    def fit(self, qpos_data: np.ndarray, margin: float = 0.05):
        """
        从数据中计算归一化参数
        
        Args:
            qpos_data: (N, seq_len, 7) 或 (N, 7) 的 qpos 数据
            margin: 扩展边界的余量（防止测试数据超出范围）
        """
        if qpos_data.ndim == 3:
            qpos_data = qpos_data.reshape(-1, qpos_data.shape[-1])
        
        self.qpos_min = np.min(qpos_data, axis=0)
        self.qpos_max = np.max(qpos_data, axis=0)
        
        # 添加边界余量
        qpos_range = self.qpos_max - self.qpos_min
        self.qpos_min -= qpos_range * margin
        self.qpos_max += qpos_range * margin
        
        self.qpos_range = self.qpos_max - self.qpos_min
        self.qpos_range[self.qpos_range < 1e-6] = 1.0
        
        print(f"QPosNormalizer fitted:")
        print(f"  Min: {self.qpos_min}")
        print(f"  Max: {self.qpos_max}")
        print(f"  Range: {self.qpos_range}")
    
    def normalize(self, qpos: np.ndarray) -> np.ndarray:
        """
        归一化 qpos 到 [-1, 1]
        
        Args:
            qpos: (..., 7) 原始 qpos
        
        Returns:
            normalized_qpos: (..., 7) 归一化后的 qpos
        """
        # Min-max 归一化到 [0, 1]
        normalized = (qpos - self.qpos_min) / self.qpos_range
        
        # 缩放到 [-1, 1]
        normalized = normalized * 2.0 - 1.0
        
        # 夹爪维度可选不归一化（如果已经是 0/1 或 -1/1）
        if not self.normalize_gripper:
            normalized[..., self.gripper_dim] = qpos[..., self.gripper_dim]
        
        return normalized
    
    def denormalize(self, normalized_qpos: np.ndarray) -> np.ndarray:
        """
        反归一化 qpos
        
        Args:
            normalized_qpos: (..., 7) 归一化的 qpos [-1, 1]
        
        Returns:
            qpos: (..., 7) 原始尺度的 qpos
        """
        # 夹爪维度特殊处理
        if not self.normalize_gripper:
            gripper = normalized_qpos[..., self.gripper_dim:self.gripper_dim+1]
        
        # 从 [-1, 1] 缩放到 [0, 1]
        unnormalized = (normalized_qpos + 1.0) / 2.0
        
        # 从 [0, 1] 恢复到原始尺度
        qpos = unnormalized * self.qpos_range + self.qpos_min
        
        # 恢复夹爪
        if not self.normalize_gripper:
            qpos[..., self.gripper_dim] = gripper.squeeze(-1)
        
        return qpos
    
    def save(self, path: str):
        """保存归一化参数"""
        np.savez(
            path,
            qpos_min=self.qpos_min,
            qpos_max=self.qpos_max,
            normalize_gripper=self.normalize_gripper,
            gripper_dim=self.gripper_dim,
        )
        print(f"QPosNormalizer saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'QPosNormalizer':
        """加载归一化参数"""
        data = np.load(path)
        normalizer = cls(
            qpos_min=data['qpos_min'],
            qpos_max=data['qpos_max'],
            normalize_gripper=bool(data['normalize_gripper']),
            gripper_dim=int(data['gripper_dim']),
        )
        print(f"QPosNormalizer loaded from {path}")
        return normalizer


class RelativeQPosTransform:
    """
    将绝对 qpos 转换为相对增量（delta qpos）
    
    优势：
    1. 更容易学习（大部分时候变化是连续且平滑的）
    2. 更好的泛化性
    3. 避免累积误差
    """
    
    def __init__(
        self,
        max_delta: np.ndarray = None,
        normalize: bool = True,
        clip: bool = True,
    ):
        """
        Args:
            max_delta: 每个关节的最大增量 (7,)，用于归一化和裁剪
            normalize: 是否归一化 delta 到 [-1, 1]
            clip: 是否裁剪超出范围的增量
        """
        self.max_delta = max_delta
        self.normalize = normalize
        self.clip = clip
    
    def fit(self, qpos_data: np.ndarray, percentile: float = 99.0):
        """
        从数据中计算最大增量
        
        Args:
            qpos_data: (N, seq_len, 7) 的 qpos 数据
            percentile: 使用百分位数而非绝对最大值（更鲁棒）
        """
        if qpos_data.ndim == 3:
            # 计算相邻帧的差异
            delta = np.diff(qpos_data, axis=1)  # (N, seq_len-1, 7)
            delta = delta.reshape(-1, delta.shape[-1])
        else:
            raise ValueError("qpos_data should be (N, seq_len, 7)")
        
        # 使用百分位数作为最大增量
        self.max_delta = np.percentile(np.abs(delta), percentile, axis=0)
        
        # 确保不为零
        self.max_delta[self.max_delta < 1e-6] = 0.1
        
        print(f"RelativeQPosTransform fitted:")
        print(f"  Max delta ({percentile}th percentile): {self.max_delta}")
    
    def to_relative(
        self,
        qpos_target: np.ndarray,
        qpos_current: np.ndarray,
    ) -> np.ndarray:
        """
        计算相对增量
        
        Args:
            qpos_target: (..., 7) 目标 qpos
            qpos_current: (..., 7) 当前 qpos
        
        Returns:
            delta_qpos: (..., 7) 相对增量
        """
        delta = qpos_target - qpos_current
        
        if self.clip:
            delta = np.clip(delta, -self.max_delta, self.max_delta)
        
        if self.normalize:
            delta = delta / self.max_delta  # 归一化到 [-1, 1]
        
        return delta
    
    def to_absolute(
        self,
        delta_qpos: np.ndarray,
        qpos_current: np.ndarray,
    ) -> np.ndarray:
        """
        从相对增量恢复绝对 qpos
        
        Args:
            delta_qpos: (..., 7) 预测的相对增量 [-1, 1]
            qpos_current: (..., 7) 当前 qpos
        
        Returns:
            qpos_target: (..., 7) 目标 qpos
        """
        if self.normalize:
            delta = delta_qpos * self.max_delta  # 反归一化
        else:
            delta = delta_qpos
        
        qpos_target = qpos_current + delta
        
        return qpos_target
    
    def save(self, path: str):
        """保存参数"""
        np.savez(
            path,
            max_delta=self.max_delta,
            normalize=self.normalize,
            clip=self.clip,
        )
        print(f"RelativeQPosTransform saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'RelativeQPosTransform':
        """加载参数"""
        data = np.load(path)
        transform = cls(
            max_delta=data['max_delta'],
            normalize=bool(data['normalize']),
            clip=bool(data['clip']),
        )
        print(f"RelativeQPosTransform loaded from {path}")
        return transform


def compute_qpos_statistics(qpos_data: np.ndarray) -> Dict:
    """
    计算 qpos 数据的统计信息
    
    Args:
        qpos_data: (N, seq_len, 7) 或 (N, 7) 的 qpos 数据
    
    Returns:
        统计信息字典
    """
    if qpos_data.ndim == 3:
        qpos_flat = qpos_data.reshape(-1, qpos_data.shape[-1])
    else:
        qpos_flat = qpos_data
    
    stats = {
        'min': np.min(qpos_flat, axis=0),
        'max': np.max(qpos_flat, axis=0),
        'mean': np.mean(qpos_flat, axis=0),
        'std': np.std(qpos_flat, axis=0),
        'median': np.median(qpos_flat, axis=0),
        'percentile_1': np.percentile(qpos_flat, 1, axis=0),
        'percentile_99': np.percentile(qpos_flat, 99, axis=0),
    }
    
    # 计算相邻帧变化（如果是序列数据）
    if qpos_data.ndim == 3:
        delta = np.diff(qpos_data, axis=1)
        delta_flat = delta.reshape(-1, delta.shape[-1])
        
        stats['delta_mean'] = np.mean(delta_flat, axis=0)
        stats['delta_std'] = np.std(delta_flat, axis=0)
        stats['delta_max'] = np.max(np.abs(delta_flat), axis=0)
        stats['delta_percentile_99'] = np.percentile(np.abs(delta_flat), 99, axis=0)
    
    return stats


def print_qpos_statistics(stats: Dict, joint_names: list = None):
    """
    打印 qpos 统计信息
    """
    if joint_names is None:
        joint_names = [f"Joint_{i}" for i in range(len(stats['min']))]
    
    print("\n" + "="*80)
    print("QPOs Statistics Summary")
    print("="*80)
    
    print(f"\n{'Joint':<15} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Range':>10}")
    print("-"*80)
    
    for i, name in enumerate(joint_names):
        range_val = stats['max'][i] - stats['min'][i]
        print(f"{name:<15} {stats['min'][i]:>10.4f} {stats['max'][i]:>10.4f} "
              f"{stats['mean'][i]:>10.4f} {stats['std'][i]:>10.4f} {range_val:>10.4f}")
    
    if 'delta_max' in stats:
        print("\n" + "-"*80)
        print("Delta Statistics (frame-to-frame changes)")
        print("-"*80)
        print(f"{'Joint':<15} {'Mean Delta':>12} {'Std Delta':>12} {'Max Delta':>12} {'99th %':>12}")
        print("-"*80)
        
        for i, name in enumerate(joint_names):
            print(f"{name:<15} {stats['delta_mean'][i]:>12.6f} {stats['delta_std'][i]:>12.6f} "
                  f"{stats['delta_max'][i]:>12.6f} {stats['delta_percentile_99'][i]:>12.6f}")
    
    print("="*80 + "\n")


# 示例使用
if __name__ == '__main__':
    # 你的示例数据
    qpos_example = torch.tensor([
        [ 0.4181,  1.3532, -1.4948, -0.1664,  1.0980,  0.1267,  0.0000],
        [ 0.4329,  1.3868, -1.5244, -0.1664,  1.1082,  0.1267,  0.0000],
        [ 0.4520,  1.4363, -1.5675, -0.1664,  1.1170,  0.1267,  0.0000],
        [ 0.4623,  1.4682, -1.5957, -0.1646,  1.1237,  0.1270,  0.0000],
        [ 0.4729,  1.5072, -1.6297, -0.1648,  1.1297,  0.1310,  0.0000]
    ]).numpy()
    
    # 假设有更多数据用于统计
    qpos_data = np.random.randn(100, 10, 7) * 0.5 + qpos_example.mean(axis=0)
    
    print("示例 1: 绝对 qpos 归一化")
    print("-" * 80)
    normalizer = QPosNormalizer()
    normalizer.fit(qpos_data, margin=0.05)
    
    normalized = normalizer.normalize(qpos_example)
    print(f"\n原始 qpos[0]: {qpos_example[0]}")
    print(f"归一化后[0]: {normalized[0]}")
    
    denormalized = normalizer.denormalize(normalized)
    print(f"反归一化后[0]: {denormalized[0]}")
    print(f"误差: {np.abs(qpos_example[0] - denormalized[0]).max():.6f}")
    
    print("\n" + "="*80)
    print("示例 2: 相对 qpos (delta) 转换")
    print("-" * 80)
    
    rel_transform = RelativeQPosTransform()
    rel_transform.fit(qpos_data, percentile=99.0)
    
    # 计算相对增量
    qpos_current = qpos_example[0]
    qpos_target = qpos_example[1]
    
    delta = rel_transform.to_relative(qpos_target, qpos_current)
    print(f"\n当前 qpos: {qpos_current}")
    print(f"目标 qpos: {qpos_target}")
    print(f"相对增量 (归一化): {delta}")
    
    # 恢复绝对值
    recovered = rel_transform.to_absolute(delta, qpos_current)
    print(f"恢复的目标: {recovered}")
    print(f"误差: {np.abs(qpos_target - recovered).max():.6f}")
    
    print("\n" + "="*80)
    print("示例 3: 数据统计分析")
    print("-" * 80)
    
    stats = compute_qpos_statistics(qpos_data)
    joint_names = ['Waist', 'Shoulder', 'Elbow', 'Wrist1', 'Wrist2', 'Wrist3', 'Gripper']
    print_qpos_statistics(stats, joint_names)
