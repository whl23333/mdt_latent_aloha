from __future__ import annotations

from typing import Iterable, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Sampler


def _get_dist_info() -> tuple[int, int]:
    """Return (world_size, rank). If not distributed, returns (1, 0)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    return 1, 0


def _pad_to_divisible(indices: np.ndarray, divisor: int, rng: np.random.RandomState) -> np.ndarray:
    if divisor <= 1:
        return indices
    n = len(indices)
    remainder = n % divisor
    if remainder == 0:
        return indices
    need = divisor - remainder
    if n == 0:
        # nothing to pad; create a tiny sequence of zeros
        pad = np.zeros(need, dtype=np.int64)
    else:
        pad_idx = rng.randint(0, n, size=need)
        pad = indices[pad_idx]
    return np.concatenate([indices, pad], axis=0)


class EpochRandomSubsetSampler(Sampler[int]):
    """Sample a fixed fraction of dataset indices each epoch.

    - Keeps dataset intact; reduces effective data per epoch.
    - Resamples a fresh random subset per epoch when `resample_each_epoch=True`.
    - Safe with window logic because the underlying dataset indices remain valid.

    Args:
        data_source: Dataset to sample from.
        fraction: 0 < fraction <= 1, proportion of examples per epoch.
        seed: Base RNG seed.
        resample_each_epoch: If True, subset changes every epoch via `set_epoch()`.
        replacement: If True, sample with replacement.
        shuffle: Shuffle chosen indices (ignored when replacement=True).
        drop_remainder: If True, drop extra indices to make each rank equal in DDP.
                         If False, pad by sampling with replacement for equal lengths.
    """

    def __init__(
        self,
        data_source: Iterable,
        fraction: float,
        seed: int = 0,
        resample_each_epoch: bool = True,
        replacement: bool = False,
        shuffle: bool = True,
        drop_remainder: bool = False,
    ) -> None:
        assert 0.0 < fraction <= 1.0, "fraction must be in (0, 1]"
        self.data_source = data_source
        self.fraction = fraction
        self.seed = seed
        self.resample_each_epoch = resample_each_epoch
        self.replacement = replacement
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder
        self.epoch = 0
        # Will also refresh world_size/rank in __len__/__iter__ for robustness
        self.world_size, self.rank = _get_dist_info()

        # Fixed advertised length for DataLoader bookkeeping
        self._length = max(1, int(round(len(self.data_source) * self.fraction)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:  # type: ignore[override]
        # Return approximate length (actual may differ when using episode sampler etc.)
        # Here we keep it simple and constant across epochs.
        n = self._length
        world_size, _ = _get_dist_info()
        # Divide evenly across ranks for DDP bookkeeping
        return (n + world_size - 1) // world_size

    def __iter__(self) -> Iterator[int]:  # type: ignore[override]
        n_total = len(self.data_source)
        n_keep = max(1, int(round(n_total * self.fraction)))
        rng = np.random.RandomState(self.seed + (self.epoch if self.resample_each_epoch else 0))

        if self.replacement:
            chosen = rng.randint(0, n_total, size=n_keep, dtype=np.int64)
        else:
            chosen = rng.choice(n_total, size=n_keep, replace=False).astype(np.int64)
            if self.shuffle:
                rng.shuffle(chosen)

        world_size, rank = _get_dist_info()
        print(f"world_size: {world_size}, rank: {rank}, seed: {self.seed}, epoch: {self.epoch}")
        if world_size > 1:
            if self.drop_remainder:
                # Trim to be divisible
                cut = (len(chosen) // world_size) * world_size
                chosen = chosen[:cut]
            else:
                chosen = _pad_to_divisible(chosen, world_size, rng)
            # Shard by rank (contiguous chunk)
            per_rank = len(chosen) // world_size
            start = rank * per_rank
            end = start + per_rank
            chosen = chosen[start:end]

        return iter(chosen.tolist())


def _compute_episode_slices_from_lookup(episode_lookup: np.ndarray) -> List[slice]:
    """Compute contiguous dataset-index ranges per episode from episode_lookup.

    Two consecutive dataset indices belong to the same episode iff
    episode_lookup[i] == episode_lookup[i-1] + 1.
    """
    if len(episode_lookup) == 0:
        return []
    boundaries: List[int] = [0]
    for i in range(1, len(episode_lookup)):
        if episode_lookup[i] != episode_lookup[i - 1] + 1:
            boundaries.append(i)
    boundaries.append(len(episode_lookup))
    return [slice(boundaries[j], boundaries[j + 1]) for j in range(len(boundaries) - 1)]


class EpisodeRandomSubsetSampler(Sampler[int]):
    """Sample a fraction of entire episodes each epoch and yield all their indices.

    - 保持每个 episode 内的起始帧连续性，兼容 `BaseDataset._get_window_size` 的边界判断。
    - 每个 epoch 随机挑选若干 episode，避免固定子集导致过拟合。
    """

    def __init__(
        self,
        dataset,
        fraction: float,
        seed: int = 0,
        resample_each_epoch: bool = True,
        shuffle_within: bool = True,
        drop_remainder: bool = False,
    ) -> None:
        assert 0.0 < fraction <= 1.0, "fraction must be in (0, 1]"
        assert hasattr(dataset, "episode_lookup"), "dataset must expose episode_lookup (np.ndarray-like)"
        self.dataset = dataset
        self.fraction = fraction
        self.seed = seed
        self.resample_each_epoch = resample_each_epoch
        self.shuffle_within = shuffle_within
        self.drop_remainder = drop_remainder
        self.epoch = 0
        # Will also refresh in __len__/__iter__ for robustness
        self.world_size, self.rank = _get_dist_info()

        ep_lookup = np.asarray(self.dataset.episode_lookup)
        self._episode_slices = _compute_episode_slices_from_lookup(ep_lookup)
        self._length = max(1, int(round(len(self.dataset) * self.fraction)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:  # type: ignore[override]
        n = self._length
        world_size, _ = _get_dist_info()
        return (n + world_size - 1) // world_size

    def __iter__(self) -> Iterator[int]:  # type: ignore[override]
        rng = np.random.RandomState(self.seed + (self.epoch if self.resample_each_epoch else 0))
        n_episodes = len(self._episode_slices)
        keep_eps = max(1, int(round(n_episodes * self.fraction)))
        chosen_eps = rng.choice(n_episodes, size=keep_eps, replace=False)

        # Gather all indices from selected episodes
        indices: List[int] = []
        for e in chosen_eps:
            sl = self._episode_slices[e]
            if self.shuffle_within:
                # Shuffle within episode to avoid positional bias
                idxs = np.arange(sl.start, sl.stop, dtype=np.int64)
                rng.shuffle(idxs)
                indices.extend(idxs.tolist())
            else:
                indices.extend(range(sl.start, sl.stop))

        indices_np = np.array(indices, dtype=np.int64)
        # Global shuffle across episodes to enhance mixing
        rng.shuffle(indices_np)

        world_size, rank = _get_dist_info()
        print(f"world_size: {world_size}, rank: {rank}")
        if world_size > 1:
            if self.drop_remainder:
                cut = (len(indices_np) // world_size) * world_size
                indices_np = indices_np[:cut]
            else:
                indices_np = _pad_to_divisible(indices_np, world_size, rng)
            per_rank = len(indices_np) // world_size
            start = rank * per_rank
            end = start + per_rank
            indices_np = indices_np[start:end]

        return iter(indices_np.tolist())


class StrideStartIndexSampler(Sampler[int]):
    """Sample every `stride`-th starting index, optionally with a random offset per epoch.

    - 在不动数据集内部索引逻辑的前提下稀疏化起始帧，等价于降低窗口密度。
    - `random_offset=True` 时，每个 epoch 会在 [0, stride-1] 内随机选择偏移，保持随机性。
    """

    def __init__(
        self,
        data_source: Iterable,
        stride: int,
        random_offset: bool = True,
        seed: int = 0,
        resample_each_epoch: bool = True,
        drop_remainder: bool = False,
    ) -> None:
        assert stride >= 1, "stride must be >= 1"
        self.data_source = data_source
        self.stride = stride
        self.random_offset = random_offset
        self.seed = seed
        self.resample_each_epoch = resample_each_epoch
        self.drop_remainder = drop_remainder
        self.epoch = 0
        # Will also refresh in __len__/__iter__ for robustness
        self.world_size, self.rank = _get_dist_info()

        approx = max(1, int(np.ceil(len(self.data_source) / self.stride)))
        self._length = (approx + self.world_size - 1) // self.world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:  # type: ignore[override]
        world_size, _ = _get_dist_info()
        # keep approximate per-rank length consistent with current world size
        return (self._length + world_size - 1) // world_size

    def __iter__(self) -> Iterator[int]:  # type: ignore[override]
        n_total = len(self.data_source)
        rng = np.random.RandomState(self.seed + (self.epoch if self.resample_each_epoch else 0))
        if self.random_offset:
            offset = int(rng.randint(0, self.stride))
        else:
            offset = 0
        base = np.arange(offset, n_total, self.stride, dtype=np.int64)

        world_size, rank = _get_dist_info()
        if world_size > 1:
            if self.drop_remainder:
                cut = (len(base) // world_size) * world_size
                base = base[:cut]
            else:
                base = _pad_to_divisible(base, world_size, rng)
            per_rank = len(base) // world_size
            start = rank * per_rank
            end = start + per_rank
            base = base[start:end]

        return iter(base.tolist())
