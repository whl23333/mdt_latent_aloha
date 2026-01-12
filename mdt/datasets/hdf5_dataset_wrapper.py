from typing import Dict, Optional, Union

import torch
from torch.utils.data import Dataset


class HDF5DatasetWrapper(Dataset):
    """
    Wrap an HDF5Dataset_for_MotoGPT_CALVINLike to match the batch format
    of the existing CALVIN-like datasets.

    Mappings:
    - rgb_obs.rgb_static: [2, 3, H, W] built from initial + selected future frame
    - rgb_obs.rgb_gripper: [2, 3, H, W] built from initial + selected future frame
    - gen_static/gen_gripper: [3, H, W] taken from a selected future frame (default first)
    - actions: [seq_len, 7] merged from [seq_len, chunk_size, 7] by first/last/mean (mask-aware)
    - language_text: preserve from underlying sample['lang']
    - rgb_left_initial/rgb_left_future: preserved as-is from HDF5 dataset

    Options:
    - future_concat_index: int index into rgb_future (and rgb_future_gripper) or 'last'
      meaning last valid according to latent_mask; default 'last'.
    - gen_index: int index for gen_* taken from rgb_future; default 0 (first).
    - action_merge_mode: 'first' | 'last' | 'mean' for merging chunk dimension; default 'last'.
    """

    def __init__(
        self,
        base: Dataset,
        future_concat_index: Union[int, str] = "last",
        gen_index: int = 0,
        action_merge_mode: str = "mean",
        transforms: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__()
        assert action_merge_mode in ("first", "last", "mean")
        self.base = base
        self.future_concat_index = future_concat_index
        self.gen_index = gen_index
        self.action_merge_mode = action_merge_mode
        self.transforms = transforms or {}

    def __len__(self) -> int:
        return len(self.base)

    def _select_future_idx(self, latent_mask: Optional[torch.Tensor]) -> int:
        if isinstance(self.future_concat_index, int):
            return int(self.future_concat_index)
        # default: 'last' valid according to latent_mask if provided
        if latent_mask is None or latent_mask.numel() == 0:
            return -1
        valid = torch.nonzero(latent_mask > 0, as_tuple=False).view(-1)
        if valid.numel() == 0:
            return -1
        return int(valid[-1].item())

    def _merge_actions(self, actions: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # actions: [S, C, 7]; mask: [S, C]
        S, C, D = actions.shape
        out = torch.zeros(S, D, dtype=actions.dtype, device=actions.device)
        if mask is None:
            mask = torch.ones(S, C, dtype=actions.dtype, device=actions.device)
        for i in range(S):
            m = mask[i]
            if self.action_merge_mode == "mean":
                wsum = m.sum()
                if wsum > 0:
                    out[i] = (actions[i] * m.view(-1, 1)).sum(dim=0) / wsum
                else:
                    out[i] = actions[i].mean(dim=0)
            elif self.action_merge_mode == "first":
                idx = torch.nonzero(m > 0, as_tuple=False)
                j = int(idx[0].item()) if idx.numel() > 0 else 0
                out[i] = actions[i, j]
            else:  # 'last'
                idx = torch.nonzero(m > 0, as_tuple=False)
                j = int(idx[-1].item()) if idx.numel() > 0 else (C - 1)
                out[i] = actions[i, j]
        return out

    def __getitem__(self, idx: int) -> Dict:
        s = self.base[idx]

        rgb_init: torch.Tensor = s["rgb_initial"][0]
        rgb_fut: torch.Tensor = s["rgb_future"]
        rgb_init_gr: torch.Tensor = s["rgb_initial_gripper"][0]
        rgb_fut_gr: torch.Tensor = s["rgb_future_gripper"]
        latent_mask: Optional[torch.Tensor] = s.get("latent_mask", None)

        fut_idx = self._select_future_idx(latent_mask)
        if fut_idx < 0 or fut_idx >= rgb_fut.shape[0]:
            fut_idx = rgb_fut.shape[0] - 1

        rgb_static = torch.stack([rgb_init, rgb_fut[fut_idx]], dim=0)
        rgb_gripper = torch.stack([rgb_init_gr, rgb_fut_gr[fut_idx]], dim=0)

        gen_idx = max(0, min(self.gen_index, rgb_fut.shape[0] - 1))
        gen_static = rgb_fut[gen_idx].unsqueeze(0) # [1, 3, H, W]
        gen_gripper = rgb_fut_gr[gen_idx].unsqueeze(0) # [1, 3, H, W]

        # Apply optional transforms frame-wise like BaseDataset
        tf_static = self.transforms.get("rgb_static")
        tf_gripper = self.transforms.get("rgb_gripper")
        tf_gen_static = self.transforms.get("gen_static")
        tf_gen_gripper = self.transforms.get("gen_gripper")

        if tf_static is not None:
            rgb_static = torch.stack([tf_static(rgb_static[0]), tf_static(rgb_static[1])], dim=0)
        if tf_gripper is not None:
            rgb_gripper = torch.stack([tf_gripper(rgb_gripper[0]), tf_gripper(rgb_gripper[1])], dim=0)
        if tf_gen_static is not None:
            gen_static = tf_gen_static(gen_static)
        if tf_gen_gripper is not None:
            gen_gripper = tf_gen_gripper(gen_gripper)

        actions = s.get("actions")
        mask = s.get("mask")
        actions_merged = self._merge_actions(actions, mask)

        out: Dict = {
            "rgb_obs": {
                "rgb_static": rgb_static,
                "rgb_gripper": rgb_gripper,
                "gen_static": gen_static,
                "gen_gripper": gen_gripper,
            },
            "actions": actions_merged,
            # language keys used by MDTAgent
            "lang_text": s.get("lang", ""),
            "lang": s.get("lang", ""),
            # extras preserved
            "rgb_left_initial": s.get("rgb_initial_left"),
            "rgb_left_future": s.get("rgb_future_left"),
            "idx": s.get("idx", idx),
            "delta_t": s.get("delta_t"),
            "start_local_step": s.get("start_local_step"),
            # expose selected future offset for image-gen loss
            "future_frame_diff": torch.tensor(int(fut_idx)),
        }

        return out
