import os
import re
import bisect
import random
import h5py
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from einops import rearrange


class HDF5Dataset_for_MotoGPT_CALVINLike(Dataset):
    def __init__(
        self,
        hdf5_dir,
        split,
        skip_frame,
        sequence_length,
        chunk_size=3,
        act_dim=7,
        do_extract_future_frames=True,
        do_extract_action=True,
        rgb_shape=(224, 224),
        rgb_preprocessor=None,
        max_skip_frame=None,
        no_repeat_data=False,
        camera_key="observations/images/cam_high",
        qpos_key="observations/qpos",
        camera_gripper_key="observations/images/cam_right_wrist",
        camera_left_key="observations/images/cam_left_wrist",
        debug=True,
        debug_sample_idx=None,
        max_attempts=50,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.skip_frame = skip_frame
        self.max_skip_frame = max_skip_frame
        self.do_extract_future_frames = do_extract_future_frames
        self.do_extract_action = do_extract_action
        self.rgb_shape = rgb_shape
        self.rgb_preprocessor = rgb_preprocessor
        self.no_repeat_data = no_repeat_data
        self.debug = debug
        self.debug_sample_idx = debug_sample_idx
        self.max_attempts = max_attempts

        self.camera_key = camera_key
        self.qpos_key = qpos_key
        self.camera_gripper_key = camera_gripper_key
        self.camera_left_key = camera_left_key

        # cache of open h5py.File handles (per-process / per-worker)
        # key: file path, value: h5py.File object opened in read-only mode
        self._file_cache = {}

        # preallocate dummies (match CALVIN output types and shapes)
        self.dummy_rgb_initial = torch.zeros(1, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_rgb_future = torch.zeros(sequence_length, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_actions = torch.zeros(sequence_length, chunk_size, act_dim, dtype=torch.float32)
        self.dummy_mask = torch.zeros(sequence_length, chunk_size, dtype=torch.float32)
        self.dummy_latent_mask = torch.zeros(sequence_length, dtype=torch.float32)
        self.dummy_rgb_initial_gripper = torch.zeros(1, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_rgb_future_gripper = torch.zeros(sequence_length, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_rgb_initial_left = torch.zeros(1, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_rgb_future_left = torch.zeros(sequence_length, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        assert split in ['train', 'val'], "split must be 'train' or 'val'"
        self.split = split
        if self._has_subsplits(hdf5_dir):
            hdf5_dir = os.path.join(hdf5_dir, split)
        else:
            raise ValueError("HDF5 dataset directory must contain 'train' and 'val' subdirectories")

        # discover episodes across subfolders and pair language from instr.txt when present
        episode_items = self._discover_episodes_with_lang(hdf5_dir)
        self.episodes = episode_items  # list of tuples (file_path, lang)
        if self.max_skip_frame is not None:
            assert self.max_skip_frame >= self.skip_frame, "max_skip_frame must be >= skip_frame"
            max_boundary = max(self.sequence_length * self.max_skip_frame, (self.sequence_length - 1) * self.max_skip_frame + self.chunk_size)
        else:
            max_boundary = max(self.sequence_length * self.skip_frame, (self.sequence_length - 1) * self.skip_frame + self.chunk_size)

        # precompute per-episode frame counts and available start indices
        self.episode_lengths = []  # number of frames in cam_high
        self.available_starts = []  # per-episode valid start count respecting minimal delta and chunk
        for ep_path, _lang in self.episodes:
            n = self._read_num_frames(ep_path, self.camera_key)
            self.episode_lengths.append(n)
            # boundary considering skip_frame and chunk_size
            avail = max(0, n - max_boundary)
            # avail = max(0, n - self.sequence_length * self.skip_frame - self.chunk_size)
            self.available_starts.append(avail)

        # build cumulative counts for binary search mapping
        self.cumulative = []
        total = 0
        for c in self.available_starts:
            total += c
            self.cumulative.append(total)
        self.dataset_len = total

    def _has_subsplits(self, root):
        return os.path.isdir(os.path.join(root, "train")) and os.path.isdir(os.path.join(root, "val"))

    def _discover_task_dirs(self, root):
        if not os.path.isdir(root):
            return []
        dirs = []
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                dirs.append(path)
        return dirs

    def _read_instr_map(self, task_dir):
        instr_path = os.path.join(task_dir, "instr.txt")
        mapping = {}
        if os.path.isfile(instr_path):
            try:
                with open(instr_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        m = re.match(r"^(\d+)\s+(.*)$", line)
                        if m:
                            idx = int(m.group(1))
                            text = m.group(2).strip()
                            mapping[idx] = text
            except Exception:
                # if malformed, fall back to empty mapping
                pass
        return mapping

    def _discover_episodes_in_dir(self, task_dir):
        # return list of (file_path, lang)
        items = []
        instr_map = self._read_instr_map(task_dir)
        for name in os.listdir(task_dir):
            path = os.path.join(task_dir, name)
            if os.path.isfile(path) and re.match(r"episode_\d+\.hdf5$", name):
                m = re.match(r"episode_(\d+)\.hdf5$", name)
                ep_idx = int(m.group(1)) if m else -1
                lang = instr_map.get(ep_idx, "")
                items.append((path, lang))
        # sort by episode index inferred from filename for determinism
        items.sort(key=lambda x: int(re.search(r"episode_(\d+)\.hdf5$", os.path.basename(x[0])).group(1)))
        return items

    def _discover_episodes_with_lang(self, root):
        if not os.path.isdir(root):
            return []
        # if episodes directly under root, use them
        direct = self._discover_episodes_in_dir(root)
        if direct:
            return direct
        # else aggregate across subfolders (tasks)
        items = []
        for task_dir in self._discover_task_dirs(root):
            items.extend(self._discover_episodes_in_dir(task_dir))
        # stable sort by (task dir path, episode index)
        items.sort(key=lambda x: (os.path.dirname(x[0]), int(re.search(r"episode_(\d+)\.hdf5$", os.path.basename(x[0])).group(1))))
        return items

    def _read_num_frames(self, file_path, camera_key):
        with h5py.File(file_path, "r") as f:
            ds = f[camera_key]
            return int(ds.shape[0])

    def _read_frame(self, file_path, frame_idx):
        with h5py.File(file_path, "r") as f:
            img = f[self.camera_key][frame_idx]
        frame = torch.from_numpy(rearrange(img, "h w c -> c h w"))
        if self.rgb_preprocessor is not None:
            frame = self.rgb_preprocessor(frame)
        return frame

    def _read_qpos(self, file_path, idx):
        with h5py.File(file_path, "r") as f:
            qpos = f[self.qpos_key][idx]
        # take last 7 dims as action
        act = np.asarray(qpos[-7:], dtype=np.float32)
        return torch.from_numpy(act)

    # Optimized readers using an already opened file handle
    def _read_frame_f(self, f, frame_idx):
        img = f[self.camera_key][frame_idx]
        frame = torch.from_numpy(rearrange(img, "h w c -> c h w"))
        if self.rgb_preprocessor is not None:
            frame = self.rgb_preprocessor(frame)
        return frame
    
    def _read_frame_gripper_f(self, f, frame_idx):
        img = f[self.camera_gripper_key][frame_idx]
        frame = torch.from_numpy(rearrange(img, "h w c -> c h w"))
        if self.rgb_preprocessor is not None:
            frame = self.rgb_preprocessor(frame)
        return frame
    
    def _read_frame_left_f(self, f, frame_idx):
        img = f[self.camera_left_key][frame_idx]
        frame = torch.from_numpy(rearrange(img, "h w c -> c h w"))
        if self.rgb_preprocessor is not None:
            frame = self.rgb_preprocessor(frame)
        return frame

    def _read_qpos_f(self, f, idx):
        qpos = f[self.qpos_key][idx]
        act = np.asarray(qpos[-7:], dtype=np.float32)
        return torch.from_numpy(act)

    def _get_file(self, file_path):
        """Get (and cache) an open h5py.File handle for the given path.

        This avoids repeatedly opening/closing the same HDF5 file for every
        sample, which can be a significant overhead when episodes are small
        and accessed many times per epoch. The cache is per Dataset instance
        and thus per DataLoader worker process.
        """
        f = self._file_cache.get(file_path, None)
        # h5py File objects have an "id" attribute; when the file is closed,
        # "id" evaluates to False. We reopen if missing or already closed.
        if f is None or not f.id:
            f = h5py.File(file_path, "r")
            self._file_cache[file_path] = f
        return f

    def _map_global_to_episode(self, global_idx):
        # binary search on cumulative to find episode index
        ep_idx = bisect.bisect_right(self.cumulative, global_idx)
        prev_total = 0 if ep_idx == 0 else self.cumulative[ep_idx - 1]
        local_start = global_idx - prev_total
        return ep_idx, local_start

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # bounded attempts to avoid infinite loops in workers
        attempts = 0
        while attempts < self.max_attempts:
            try:
                # map to episode and local start
                ep_idx, local_start = self._map_global_to_episode(idx)
                file_path, lang = self.episodes[ep_idx]
                num_frames = self.episode_lengths[ep_idx]

                # pick delta_t
                if self.max_skip_frame is None:
                    delta_t = self.skip_frame
                else:
                    max_sf = max(self.skip_frame, min(num_frames - 1, self.max_skip_frame))
                    delta_t = random.randint(self.skip_frame, max_sf)

                # start within boundary (ensure at least one future frame with min skip);
                # clip local_start if episode got shorter
                max_start = max(0, num_frames - self.skip_frame - 1)
                start_local_step = min(local_start, max_start)

                # dummies per sample
                rgb_initial = self.dummy_rgb_initial.clone()
                rgb_future = self.dummy_rgb_future.clone()
                actions = self.dummy_actions.clone()
                mask = self.dummy_mask.clone()
                latent_mask = self.dummy_latent_mask.clone()
                rgb_initial_gripper = self.dummy_rgb_initial_gripper.clone()
                rgb_future_gripper = self.dummy_rgb_future_gripper.clone()
                rgb_initial_left = self.dummy_rgb_initial_left.clone()
                rgb_future_left = self.dummy_rgb_future_left.clone()

                # lang paired from instr.txt per episode when available; otherwise empty

                # get (and cache) an open file handle for this episode
                f = self._get_file(file_path)

                # initial frame
                rgb_initial[0] = self._read_frame_f(f, start_local_step)
                rgb_initial_gripper[0] = self._read_frame_gripper_f(f, start_local_step)
                rgb_initial_left[0] = self._read_frame_left_f(f, start_local_step)

                # future frames
                if self.do_extract_future_frames:
                    for i in range(self.sequence_length):
                        next_idx = start_local_step + (i + 1) * delta_t
                        if next_idx < num_frames:
                            rgb_future[i] = self._read_frame_f(f, next_idx)
                            rgb_future_gripper[i] = self._read_frame_gripper_f(f, next_idx)
                            rgb_future_left[i] = self._read_frame_left_f(f, next_idx)
                            latent_mask[i] = 1
                        else:
                            break

                # actions (from qpos last 7 dims)
                if self.do_extract_action:
                    for i in range(self.sequence_length):
                        for j in range(self.chunk_size):
                            cur_idx = start_local_step + i * delta_t + j
                            if cur_idx < num_frames:
                                mask[i, j] = 1
                                actions[i, j] = self._read_qpos_f(f, cur_idx)

                if not self.no_repeat_data:
                    if self.do_extract_future_frames and (not self.do_extract_action) and latent_mask.sum() == 0:
                        raise RuntimeError("latent_mask should be larger than zero!")

                return {
                    "lang": lang,
                    "rgb_initial": rgb_initial,
                    "rgb_future": rgb_future,
                    "rgb_initial_gripper": rgb_initial_gripper,
                    "rgb_future_gripper": rgb_future_gripper,
                    "rgb_initial_left": rgb_initial_left,
                    "rgb_future_left": rgb_future_left,
                    "actions": actions,
                    "mask": mask,
                    "latent_mask": latent_mask,
                    "idx": idx,
                    "delta_t": delta_t,
                    "start_local_step": start_local_step,
                }

            except Exception as e:
                # resample randomly on failure
                attempts += 1
                if self.debug:
                    import traceback
                    print(f"[HDF5Dataset DEBUG] attempt={attempts} idx={idx} error={e}")
                    traceback.print_exc()
                if self.dataset_len <= 0:
                    raise
                idx = random.randint(0, self.dataset_len - 1)

        # if we reach here, give up to avoid hanging
        raise RuntimeError(f"Failed to fetch sample after {self.max_attempts} attempts.")

    def __del__(self):
        # best-effort cleanup of cached file handles
        file_cache = getattr(self, "_file_cache", None)
        if not file_cache:
            return
        for f in file_cache.values():
            try:
                f.close()
            except Exception:
                pass
