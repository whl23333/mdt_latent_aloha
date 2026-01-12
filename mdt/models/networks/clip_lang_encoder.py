from typing import List

import torch
import torch.nn as nn

from mdt.models.networks.clip import build_model, load_clip, tokenize


class LangClip(nn.Module):
    def __init__(self, freeze_backbone: bool = True, model_name: str = "RN50"):
        super(LangClip, self).__init__()
        # Load CLIP model
        print(f"loading language CLIP model with backbone: {model_name}")
        self._load_clip(model_name)
        if freeze_backbone:
            for param in self.clip_rn50.parameters():
                param.requires_grad = False

    def _load_clip(self, model_name: str) -> None:
        # Load on CPU; Lightning will move the module to the correct device per process
        model, _ = load_clip(model_name, device="cpu")
        self.clip_rn50 = build_model(model.state_dict())

    def forward(self, x: List) -> torch.Tensor:
        with torch.no_grad():
            device = next(self.clip_rn50.parameters()).device
            tokens = tokenize(x).to(device)
            emb = self.clip_rn50.encode_text(tokens)
        return torch.unsqueeze(emb, 1)
