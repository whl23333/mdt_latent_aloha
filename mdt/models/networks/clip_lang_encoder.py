from typing import List

import torch
import torch.nn as nn

from mdt.models.networks.clip import build_model, load_clip, tokenize


class LangClip(nn.Module):
    def __init__(self, freeze_backbone: bool = True, model_name: str = "RN50"):
        super(LangClip, self).__init__()
        # Load CLIP model on CPU first to avoid eager CUDA init; PL will move module later
        print(f"loading language CLIP model with backbone: {model_name}")
        self._load_clip(model_name)
        if freeze_backbone:
            for param in self.clip_rn50.parameters():
                param.requires_grad = False

    def _load_clip(self, model_name: str) -> None:
        # Build on CPU; Lightning will place on the correct device later
        model, _ = load_clip(model_name, device=torch.device('cpu'))
        self.clip_rn50 = build_model(model.state_dict())

    def forward(self, x: List) -> torch.Tensor:
        # Use current module device determined by PL
        dev = next(self.clip_rn50.parameters()).device
        with torch.no_grad():
            tokens = tokenize(x).to(dev)
            emb = self.clip_rn50.encode_text(tokens)
        return torch.unsqueeze(emb, 1)
