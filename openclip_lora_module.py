# openclip_lora_module.py

import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, orig_module: nn.Linear, r: int = 8, alpha: float = 16):
        super().__init__()
        self.orig_module = orig_module
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        # Freeze original
        for param in self.orig_module.parameters():
            param.requires_grad = False

        self.lora_A = nn.Linear(orig_module.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, orig_module.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # 🔥 Patch to ensure weights are on same device as input
        self.lora_A.to(x.device)
        self.lora_B.to(x.device)
        return self.orig_module(x) + self.lora_B(self.lora_A(x)) * self.scale

    @property
    def weight(self):
        return self.orig_module.weight

    @property
    def bias(self):
        return self.orig_module.bias



