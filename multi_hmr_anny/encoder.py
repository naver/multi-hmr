# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
from torch import nn
import numpy as np
from utils import unpatch
import torch.nn.functional as F
import math

class Encoder(nn.Module):
    """
    Image encoding with a ViT followed by patch-level detection and camera instrinsics estimation.
    """
    def __init__(self, name='dinov2_vits14', pretrained=False, *args, **kwargs):
        super().__init__()
        self.name = name

        # ViT-Backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', self.name, pretrained=pretrained)
        self.patch_size = self.backbone.patch_size
        self.embed_dim = self.backbone.embed_dim

        # Patch-level detection
        self.mlp_det = nn.Sequential(*[nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim,1)])

        # FOV
        self.mlp_fov_unique = nn.Sequential(*[nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim,1)]) # between 0 and np.pi
        fov_max = torch.tensor([math.pi])
        self.register_buffer("fov_max", fov_max)

    def forward(self, x):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        assert len(x.shape) == 4
        img_size = x.shape[-1]

        # Encode RGB image using a ViT
        y, cls = self.backbone.get_intermediate_layers(x, return_class_token=True)[0] # [bs,np,emb]
        y = unpatch(y, patch_size=1, c=y.shape[2], img_size=int(np.sqrt(y.shape[1]))) # [bs,emb,sqrt(np),sqrt(np)]
        y = y.permute(0,2,3,1) # [bs,sqrt(np),sqrt(np),emb]

        # Field-of-view prediction
        fov = self.fov_max * torch.sigmoid(self.mlp_fov_unique(cls)) # range [0,fov_max]
        focal_length = (img_size / 2) / torch.tan(fov / 2)

        # Camera intrinsics 3x3 matrix
        K = torch.eye(3).float().to(cls.device).reshape(1,3,3).repeat(cls.shape[0], 1, 1)
        K[:,[0,1],[0,1]] = focal_length[:,[0]] # FOV-x = FOV-y
        K[:,[0,1],[-1,-1]] = img_size / 2.

        # Patch-level detection
        scores_logits = self.mlp_det(y)[...,0]
        scores = torch.sigmoid(scores_logits)

        return {
            'scores_logits': scores_logits, # [bs,sqrt(np),sqrt(np)]
            'scores': scores, # [bs,sqrt(np),sqrt(np)]
            'K': K, # [bs,3,3]
            'fov': fov, # [bs,1]
            'feat': y, # [bs,sqrt(np),sqrt(np),emb]
                }