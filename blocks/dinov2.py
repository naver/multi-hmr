# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
from torch import nn

class Dinov2Backbone(nn.Module):
    def __init__(self, name='dinov2_vitb14', *args, **kwargs):
        super().__init__()
        self.name = name
        torch.hub.set_dir('/home/glxadmin/.cache/torch/hub')
        self.encoder = torch.hub.load('/home/glxadmin/.cache/torch/hub/facebookresearch_dinov2_main', self.name, pretrained=False, source='local')
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.encoder.embed_dim

    def forward(self, x):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        assert len(x.shape) == 4
        y = self.encoder.get_intermediate_layers(x)[0] # ViT-L+896x896: [bs,4096,1024] - [bs,nb_patches,emb]
        return y

