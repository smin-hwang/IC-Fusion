"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]

@register
class RTDETR(nn.Module):
    __inject__ = ['backbone_rgb', 'backbone_ir', 'encoder', 'decoder', ]

    def __init__(
        self, 
        backbone_rgb: nn.Module,
        backbone_ir : nn.Module,
        encoder,
        decoder,
        multi_scale=None
    ):
        super().__init__()
        self.backbone_rgb = backbone_rgb
        self.backbone_ir = backbone_ir
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x_rgb, x_ir, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x_rgb = F.interpolate(x_rgb, size=[sz, sz])
            x_ir = F.interpolate(x_ir, size=[sz, sz])


        x_rgb = self.backbone_rgb(x_rgb)
        x_ir = self.backbone_ir(x_ir)      
        x = self.encoder(x_rgb, x_ir)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self