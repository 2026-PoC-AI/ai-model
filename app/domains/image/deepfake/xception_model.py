# app/domains/image/deepfake/xception_model.py
import torch
import torch.nn as nn
import timm

class XceptionNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=1 
        )

    def forward(self, x):
        return self.backbone(x)