# app/domains/image/deepfake/xception_model.py
import torch
import torch.nn as nn
import timm

class XceptionNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=0,  
            global_pool="avg"
        )
        self.classifier = nn.Linear(2048, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)