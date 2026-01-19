# app/domains/image/deepfake/xception_model.py
import torch
import torch.nn as nn
import timm


class XceptionNet(nn.Module):
    """
    Xception backbone + binary classifier
    Input : (B, 3, 299, 299)
    Output: (B, 1) logits
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            "xception",
            pretrained=pretrained,  
            num_classes=0,
            global_pool="avg",
        )

        self.classifier = nn.Linear(2048, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.classifier(feat)
