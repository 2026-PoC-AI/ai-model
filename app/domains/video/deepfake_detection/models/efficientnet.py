import torch
import torch.nn as nn
from torchvision import models

class EfficientNetB4(nn.Module):
    """
    EfficientNet-B4 기반 딥페이크 탐지 모델
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super(EfficientNetB4, self).__init__()
        
        # EfficientNet-B4 백본 로드
        if pretrained:
            weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b4(weights=weights)
        else:
            self.backbone = models.efficientnet_b4(weights=None)
        
        # 마지막 분류 레이어 교체
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """백본 레이어 고정"""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """백본 레이어 학습 가능하게 변경"""
        for param in self.backbone.features.parameters():
            param.requires_grad = True