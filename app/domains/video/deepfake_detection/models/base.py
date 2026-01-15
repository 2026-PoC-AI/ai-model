import timm
import torch.nn as nn

class DeepfakeDetector(nn.Module):
    """
    딥페이크 탐지를 위한 베이스 모델 클래스
    다양한 백본을 사용할 수 있도록 설계
    """
    def __init__(self, backbone_name='xception', num_classes=2, pretrained=True, dropout=0.5):
        super(DeepfakeDetector, self).__init__()
        
        self.backbone_name = backbone_name
        
        # timm을 통한 백본 로드
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # feature extractor 모드
            global_pool='avg'
        )
        
        # 백본의 출력 차원
        in_features = self.backbone.num_features
        
        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def extract_features(self, x):
        """
        특징 벡터만 추출 (앙상블에서 사용)
        """
        return self.backbone(x)
    
    def get_num_features(self):
        """
        백본의 특징 차원 반환
        """
        return self.backbone.num_features