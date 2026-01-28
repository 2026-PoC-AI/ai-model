from .base import DeepfakeDetector

class XceptionNet(nn.Module):
    def __init__(self, pretrained=False):
        super(XceptionNet, self).__init__()
        # backbone만 가져오고 fc는 제거
        base_model = timm.create_model('xception', pretrained=pretrained, num_classes=0)
        self.model = base_model
        
        # 학습 시 사용한 classifier 구조 재현
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),           # classifier.0
            nn.Linear(2048, 512),      # classifier.1
            nn.ReLU(),                 # classifier.2
            nn.Dropout(0.5),           # classifier.3
            nn.Linear(512, 2)          # classifier.4
        )
    
    def forward(self, x):
        features = self.model(x)
        return self.classifier(features)