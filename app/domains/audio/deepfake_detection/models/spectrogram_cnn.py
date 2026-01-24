import torch
import torch.nn as nn

class LightweightAudioCNN(nn.Module):
    """
    경량 CNN 기반 오디오 딥페이크 탐지 모델
    ASVspoof 스타일의 간단한 아키텍처
    """
    def __init__(self, num_classes=2, dropout=0.3):
        super(LightweightAudioCNN, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 1, height, width) - Mel-spectrogram
        Returns:
            output: (batch_size, num_classes)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(model_name='lightweight_cnn', num_classes=2, **kwargs):
    """
    CNN 모델 생성 헬퍼 함수
    """
    if model_name == 'lightweight_cnn':
        model = LightweightAudioCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model