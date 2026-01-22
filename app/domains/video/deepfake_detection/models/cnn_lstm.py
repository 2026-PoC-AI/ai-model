import torch
import torch.nn as nn
import torchvision.models as models

class CNN_LSTM(nn.Module):
    """
    CNN(ResNet) + LSTM 기반 Deepfake Detection 모델
    """
    def __init__(self, num_classes=2, hidden_size=512, num_layers=2, dropout=0.5, pretrained=True):
        super(CNN_LSTM, self).__init__()
        
        # CNN Feature Extractor (ResNet50)
        resnet = models.resnet50(pretrained=pretrained)
        # 마지막 FC layer 제거
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_output_size = 2048  # ResNet50의 출력 차원
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_frames, channels, height, width)
        Returns:
            output: (batch_size, num_classes)
        """
        batch_size, num_frames, c, h, w = x.shape
        
        # CNN feature extraction for each frame
        # (batch_size * num_frames, channels, height, width)
        x = x.view(batch_size * num_frames, c, h, w)
        
        # (batch_size * num_frames, cnn_output_size, 1, 1)
        cnn_features = self.cnn(x)
        
        # (batch_size * num_frames, cnn_output_size)
        cnn_features = cnn_features.view(batch_size * num_frames, -1)
        
        # (batch_size, num_frames, cnn_output_size)
        cnn_features = cnn_features.view(batch_size, num_frames, -1)
        
        # LSTM
        # lstm_out: (batch_size, num_frames, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        
        # 마지막 타임스텝의 출력 사용
        # (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        # Classification
        # (batch_size, num_classes)
        output = self.fc(last_output)
        
        return output

class ResNetLSTM(nn.Module):
    """
    경량화된 ResNet18 + LSTM 모델
    """
    def __init__(self, num_classes=2, hidden_size=256, num_layers=1, dropout=0.5, pretrained=True):  # 0.3 → 0.5
        super(ResNetLSTM, self).__init__()
        
        # CNN Feature Extractor (ResNet18)
        resnet = models.resnet18(pretrained=pretrained)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_output_size = 512
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout),  # 0.3 → 0.5
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        
        # CNN
        x = x.view(batch_size * num_frames, c, h, w)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size * num_frames, -1)
        cnn_features = cnn_features.view(batch_size, num_frames, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_features)
        last_output = lstm_out[:, -1, :]
        
        # Classification
        output = self.fc(last_output)
        
        return output

def get_model(model_name='cnn_lstm', num_classes=2, pretrained=True, **kwargs):
    """
    모델 생성 헬퍼 함수
    """
    if model_name == 'cnn_lstm':
        model = CNN_LSTM(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet_lstm':
        model = ResNetLSTM(num_classes=num_classes, pretrained=pretrained, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model