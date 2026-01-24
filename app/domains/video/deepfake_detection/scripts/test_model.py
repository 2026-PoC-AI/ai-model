# scripts/test_model.py
from pathlib import Path
import sys
import torch
sys.path.append(str(Path(__file__).parent.parent))

from models.cnn_lstm import get_model

if __name__ == "__main__":
    print("Testing CNN_LSTM model...")
    
    # 모델 생성
    model = get_model('cnn_lstm', num_classes=2, hidden_size=512, num_layers=2, dropout=0.5)
    
    # 더미 입력 생성 (batch_size=2, num_frames=16, C=3, H=224, W=224)
    dummy_input = torch.randn(2, 16, 3, 224, 224)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n" + "="*50)
    print("Testing ResNetLSTM model...")
    
    # 경량 모델 테스트
    model_light = get_model('resnet_lstm', num_classes=2, hidden_size=256, num_layers=1, dropout=0.3)
    output_light = model_light(dummy_input)
    
    print(f"Output shape: {output_light.shape}")
    
    total_params_light = sum(p.numel() for p in model_light.parameters())
    trainable_params_light = sum(p.numel() for p in model_light.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params_light:,}")
    print(f"Trainable parameters: {trainable_params_light:,}")
    
    print("\nModel test successful!")