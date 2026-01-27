# check_weights.py
# python check_weights.py 로 실행하여 가중치 파일 점검
# check_weights.py
import torch

print("="*60)
print("XceptionNet Checkpoint - Classifier Keys")
print("="*60)
xception_checkpoint = torch.load(
    "app/domains/video/deepfake_detection/weights/xception/xception_best_20260116.pth",
    map_location='cpu',
    weights_only=False
)

print("\nKeys containing 'classifier' or 'fc':")
state_dict = xception_checkpoint['model_state_dict']
for key in state_dict.keys():
    if 'classifier' in key or 'fc' in key:
        print(f"  {key}: {state_dict[key].shape}")

print("\n" + "="*60)
print("EfficientNet Checkpoint - Classifier Keys")
print("="*60)
efficientnet_checkpoint = torch.load(
    "app/domains/video/deepfake_detection/weights/efficientnet/efficientnet_best.pth",
    map_location='cpu',
    weights_only=False
)

print("\nKeys containing 'classifier' or 'fc':")
state_dict = efficientnet_checkpoint['model_state_dict']
for key in state_dict.keys():
    if 'classifier' in key or 'fc' in key:
        print(f"  {key}: {state_dict[key].shape}")

print("\n" + "="*60)
print("CNN-LSTM Checkpoint - All Keys (last 20)")
print("="*60)
cnn_lstm_checkpoint = torch.load(
    "app/domains/video/deepfake_detection/weights/cnn-lstm/improved/best_model_latest.pth",
    map_location='cpu',
    weights_only=False
)

state_dict = cnn_lstm_checkpoint['model_state_dict']
all_keys = list(state_dict.keys())
print("\nLast 20 keys:")
for key in all_keys[-20:]:
    print(f"  {key}: {state_dict[key].shape}")

print("\n" + "="*60)
print("Current Models - Classifier Keys")
print("="*60)
from app.domains.video.model import XceptionNet, EfficientNetB4, CNNLSTMModel

xception_model = XceptionNet(pretrained=False)
print("\nXceptionNet - keys with 'fc' or 'classifier':")
for key in xception_model.state_dict().keys():
    if 'fc' in key or 'classifier' in key:
        print(f"  {key}: {xception_model.state_dict()[key].shape}")

efficientnet_model = EfficientNetB4(pretrained=False)
print("\nEfficientNet - keys with 'fc' or 'classifier':")
for key in efficientnet_model.state_dict().keys():
    if 'fc' in key or 'classifier' in key:
        print(f"  {key}: {efficientnet_model.state_dict()[key].shape}")

cnn_lstm_model = CNNLSTMModel()
print("\nCNN-LSTM - last 20 keys:")
all_keys = list(cnn_lstm_model.state_dict().keys())
for key in all_keys[-20:]:
    print(f"  {key}: {cnn_lstm_model.state_dict()[key].shape}")