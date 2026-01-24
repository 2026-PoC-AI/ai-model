import torch
import numpy as np
from pathlib import Path

print(f"Current NumPy version: {np.__version__}")

# Xception 모델 재저장
xception_path = Path("xception/xception_best_20260116.pth")
if xception_path.exists():
    print(f"Loading {xception_path}...")
    checkpoint = torch.load(xception_path, map_location='cpu', weights_only=False)
    
    # 백업
    backup_path = xception_path.parent / f"{xception_path.stem}_backup.pth"
    torch.save(checkpoint, backup_path)
    print(f"Backup saved: {backup_path}")
    
    # 재저장
    torch.save(checkpoint, xception_path)
    print(f"Fixed: {xception_path}")

# EfficientNet 모델 재저장
efficientnet_path = Path("efficientnet/efficientnet_best_20260116.pth")
if efficientnet_path.exists():
    print(f"Loading {efficientnet_path}...")
    checkpoint = torch.load(efficientnet_path, map_location='cpu', weights_only=False)
    
    # 백업
    backup_path = efficientnet_path.parent / f"{efficientnet_path.stem}_backup.pth"
    torch.save(checkpoint, backup_path)
    print(f"Backup saved: {backup_path}")
    
    # 재저장
    torch.save(checkpoint, efficientnet_path)
    print(f"Fixed: {efficientnet_path}")

print("Done!")