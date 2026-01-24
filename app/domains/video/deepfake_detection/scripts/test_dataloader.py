from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import get_data_loaders

if __name__ == "__main__":
    # 경로 설정
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root.parent / 'data' / 'processed' / 'frames_16'
    splits_file = project_root.parent / 'data' / 'splits' / 'splits.json'
    
    print(f"Data directory: {data_dir}")
    print(f"Splits file: {splits_file}")
    
    # DataLoader 생성
    train_loader, val_loader, test_loader = get_data_loaders(
        str(data_dir),
        str(splits_file),
        batch_size=4,
        num_workers=0,  # Windows에서는 0으로 설정
        image_size=224
    )
    
    # 첫 번째 배치 테스트
    print("\nTesting first batch...")
    frames, labels = next(iter(train_loader))
    print(f"Frames shape: {frames.shape}")  # (batch_size, 16, 3, 224, 224)
    print(f"Labels shape: {labels.shape}")  # (batch_size,)
    print(f"Labels: {labels}")
    
    print("\nDataLoader test successful!")