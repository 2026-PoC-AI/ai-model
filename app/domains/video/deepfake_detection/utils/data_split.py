import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_splits(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    data_dir: processed_16frames_1000 디렉토리 경로
    fake와 real 폴더 안의 비디오들을 train/val/test로 분할
    """
    fake_dir = Path(data_dir) / 'fake'
    real_dir = Path(data_dir) / 'real'
    
    # 비디오 목록 가져오기
    fake_videos = sorted([d.name for d in fake_dir.iterdir() if d.is_dir()])
    real_videos = sorted([d.name for d in real_dir.iterdir() if d.is_dir()])
    
    # fake 분할
    fake_train, fake_temp = train_test_split(fake_videos, test_size=(val_ratio + test_ratio), random_state=random_state)
    fake_val, fake_test = train_test_split(fake_temp, test_size=test_ratio/(val_ratio + test_ratio), random_state=random_state)
    
    # real 분할
    real_train, real_temp = train_test_split(real_videos, test_size=(val_ratio + test_ratio), random_state=random_state)
    real_val, real_test = train_test_split(real_temp, test_size=test_ratio/(val_ratio + test_ratio), random_state=random_state)
    
    splits = {
        'train': {'fake': fake_train, 'real': real_train},
        'val': {'fake': fake_val, 'real': real_val},
        'test': {'fake': fake_test, 'real': real_test}
    }
    
    return splits

def save_splits(splits, output_dir='./data/splits'):
    """분할 정보를 JSON으로 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/splits.json', 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Train: {len(splits['train']['fake'])} fake, {len(splits['train']['real'])} real")
    print(f"Val: {len(splits['val']['fake'])} fake, {len(splits['val']['real'])} real")
    print(f"Test: {len(splits['test']['fake'])} fake, {len(splits['test']['real'])} real")