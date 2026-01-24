import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path

class AudioSpectrogramDataset(Dataset):
    """
    스펙트로그램 데이터셋
    """
    def __init__(self, data_dir, split_info, transform=None):
        """
        Args:
            data_dir: 전처리된 스펙트로그램 디렉토리
            split_info: {'real': [...], 'fake': [...]} 형식의 파일 리스트
            transform: 추가 변환 (optional)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # 데이터 리스트 생성
        self.samples = []
        for label, files in split_info.items():
            label_idx = 0 if label == 'real' else 1
            for file in files:
                self.samples.append((file, label_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # 스펙트로그램 로드
        spec_path = self.data_dir / f"{Path(file_path).stem}.npy"
        
        try:
            spec = np.load(spec_path)
        except FileNotFoundError:
            print(f"Warning: File not found: {spec_path}")
            return self.__getitem__((idx + 1) % len(self))
        
        # (1, H, W) 형태로 변환
        spec = torch.from_numpy(spec).unsqueeze(0).float()
        
        if self.transform:
            spec = self.transform(spec)
        
        return spec, label


def get_data_loaders(data_dir, splits_file, batch_size=8, num_workers=0):
    """
    데이터 로더 생성
    
    Args:
        data_dir: 전처리된 스펙트로그램 디렉토리
        splits_file: train/val/test split 정보 JSON 파일
        batch_size: 배치 크기
        num_workers: 워커 수
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Split 정보 로드
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    # 데이터셋 생성
    train_dataset = AudioSpectrogramDataset(data_dir, splits['train'])
    val_dataset = AudioSpectrogramDataset(data_dir, splits['val'])
    test_dataset = AudioSpectrogramDataset(data_dir, splits['test'])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader