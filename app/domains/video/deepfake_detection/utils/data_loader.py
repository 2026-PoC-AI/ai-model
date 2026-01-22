import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import json
import torchvision.transforms as transforms

class VideoFrameDataset(Dataset):
    """
    16프레임 비디오 데이터셋
    """
    def __init__(self, data_dir, splits_file, split='train', transform=None):
        """
        Args:
            data_dir: frames_16 디렉토리 경로
            splits_file: splits.json 파일 경로
            split: 'train', 'val', 'test' 중 하나
            transform: 이미지 변환
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # splits 로드
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        # 현재 split의 비디오 목록
        self.samples = []
        
        # fake 비디오 (label=0)
        for video_name in splits[split]['fake']:
            video_path = self.data_dir / 'fake' / video_name
            if video_path.exists():
                self.samples.append((video_path, 0))
        
        # real 비디오 (label=1)
        for video_name in splits[split]['real']:
            video_path = self.data_dir / 'real' / video_name
            if video_path.exists():
                self.samples.append((video_path, 1))
        
        print(f"{split} dataset: {len(self.samples)} videos")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]
        
        # 프레임 로드 (16개)
        frames = sorted(video_dir.glob('*.jpg'))[:16]
        
        frame_tensors = []
        for frame_path in frames:
            img = Image.open(frame_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frame_tensors.append(img)
        
        # (16, C, H, W) 형태로 스택
        frames_tensor = torch.stack(frame_tensors)
        
        return frames_tensor, label

def get_transforms(image_size=224, is_training=True):
    """
    데이터 변환 정의 - 강화된 버전
    """
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),              # 224 → 256 (crop 위해)
            transforms.RandomCrop((image_size, image_size)),  # Random Crop 추가
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,  # 0.2 → 0.3
                contrast=0.3,    # 0.2 → 0.3
                saturation=0.3,  # 0.2 → 0.3
                hue=0.1          # 추가
            ),
            transforms.RandomRotation(degrees=10),      # 추가
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def get_data_loaders(data_dir, splits_file, batch_size=16, num_workers=4, image_size=224):
    """
    Train, Val, Test DataLoader 생성
    """
    # Transforms
    train_transform = get_transforms(image_size=image_size, is_training=True)
    test_transform = get_transforms(image_size=image_size, is_training=False)
    
    # Datasets
    train_dataset = VideoFrameDataset(data_dir, splits_file, split='train', transform=train_transform)
    val_dataset = VideoFrameDataset(data_dir, splits_file, split='val', transform=test_transform)
    test_dataset = VideoFrameDataset(data_dir, splits_file, split='test', transform=test_transform)
    
    # DataLoaders
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