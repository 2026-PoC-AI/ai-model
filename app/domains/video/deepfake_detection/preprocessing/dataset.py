import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os
from PIL import Image

class DeepfakeDataset(Dataset):
    """
    딥페이크 탐지를 위한 데이터셋 클래스
    """
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # 실제 이미지와 가짜 이미지 경로 수집
        self.real_images = self._load_image_paths(
            os.path.join(data_dir, split, 'real')
        )
        self.fake_images = self._load_image_paths(
            os.path.join(data_dir, split, 'fake')
        )
        
        # 레이블 생성: 0=real, 1=fake
        self.samples = [(img, 0) for img in self.real_images] + \
                      [(img, 1) for img in self.fake_images]
        
        print(f"{split} dataset: {len(self.real_images)} real, {len(self.fake_images)} fake")
    
    def _load_image_paths(self, directory):
        """
        디렉토리에서 이미지 경로 로드
        """
        paths = []
        if not os.path.exists(directory):
            print(f"Warning: {directory} does not exist")
            return paths
            
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(root, file))
        return paths
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 이미지 로드
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # PIL Image로 변환
        image = Image.fromarray(image)
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(split='train'):
    """
    데이터 증강 및 전처리 변환
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.RandomCrop((299, 299)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])