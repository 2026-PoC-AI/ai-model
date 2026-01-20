# 5프레임 시퀀스로 C3D를 학습

video_dataset_code = '''import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from collections import defaultdict
from torchvision import transforms

class VideoDataset(Dataset):
    """
    비디오 시퀀스 데이터셋
    각 비디오의 프레임들을 시퀀스로 묶어서 반환
    """
    def __init__(self, data_dir, split='train', sequence_length=5, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.sequence_length = sequence_length
        self.transform = transform
        
        # 비디오별로 프레임 그룹화
        self.videos = self._group_frames()
        
    def _group_frames(self):
        """비디오 ID별로 프레임들을 그룹화"""
        videos = {'real': defaultdict(list), 'fake': defaultdict(list)}
        
        for label in ['real', 'fake']:
            frame_dir = os.path.join(self.data_dir, self.split, label)
            
            for filename in sorted(os.listdir(frame_dir)):
                if not filename.endswith('.jpg'):
                    continue
                
                # 파일명에서 비디오 ID 추출: id0_id16_0000_0.jpg -> id0_id16_0000
                parts = filename.rsplit('_', 1)
                video_id = parts[0]
                frame_num = int(parts[1].split('.')[0])
                
                videos[label][video_id].append({
                    'path': os.path.join(frame_dir, filename),
                    'frame_num': frame_num
                })
        
        # 프레임 번호순으로 정렬
        for label in ['real', 'fake']:
            for video_id in videos[label]:
                videos[label][video_id].sort(key=lambda x: x['frame_num'])
        
        # (video_id, label) 리스트로 변환
        video_list = []
        for label in ['real', 'fake']:
            for video_id, frames in videos[label].items():
                if len(frames) >= self.sequence_length:
                    video_list.append({
                        'video_id': video_id,
                        'label': 1 if label == 'fake' else 0,
                        'frames': frames[:self.sequence_length]  # 정확히 sequence_length만큼만
                    })
        
        return video_list
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_info = self.videos[idx]
        frames = video_info['frames']
        label = video_info['label']
        
        # 프레임 로드 및 변환
        frame_tensors = []
        for frame_info in frames:
            img = Image.open(frame_info['path']).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frame_tensors.append(img)
        
        # (T, C, H, W) -> (C, T, H, W)로 변환
        frames_tensor = torch.stack(frame_tensors)  # (T, C, H, W)
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
        
        return frames_tensor, label

def get_video_transforms(split='train', img_size=112):
    """비디오용 Transform"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
'''

# 파일 저장
with open('preprocessing/video_dataset.py', 'w') as f:
    f.write(video_dataset_code)

print("Video dataset created!")