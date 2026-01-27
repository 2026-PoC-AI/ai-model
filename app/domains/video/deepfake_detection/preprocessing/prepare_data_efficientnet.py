import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from pathlib import Path
from tqdm import tqdm

class FaceExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        self.mtcnn = MTCNN(
            image_size=224,
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=False
        )
        print(f"MTCNN initialized on {device}")  # ✓ 제거
    
    def extract_faces_from_video(self, video_path, num_frames=10, frame_interval=30):
        """
        비디오에서 얼굴을 추출
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < num_frames * frame_interval:
            frame_interval = max(1, total_frames // num_frames)
        
        faces = []
        frame_indices = list(range(0, total_frames, frame_interval))[:num_frames]
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                face = self.mtcnn(frame_rgb)
                if face is not None:
                    face_np = face.permute(1, 2, 0).cpu().numpy()
                    face_np = ((face_np + 1) * 127.5).astype(np.uint8)
                    faces.append(face_np)
            except:
                continue
        
        cap.release()
        return faces

def preprocess_dataset(
    raw_dir,
    processed_dir,
    num_real_videos=500,
    num_fake_videos=500,
    frames_per_video=10,
    frame_interval=30
):
    """
    비디오 데이터셋을 전처리하여 이미지로 변환
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = FaceExtractor(device=device)
    
    # Real 비디오 수집
    print(f"\n{'='*60}")
    print("Collecting Real videos...")
    print(f"{'='*60}")
    
    real_videos = []
    
    celeb_real_dir = os.path.join(raw_dir, 'Celeb-real')
    if os.path.exists(celeb_real_dir):
        videos = [os.path.join(celeb_real_dir, f) for f in os.listdir(celeb_real_dir) 
                    if f.endswith(('.mp4', '.avi'))]
        real_videos.extend(videos)
        print(f"Found {len(videos)} videos in Celeb-real")
    
    youtube_real_dir = os.path.join(raw_dir, 'YouTube-real')
    if os.path.exists(youtube_real_dir):
        videos = [os.path.join(youtube_real_dir, f) for f in os.listdir(youtube_real_dir) 
                    if f.endswith(('.mp4', '.avi'))]
        real_videos.extend(videos)
        print(f"Found {len(videos)} videos in YouTube-real")
    
    real_videos = real_videos[:num_real_videos]
    print(f"Selected {len(real_videos)} real videos")
    
    # Fake 비디오 수집
    print(f"\n{'='*60}")
    print("Collecting Fake videos...")
    print(f"{'='*60}")
    
    fake_dir = os.path.join(raw_dir, 'Celeb-synthesis')
    fake_videos = []
    if os.path.exists(fake_dir):
        videos = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                    if f.endswith(('.mp4', '.avi'))]
        fake_videos.extend(videos)
        print(f"Found {len(videos)} videos in Celeb-synthesis")
    
    fake_videos = fake_videos[:num_fake_videos]
    print(f"Selected {len(fake_videos)} fake videos")
    
    # 디렉토리 생성
    train_real_dir = os.path.join(processed_dir, 'train', 'real')
    train_fake_dir = os.path.join(processed_dir, 'train', 'fake')
    val_real_dir = os.path.join(processed_dir, 'val', 'real')
    val_fake_dir = os.path.join(processed_dir, 'val', 'fake')
    
    for d in [train_real_dir, train_fake_dir, val_real_dir, val_fake_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Real 비디오 처리
    print(f"\n{'='*60}")
    print("Processing Real videos...")
    print(f"{'='*60}")
    
    real_count = {'train': 0, 'val': 0}
    failed_real = 0
    
    for idx, video_path in enumerate(tqdm(real_videos, desc="Real videos")):
        try:
            faces = extractor.extract_faces_from_video(
                video_path, 
                num_frames=frames_per_video,
                frame_interval=frame_interval
            )
            
            if len(faces) == 0:
                failed_real += 1
                continue
            
            split = 'train' if idx < len(real_videos) * 0.8 else 'val'
            output_dir = train_real_dir if split == 'train' else val_real_dir
            
            for face_idx, face in enumerate(faces):
                video_name = Path(video_path).stem
                filename = f"{video_name}_frame_{face_idx:03d}.jpg"
                cv2.imwrite(
                    os.path.join(output_dir, filename),
                    cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                )
                real_count[split] += 1
        except Exception as e:
            failed_real += 1
            print(f"Error processing {Path(video_path).name}: {e}")
            continue
    
    print(f"Failed real videos: {failed_real}")
    
    # Fake 비디오 처리
    print(f"\n{'='*60}")
    print("Processing Fake videos...")
    print(f"{'='*60}")
    
    fake_count = {'train': 0, 'val': 0}
    failed_fake = 0
    
    for idx, video_path in enumerate(tqdm(fake_videos, desc="Fake videos")):
        try:
            faces = extractor.extract_faces_from_video(
                video_path,
                num_frames=frames_per_video,
                frame_interval=frame_interval
            )
            
            if len(faces) == 0:
                failed_fake += 1
                continue
            
            split = 'train' if idx < len(fake_videos) * 0.8 else 'val'
            output_dir = train_fake_dir if split == 'train' else val_fake_dir
            
            for face_idx, face in enumerate(faces):
                video_name = Path(video_path).stem
                filename = f"{video_name}_frame_{face_idx:03d}.jpg"
                cv2.imwrite(
                    os.path.join(output_dir, filename),
                    cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                )
                fake_count[split] += 1
        except Exception as e:
            failed_fake += 1
            print(f"Error processing {Path(video_path).name}: {e}")
            continue
    
    print(f"Failed fake videos: {failed_fake}")
    
    # 최종 결과
    print(f"\n{'='*60}")
    print("Preprocessing Complete!")
    print(f"{'='*60}")
    print(f"Train: {real_count['train']} real + {fake_count['train']} fake = {real_count['train'] + fake_count['train']}")
    print(f"Val: {real_count['val']} real + {fake_count['val']} fake = {real_count['val'] + fake_count['val']}")
    print(f"Total: {sum(real_count.values()) + sum(fake_count.values())} images")
    print(f"Failed: {failed_real} real + {failed_fake} fake = {failed_real + failed_fake}")

if __name__ == '__main__':
    preprocess_dataset(
        raw_dir='../data/raw/celeb-df', 
        processed_dir='../data/processed_efficientnet', 
        num_real_videos=500,
        num_fake_videos=500,
        frames_per_video=10,
        frame_interval=30
    )