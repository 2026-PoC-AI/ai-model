"""
간단한 전처리 - 독립 실행형
"""
import cv2
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

# facenet-pytorch 임포트
from facenet_pytorch import MTCNN
import torch

class FaceDetector:
    """얼굴 검출 및 정렬"""
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(
            keep_all=False,
            device=self.device,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            post_process=False
        )
        print(f"Face detector initialized on {self.device}")
    
    def detect_and_align(self, image):
        """얼굴 검출 및 정렬"""
        # BGR to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        
        if boxes is None:
            return None, None
        
        # 가장 큰 얼굴
        max_idx = np.argmax(probs)
        box = boxes[max_idx]
        
        # 얼굴 영역 크롭
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        face = image[y1:y2, x1:x2]
        
        # 299x299로 리사이즈
        if face.size > 0:
            face = cv2.resize(face, (299, 299))
            return face, box
        
        return None, None

def quick_preprocess(raw_dir, processed_dir, max_videos=10):
    """
    빠른 전처리
    """
    print("="*50)
    print("Quick Preprocessing - Celeb-DF")
    print("="*50)
    
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    
    print(f"\nRaw directory: {raw_dir}")
    print(f"Processed directory: {processed_dir}")
    
    if not raw_dir.exists():
        print(f"\nError: {raw_dir} does not exist!")
        return
    
    # 얼굴 검출기 초기화
    print("\nInitializing face detector...")
    face_detector = FaceDetector()
    
    # 카테고리별 처리
    categories = {
        'real': ['Celeb-real', 'YouTube-real'],
        'fake': ['Celeb-synthesis']
    }
    
    for label, dir_names in categories.items():
        print(f"\n{'='*50}")
        print(f"Processing {label.upper()} videos")
        print(f"{'='*50}")
        
        processed_count = 0
        
        for dir_name in dir_names:
            video_dir = raw_dir / dir_name
            
            if not video_dir.exists():
                print(f"Warning: {video_dir} does not exist")
                continue
            
            videos = list(video_dir.glob('*.mp4'))[:max_videos]
            
            print(f"\nProcessing {len(videos)} videos from {dir_name}...")
            
            for video_path in tqdm(videos):
                cap = cv2.VideoCapture(str(video_path))
                frame_count = 0
                saved_count = 0
                
                # 각 영상에서 5개 프레임만
                while cap.isOpened() and saved_count < 5:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % 30 == 0:
                        aligned_face, _ = face_detector.detect_and_align(frame)
                        
                        if aligned_face is not None:
                            # train/val 8:2 분할
                            split = 'train' if processed_count < max_videos * 0.8 else 'val'
                            
                            output_dir = processed_dir / split / label
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            output_file = output_dir / f"{video_path.stem}_{saved_count}.jpg"
                            cv2.imwrite(
                                str(output_file),
                                cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
                            )
                            saved_count += 1
                    
                    frame_count += 1
                
                cap.release()
                processed_count += 1
    
    # 결과 출력
    print("\n" + "="*50)
    print("Preprocessing Complete!")
    print("="*50)
    
    for split in ['train', 'val']:
        print(f"\n{split.upper()}:")
        for label in ['real', 'fake']:
            img_dir = processed_dir / split / label
            if img_dir.exists():
                count = len(list(img_dir.glob('*.jpg')))
                print(f"  {label}: {count} images")

if __name__ == '__main__':
    # 절대 경로로 직접 지정
    quick_preprocess(
        raw_dir='app/domains/video/data/raw/celeb-df',
        processed_dir='app/domains/video/data/processed',
        max_videos=10
    )