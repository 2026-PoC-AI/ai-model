import cv2
import os
from pathlib import Path
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split
import argparse
import random

from face_detector import FaceDetector

class CelebDFPreprocessor:
    """
    Celeb-DF 데이터셋 전처리
    영상에서 얼굴 추출 및 train/val 분할
    """
    def __init__(self, raw_dir='video/data/raw/celeb-df', 
                    processed_dir='video/data/processed',
                    frame_interval=30,
                    num_real_videos=300,
                    num_fake_videos=300):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.frame_interval = frame_interval
        self.num_real_videos = num_real_videos
        self.num_fake_videos = num_fake_videos
        
        # 얼굴 검출기 초기화
        print("Initializing face detector...")
        self.face_detector = FaceDetector()
        
    def extract_faces_from_video(self, video_path, output_dir, max_faces=10):
        """
        영상에서 얼굴 프레임 추출
        
        Args:
            video_path: 영상 파일 경로
            output_dir: 저장 디렉토리
            max_faces: 영상당 최대 저장 개수
        
        Returns:
            저장된 얼굴 개수
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return 0
        
        video_name = video_path.stem
        output_path = Path(output_dir) / video_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened() and saved_count < max_faces:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 일정 간격으로 샘플링
            if frame_count % self.frame_interval == 0:
                # 얼굴 검출 및 정렬
                aligned_face, box = self.face_detector.detect_and_align(frame)
                
                if aligned_face is not None:
                    # 저장
                    output_file = output_path / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(
                        str(output_file),
                        cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
                    )
                    saved_count += 1
            
            frame_count += 1
        
        cap.release()
        return saved_count
    
    def process_video_directory(self, video_dir, temp_dir, label, max_videos=None):
        """
        디렉토리의 모든 영상 처리
        
        Args:
            video_dir: 영상 디렉토리
            temp_dir: 임시 저장 디렉토리
            label: 'real' or 'fake'
            max_videos: 최대 처리 비디오 수
        
        Returns:
            처리된 영상 수
        """
        video_files = list(Path(video_dir).glob('*.mp4')) + \
                     list(Path(video_dir).glob('*.avi'))
        
        if len(video_files) == 0:
            print(f"No videos found in {video_dir}")
            return 0
        
        # 랜덤 샘플링 (재현성을 위해 seed 고정)
        if max_videos and len(video_files) > max_videos:
            random.seed(42)
            video_files = random.sample(video_files, max_videos)
        
        print(f"\nProcessing {len(video_files)} {label} videos from {video_dir.name}...")
        
        processed_count = 0
        total_faces = 0
        
        for video_path in tqdm(video_files, desc=f"Processing {label}"):
            try:
                saved = self.extract_faces_from_video(
                    video_path,
                    temp_dir / label,
                    max_faces=10
                )
                
                if saved > 0:
                    processed_count += 1
                    total_faces += saved
                    
            except Exception as e:
                print(f"\nError processing {video_path.name}: {e}")
        
        print(f"Processed {processed_count}/{len(video_files)} videos")
        print(f"Extracted {total_faces} faces")
        
        return processed_count
    
    def split_dataset(self, temp_dir, train_ratio=0.8):
        """
        train/val 분할
        
        Args:
            temp_dir: 임시 디렉토리
            train_ratio: train 비율
        """
        print("\nSplitting dataset into train/val...")
        
        for label in ['real', 'fake']:
            label_dir = temp_dir / label
            
            if not label_dir.exists():
                print(f"Warning: {label_dir} does not exist")
                continue
            
            # 각 비디오 폴더 수집
            video_folders = [d for d in label_dir.iterdir() if d.is_dir()]
            
            if len(video_folders) == 0:
                print(f"Warning: No video folders in {label_dir}")
                continue
            
            # train/val 분할
            train_folders, val_folders = train_test_split(
                video_folders,
                train_size=train_ratio,
                random_state=42
            )
            
            print(f"{label}: {len(train_folders)} train, {len(val_folders)} val videos")
            
            # train 폴더로 복사
            train_dir = self.processed_dir / 'train' / label
            train_dir.mkdir(parents=True, exist_ok=True)
            
            for folder in tqdm(train_folders, desc=f"Copying train {label}"):
                for img_file in folder.glob('*.jpg'):
                    shutil.copy(img_file, train_dir / f"{folder.name}_{img_file.name}")
            
            # val 폴더로 복사
            val_dir = self.processed_dir / 'val' / label
            val_dir.mkdir(parents=True, exist_ok=True)
            
            for folder in tqdm(val_folders, desc=f"Copying val {label}"):
                for img_file in folder.glob('*.jpg'):
                    shutil.copy(img_file, val_dir / f"{folder.name}_{img_file.name}")
    
    def run(self, clean_temp=True):
        """
        전체 전처리 실행
        
        Args:
            clean_temp: 임시 디렉토리 삭제 여부
        """
        print("="*50)
        print("Celeb-DF Dataset Preprocessing")
        print("="*50)
        print(f"Target: {self.num_real_videos} real + {self.num_fake_videos} fake videos")
        print(f"Frame interval: {self.frame_interval}")
        print(f"Max faces per video: 10")
        print("="*50)
        
        # 임시 디렉토리
        temp_dir = Path('video/data/temp')
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Real 영상 처리 (Celeb-real + YouTube-real 합쳐서 num_real_videos개)
        celeb_real_dir = self.raw_dir / 'Celeb-real'
        youtube_real_dir = self.raw_dir / 'YouTube-real'
        
        # Real 비디오 수집
        real_videos = []
        if celeb_real_dir.exists():
            real_videos.extend(list(celeb_real_dir.glob('*.mp4')) + list(celeb_real_dir.glob('*.avi')))
        if youtube_real_dir.exists():
            real_videos.extend(list(youtube_real_dir.glob('*.mp4')) + list(youtube_real_dir.glob('*.avi')))
        
        # 랜덤 샘플링
        if len(real_videos) > self.num_real_videos:
            random.seed(42)
            real_videos = random.sample(real_videos, self.num_real_videos)
        
        print(f"\nProcessing {len(real_videos)} real videos...")
        processed_count = 0
        total_faces = 0
        
        for video_path in tqdm(real_videos, desc="Processing real"):
            try:
                saved = self.extract_faces_from_video(
                    video_path,
                    temp_dir / 'real',
                    max_faces=10
                )
                
                if saved > 0:
                    processed_count += 1
                    total_faces += saved
                    
            except Exception as e:
                print(f"\nError processing {video_path.name}: {e}")
        
        print(f"Processed {processed_count}/{len(real_videos)} real videos")
        print(f"Extracted {total_faces} real faces")
        
        # 2. Fake 영상 처리 (Celeb-synthesis에서 num_fake_videos개)
        celeb_synthesis_dir = self.raw_dir / 'Celeb-synthesis'
        
        if celeb_synthesis_dir.exists():
            self.process_video_directory(
                celeb_synthesis_dir, 
                temp_dir, 
                'fake',
                max_videos=self.num_fake_videos
            )
        
        # 3. Train/Val 분할
        self.split_dataset(temp_dir, train_ratio=0.8)
        
        # 4. 임시 디렉토리 삭제
        if clean_temp:
            print("\nCleaning up temporary directory...")
            shutil.rmtree(temp_dir)
        
        # 5. 결과 출력
        print("\n" + "="*50)
        print("Preprocessing Complete!")
        print("="*50)
        
        for split in ['train', 'val']:
            for label in ['real', 'fake']:
                img_dir = self.processed_dir / split / label
                if img_dir.exists():
                    count = len(list(img_dir.glob('*.jpg')))
                    print(f"{split}/{label}: {count} images")
        
        print("\nData directory structure:")
        print(f"{self.processed_dir}/")
        print("├── train/")
        print("│   ├── real/")
        print("│   └── fake/")
        print("└── val/")
        print("    ├── real/")
        print("    └── fake/")

def main():
    parser = argparse.ArgumentParser(description='Preprocess Celeb-DF dataset')
    parser.add_argument(
        '--raw_dir',
        type=str,
        default='video/data/raw/celeb-df',
        help='Raw data directory'
    )
    parser.add_argument(
        '--processed_dir',
        type=str,
        default='video/data/processed',
        help='Processed data directory'
    )
    parser.add_argument(
        '--frame_interval',
        type=int,
        default=30,
        help='Frame sampling interval'
    )
    parser.add_argument(
        '--num_real_videos',
        type=int,
        default=300,
        help='Number of real videos to process'
    )
    parser.add_argument(
        '--num_fake_videos',
        type=int,
        default=300,
        help='Number of fake videos to process'
    )
    parser.add_argument(
        '--keep_temp',
        action='store_true',
        help='Keep temporary directory'
    )
    
    args = parser.parse_args()
    
    preprocessor = CelebDFPreprocessor(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        frame_interval=args.frame_interval,
        num_real_videos=args.num_real_videos,
        num_fake_videos=args.num_fake_videos
    )
    
    preprocessor.run(clean_temp=not args.keep_temp)

if __name__ == '__main__':
    main()