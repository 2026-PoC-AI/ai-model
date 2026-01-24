import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import os
import json
import sys

class FFTPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def extract_fft_spectrum(self, image):
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # FFT 적용
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # 주파수 스펙트럼 계산
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # 정규화
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        
        return magnitude_spectrum.astype(np.uint8)
    
    def process_image(self, image_path):
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지 로드 실패: {image_path}")
        
        # 리사이즈
        resized = cv2.resize(image, self.target_size)
        
        # RGB 각 채널에 대해 FFT 적용
        fft_channels = []
        for i in range(3):
            channel = resized[:, :, i]
            fft_spectrum = self.extract_fft_spectrum(channel)
            fft_channels.append(fft_spectrum)
        
        # 3채널로 합성
        fft_image = np.stack(fft_channels, axis=-1)
        
        return fft_image


def load_split_data(data_root, split='train'):
    """
    splits.json 파일에서 train/val/test 분할 정보 로드
    """
    splits_json_path = data_root / 'splits' / 'splits.json'
    processed_frames_path = data_root / 'processed' / 'frames_16'
    
    data_list = []
    
    print(f"\n{split} 데이터 로드 중...")
    print(f"  Splits JSON: {splits_json_path}")
    print(f"  Frames 경로: {processed_frames_path}")
    
    # JSON 파일 로드
    if not splits_json_path.exists():
        print(f"  오류: splits.json 파일이 없습니다: {splits_json_path}")
        return data_list
    
    with open(splits_json_path, 'r') as f:
        splits_data = json.load(f)
    
    if split not in splits_data:
        print(f"  오류: '{split}' 키가 splits.json에 없습니다")
        return data_list
    
    split_info = splits_data[split]
    
    # fake 데이터
    if 'fake' in split_info:
        fake_ids = split_info['fake']
        print(f"  Fake 비디오 수: {len(fake_ids)}개")
        
        for fake_id in fake_ids:
            video_frames_path = processed_frames_path / 'fake' / fake_id
            
            if video_frames_path.exists():
                frame_files = sorted([f for f in video_frames_path.glob('*.jpg')])
                for frame_file in frame_files:
                    data_list.append({
                        'image_path': str(frame_file),
                        'label': 1,  # fake
                        'video_id': fake_id,
                        'frame': frame_file.name
                    })
            else:
                print(f"    경고: 프레임 폴더 없음 - {video_frames_path}")
    
    # real 데이터
    if 'real' in split_info:
        real_ids = split_info['real']
        print(f"  Real 비디오 수: {len(real_ids)}개")
        
        for real_id in real_ids:
            video_frames_path = processed_frames_path / 'real' / real_id
            
            if video_frames_path.exists():
                frame_files = sorted([f for f in video_frames_path.glob('*.jpg')])
                for frame_file in frame_files:
                    data_list.append({
                        'image_path': str(frame_file),
                        'label': 0,  # real
                        'video_id': real_id,
                        'frame': frame_file.name
                    })
            else:
                print(f"    경고: 프레임 폴더 없음 - {video_frames_path}")
    
    return data_list
    
    # real 데이터
    real_split_path = splits_path / 'real'
    if real_split_path.exists():
        real_files = [f for f in real_split_path.iterdir() if f.is_file()]
        print(f"  Real split 파일 수: {len(real_files)}")
        
        for real_file in real_files:
            real_id = real_file.stem
            video_frames_path = processed_frames_path / 'real' / real_id
            
            if video_frames_path.exists():
                frame_files = sorted([f for f in video_frames_path.glob('*.jpg')])
                for frame_file in frame_files:
                    data_list.append({
                        'image_path': str(frame_file),
                        'label': 0,  # real
                        'video_id': real_id,
                        'frame': frame_file.name
                    })
    
    return data_list


def process_and_save_fft_dataset(data_list, output_dir, split_name, 
                                    preprocessor, batch_size=500, resume_from=0):
    """
    FFT 전처리를 수행하고 배치 단위로 저장 (재시작 지원)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 진행 상황 저장 파일
    progress_file = Path(output_dir) / f'{split_name}_progress.json'
    
    # 이전 진행 상황 로드
    if progress_file.exists() and resume_from == 0:
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            if not progress.get('completed', False):
                resume_from = progress.get('last_batch', 0) + 1
                print(f"이전 진행 상황 발견: 배치 {resume_from}부터 재시작")
    
    batch_data = []
    batch_labels = []
    batch_video_ids = []
    batch_num = resume_from
    
    # 이미 처리된 항목 건너뛰기
    start_idx = resume_from * batch_size
    
    pbar = tqdm(data_list[start_idx:], 
                desc=f"Processing {split_name}",
                initial=start_idx, total=len(data_list))
    
    for idx, item in enumerate(pbar):
        try:
            # FFT 스펙트럼 추출
            fft_image = preprocessor.process_image(item['image_path'])
            
            batch_data.append(fft_image)
            batch_labels.append(item['label'])
            batch_video_ids.append(item['video_id'])
            
            # 배치 크기에 도달하면 저장
            if len(batch_data) >= batch_size:
                save_path = Path(output_dir) / f'{split_name}_batch_{batch_num}.npz'
                np.savez_compressed(
                    save_path,
                    data=np.array(batch_data),
                    labels=np.array(batch_labels),
                    video_ids=batch_video_ids
                )
                
                file_size = save_path.stat().st_size / (1024**2)
                pbar.write(f"배치 {batch_num} 저장: {len(batch_data)}개, {file_size:.1f}MB")
                
                # 진행 상황 저장
                with open(progress_file, 'w') as f:
                    json.dump({
                        'last_batch': batch_num, 
                        'total_processed': start_idx + idx + 1
                    }, f)
                
                batch_data = []
                batch_labels = []
                batch_video_ids = []
                batch_num += 1
                
        except Exception as e:
            pbar.write(f"처리 실패 - {item['image_path']}: {e}")
            continue
    
    # 남은 데이터 저장
    if batch_data:
        save_path = Path(output_dir) / f'{split_name}_batch_{batch_num}.npz'
        np.savez_compressed(
            save_path,
            data=np.array(batch_data),
            labels=np.array(batch_labels),
            video_ids=batch_video_ids
        )
        file_size = save_path.stat().st_size / (1024**2)
        print(f"최종 배치 저장: {len(batch_data)}개, {file_size:.1f}MB")
    
    # 완료 표시
    with open(progress_file, 'w') as f:
        json.dump({'completed': True, 'total_batches': batch_num + 1}, f)
    
    print(f"{split_name} FFT 전처리 완료!")


if __name__ == '__main__':
    # 경로 설정 (현재 파일 기준)
    current_file = Path(__file__)
    # preprocessing -> deepfake_detection -> video -> domains -> app
    DATA_ROOT = current_file.parent.parent.parent / 'data'
    OUTPUT_PATH = DATA_ROOT / 'fft_processed'
    
    print("=" * 60)
    print("FFT 전처리 시작")
    print("=" * 60)
    print(f"데이터 경로: {DATA_ROOT.absolute()}")
    print(f"출력 경로: {OUTPUT_PATH.absolute()}")
    
    # 경로 확인
    if not DATA_ROOT.exists():
        print(f"\n오류: 데이터 경로가 존재하지 않습니다: {DATA_ROOT.absolute()}")
        print("\n사용 가능한 경로:")
        parent = current_file.parent.parent.parent
        for item in parent.iterdir():
            print(f"  {item.name}")
        sys.exit(1)
    
    # 전처리 객체 생성
    preprocessor = FFTPreprocessor(target_size=(224, 224))
    
    # 데이터 로드
    print("\n데이터 로드 중...")
    train_data = load_split_data(DATA_ROOT, 'train')
    val_data = load_split_data(DATA_ROOT, 'val')
    test_data = load_split_data(DATA_ROOT, 'test')
    
    print(f"\n{'='*60}")
    print("데이터 로드 완료")
    print(f"{'='*60}")
    print(f"Train: {len(train_data)}개 프레임")
    if train_data:
        print(f"  - Fake: {sum(1 for d in train_data if d['label'] == 1)}개")
        print(f"  - Real: {sum(1 for d in train_data if d['label'] == 0)}개")
    
    print(f"\nVal: {len(val_data)}개 프레임")
    if val_data:
        print(f"  - Fake: {sum(1 for d in val_data if d['label'] == 1)}개")
        print(f"  - Real: {sum(1 for d in val_data if d['label'] == 0)}개")
    
    print(f"\nTest: {len(test_data)}개 프레임")
    if test_data:
        print(f"  - Fake: {sum(1 for d in test_data if d['label'] == 1)}개")
        print(f"  - Real: {sum(1 for d in test_data if d['label'] == 0)}개")
    
    if len(train_data) == 0:
        print("\n오류: 데이터를 찾을 수 없습니다.")
        print(f"확인할 경로:")
        print(f"  - {DATA_ROOT / 'raw' / 'splits'}")
        print(f"  - {DATA_ROOT / 'processed' / 'frames_16'}")
        sys.exit(1)
    
    # FFT 전처리 실행
    print("\n" + "=" * 60)
    print("Train 데이터 처리 시작")
    print("=" * 60)
    process_and_save_fft_dataset(train_data, OUTPUT_PATH, 'train', 
                                    preprocessor, batch_size=500)
    
    print("\n" + "=" * 60)
    print("Val 데이터 처리 시작")
    print("=" * 60)
    process_and_save_fft_dataset(val_data, OUTPUT_PATH, 'val', 
                                    preprocessor, batch_size=500)
    
    print("\n" + "=" * 60)
    print("Test 데이터 처리 시작")
    print("=" * 60)
    process_and_save_fft_dataset(test_data, OUTPUT_PATH, 'test', 
                                    preprocessor, batch_size=500)
    
    print("\n" + "=" * 60)
    print("모든 FFT 전처리 완료!")
    print("=" * 60)
    
    # 결과 요약
    output_files = list(Path(OUTPUT_PATH).glob('*.npz'))
    total_size = sum(f.stat().st_size for f in output_files) / (1024**3)
    print(f"\n생성된 파일: {len(output_files)}개")
    print(f"총 크기: {total_size:.2f} GB")
    print(f"저장 위치: {OUTPUT_PATH.absolute()}")