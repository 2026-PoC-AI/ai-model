# 데이터 분할 (train/val/test) - splits 생성
from pathlib import Path
import json
import traceback
from sklearn.model_selection import train_test_split

def create_splits(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    fake와 real 폴더의 비디오들을 train/val/test로 분할
    """
    data_path = Path(data_dir)
    print(f"Looking for data in: {data_path.absolute()}")
    
    fake_dir = data_path / 'fake'
    real_dir = data_path / 'real'
    
    print(f"Fake dir exists: {fake_dir.exists()}")
    print(f"Real dir exists: {real_dir.exists()}")
    
    if not fake_dir.exists() or not real_dir.exists():
        raise FileNotFoundError(f"Data directories not found at {data_path}")
    
    # 비디오 목록 가져오기
    fake_videos = sorted([d.name for d in fake_dir.iterdir() if d.is_dir()])
    real_videos = sorted([d.name for d in real_dir.iterdir() if d.is_dir()])
    
    print(f"Total fake videos: {len(fake_videos)}")
    print(f"Total real videos: {len(real_videos)}")
    
    # fake 분할
    fake_train, fake_temp = train_test_split(
        fake_videos, 
        test_size=(val_ratio + test_ratio), 
        random_state=random_state
    )
    fake_val, fake_test = train_test_split(
        fake_temp, 
        test_size=test_ratio/(val_ratio + test_ratio), 
        random_state=random_state
    )
    
    # real 분할
    real_train, real_temp = train_test_split(
        real_videos, 
        test_size=(val_ratio + test_ratio), 
        random_state=random_state
    )
    real_val, real_test = train_test_split(
        real_temp, 
        test_size=test_ratio/(val_ratio + test_ratio), 
        random_state=random_state
    )
    
    splits = {
        'train': {
            'fake': fake_train,
            'real': real_train
        },
        'val': {
            'fake': fake_val,
            'real': real_val
        },
        'test': {
            'fake': fake_test,
            'real': real_test
        }
    }
    
    return splits

def save_splits(splits, output_dir):
    """
    분할 정보를 JSON으로 저장
    """
    output_path = Path(output_dir)
    print(f"Creating output directory: {output_path.absolute()}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / 'splits.json'
    print(f"Saving splits to: {output_file.absolute()}")
    
    # JSON 파일로 저장
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    # 통계 출력
    print("\n=== Split Statistics ===")
    for split_name in ['train', 'val', 'test']:
        fake_count = len(splits[split_name]['fake'])
        real_count = len(splits[split_name]['real'])
        total = fake_count + real_count
        print(f"{split_name.upper():5s}: {total:4d} videos (fake: {fake_count:3d}, real: {real_count:3d})")
    
    print(f"\nSplits saved successfully!")

if __name__ == "__main__":
    try:
        # 현재 스크립트 위치 기준으로 경로 계산
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent  # deepfake_detection
        
        print(f"Script directory: {script_dir}")
        print(f"Project root: {project_root}")
        
        # 데이터는 deepfake_detection이 아니라 video 레벨에 있음
        data_dir = project_root.parent / 'data' / 'processed' / 'frames_16'
        output_dir = project_root.parent / 'data' / 'splits'
        
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {output_dir}")
        
        # splits 생성
        print("\n=== Creating data splits ===")
        splits = create_splits(
            str(data_dir),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42
        )
        
        # 저장
        save_splits(splits, str(output_dir))
        
        print("\n=== Done! ===")
        
    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"Error: {e}")
        traceback.print_exc()