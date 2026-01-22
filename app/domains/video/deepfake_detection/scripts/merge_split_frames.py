# 압축 된 프레임 폴더에서 'copy' 폴더의 프레임들을 원본 폴더로 병합하는 스크립트
from pathlib import Path
import shutil

def merge_copy_folders(base_dir):
    """
    원본 폴더와 'copy' 폴더의 프레임들을 하나로 병합
    """
    base_path = Path(base_dir)
    
    for video_dir in sorted(base_path.iterdir()):
        if not video_dir.is_dir() or 'copy' not in video_dir.name:
            continue
        
        # copy 폴더와 원본 폴더 찾기
        copy_folder = video_dir
        original_name = video_dir.name.replace(' copy', '')
        original_folder = base_path / original_name
        
        if not original_folder.exists():
            print(f"Warning: Original folder not found for {copy_folder.name}")
            continue
        
        # copy 폴더의 프레임들을 원본 폴더로 이동
        frames_moved = 0
        for frame in sorted(copy_folder.glob('*.jpg')):
            dest = original_folder / frame.name
            if not dest.exists():
                shutil.move(str(frame), str(dest))
                frames_moved += 1
        
        # copy 폴더 삭제
        if frames_moved > 0:
            copy_folder.rmdir()
            print(f"Merged {frames_moved} frames: {original_name}")
        
        # 최종 프레임 개수 확인
        total_frames = len(list(original_folder.glob('*.jpg')))
        if total_frames != 16:
            print(f"  WARNING: {original_name} has {total_frames} frames (expected 16)")

if __name__ == "__main__":
    # fake 폴더 처리
    fake_dir = './app/domains/video/data/processed/frames_16/fake'
    print("Processing fake videos...")
    merge_copy_folders(fake_dir)
    
    # real 폴더도 있다면 처리
    real_dir = './app/domains/video/data/processed/frames_16/real'
    if Path(real_dir).exists():
        print("\nProcessing real videos...")
        merge_copy_folders(real_dir)
    
    print("\nMerge complete!")