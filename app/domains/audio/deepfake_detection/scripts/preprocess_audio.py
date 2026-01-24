import sys
from pathlib import Path

current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

import numpy as np
import json
from tqdm import tqdm
import argparse

from preprocessing.audio_preprocessor import AudioPreprocessor

def preprocess_dataset(raw_audio_dirs, splits_file, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        n_fft=512,
        hop_length=256,
        n_mels=64,
        duration=4.0,
        spec_height=64,
        spec_width=256
    )
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    all_files = set()
    for split_name, split_data in splits.items():
        all_files.update(split_data['real'])
        all_files.update(split_data['fake'])
    
    print(f"Total unique files to process: {len(all_files)}")
    
    success_count = 0
    error_count = 0
    
    for filename in tqdm(all_files, desc="Processing audio files"):
        audio_path = None
        for raw_dir in raw_audio_dirs:
            potential_path = Path(raw_dir) / filename
            if potential_path.exists():
                audio_path = potential_path
                break
        
        if audio_path is None:
            error_count += 1
            continue
        
        try:
            spec = preprocessor.preprocess(str(audio_path))
            output_path = output_dir / f"{audio_path.stem}.npy"
            np.save(output_path, spec)
            success_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            error_count += 1
    
    print(f"\n{'='*50}")
    print(f"Success: {success_count}/{len(all_files)}")
    print(f"Errors: {error_count}/{len(all_files)}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dirs', type=str, nargs='+',
                        default=[
                            '../../data/raw/LA/ASVspoof2019_LA_train/flac',
                            '../../data/raw/LA/ASVspoof2019_LA_dev/flac',
                            '../../data/raw/LA/ASVspoof2019_LA_eval/flac'
                        ])
    parser.add_argument('--splits_file', type=str,
                        default='../../data/splits/splits.json')
    parser.add_argument('--output_dir', type=str,
                        default='../../data/processed/spectrograms')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    raw_dirs = [(script_dir / d).resolve() for d in args.raw_dirs]
    splits_file = (script_dir / args.splits_file).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    
    preprocess_dataset(raw_dirs, splits_file, output_dir)