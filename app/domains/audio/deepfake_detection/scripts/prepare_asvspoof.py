import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

def parse_protocol_file(protocol_file):
    """
    ASVspoof 프로토콜 파일 파싱
    
    Format: SPEAKER AUDIO_FILE_NAME - SYSTEM_ID LABEL
    Example: LA_0079 LA_E_5332195 - - bonafide
    """
    data = {'real': [], 'fake': []}
    
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            audio_file = parts[1]
            label = parts[4]
            
            if label == 'bonafide':
                data['real'].append(f"{audio_file}.flac")
            elif label == 'spoof':
                data['fake'].append(f"{audio_file}.flac")
    
    return data

def create_splits(data_dir, output_file):
    """
    ASVspoof 프로토콜 파일을 splits.json으로 변환
    """
    data_dir = Path(data_dir)
    protocol_dir = data_dir / 'ASVspoof2019_LA_asv_protocols'
    
    splits = {}
    
    # Train
    print("Parsing train protocol...")
    train_protocol = protocol_dir / 'ASVspoof2019.LA.asv.train.gi.trl.txt'
    splits['train'] = parse_protocol_file(train_protocol)
    print(f"Train - Real: {len(splits['train']['real'])}, Fake: {len(splits['train']['fake'])}")
    
    # Dev (Validation)
    print("Parsing dev protocol...")
    dev_protocol = protocol_dir / 'ASVspoof2019.LA.asv.dev.gi.trl.txt'
    splits['val'] = parse_protocol_file(dev_protocol)
    print(f"Val - Real: {len(splits['val']['real'])}, Fake: {len(splits['val']['fake'])}")
    
    # Eval (Test)
    print("Parsing eval protocol...")
    eval_protocol = protocol_dir / 'ASVspoof2019.LA.asv.eval.gi.trl.txt'
    splits['test'] = parse_protocol_file(eval_protocol)
    print(f"Test - Real: {len(splits['test']['real'])}, Fake: {len(splits['test']['fake'])}")
    
    # 저장
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nSplits saved to: {output_file}")
    
    return splits

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'LA'
    output_file = Path(__file__).parent.parent.parent / 'data' / 'splits' / 'splits.json'
    
    create_splits(data_dir, output_file)