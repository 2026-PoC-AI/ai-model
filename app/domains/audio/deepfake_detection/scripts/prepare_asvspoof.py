import json
from pathlib import Path
import argparse

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

def create_splits(raw_dir, output_file):
    """
    ASVspoof 프로토콜 파일을 splits.json으로 변환
    """
    raw_dir = Path(raw_dir)
    protocol_dir = raw_dir / 'ASVspoof2019_LA_cm_protocols'
    
    if not protocol_dir.exists():
        print(f"Error: Protocol directory not found: {protocol_dir}")
        return None
    
    splits = {}
    
    # Train
    print("Parsing train protocol...")
    train_protocol = protocol_dir / 'ASVspoof2019.LA.cm.train.trn.txt'
    if train_protocol.exists():
        splits['train'] = parse_protocol_file(train_protocol)
        print(f"Train - Real: {len(splits['train']['real'])}, Fake: {len(splits['train']['fake'])}")
    else:
        print(f"Warning: Train protocol not found: {train_protocol}")
        splits['train'] = {'real': [], 'fake': []}
    
    # Dev (Validation)
    print("Parsing dev protocol...")
    dev_protocol = protocol_dir / 'ASVspoof2019.LA.cm.dev.trl.txt'
    if dev_protocol.exists():
        splits['val'] = parse_protocol_file(dev_protocol)
        print(f"Val - Real: {len(splits['val']['real'])}, Fake: {len(splits['val']['fake'])}")
    else:
        print(f"Warning: Dev protocol not found: {dev_protocol}")
        splits['val'] = {'real': [], 'fake': []}
    
    # Eval (Test)
    print("Parsing eval protocol...")
    eval_protocol = protocol_dir / 'ASVspoof2019.LA.cm.eval.trl.txt'
    if eval_protocol.exists():
        splits['test'] = parse_protocol_file(eval_protocol)
        print(f"Test - Real: {len(splits['test']['real'])}, Fake: {len(splits['test']['fake'])}")
    else:
        print(f"Warning: Eval protocol not found: {eval_protocol}")
        splits['test'] = {'real': [], 'fake': []}
    
    # 저장
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nSplits saved to: {output_file}")
    
    return splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare ASVspoof dataset splits')
    parser.add_argument('--raw_dir', type=str, 
                       default='../../data/raw/LA',
                       help='Raw data directory containing protocol files')
    parser.add_argument('--output_file', type=str,
                       default='../../data/splits/splits.json',
                       help='Output splits JSON file')
    
    args = parser.parse_args()
    
    # 상대 경로를 절대 경로로 변환
    script_dir = Path(__file__).parent
    raw_dir = (script_dir / args.raw_dir).resolve()
    output_file = (script_dir / args.output_file).resolve()
    
    create_splits(raw_dir, output_file)