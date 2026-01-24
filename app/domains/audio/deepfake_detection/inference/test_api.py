import requests
import json
from pathlib import Path

# API 엔드포인트
BASE_URL = "http://localhost:8000"

def test_health_check():
    """
    헬스 체크 테스트
    """
    print("\n" + "="*50)
    print("Testing Health Check")
    print("="*50)
    
    response = requests.get(f"/api/v1/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_models_info():
    """
    모델 정보 조회 테스트
    """
    print("\n" + "="*50)
    print("Testing Models Info")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/models/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_single_prediction(audio_path: str):
    """
    단일 예측 테스트
    """
    print("\n" + "="*50)
    print(f"Testing Single Prediction: {audio_path}")
    print("="*50)
    
    if not Path(audio_path).exists():
        print(f"Error: File not found - {audio_path}")
        return
    
    with open(audio_path, 'rb') as f:
        files = {'file': (Path(audio_path).name, f, 'audio/flac')}
        response = requests.post(f"{BASE_URL}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"\nProbabilities:")
        print(f"  Real: {result['probabilities']['real']:.4f}")
        print(f"  Fake: {result['probabilities']['fake']:.4f}")
        print(f"\nModel Outputs:")
        print(f"  Mel-spectrogram CNN:")
        print(f"    Real: {result['model_outputs']['mel']['real']:.4f}")
        print(f"    Fake: {result['model_outputs']['mel']['fake']:.4f}")
        print(f"  LFCC CNN:")
        print(f"    Real: {result['model_outputs']['lfcc']['real']:.4f}")
        print(f"    Fake: {result['model_outputs']['lfcc']['fake']:.4f}")
        print(f"\nProcessing Time: {result['processing_time']:.3f}s")
    else:
        print(f"Error: {response.text}")


def test_batch_prediction(audio_paths: list):
    """
    배치 예측 테스트
    """
    print("\n" + "="*50)
    print(f"Testing Batch Prediction ({len(audio_paths)} files)")
    print("="*50)
    
    files = []
    for audio_path in audio_paths:
        if Path(audio_path).exists():
            files.append(
                ('files', (Path(audio_path).name, open(audio_path, 'rb'), 'audio/flac'))
            )
    
    if not files:
        print("Error: No valid files found")
        return
    
    response = requests.post(f"{BASE_URL}/predict/batch", files=files)
    
    # 파일 닫기
    for _, (_, f, _) in files:
        f.close()
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()['results']
        print(f"\nProcessed {len(results)} files:")
        
        for i, result in enumerate(results, 1):
            if 'error' in result:
                print(f"\n{i}. {result.get('filename', 'unknown')}: ERROR")
                print(f"   {result['error']}")
            else:
                print(f"\n{i}. {result.get('filename', 'unknown')}")
                print(f"   Prediction: {result['prediction'].upper()}")
                print(f"   Confidence: {result['confidence']:.4f}")
    else:
        print(f"Error: {response.text}")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Audio Deepfake Detection API Test")
    print("="*50)
    
    # 1. 헬스 체크
    test_health_check()
    
    # 2. 모델 정보
    test_models_info()
    
    # 3. 단일 예측 테스트
    # 테스트 오디오 파일 경로 (실제 경로로 변경 필요)
    test_audio = "../../data/raw/LA/ASVspoof2019_LA_dev/flac/LA_D_2776056.flac"
    test_single_prediction(test_audio)
    
    # 4. 배치 예측 테스트 (선택사항)
    # test_audio_list = [
    #     "path/to/audio1.flac",
    #     "path/to/audio2.flac",
    # ]
    # test_batch_prediction(test_audio_list)
    
    print("\n" + "="*50)
    print("Test Completed!")
    print("="*50 + "\n")