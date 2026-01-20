import os
from video_model import load_video_model

def main():
    # 모델 경로 설정
    xception_path = "weights/xception/xception_best_20260116.pth"
    efficientnet_path = "weights/efficientnet/efficientnet_best_20260116.pth"
    
    # 테스트 비디오 경로
    test_video = "../data/raw/Celeb-real/videos/id0_id1_0000.mp4"
    
    print("="*60)
    print("Video Deepfake Detection - Model Verification")
    print("="*60)
    
    # 모델 로드
    print("\n[1] Loading ensemble model...")
    model = load_video_model(
        xception_path=xception_path,
        efficientnet_path=efficientnet_path,
        ensemble_method='soft_voting',
        weights=[0.5, 0.5]
    )
    print("✓ Model loaded successfully!")
    
    # 비디오 분석
    print(f"\n[2] Analyzing video: {test_video}")
    result = model.analyze_video(test_video, frame_interval=10)
    
    # 결과 출력
    print("\n[3] Analysis Results:")
    print(f"  Total Frames: {result['total_frames']}")
    print(f"  Analyzed Frames: {result['analyzed_frames']}")
    print(f"  Fake Frame Ratio: {result['fake_frame_ratio']:.2%}")
    print(f"  Average Confidence: {result['average_confidence']:.4f}")
    print(f"  Video Classification: {'FAKE' if result['is_deepfake_video'] else 'REAL'}")
    
    # 프레임별 샘플 결과
    print("\n[4] Frame-level Analysis (first 5 frames):")
    for frame_result in result['frame_results'][:5]:
        print(f"  Frame {frame_result['frame_idx']} ({frame_result['timestamp']:.2f}s):")
        print(f"    Prediction: {'FAKE' if frame_result['is_deepfake'] else 'REAL'}")
        print(f"    Confidence: {frame_result['confidence']:.4f}")
        print(f"    Anomaly: {frame_result['anomaly_type']}")
    
    print("\n" + "="*60)
    print("Verification completed!")
    print("="*60)

if __name__ == "__main__":
    main()