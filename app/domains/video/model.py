import cv2
import numpy as np
import torch
from typing import List, Tuple
from PIL import Image

# 상대 경로로 수정
from .deepfake_detection.inference.ensemble_predictor import EnsemblePredictor
from .deepfake_detection.preprocessing.dataset import get_transforms

class VideoModel:
    """영상 딥페이크 탐지 모델"""
    
    def __init__(self, xception_path: str, efficientnet_path: str, 
                    ensemble_method: str = 'soft_voting', weights: List[float] = None,
                    device: str = 'cuda'):
        """
        Args:
            xception_path: XceptionNet 체크포인트 경로
            efficientnet_path: EfficientNet-B4 체크포인트 경로
            ensemble_method: 앙상블 방식 ('soft_voting', 'hard_voting')
            weights: 앙상블 가중치 [w1, w2]
            device: 사용할 디바이스
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.ensemble_predictor = EnsemblePredictor(
            xception_path=xception_path,
            efficientnet_path=efficientnet_path,
            weights=weights,
            ensemble_method=ensemble_method,
            device=str(self.device)
        )
        self.transform = get_transforms('val')
        
    def analyze_frame(self, frame: np.ndarray) -> Tuple[bool, float, str]:
        """
        단일 프레임 분석
        
        Args:
            frame: OpenCV로 읽은 프레임 (numpy array)
            
        Returns:
            (is_deepfake, confidence, anomaly_type)
        """
        # BGR to RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # PIL Image로 변환
        pil_image = Image.fromarray(frame_rgb)
        
        # 전처리 적용
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # 예측 수행
        with torch.no_grad():
            outputs = self.ensemble_predictor.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
            fake_prob = probabilities[1].item()
            real_prob = probabilities[0].item()
            
            is_deepfake = fake_prob > 0.5
            confidence = fake_prob if is_deepfake else real_prob
        
        # 이상 유형 결정
        if is_deepfake:
            if fake_prob > 0.9:
                anomaly_type = "high_confidence_fake"
            elif fake_prob > 0.7:
                anomaly_type = "face_boundary_blur"
            else:
                anomaly_type = "frame_inconsistency"
        else:
            anomaly_type = "normal"
        
        return is_deepfake, confidence, anomaly_type
    
    def analyze_video(self, video_path: str, frame_interval: int = 30) -> dict:
        """
        비디오 전체 분석
        
        Args:
            video_path: 비디오 파일 경로
            frame_interval: 분석할 프레임 간격
            
        Returns:
            분석 결과 딕셔너리
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        results = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 지정된 간격마다 프레임 분석
            if frame_idx % frame_interval == 0:
                is_deepfake, confidence, anomaly_type = self.analyze_frame(frame)
                results.append({
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps,
                    'is_deepfake': is_deepfake,
                    'confidence': confidence,
                    'anomaly_type': anomaly_type
                })
            
            frame_idx += 1
        
        cap.release()
        
        # 전체 비디오 판단
        fake_count = sum(1 for r in results if r['is_deepfake'])
        fake_ratio = fake_count / len(results) if results else 0
        avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
        
        video_result = {
            'video_path': video_path,
            'total_frames': total_frames,
            'analyzed_frames': len(results),
            'fake_frame_ratio': fake_ratio,
            'average_confidence': avg_confidence,
            'is_deepfake_video': fake_ratio > 0.5,
            'frame_results': results
        }
        
        return video_result

def load_video_model(xception_path: str, efficientnet_path: str, 
                    ensemble_method: str = 'soft_voting',
                    weights: List[float] = None) -> VideoModel:
    """영상 분석 모델 로드"""
    model = VideoModel(
        xception_path=xception_path,
        efficientnet_path=efficientnet_path,
        ensemble_method=ensemble_method,
        weights=weights
    )
    return model