import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import pickle
import sys

# NumPy 호환성 패치
class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # numpy._core를 numpy.core로 리다이렉트
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

def safe_torch_load(path, map_location):
    """NumPy 호환성 문제 해결하는 torch.load"""
    # 파일 읽기
    with open(path, 'rb') as f:
        data = f.read()
    
    # BytesIO로 변환
    from io import BytesIO
    buffer = BytesIO(data)
    
    # 커스텀 unpickler 사용
    unpickler = NumpyCompatUnpickler(buffer)
    
    # torch.load 대신 수동으로 unpickle
    # 하지만 torch의 persistent_load 필요
    # 그래서 monkey patch 방식 사용
    original_unpickler = pickle.Unpickler
    
    def patched_unpickler(file, **kwargs):
        return NumpyCompatUnpickler(file, **kwargs)
    
    pickle.Unpickler = patched_unpickler
    
    try:
        result = torch.load(BytesIO(data), map_location=map_location, weights_only=False)
    finally:
        pickle.Unpickler = original_unpickler
    
    return result

# 상대 경로
from ..models.ensemble import EnsembleModel
from ..models.xception import XceptionNet
from ..models.efficientnet import EfficientNetB4
from ..preprocessing.face_detector import FaceDetector
from ..preprocessing.dataset import get_transforms
from .utils import aggregate_predictions

class EnsemblePredictor:
    """
    앙상블 모델 기반 딥페이크 탐지 추론 클래스
    """
    def __init__(self, xception_path, efficientnet_path, 
                    weights=None, ensemble_method='soft_voting', device='cuda'):
        """
        Args:
            xception_path: XceptionNet 체크포인트 경로
            efficientnet_path: EfficientNet-B4 체크포인트 경로
            weights: 앙상블 가중치 [w1, w2]
            ensemble_method: 앙상블 방식
            device: 사용할 디바이스
        """
        self.device = torch.device(device)
        
        # XceptionNet 로드
        print(f"Loading XceptionNet from {xception_path}")
        xception_ckpt = safe_torch_load(xception_path, self.device)
        
        xception_config = xception_ckpt.get('config', {})
        xception_model = XceptionNet(
            num_classes=xception_config.get('num_classes', 2),
            pretrained=False,
            dropout=xception_config.get('dropout', 0.5)
        )
        xception_model.load_state_dict(xception_ckpt['model_state_dict'])
        
        # EfficientNet-B4 로드
        print(f"Loading EfficientNet-B4 from {efficientnet_path}")
        efficientnet_ckpt = safe_torch_load(efficientnet_path, self.device)
        
        efficientnet_config = efficientnet_ckpt.get('config', {})
        efficientnet_model = EfficientNetB4(
            num_classes=efficientnet_config.get('num_classes', 2),
            pretrained=False,
            dropout=efficientnet_config.get('dropout', 0.5)
        )
        efficientnet_model.load_state_dict(efficientnet_ckpt['model_state_dict'])
        
        # 앙상블 모델 생성
        self.model = EnsembleModel(
            xception_model, 
            efficientnet_model,
            weights=weights,
            ensemble_method=ensemble_method
        )
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Ensemble model loaded successfully")
        print(f"  - Method: {ensemble_method}")
        print(f"  - Weights: {weights if weights else [0.5, 0.5]}")
        
        # 얼굴 검출기
        self.face_detector = FaceDetector(device=device)
        
        # 전처리 변환
        self.transform = get_transforms('val')
    
    def predict_image(self, image_path):
        """
        단일 이미지에 대한 앙상블 예측
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            result: 예측 결과 딕셔너리
            error: 에러 메시지 (에러 없으면 None)
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        
        if image is None:
            return None, f"Failed to load image: {image_path}"
        
        # 얼굴 검출 및 정렬
        faces = self.face_detector.detect_and_align(image)
        
        if len(faces) == 0:
            return None, "No face detected in the image"
        
        # 첫 번째 얼굴만 사용
        face = faces[0]
        
        # PIL Image로 변환
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        
        # 전처리 및 배치 차원 추가
        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
        
        # 앙상블 예측
        with torch.no_grad():
            ensemble_probs = self.model(face_tensor)
            fake_prob = ensemble_probs[0][1].item()
        
        result = {
            'is_fake': fake_prob > 0.5,
            'fake_probability': fake_prob,
            'real_probability': 1 - fake_prob,
            'confidence': max(fake_prob, 1 - fake_prob)
        }
        
        return result, None
    
    def predict_video(self, video_path, sample_rate=5, aggregation='mean'):
        """
        비디오에 대한 앙상블 예측
        
        Args:
            video_path: 비디오 파일 경로
            sample_rate: 프레임 샘플링 비율
            aggregation: 프레임별 예측 집계 방식
        
        Returns:
            result: 예측 결과 딕셔너리
            error: 에러 메시지 (에러 없으면 None)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None, f"Failed to open video: {video_path}"
        
        frame_predictions = []
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # 샘플링
            if frame_count % sample_rate != 0:
                continue
            
            # 얼굴 검출 및 정렬
            faces = self.face_detector.detect_and_align(frame)
            
            if len(faces) == 0:
                continue
            
            # 첫 번째 얼굴만 사용
            face = faces[0]
            
            # PIL Image로 변환
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            
            # 전처리 및 배치 차원 추가
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            # 앙상블 예측
            with torch.no_grad():
                ensemble_probs = self.model(face_tensor)
                fake_prob = ensemble_probs[0][1].item()
            
            frame_predictions.append(fake_prob)
            processed_count += 1
        
        cap.release()
        
        if len(frame_predictions) == 0:
            return None, "No faces detected in video frames"
        
        # 프레임별 예측 집계
        aggregated_prob = aggregate_predictions(frame_predictions, method=aggregation)
        
        result = {
            'is_fake': aggregated_prob > 0.5,
            'fake_probability': aggregated_prob,
            'real_probability': 1 - aggregated_prob,
            'confidence': max(aggregated_prob, 1 - aggregated_prob),
            'total_frames': frame_count,
            'processed_frames': processed_count,
            'frame_predictions': frame_predictions
        }
        
        return result, None