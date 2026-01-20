import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

# 상대 경로로 수정
from ..models.xception import XceptionNet
from ..preprocessing.face_detector import FaceDetector
from ..preprocessing.dataset import get_transforms
from .utils import aggregate_predictions


class XceptionPredictor:
    """
    Xception 기반 딥페이크 탐지 추론 클래스
    """
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device)
        
        # 모델 로드
        print(f"Loading Xception model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 모델 초기화
        model_config = checkpoint.get('config', {})
        self.model = XceptionNet(
            num_classes=model_config.get('num_classes', 2),
            pretrained=False,
            dropout=model_config.get('dropout', 0.5)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Xception model loaded successfully (epoch {checkpoint['epoch']})")
        
        # 얼굴 검출기
        self.face_detector = FaceDetector(device=device)
        
        # 전처리 변환
        self.transform = get_transforms('val')
    
    def predict_image(self, image_path):
        """
        단일 이미지에 대한 딥페이크 예측
        
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
        
        # 예측
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probs = F.softmax(outputs, dim=1)
            fake_prob = probs[0][1].item()
        
        result = {
            'is_fake': fake_prob > 0.5,
            'fake_probability': fake_prob,
            'real_probability': 1 - fake_prob,
            'confidence': max(fake_prob, 1 - fake_prob)
        }
        
        return result, None
    
    def predict_video(self, video_path, sample_rate=5, aggregation='mean'):
        """
        비디오에 대한 딥페이크 예측
        
        Args:
            video_path: 비디오 파일 경로
            sample_rate: 프레임 샘플링 비율 (매 N 프레임마다 분석)
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
            
            # 예측
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probs = F.softmax(outputs, dim=1)
                fake_prob = probs[0][1].item()
            
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