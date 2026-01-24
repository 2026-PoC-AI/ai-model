import torch
import pickle
import numpy as np
import io

def safe_torch_load(path, map_location):
    """
    NumPy 2.0 환경에서 구버전 가중치 파일을 읽을 때 발생하는 
    Pickle/Code 인자 타입 에러를 해결하는 로더
    """
    # // 1. 파일 내용을 바이너리로 읽음
    with open(path, 'rb') as f:
        data = f.read()

    # // 2. NumPy 2.0에서 변경된 내부 경로 리다이렉트를 위한 커스텀 Unpickler
    class CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # // NumPy 2.0의 _core 구조를 구버전 core 구조로 매핑
            if module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core')
            elif module == 'numpy' and name in ['float', 'int', 'bool']:
                # // NumPy 2.0에서 사라진 별칭들을 기본 파이썬 타입으로 복구
                return getattr(__builtins__, name)
            return super().find_class(module, name)

    # // 3. torch.load 내부에서 사용할 pickle 객체를 몽키 패칭
    def custom_load(file, **kwargs):
        return CompatUnpickler(file, **kwargs).load()

    try:
        # // 먼저 가장 안전한 weights_only=True 시도
        return torch.load(io.BytesIO(data), map_location=map_location, weights_only=True)
    except Exception:
        # // 실패 시, 커스텀 unpickler를 강제로 주입하여 로드
        try:
            # // pickle.load 대신 커스텀 unpickler를 사용하도록 유도
            # // pickle_module 인자를 통해 직접 전달
            return torch.load(
                io.BytesIO(data), 
                map_location=map_location, 
                weights_only=False,
                pickle_module=type('CustomPickle', (), {'Unpickler': CompatUnpickler, 'load': custom_load})
            )
        except Exception as e:
            # // 최후의 수단: 에러가 'code' 관련이면 CPU로 로드 후 복사
            if "argument 'code' must be code" in str(e):
                return torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
            raise e

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