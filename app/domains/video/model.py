# app/domains/video/model.py
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from PIL import Image
import timm

class XceptionNet(nn.Module):
    def __init__(self, pretrained=False):
        super(XceptionNet, self).__init__()
        self.model = timm.create_model('xception', pretrained=pretrained, num_classes=2)
    
    def forward(self, x):
        return self.model(x)

class EfficientNetB4(nn.Module):
    def __init__(self, pretrained=False):
        super(EfficientNetB4, self).__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=2)
    
    def forward(self, x):
        return self.model(x)

class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256, num_layers=2, dropout=0.5):
        super(CNNLSTMModel, self).__init__()
        
        # ResNet18 백본
        resnet = timm.create_model('resnet18', pretrained=False)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # CNN 특징 추출
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.features(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(features)
        
        # Attention
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 분류
        output = self.classifier(context)
        
        return output, attention_weights.squeeze(-1)

class VideoModel:
    """영상 딥페이크 탐지 3-모델 앙상블"""
    
    def __init__(
        self, 
        xception_path: str, 
        efficientnet_path: str,
        cnn_lstm_path: str,
        device: str = 'cuda'
    ):
        """
        Args:
            xception_path: XceptionNet 체크포인트 경로
            efficientnet_path: EfficientNet-B4 체크포인트 경로
            cnn_lstm_path: CNN-LSTM 체크포인트 경로
            device: 사용할 디바이스
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        self.xception = XceptionNet(pretrained=False).to(self.device)
        self.efficientnet = EfficientNetB4(pretrained=False).to(self.device)
        self.cnn_lstm = CNNLSTMModel().to(self.device)
        
        # 가중치 로드
        print(f"Loading XceptionNet from {xception_path}")
        xception_checkpoint = torch.load(
            xception_path, 
            map_location=self.device,
            weights_only=False  
        )

        # state_dict 키 확인 및 처리
        if 'model_state_dict' in xception_checkpoint:
            state_dict = xception_checkpoint['model_state_dict']
        else:
            state_dict = xception_checkpoint

        # 키 매핑 확인
        try:
            self.xception.load_state_dict(state_dict)
        except RuntimeError as e:
            if "Missing key(s) in state_dict" in str(e):
                print(f"Warning: State dict keys mismatch, loading with strict=False")
                self.xception.load_state_dict(state_dict, strict=False)
            else:
                raise

        print(f"Loading EfficientNet-B4 from {efficientnet_path}")
        efficientnet_checkpoint = torch.load(
            efficientnet_path, 
            map_location=self.device,
            weights_only=False  
        )

        if 'model_state_dict' in efficientnet_checkpoint:
            state_dict = efficientnet_checkpoint['model_state_dict']
        else:
            state_dict = efficientnet_checkpoint

        try:
            self.efficientnet.load_state_dict(state_dict)
        except RuntimeError as e:
            if "Missing key(s) in state_dict" in str(e):
                print(f"Warning: State dict keys mismatch, loading with strict=False")
                self.efficientnet.load_state_dict(state_dict, strict=False)
            else:
                raise

        print(f"Loading XceptionNet from {xception_path}")
        xception_checkpoint = torch.load(
            xception_path, 
            map_location=self.device,
            weights_only=False  
        )

        # 디버깅: 키 구조 출력
        print("Checkpoint keys:", xception_checkpoint.keys())
        if 'model_state_dict' in xception_checkpoint:
            state_dict = xception_checkpoint['model_state_dict']
            print("First 5 state_dict keys:")
            for i, key in enumerate(list(state_dict.keys())[:5]):
                print(f"  {key}")

        print(f"Loading CNN-LSTM from {cnn_lstm_path}")
        cnn_lstm_checkpoint = torch.load(
            cnn_lstm_path, 
            map_location=self.device,
            weights_only=False  
        )

        if 'model_state_dict' in cnn_lstm_checkpoint:
            state_dict = cnn_lstm_checkpoint['model_state_dict']
        else:
            state_dict = cnn_lstm_checkpoint

        try:
            self.cnn_lstm.load_state_dict(state_dict)
        except RuntimeError as e:
            if "Missing key(s) in state_dict" in str(e):
                print(f"Warning: State dict keys mismatch, loading with strict=False")
                self.cnn_lstm.load_state_dict(state_dict, strict=False)
            else:
                raise

        # Evaluation 모드
        self.xception.eval()
        self.efficientnet.eval()
        self.cnn_lstm.eval()
        
        # 앙상블 가중치
        self.weights = {
            'xception': 0.4,
            'efficientnet': 0.3,
            'cnn_lstm': 0.3
        }
    
    def _preprocess_single_frame(self, frame: np.ndarray) -> torch.Tensor:
        """단일 프레임 전처리 (XceptionNet, EfficientNet용 - 299x299)"""
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 299x299로 리사이즈
        frame_resized = cv2.resize(frame_rgb, (299, 299))
        
        # 정규화
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # ImageNet 정규화
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame_normalized = (frame_normalized - mean) / std
        
        # (H, W, C) -> (C, H, W)
        frame_tensor = torch.from_numpy(frame_normalized.transpose(2, 0, 1))
        
        return frame_tensor.unsqueeze(0).float().to(self.device)
    
    def _preprocess_frame_sequence(self, frames: List[np.ndarray]) -> torch.Tensor:
        """프레임 시퀀스 전처리 (CNN-LSTM용 - 16x112x112)"""
        processed_frames = []
        
        for frame in frames:
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 112x112로 리사이즈
            frame_resized = cv2.resize(frame_rgb, (112, 112))
            
            # 정규화
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            
            # ImageNet 정규화
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame_normalized = (frame_normalized - mean) / std
            
            processed_frames.append(frame_normalized)
        
        # (16, H, W, C) -> (16, C, H, W)
        frames_array = np.array(processed_frames)
        frames_tensor = torch.from_numpy(frames_array.transpose(0, 3, 1, 2))
        
        # (1, 16, C, H, W)
        return frames_tensor.unsqueeze(0).float().to(self.device)
    
    def predict_single_frame(self, frame: torch.Tensor) -> Dict:
        """단일 프레임 예측 (XceptionNet, EfficientNet)"""
        with torch.no_grad():
            # XceptionNet
            xception_out = self.xception(frame)
            xception_prob = F.softmax(xception_out, dim=1)
            xception_fake_prob = xception_prob[:, 1].item()
            
            # EfficientNet
            efficientnet_out = self.efficientnet(frame)
            efficientnet_prob = F.softmax(efficientnet_out, dim=1)
            efficientnet_fake_prob = efficientnet_prob[:, 1].item()
        
        return {
            'xception': {
                'fake_probability': xception_fake_prob,
                'prediction': 'fake' if xception_fake_prob > 0.5 else 'real',
                'confidence': max(xception_fake_prob, 1 - xception_fake_prob),
                'detected_patterns': self._get_xception_patterns(xception_fake_prob)
            },
            'efficientnet': {
                'fake_probability': efficientnet_fake_prob,
                'prediction': 'fake' if efficientnet_fake_prob > 0.5 else 'real',
                'confidence': max(efficientnet_fake_prob, 1 - efficientnet_fake_prob),
                'detected_patterns': self._get_efficientnet_patterns(efficientnet_fake_prob)
            }
        }
    
    def predict_sequence(self, frames: torch.Tensor) -> Dict:
        """16프레임 시퀀스 예측 (CNN-LSTM)"""
        with torch.no_grad():
            cnn_lstm_out, attention_weights = self.cnn_lstm(frames)
            cnn_lstm_prob = F.softmax(cnn_lstm_out, dim=1)
            cnn_lstm_fake_prob = cnn_lstm_prob[:, 1].item()
            
            # Attention 분석
            attention_weights = attention_weights.cpu().numpy()[0]
            suspicious_frames = self._analyze_attention(attention_weights)
        
        return {
            'cnn_lstm': {
                'fake_probability': cnn_lstm_fake_prob,
                'prediction': 'fake' if cnn_lstm_fake_prob > 0.5 else 'real',
                'confidence': max(cnn_lstm_fake_prob, 1 - cnn_lstm_fake_prob),
                'detected_patterns': self._get_cnn_lstm_patterns(cnn_lstm_fake_prob),
                'suspicious_frames': suspicious_frames,
                'attention_weights': attention_weights.tolist()
            }
        }
    
    def predict_ensemble(
        self, 
        single_frame_raw: np.ndarray, 
        frame_sequence_raw: List[np.ndarray]
    ) -> Dict:
        """전체 앙상블 예측"""
        
        # 전처리
        single_frame = self._preprocess_single_frame(single_frame_raw)
        frame_sequence = self._preprocess_frame_sequence(frame_sequence_raw)
        
        # 개별 모델 예측
        single_results = self.predict_single_frame(single_frame)
        sequence_results = self.predict_sequence(frame_sequence)
        
        # 앙상블 확률
        ensemble_fake_prob = (
            single_results['xception']['fake_probability'] * self.weights['xception'] +
            single_results['efficientnet']['fake_probability'] * self.weights['efficientnet'] +
            sequence_results['cnn_lstm']['fake_probability'] * self.weights['cnn_lstm']
        )
        
        # 모델 합의도
        predictions = [
            single_results['xception']['prediction'],
            single_results['efficientnet']['prediction'],
            sequence_results['cnn_lstm']['prediction']
        ]
        agreement = predictions.count(predictions[0]) / len(predictions)
        
        final_prediction = 'fake' if ensemble_fake_prob > 0.5 else 'real'
        
        return {
            'final_prediction': final_prediction,
            'ensemble_fake_probability': ensemble_fake_prob,
            'ensemble_confidence': max(ensemble_fake_prob, 1 - ensemble_fake_prob),
            'model_agreement': agreement,
            'individual_models': {
                **single_results,
                **sequence_results
            },
            'detected_artifacts': self._aggregate_artifacts(single_results, sequence_results),
            'risk_level': self._calculate_risk_level(ensemble_fake_prob, agreement)
        }
    
    def _get_xception_patterns(self, prob: float) -> List[str]:
        """XceptionNet 탐지 패턴"""
        patterns = []
        if prob > 0.7:
            patterns.extend([
                "얼굴 경계의 부자연스러운 블렌딩 감지",
                "피부 텍스처의 비정상적 매끄러움"
            ])
        elif prob > 0.5:
            patterns.append("미세한 공간적 아티팩트 감지")
        return patterns
    
    def _get_efficientnet_patterns(self, prob: float) -> List[str]:
        """EfficientNet 탐지 패턴"""
        patterns = []
        if prob > 0.7:
            patterns.extend([
                "다층 스케일에서 구조적 불일치 감지",
                "조명과 그림자의 불일치"
            ])
        elif prob > 0.5:
            patterns.append("전체적 특징 분포 이상")
        return patterns
    
    def _get_cnn_lstm_patterns(self, prob: float) -> List[str]:
        """CNN-LSTM 탐지 패턴"""
        patterns = []
        if prob > 0.7:
            patterns.extend([
                "프레임 간 불연속적 변화 감지",
                "시간적 일관성 결여",
                "비정상적인 움직임 패턴"
            ])
        elif prob > 0.5:
            patterns.append("시간적 아티팩트 감지")
        return patterns
    
    def _analyze_attention(self, attention_weights: np.ndarray) -> List[int]:
        """Attention 높은 프레임 추출"""
        threshold = np.mean(attention_weights) + np.std(attention_weights)
        suspicious = np.where(attention_weights > threshold)[0].tolist()
        return suspicious
    
    def _aggregate_artifacts(self, single_results: Dict, sequence_results: Dict) -> Dict:
        """아티팩트 종합"""
        artifacts = {
            'spatial': {'detected': False, 'sources': [], 'patterns': []},
            'temporal': {'detected': False, 'sources': [], 'patterns': []},
            'structural': {'detected': False, 'sources': [], 'patterns': []}
        }
        
        if single_results['xception']['fake_probability'] > 0.5:
            artifacts['spatial']['detected'] = True
            artifacts['spatial']['sources'].append('XceptionNet')
            artifacts['spatial']['patterns'].extend(
                single_results['xception']['detected_patterns']
            )
        
        if single_results['efficientnet']['fake_probability'] > 0.5:
            artifacts['structural']['detected'] = True
            artifacts['structural']['sources'].append('EfficientNet-B4')
            artifacts['structural']['patterns'].extend(
                single_results['efficientnet']['detected_patterns']
            )
        
        if sequence_results['cnn_lstm']['fake_probability'] > 0.5:
            artifacts['temporal']['detected'] = True
            artifacts['temporal']['sources'].append('CNN-LSTM')
            artifacts['temporal']['patterns'].extend(
                sequence_results['cnn_lstm']['detected_patterns']
            )
        
        return artifacts
    
    def _calculate_risk_level(self, prob: float, agreement: float) -> str:
        """위험도 계산"""
        if prob > 0.8 and agreement >= 0.67:
            return 'HIGH'
        elif prob > 0.6 or (prob > 0.5 and agreement >= 0.67):
            return 'MEDIUM'
        elif prob > 0.5:
            return 'LOW'
        else:
            return 'SAFE'
    
    # ========================================
    # 기존 호환성 메서드 (삭제 예정)
    # ========================================
    def analyze_frame(self, frame: np.ndarray) -> Tuple[bool, float, str]:
        """
        [DEPRECATED] 기존 호환성을 위한 메서드
        새로운 코드에서는 predict_ensemble() 사용
        """
        single_frame = self._preprocess_single_frame(frame)
        results = self.predict_single_frame(single_frame)
        
        # 간단히 XceptionNet 결과만 반환
        xception_result = results['xception']
        is_deepfake = xception_result['prediction'] == 'fake'
        confidence = xception_result['confidence']
        
        if is_deepfake:
            anomaly_type = "spatial_artifact"
        else:
            anomaly_type = "normal"
        
        return is_deepfake, confidence, anomaly_type

def load_video_model(
    xception_path: str, 
    efficientnet_path: str,
    cnn_lstm_path: str,
    device: str = 'cuda'
) -> VideoModel:
    """영상 분석 3-모델 앙상블 로드"""
    model = VideoModel(
        xception_path=xception_path,
        efficientnet_path=efficientnet_path,
        cnn_lstm_path=cnn_lstm_path,
        device=device
    )
    return model