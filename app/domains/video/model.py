# app/domains/video/model.py
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from PIL import Image
import timm
import logging

logger = logging.getLogger(__name__)

class XceptionNet(nn.Module):
    def __init__(self, pretrained=False):
        super(XceptionNet, self).__init__()
        # backbone (features only, no classifier)
        base_model = timm.create_model('xception', pretrained=pretrained, num_classes=0)
        self.backbone = base_model
        
        # 학습 시 사용한 classifier 구조
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),           # 0
            nn.Linear(2048, 512),      # 1
            nn.ReLU(),                 # 2
            nn.Dropout(0.5),           # 3
            nn.Linear(512, 2)          # 4
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class EfficientNetB4(nn.Module):
    def __init__(self, pretrained=False):
        super(EfficientNetB4, self).__init__()
        
        # timm 사용 (학습 시와 동일)
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained)
        
        # 학습 시 사용한 classifier 구조
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 2)
        )
    
    def forward(self, x):
        return self.backbone(x)

class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256, num_layers=2, dropout=0.5):
        super(CNNLSTMModel, self).__init__()
        
        # ResNet18 백본을 cnn으로 명명 (체크포인트와 일치)
        resnet = timm.create_model('resnet18', pretrained=False)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention (체크포인트에는 없지만 추론 시 필요)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 학습 시 사용한 fc 구조 (단순 Linear)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),       # 0
            nn.Linear(hidden_size, 2)  # 1
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # CNN 특징 추출
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(features)
        
        # 학습 시에는 attention 없이 마지막 hidden state만 사용
        # 추론 시에는 attention 사용 가능
        if self.training:
            context = lstm_out[:, -1, :]  # 마지막 시퀀스만
        else:
            # Attention
            attention_weights = self.attention(lstm_out)
            attention_weights = F.softmax(attention_weights, dim=1)
            context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 분류
        output = self.fc(context)
        
        # attention_weights 반환 (추론 시)
        if not self.training:
            return output, attention_weights.squeeze(-1)
        return output, None

class VideoModel:
    """영상 딥페이크 탐지 3-모델 앙상블"""
    
    def __init__(
        self, 
        xception_path: str, 
        efficientnet_path: str,
        cnn_lstm_path: str,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.xception = XceptionNet(pretrained=False).to(self.device)
        self.efficientnet = EfficientNetB4(pretrained=False).to(self.device)
        self.cnn_lstm = CNNLSTMModel().to(self.device)
        
        # XceptionNet 로드
        print(f"Loading XceptionNet from {xception_path}")
        xception_checkpoint = torch.load(xception_path, map_location=self.device, weights_only=False)
        state_dict = xception_checkpoint.get('model_state_dict', xception_checkpoint)

        # 체크포인트 키에는 'backbone.' 접두사가 없으므로 추가해야 함
        new_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith('backbone.') and not key.startswith('classifier.'):
                # backbone 레이어에 'backbone.' 접두사 추가
                new_key = f'backbone.{key}'
                new_state_dict[new_key] = value
            else:
                # classifier는 그대로
                new_state_dict[key] = value

        self.xception.load_state_dict(new_state_dict, strict=True)
        print(f"✓ XceptionNet loaded successfully")
        
        # EfficientNet 로드
        print(f"Loading EfficientNet-B4 from {efficientnet_path}")
        efficientnet_checkpoint = torch.load(efficientnet_path, map_location=self.device, weights_only=False)
        state_dict = efficientnet_checkpoint.get('model_state_dict', efficientnet_checkpoint)

        # torchvision 모델이므로 키 변환 없이 그대로 로드
        self.efficientnet.load_state_dict(state_dict, strict=True)
        print(f"✓ EfficientNet-B4 loaded successfully")

        # CNN-LSTM 로드
        print(f"Loading CNN-LSTM from {cnn_lstm_path}")
        cnn_lstm_checkpoint = torch.load(cnn_lstm_path, map_location=self.device, weights_only=False)
        state_dict = cnn_lstm_checkpoint.get('model_state_dict', cnn_lstm_checkpoint)

        # 'cnn.' 그대로 사용 (모델 정의와 일치)
        self.cnn_lstm.load_state_dict(state_dict, strict=False)
        print(f"✓ CNN-LSTM loaded successfully (attention initialized randomly)")

        self.xception.eval()
        self.efficientnet.eval()
        self.cnn_lstm.eval()
        
        self.weights = {
            'xception': 0.4,
            'efficientnet': 0.3,
            'cnn_lstm': 0.3
        }
        
        print("✓ All models loaded successfully!")
    
    def _preprocess_single_frame(self, frame: np.ndarray) -> torch.Tensor:
        """단일 프레임 전처리 (299x299)"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (299, 299))
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame_normalized = (frame_normalized - mean) / std
        
        frame_tensor = torch.from_numpy(frame_normalized.transpose(2, 0, 1))
        return frame_tensor.unsqueeze(0).float().to(self.device)
    
    def _preprocess_frame_sequence(self, frames: List[np.ndarray]) -> torch.Tensor:
        """프레임 시퀀스 전처리 (16x112x112)"""
        processed_frames = []
        
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (112, 112))
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame_normalized = (frame_normalized - mean) / std
            
            processed_frames.append(frame_normalized)
        
        frames_array = np.array(processed_frames)
        frames_tensor = torch.from_numpy(frames_array.transpose(0, 3, 1, 2))
        
        return frames_tensor.unsqueeze(0).float().to(self.device)
    
    def predict_single_frame(self, frame: torch.Tensor) -> Dict:
        """단일 프레임 예측"""
        with torch.no_grad():
            xception_out = self.xception(frame)
            xception_prob = F.softmax(xception_out, dim=1)
            xception_fake_prob = xception_prob[:, 1].item()
            
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
        """16프레임 시퀀스 예측"""
        with torch.no_grad():
            cnn_lstm_out, attention_weights = self.cnn_lstm(frames)
            cnn_lstm_prob = F.softmax(cnn_lstm_out, dim=1)
            cnn_lstm_fake_prob = cnn_lstm_prob[:, 1].item()
            
            attention_weights = attention_weights.cpu().numpy()[0] if attention_weights is not None else np.ones(16) / 16
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
    
    def predict_video(
        self, 
        video_path: str, 
        sample_rate: int = 5, 
        aggregation: str = 'mean',
        progress_callback=None, 
        analysis_id=None
    ) -> Tuple[Dict, str]:
        """비디오 딥페이크 분석"""
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None, f"Failed to open video: {video_path}"
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if progress_callback:
            progress_callback(0, "video_upload", "영상 업로드 완료", analysis_id)
        
        frame_predictions = []
        frame_details = []
        sequence_buffer = []
        sequence_predictions = []
        
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % sample_rate != 0:
                continue
            
            current_progress = int((frame_count / total_frames) * 100)
            
            try:
                single_frame = self._preprocess_single_frame(frame)
                single_result = self.predict_single_frame(single_frame)
                
                sequence_buffer.append(frame)
                if len(sequence_buffer) > 16:
                    sequence_buffer.pop(0)
                
                cnn_lstm_result = None
                if len(sequence_buffer) == 16:
                    sequence_tensor = self._preprocess_frame_sequence(sequence_buffer)
                    cnn_lstm_result = self.predict_sequence(sequence_tensor)
                    sequence_predictions.append(cnn_lstm_result['cnn_lstm'])
                
                if cnn_lstm_result:
                    frame_ensemble_prob = (
                        single_result['xception']['fake_probability'] * 0.4 +
                        single_result['efficientnet']['fake_probability'] * 0.3 +
                        cnn_lstm_result['cnn_lstm']['fake_probability'] * 0.3
                    )
                else:
                    frame_ensemble_prob = (
                        single_result['xception']['fake_probability'] * 0.57 +
                        single_result['efficientnet']['fake_probability'] * 0.43
                    )
                
                frame_predictions.append(frame_ensemble_prob)
                
                frame_details.append({
                    'frame_number': frame_count - 1,
                    'timestamp': (frame_count - 1) / fps if fps > 0 else 0,
                    'xception': single_result['xception'],
                    'efficientnet': single_result['efficientnet'],
                    'cnn_lstm': cnn_lstm_result['cnn_lstm'] if cnn_lstm_result else None,
                    'ensemble_prob': frame_ensemble_prob
                })
                
                processed_count += 1
                
                if progress_callback and processed_count % 5 == 0:
                    progress_callback(
                        current_progress, 
                        "ai_analysis", 
                        f"AI 모델 분석 중 ({processed_count}개 프레임 완료)",
                        analysis_id
                    )
                    
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                continue
        
        cap.release()
        
        if len(frame_predictions) == 0:
            return None, "No frames could be processed"
        
        if progress_callback:
            progress_callback(95, "result_generation", "결과 생성 중", analysis_id)
        
        aggregated_prob = self._aggregate_predictions(frame_predictions, method=aggregation)
        
        xception_avg = np.mean([f['xception']['fake_probability'] for f in frame_details])
        efficientnet_avg = np.mean([f['efficientnet']['fake_probability'] for f in frame_details])
        
        cnn_lstm_probs = [f['cnn_lstm']['fake_probability'] for f in frame_details if f['cnn_lstm'] is not None]
        cnn_lstm_avg = np.mean(cnn_lstm_probs) if cnn_lstm_probs else 0.5
        
        # 개별 모델 판정
        predictions = [
            1 if xception_avg > 0.5 else 0,
            1 if efficientnet_avg > 0.5 else 0,
            1 if cnn_lstm_avg > 0.5 else 0
        ]

        # 모델 합의도
        model_agreement = sum(predictions) / 3.0

        # 최대 개별 확률
        max_individual_prob = max(xception_avg, efficientnet_avg, cnn_lstm_avg)

        # 딥페이크 판정 모델 수
        fake_votes = sum(predictions)

        # 매우 민감한 딥페이크 판정 로직
        # 1) 하나라도 0.5 이상이면 딥페이크
        # 2) 또는 앙상블 평균이 0.3 이상
        # 3) 또는 최대값이 0.45 이상
        is_fake = (fake_votes >= 1) or (aggregated_prob > 0.3) or (max_individual_prob > 0.45)

        suspicious_frames = []
        if sequence_predictions:
            for sp in sequence_predictions:
                if 'suspicious_frames' in sp:
                    suspicious_frames.extend(sp['suspicious_frames'])
        
        result = {
            'is_fake': is_fake,  # 새로운 판정 로직 적용
            'fake_probability': aggregated_prob,
            'real_probability': 1 - aggregated_prob,
            'confidence': max(aggregated_prob, 1 - aggregated_prob),
            'total_frames': frame_count,
            'processed_frames': processed_count,
            'frame_predictions': frame_predictions,
            'frame_details': frame_details,
            'individual_models': {
                'xception': {
                    'prediction': 'fake' if xception_avg > 0.5 else 'real',
                    'confidence': float(xception_avg),
                    'fake_probability': float(xception_avg),
                    'detected_patterns': self._get_xception_patterns(xception_avg)
                },
                'efficientnet': {
                    'prediction': 'fake' if efficientnet_avg > 0.5 else 'real',
                    'confidence': float(efficientnet_avg),
                    'fake_probability': float(efficientnet_avg),
                    'detected_patterns': self._get_efficientnet_patterns(efficientnet_avg)
                },
                'cnn_lstm': {
                    'prediction': 'fake' if cnn_lstm_avg > 0.5 else 'real',
                    'confidence': float(cnn_lstm_avg),
                    'fake_probability': float(cnn_lstm_avg),
                    'detected_patterns': self._get_cnn_lstm_patterns(cnn_lstm_avg),
                    'suspicious_frames': suspicious_frames
                }
            },
            'model_agreement': model_agreement,
            'risk_level': self._calculate_risk_level(aggregated_prob, model_agreement, max_individual_prob),
            'detected_artifacts': self._get_artifacts(xception_avg, efficientnet_avg, cnn_lstm_avg)
        }
        
        if progress_callback:
            progress_callback(100, "completed", "분석 완료", analysis_id)
        
        return result, None
    
    def _aggregate_predictions(self, predictions: List[float], method: str = 'mean') -> float:
        """프레임별 예측 집계"""
        if method == 'mean':
            return float(np.mean(predictions))
        elif method == 'max':
            return float(np.max(predictions))
        elif method == 'median':
            return float(np.median(predictions))
        else:
            return float(np.mean(predictions))
    
    def _get_artifacts(self, xception: float, efficientnet: float, cnn_lstm: float) -> Dict:
        """탐지된 아티팩트"""
        return {
            'spatial': {
                'detected': xception > 0.5,
                'sources': ['XceptionNet'] if xception > 0.5 else [],
                'patterns': self._get_xception_patterns(xception)
            },
            'structural': {
                'detected': efficientnet > 0.5,
                'sources': ['EfficientNet-B4'] if efficientnet > 0.5 else [],
                'patterns': self._get_efficientnet_patterns(efficientnet)
            },
            'temporal': {
                'detected': cnn_lstm > 0.5,
                'sources': ['CNN-LSTM'] if cnn_lstm > 0.5 else [],
                'patterns': self._get_cnn_lstm_patterns(cnn_lstm)
            }
        }
    
    def _get_xception_patterns(self, prob: float) -> List[str]:
        patterns = []
        if prob > 0.7:
            patterns.extend(["얼굴 경계의 부자연스러운 블렌딩 감지", "피부 텍스처의 비정상적 매끄러움"])
        elif prob > 0.5:
            patterns.append("미세한 공간적 아티팩트 감지")
        return patterns
    
    def _get_efficientnet_patterns(self, prob: float) -> List[str]:
        patterns = []
        if prob > 0.7:
            patterns.extend(["다층 스케일에서 구조적 불일치 감지", "조명과 그림자의 불일치"])
        elif prob > 0.5:
            patterns.append("전체적 특징 분포 이상")
        return patterns
    
    def _get_cnn_lstm_patterns(self, prob: float) -> List[str]:
        patterns = []
        if prob > 0.7:
            patterns.extend(["프레임 간 불연속적 변화 감지", "시간적 일관성 결여", "비정상적인 움직임 패턴"])
        elif prob > 0.5:
            patterns.append("시간적 아티팩트 감지")
        return patterns
    
    def _analyze_attention(self, attention_weights: np.ndarray) -> List[int]:
        threshold = np.mean(attention_weights) + np.std(attention_weights)
        suspicious = np.where(attention_weights > threshold)[0].tolist()
        return suspicious
    
    def _calculate_risk_level(self, prob: float, agreement: float, max_individual: float = 0) -> str:
        """위험도 레벨 계산"""
        # 앙상블 확률이 높거나, 개별 모델이 강하게 탐지한 경우
        if prob > 0.7 or max_individual > 0.8:
            return 'HIGH'
        elif prob > 0.5 or max_individual > 0.65:
            return 'MEDIUM'  
        elif prob > 0.3 or max_individual > 0.5:
            return 'LOW'
        else:
            return 'SAFE'

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