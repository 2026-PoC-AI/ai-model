# app/domains/video/deepfake_detection/models/ensemble.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Tuple
import numpy as np

class XceptionNet(nn.Module):
    def __init__(self, pretrained=True):
        super(XceptionNet, self).__init__()
        self.model = timm.create_model('xception', pretrained=pretrained, num_classes=2)
    
    def forward(self, x):
        return self.model(x)

class EfficientNetB4(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetB4, self).__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=2)
    
    def forward(self, x):
        return self.model(x)

class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256, num_layers=2, dropout=0.5):
        super(CNNLSTMModel, self).__init__()
        
        # ResNet18 백본
        resnet = timm.create_model('resnet18', pretrained=True)
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

class DeepfakeEnsemble:
    def __init__(self, weights_dir: str, device: str = 'cuda'):
        self.device = device
        
        # 모델 로드
        self.xception = XceptionNet(pretrained=False).to(device)
        self.efficientnet = EfficientNetB4(pretrained=False).to(device)
        self.cnn_lstm = CNNLSTMModel().to(device)
        
        # 가중치 로드
        xception_weights = torch.load(f"{weights_dir}/xception/xception_best_20260116.pth", map_location=device)
        self.xception.load_state_dict(xception_weights['model_state_dict'])
        
        efficientnet_weights = torch.load(f"{weights_dir}/efficientnet/efficientnet_best_20260116.pth", map_location=device)
        self.efficientnet.load_state_dict(efficientnet_weights['model_state_dict'])
        
        cnn_lstm_weights = torch.load(f"{weights_dir}/cnn-lstm/improved/best_model_latest.pth", map_location=device)
        self.cnn_lstm.load_state_dict(cnn_lstm_weights['model_state_dict'])
        
        # Evaluation 모드
        self.xception.eval()
        self.efficientnet.eval()
        self.cnn_lstm.eval()
        
        # 앙상블 가중치 (validation accuracy 기반)
        self.weights = {
            'xception': 0.4,      # 91.75%
            'efficientnet': 0.3,  # 81.43%
            'cnn_lstm': 0.3       # 90.00%
        }
    
    def predict_single_frame(self, frame: torch.Tensor) -> Dict:
        """단일 프레임 예측 (XceptionNet, EfficientNet)"""
        with torch.no_grad():
            # XceptionNet 예측
            xception_out = self.xception(frame)
            xception_prob = F.softmax(xception_out, dim=1)
            xception_fake_prob = xception_prob[:, 1].item()
            
            # EfficientNet 예측
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
            
            # Attention 가중치 분석
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
        single_frame: torch.Tensor, 
        frame_sequence: torch.Tensor
    ) -> Dict:
        """전체 앙상블 예측"""
        
        # 개별 모델 예측
        single_results = self.predict_single_frame(single_frame)
        sequence_results = self.predict_sequence(frame_sequence)
        
        # 앙상블 확률 계산
        ensemble_fake_prob = (
            single_results['xception']['fake_probability'] * self.weights['xception'] +
            single_results['efficientnet']['fake_probability'] * self.weights['efficientnet'] +
            sequence_results['cnn_lstm']['fake_probability'] * self.weights['cnn_lstm']
        )
        
        # 모델 간 합의도 계산
        predictions = [
            single_results['xception']['prediction'],
            single_results['efficientnet']['prediction'],
            sequence_results['cnn_lstm']['prediction']
        ]
        agreement = predictions.count(predictions[0]) / len(predictions)
        
        # 최종 결과
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
            'detected_artifacts': self._aggregate_artifacts(
                single_results, 
                sequence_results
            ),
            'risk_level': self._calculate_risk_level(ensemble_fake_prob, agreement)
        }
    
    def _get_xception_patterns(self, prob: float) -> List[str]:
        """XceptionNet이 탐지한 패턴"""
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
        """EfficientNet이 탐지한 패턴"""
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
        """CNN-LSTM이 탐지한 패턴"""
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
        """Attention이 높은 의심 프레임 추출"""
        threshold = np.mean(attention_weights) + np.std(attention_weights)
        suspicious = np.where(attention_weights > threshold)[0].tolist()
        return suspicious
    
    def _aggregate_artifacts(
        self, 
        single_results: Dict, 
        sequence_results: Dict
    ) -> Dict:
        """전체 탐지된 아티팩트 종합"""
        artifacts = {
            'spatial': {
                'detected': False,
                'sources': [],
                'patterns': []
            },
            'temporal': {
                'detected': False,
                'sources': [],
                'patterns': []
            },
            'structural': {
                'detected': False,
                'sources': [],
                'patterns': []
            }
        }
        
        # XceptionNet - 공간적
        if single_results['xception']['fake_probability'] > 0.5:
            artifacts['spatial']['detected'] = True
            artifacts['spatial']['sources'].append('XceptionNet')
            artifacts['spatial']['patterns'].extend(
                single_results['xception']['detected_patterns']
            )
        
        # EfficientNet - 구조적
        if single_results['efficientnet']['fake_probability'] > 0.5:
            artifacts['structural']['detected'] = True
            artifacts['structural']['sources'].append('EfficientNet-B4')
            artifacts['structural']['patterns'].extend(
                single_results['efficientnet']['detected_patterns']
            )
        
        # CNN-LSTM - 시간적
        if sequence_results['cnn_lstm']['fake_probability'] > 0.5:
            artifacts['temporal']['detected'] = True
            artifacts['temporal']['sources'].append('CNN-LSTM')
            artifacts['temporal']['patterns'].extend(
                sequence_results['cnn_lstm']['detected_patterns']
            )
        
        return artifacts
    
    def _calculate_risk_level(self, prob: float, agreement: float) -> str:
        """위험도 레벨 계산"""
        if prob > 0.8 and agreement >= 0.67:
            return 'HIGH'
        elif prob > 0.6 or (prob > 0.5 and agreement >= 0.67):
            return 'MEDIUM'
        elif prob > 0.5:
            return 'LOW'
        else:
            return 'SAFE'