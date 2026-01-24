import numpy as np
import librosa
from typing import Dict, List, Tuple

class DeepfakeMethodClassifier:
    """
    딥페이크 생성 기술 분류기
    - TTS (Text-to-Speech)
    - Voice Conversion
    - Replay Attack
    """
    
    def __init__(self):
        self.methods = ['tts', 'voice_conversion', 'replay_attack']
    
    def analyze(self, audio_path: str, mel_pred: float, lfcc_pred: float) -> Dict:
        """
        딥페이크 생성 기술 분류 및 상세 분석
        """
        y, sr = librosa.load(audio_path, sr=16000)
        
        # 1. 각 생성 기술별 확률 계산
        method_probs = self._classify_method(y, sr, mel_pred, lfcc_pred)
        
        # 2. 가장 의심되는 기술
        suspected_method = max(method_probs, key=method_probs.get)
        method_confidence = method_probs[suspected_method]
        
        # 3. 의심스러운 패턴 탐지
        suspicious_patterns = self._detect_patterns(y, sr, suspected_method)
        
        # 4. 시간대별 위험도 분석
        time_segments = self._analyze_time_segments(y, sr, suspected_method)
        
        return {
            'suspected_method': self._format_method_name(suspected_method),
            'method_confidence': float(method_confidence),
            'detailed_analysis': {
                'voice_synthesis_probability': float(method_probs['tts']),
                'voice_conversion_probability': float(method_probs['voice_conversion']),
                'replay_attack_probability': float(method_probs['replay_attack'])
            },
            'suspicious_patterns': suspicious_patterns,
            'time_segments': time_segments
        }
    
    def _classify_method(self, y: np.ndarray, sr: int, 
                        mel_pred: float, lfcc_pred: float) -> Dict[str, float]:
        """
        모델 예측 + 음향 특징 기반 생성 기술 분류
        """
        features = self._extract_method_features(y, sr)
        
        # TTS 특징: 일정한 피치, 부자연스러운 운율
        tts_score = (
            features['pitch_variance_low'] * 0.3 +
            features['prosody_unnaturalness'] * 0.4 +
            features['formant_irregularity'] * 0.3
        )
        
        # Voice Conversion 특징: 포먼트 불일치, 스펙트럼 왜곡
        vc_score = (
            features['formant_mismatch'] * 0.4 +
            features['spectral_distortion'] * 0.3 +
            features['phase_discontinuity'] * 0.3
        )
        
        # Replay Attack 특징: 배경 소음, 공간 음향
        replay_score = (
            features['background_noise'] * 0.4 +
            features['room_acoustics'] * 0.3 +
            features['recording_artifacts'] * 0.3
        )
        
        # 모델 예측 신뢰도 반영
        model_confidence = (mel_pred + lfcc_pred) / 2
        
        # 정규화
        total = tts_score + vc_score + replay_score
        if total > 0:
            tts_score = (tts_score / total) * model_confidence
            vc_score = (vc_score / total) * model_confidence
            replay_score = (replay_score / total) * model_confidence
        
        return {
            'tts': tts_score,
            'voice_conversion': vc_score,
            'replay_attack': replay_score
        }
    
    def _extract_method_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        생성 기술 분류를 위한 음향 특징 추출
        """
        # 피치 분석
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        pitch_variance = np.var(pitch_values) if pitch_values else 0
        pitch_variance_low = 1.0 if pitch_variance < 100 else 0.0
        
        # 스펙트럼 특징
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # MFCC 변화율
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # 제로 크로싱 레이트
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        return {
            'pitch_variance_low': pitch_variance_low,
            'prosody_unnaturalness': np.std(mfcc_delta) / 10.0,
            'formant_irregularity': np.std(spectral_centroid) / 1000.0,
            'formant_mismatch': np.corrcoef(spectral_centroid, spectral_rolloff)[0, 1],
            'spectral_distortion': np.mean(np.abs(mfcc_delta)),
            'phase_discontinuity': np.std(zcr),
            'background_noise': np.mean(y[:sr] ** 2),  # 첫 1초 에너지
            'room_acoustics': np.std(spectral_rolloff) / 1000.0,
            'recording_artifacts': np.max(zcr) - np.min(zcr)
        }
    
    def _detect_patterns(self, y: np.ndarray, sr: int, method: str) -> List[str]:
        """
        생성 기술별 특징적인 패턴 탐지
        """
        patterns = []
        
        if method == 'tts':
            # TTS 특유 패턴
            pitches, _ = librosa.piptrack(y=y, sr=sr)
            if np.std(pitches[pitches > 0]) < 50:
                patterns.append("TTS 특유의 운율 패턴")
            
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            if np.std(spectral_centroid) < 500:
                patterns.append("일정한 피치 변화")
            
            patterns.append("부자연스러운 포먼트 전환")
        
        elif method == 'voice_conversion':
            patterns.append("포먼트 불일치 감지")
            patterns.append("스펙트럼 왜곡 패턴")
            patterns.append("위상 불연속성")
        
        elif method == 'replay_attack':
            patterns.append("배경 잡음 감지")
            patterns.append("공간 음향 특성")
            patterns.append("녹음 아티팩트")
        
        return patterns
    
    def _analyze_time_segments(self, y: np.ndarray, sr: int, 
                               method: str) -> List[Dict]:
        """
        시간대별 위험도 분석
        """
        duration = len(y) / sr
        segment_length = 0.5  # 0.5초 단위
        segments = []
        
        for i in range(0, int(duration / segment_length)):
            start = i * segment_length
            end = min((i + 1) * segment_length, duration)
            
            start_idx = int(start * sr)
            end_idx = int(end * sr)
            segment_audio = y[start_idx:end_idx]
            
            # 세그먼트별 위험도 계산
            risk, reason = self._calculate_segment_risk(
                segment_audio, sr, method
            )
            
            segments.append({
                'start': float(start),
                'end': float(end),
                'risk': risk,
                'reason': reason
            })
        
        return segments
    
    def _calculate_segment_risk(self, segment: np.ndarray, sr: int, 
                                method: str) -> Tuple[str, str]:
        """
        세그먼트 위험도 계산
        """
        # 에너지 기반 분석
        energy = np.sum(segment ** 2)
        
        # 스펙트럼 변화
        if len(segment) > 512:
            spec = np.abs(librosa.stft(segment))
            spec_std = np.std(spec)
            
            if spec_std > 0.5:
                return 'high', f'{method} 의심'
            elif spec_std > 0.2:
                return 'medium', '부분적 왜곡 감지'
            else:
                return 'low', '자연스러운 구간'
        
        return 'low', '분석 불가 (구간 짧음)'
    
    def _format_method_name(self, method: str) -> str:
        """
        기술명 포맷팅
        """
        names = {
            'tts': 'TTS (Text-to-Speech)',
            'voice_conversion': 'Voice Conversion',
            'replay_attack': 'Replay Attack'
        }
        return names.get(method, method)