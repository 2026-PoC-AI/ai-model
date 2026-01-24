import numpy as np
import librosa
from typing import List, Dict, Tuple
from scipy import signal


class AudioAnalyzer:
    """
    음성 딥페이크 상세 분석
    
    주파수 스펙트럼, 위상, 시간대별 이상 패턴 탐지
    """
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def analyze_frequency_spectrum(
        self, 
        audio: np.ndarray
    ) -> Dict[str, List]:
        """
        주파수 스펙트럼 분석
        
        고주파 아티팩트, 부자연스러운 주파수 패턴 탐지
        """
        # FFT로 주파수 분석
        fft = np.fft.fft(audio)
        frequencies = np.fft.fftfreq(len(fft), 1/self.sample_rate)
        magnitude = np.abs(fft)
        
        # 양수 주파수만
        positive_freq_idx = frequencies > 0
        frequencies = frequencies[positive_freq_idx]
        magnitude = magnitude[positive_freq_idx]
        
        # 주파수 대역별 에너지
        bands = {
            'low': (0, 500),       # 저주파
            'mid': (500, 2000),    # 중간주파
            'high': (2000, 4000),  # 고주파
            'very_high': (4000, 8000)  # 초고주파
        }
        
        suspicious_bands = []
        band_energies = {}
        
        for band_name, (low, high) in bands.items():
            mask = (frequencies >= low) & (frequencies < high)
            energy = np.mean(magnitude[mask])
            band_energies[band_name] = float(energy)
            
            # 이상 패턴 감지
            # 고주파가 비정상적으로 높거나 낮으면 의심
            if band_name in ['high', 'very_high']:
                if energy > np.percentile(magnitude, 95):
                    suspicious_bands.append(int((low + high) / 2))
        
        # 스펙트럴 플럭스 (시간에 따른 주파수 변화)
        spectral_flux = self._compute_spectral_flux(audio)
        
        return {
            'suspicious_frequency_bands': suspicious_bands,
            'band_energies': band_energies,
            'spectral_flux': float(spectral_flux),
            'has_high_freq_artifacts': len(suspicious_bands) > 0
        }
    
    def analyze_phase_coherence(
        self, 
        audio: np.ndarray
    ) -> Dict[str, any]:
        """
        위상 일관성 분석
        
        합성 음성은 위상이 부자연스러울 수 있음
        """
        # STFT로 시간-주파수 분석
        f, t, Zxx = signal.stft(audio, fs=self.sample_rate, nperseg=512)
        
        # 위상 추출
        phase = np.angle(Zxx)
        
        # 위상 차이 계산
        phase_diff = np.diff(phase, axis=1)
        
        # 위상 일관성 측정
        phase_coherence = np.std(phase_diff)
        
        # 임계값 기반 판단 (경험적 값)
        is_phase_inconsistent = phase_coherence > 1.5
        
        return {
            'phase_coherence': float(phase_coherence),
            'is_phase_inconsistent': bool(is_phase_inconsistent),
            'phase_stability': 'unstable' if is_phase_inconsistent else 'stable'
        }
    
    def detect_temporal_anomalies(
        self, 
        audio: np.ndarray,
        mel_probs: Dict[str, float],
        lfcc_probs: Dict[str, float],
        window_size: float = 0.5
    ) -> List[Dict]:
        """
        시간대별 이상 구간 탐지
        
        오디오를 작은 구간으로 나눠서 각각 분석
        """
        duration = len(audio) / self.sample_rate
        window_samples = int(window_size * self.sample_rate)
        hop_samples = window_samples // 2
        
        segments = []
        
        for start_sample in range(0, len(audio) - window_samples, hop_samples):
            end_sample = start_sample + window_samples
            segment_audio = audio[start_sample:end_sample]
            
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            
            # 세그먼트별 특징 추출
            rms_energy = np.sqrt(np.mean(segment_audio ** 2))
            zero_crossing_rate = np.mean(librosa.zero_crossings(segment_audio))
            
            # 이상 점수 계산 (간단한 휴리스틱)
            # 실제로는 모델 예측을 사용하면 더 정확
            anomaly_score = 0.0
            
            # RMS 에너지가 너무 낮거나 높으면 의심
            if rms_energy < 0.01 or rms_energy > 0.5:
                anomaly_score += 0.3
            
            # Zero crossing rate가 이상하면 의심
            if zero_crossing_rate > 0.3:
                anomaly_score += 0.2
            
            # 전체 모델 예측과 결합
            overall_fake_prob = (mel_probs['fake'] + lfcc_probs['fake']) / 2
            anomaly_score = (anomaly_score + overall_fake_prob) / 2
            
            # 위험도 판단
            if anomaly_score > 0.7:
                risk_level = 'high'
                reason = '합성 음성 특징 감지'
            elif anomaly_score > 0.4:
                risk_level = 'medium'
                reason = '부자연스러운 패턴'
            else:
                risk_level = 'low'
                reason = '정상 구간'
            
            if anomaly_score > 0.4:  # 의심 구간만 추가
                segments.append({
                    'start': round(start_time, 2),
                    'end': round(end_time, 2),
                    'risk': risk_level,
                    'score': round(anomaly_score, 3),
                    'reason': reason
                })
        
        return segments
    
    def generate_deepfake_indicators(
        self,
        freq_analysis: Dict,
        phase_analysis: Dict,
        temporal_segments: List[Dict],
        mel_probs: Dict[str, float],
        lfcc_probs: Dict[str, float]
    ) -> List[str]:
        """
        딥페이크 탐지 근거 생성
        """
        indicators = []
        
        # 모델 예측 기반
        if mel_probs['fake'] > 0.9 and lfcc_probs['fake'] > 0.9:
            indicators.append("두 모델 모두 높은 확신으로 딥페이크 판단")
        elif mel_probs['fake'] > 0.7 or lfcc_probs['fake'] > 0.7:
            indicators.append("딥페이크 패턴 감지")
        
        # 주파수 분석 기반
        if freq_analysis['has_high_freq_artifacts']:
            bands_str = ', '.join([f"{b}Hz" for b in freq_analysis['suspicious_frequency_bands']])
            indicators.append(f"비자연스러운 고주파 패턴 감지 ({bands_str})")
        
        if freq_analysis['spectral_flux'] > 0.5:
            indicators.append("주파수 스펙트럼의 급격한 변화 감지")
        
        # 위상 분석 기반
        if phase_analysis['is_phase_inconsistent']:
            indicators.append(f"위상 불일치 발견 (coherence: {phase_analysis['phase_coherence']:.2f})")
        
        # 시간대별 분석 기반
        if temporal_segments:
            high_risk_count = sum(1 for s in temporal_segments if s['risk'] == 'high')
            if high_risk_count > 0:
                indicators.append(f"{high_risk_count}개 구간에서 합성 음성 특징 발견")
        
        # LFCC가 Mel보다 확신이 높으면
        if lfcc_probs['fake'] - mel_probs['fake'] > 0.1:
            indicators.append("선형 주파수 분석에서 강한 아티팩트 탐지")
        
        if not indicators:
            indicators.append("명확한 딥페이크 증거는 발견되지 않음")
        
        return indicators
    
    def _compute_spectral_flux(self, audio: np.ndarray) -> float:
        """
        스펙트럴 플럭스 계산
        
        시간에 따른 주파수 변화율
        """
        # STFT
        hop_length = 512
        spec = np.abs(librosa.stft(audio, hop_length=hop_length))
        
        # 프레임 간 차이
        flux = np.sqrt(np.sum(np.diff(spec, axis=1) ** 2, axis=0))
        
        return np.mean(flux)
    
    def full_analysis(
        self,
        audio_path: str,
        mel_probs: Dict[str, float],
        lfcc_probs: Dict[str, float]
    ) -> Dict:
        """
        전체 분석 실행
        """
        # 오디오 로드
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # 각종 분석 실행
        freq_analysis = self.analyze_frequency_spectrum(audio)
        phase_analysis = self.analyze_phase_coherence(audio)
        temporal_segments = self.detect_temporal_anomalies(
            audio, mel_probs, lfcc_probs
        )
        indicators = self.generate_deepfake_indicators(
            freq_analysis, phase_analysis, temporal_segments,
            mel_probs, lfcc_probs
        )
        
        return {
            'frequency_analysis': freq_analysis,
            'phase_analysis': phase_analysis,
            'suspicious_time_segments': temporal_segments,
            'deepfake_indicators': indicators
        }