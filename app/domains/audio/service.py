import tempfile
import os
from pathlib import Path
from typing import BinaryIO

class AudioAnalysisService:
    """
    오디오 딥페이크 분석 서비스
    """
    
    def __init__(self, predictor):
        """
        Args:
            predictor: EnsemblePredictor 인스턴스
        """
        self.predictor = predictor
    
    async def analyze_audio(
        self,
        file_content: BinaryIO,
        filename: str,
        analysis_id: int
    ) -> dict:
        """
        오디오 파일 분석
        
        Args:
            file_content: 파일 내용
            filename: 파일명
            analysis_id: 분석 ID
            
        Returns:
            분석 결과 딕셔너리
        """
        file_ext = Path(filename).suffix.lower()
        allowed_extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
        
        if file_ext not in allowed_extensions:
            raise ValueError(f"지원하지 않는 파일 형식: {file_ext}")
        
        temp_path = None
        try:
            # 임시 파일 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                content = file_content.read()
                temp_file.write(content)
                temp_path = temp_file.name
            
            # 예측 수행 - detailed=True 추가
            result = self.predictor.predict(temp_path, detailed=True)
            
            # 응답 데이터 구성
            response = {
                'analysis_id': analysis_id,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'model_outputs': result['model_outputs'],
                'model_version': result.get('model_version', 'ensemble_v1.0'),
                'processing_time': result.get('processing_time', 0.0),
                'file_name': filename,
                'file_size': len(content),
                'status': 'completed',
                # 3단계 필드 추가
                'suspected_method': result.get('suspected_method'),
                'method_confidence': result.get('method_confidence'),
                'detailed_analysis': result.get('detailed_analysis'),
                'suspicious_patterns': result.get('suspicious_patterns'),
                'time_segments': result.get('time_segments')
            }
            
            return response
            
        finally:
            # 임시 파일 삭제
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)