import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict
import tempfile
from pathlib import Path
import sys

# 프로젝트 루트를 path에 추가
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from inference.ensemble_predictor import get_ensemble_predictor

# FastAPI 앱
app = FastAPI(
    title="Audio Deepfake Detection API",
    description="Mel-spectrogram + LFCC 앙상블 기반 음성 딥페이크 탐지",
    version="1.0.0"
)

# 앙상블 예측기 초기화
print("Initializing ensemble predictor...")
ensemble = None

try:
    ensemble = get_ensemble_predictor(
        weights_dir='../weights',
        device='cpu',
        ensemble_method='weighted_avg'
    )
    print("Ensemble predictor ready!")
except Exception as e:
    print(f"Failed to initialize ensemble: {e}")


class PredictionResponse(BaseModel):
    """
    예측 응답 스키마
    """
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_outputs: Dict[str, Dict[str, float]]
    processing_time: Optional[float] = None


@app.get("/")
async def root():
    """
    API 루트
    """
    return {
        "message": "Audio Deepfake Detection API",
        "version": "1.0.0",
        "status": "ready" if ensemble else "not ready"
    }


@app.get("/health")
async def health_check():
    """
    헬스 체크
    """
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "status": "healthy",
        "models": {
            "mel_cnn": "loaded",
            "lfcc_cnn": "loaded"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    """
    음성 딥페이크 예측
    
    Args:
        file: 오디오 파일 (.wav, .flac, .mp3 등)
        
    Returns:
        prediction: 'real' or 'fake'
        confidence: 신뢰도 (0~1)
        probabilities: {'real': float, 'fake': float}
        model_outputs: 각 모델의 개별 예측
    """
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # 파일 확장자 확인
    allowed_extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {allowed_extensions}"
        )
    
    # 임시 파일 저장
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # 예측
        import time
        start_time = time.time()
        
        result = ensemble.predict(temp_path)
        
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        
        # 임시 파일 삭제
        os.unlink(temp_path)
        
        return PredictionResponse(**result)
    
    except Exception as e:
        # 에러 시 임시 파일 정리
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_audio_batch(files: list[UploadFile] = File(...)):
    """
    배치 예측
    
    여러 오디오 파일을 한 번에 예측
    """
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    temp_paths = []
    
    try:
        # 모든 파일 임시 저장
        for file in files:
            file_ext = Path(file.filename).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_paths.append(temp_file.name)
        
        # 배치 예측
        results = ensemble.batch_predict(temp_paths)
        
        # 파일명 추가
        for i, result in enumerate(results):
            result['filename'] = files[i].filename
        
        return JSONResponse(content={"results": results})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    finally:
        # 임시 파일 정리
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@app.get("/models/info")
async def get_models_info():
    """
    모델 정보 조회
    """
    return {
        "ensemble_method": "weighted_average",
        "models": [
            {
                "name": "Mel-spectrogram CNN",
                "type": "lightweight_cnn",
                "validation_accuracy": 99.15,
                "weight": 0.49,
                "features": "Mel-scale frequency representation"
            },
            {
                "name": "LFCC CNN",
                "type": "lightweight_lfcc",
                "validation_accuracy": 99.57,
                "weight": 0.51,
                "features": "Linear frequency cepstral coefficients"
            }
        ],
        "expected_ensemble_accuracy": "99.6-99.8%"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*50)
    print("Starting Audio Deepfake Detection API")
    print("="*50)
    print("Ensemble: Mel-spectrogram CNN + LFCC CNN")
    print("Expected Accuracy: 99.6-99.8%")
    print("="*50 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )