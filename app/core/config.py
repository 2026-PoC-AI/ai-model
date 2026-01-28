# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", populate_by_name=True)

    APP_NAME: str
    APP_ENV: str
    LOG_LEVEL: str
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    SPRING_BACKEND_URL: str

    # Video Deepfake Detection Models (3-Model Ensemble)
    XCEPTION_MODEL_PATH: str = "app/domains/video/deepfake_detection/weights/xception/xception_best_20260116.pth"
    EFFICIENTNET_MODEL_PATH: str = "app/domains/video/deepfake_detection/weights/efficientnet/efficientnet_best.pth"
    CNN_LSTM_MODEL_PATH: str = "app/domains/video/deepfake_detection/weights/cnn-lstm/improved/best_model_latest.pth" 

    # Audio Deepfake Detection Model
    AUDIO_MODEL_PATH: str = "app/domains/audio/deepfake_detection/weights/audio_cnn/best_model_latest.pth"
    
    # Image Deepfake Detection Model
    IMAGE_MODEL_PATH: str = "app/domains/image/deepfake/weights/custom_xception.pth"
    IMAGE_WEIGHT_PATH: str = "app/domains/image/deepfake/weights/custom_xception.pth"  # 호환성 유지
    
    # Text Deepfake Detection Model
    TEXT_MODEL_PATH: str = "app/domains/text/fake_news/weights/model_state_dict.pth"

    # GPU 설정
    USE_GPU: bool = False
    GPU_DEVICE_ID: int = 0

    # File Size Limits
    MAX_AUDIO_MB: int = 50
    MAX_VIDEO_MB: int = 300
    MAX_IMAGE_MB: int = 20
    MAX_TEXT_LENGTH: int = 10000
    MIN_AUDIO_DURATION: float = 3.0
    
    # AWS
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    AWS_BUCKET_NAME: str
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Naver API
    naver_client_id: str = Field(default="", alias="NAVER_CLIENT_ID")
    naver_client_secret: str = Field(default="", alias="NAVER_CLIENT_SECRET")

settings = Settings()