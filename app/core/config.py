# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    APP_NAME: str
    APP_ENV: str
    LOG_LEVEL: str
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    SPRING_BACKEND_URL: str

    AUDIO_MODEL_PATH: str
    VIDEO_MODEL_PATH: str
    IMAGE_MODEL_PATH: str
    TEXT_MODEL_PATH: str
    
    # Video Deepfake Detection Models
    XCEPTION_MODEL_PATH: str = "app/domains/video/deepfake_detection/weights/xception/xception_best_20260116.pth"
    EFFICIENTNET_MODEL_PATH: str = "app/domains/video/deepfake_detection/weights/efficientnet/efficientnet_best_20260116.pth"

    # Image Deepfake Detection Model
    IMAGE_WEIGHT_PATH: str = "app/domains/image/deepfake/weights/custom_xception.pth"

    USE_GPU: bool = False
    GPU_DEVICE_ID: int = 0

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

settings = Settings()