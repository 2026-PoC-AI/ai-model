from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    APP_NAME: str
    APP_ENV: str
    LOG_LEVEL: str

    AUDIO_MODEL_PATH: str
    VIDEO_MODEL_PATH: str
    IMAGE_MODEL_PATH: str
    TEXT_MODEL_PATH: str

    USE_GPU: bool = False
    GPU_DEVICE_ID: int = 0

    MAX_AUDIO_MB: int = 50
    MAX_VIDEO_MB: int = 300
    MAX_IMAGE_MB: int = 20
    MAX_TEXT_LENGTH: int = 10000

settings = Settings()
