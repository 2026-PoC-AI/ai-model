from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.domains.audio.router import router as audio_router
from app.domains.video.router import router as video_router
from app.domains.image.router import router as image_router
from app.domains.text.router import router as text_router

router = APIRouter()
router.include_router(health_router)
router.include_router(audio_router, prefix="/audio", tags=["audio"])
# router.include_router(video_router, prefix="/video", tags=["video"])
router.include_router(image_router, prefix="/image", tags=["image"])
router.include_router(text_router, prefix="/text", tags=["text"])
