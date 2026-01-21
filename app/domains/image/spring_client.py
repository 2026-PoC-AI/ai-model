import requests
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class ImageSpringClient:
    """
    Image 도메인 전용 Spring Backend Client
    (딥페이크 분석 결과 전달)
    """

    def __init__(self):
        self.base_url = settings.SPRING_BACKEND_URL.rstrip("/")

    def send_deepfake_result(self, payload: dict) -> None:
        url = f"{self.base_url}/api/v1/images/results/deepfake"

        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=5,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(
                f"[ImageSpringClient] Failed to send result to Spring: {e}"
            )
