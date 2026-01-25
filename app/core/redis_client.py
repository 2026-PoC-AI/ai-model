import json
import redis
from typing import Optional, Dict, Any

class RedisClient:
    def __init__(self):
        self.client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True  # 중요: bytes를 str로 자동 변환
        )
    
    def set_progress(self, analysis_id: int, progress: int, stage: str, detail: str):
        """분석 진행률을 Redis에 저장"""
        key = f"video_analysis_progress:{analysis_id}"
        
        progress_data = {
            "progress": progress,
            "stage": stage,
            "detail": detail
        }
        
        # json.dumps 사용 (ensure_ascii=False로 한글 그대로)
        progress_json = json.dumps(progress_data, ensure_ascii=False)
        
        self.client.setex(key, 3600, progress_json)
    
    def get_progress(self, analysis_id: int) -> Optional[Dict[str, Any]]:
        """분석 진행률을 Redis에서 조회"""
        key = f"video_analysis_progress:{analysis_id}"
        data = self.client.get(key)
        
        if data:
            return json.loads(data)
        return None

redis_client = RedisClient()