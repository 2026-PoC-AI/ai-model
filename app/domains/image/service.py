import uuid
from datetime import datetime, timezone
from fastapi import Request
from botocore.exceptions import ClientError

from app.common.exceptions import AppError
from app.common.errors import Errors
from app.core.s3 import s3_client
from app.core.config import settings
from app.domains.image.deepfake.predictor import predict_xception
from app.domains.image.spring_client import ImageSpringClient

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _make_result_key(job_uuid: str, task: str) -> str:
    # 원하는 규칙으로 바꿔도 됨
    # image/result/{YYYY}/{MM}/{DD}/{jobUuid}/{task}.json
    dt = datetime.now(timezone.utc)
    return f"image/result/{dt:%Y/%m/%d}/{job_uuid}/{task}.json"

def interpret_deepfake(confidence: float) -> dict:
    if confidence >= 0.55:
        return {
            "decision": "FAKE",
            "risk_level": "HIGH",
            "grade": "HIGH",
            "message": "AI 분석 결과, 이미지가 합성 또는 생성되었을 가능성이 높습니다."
        }

    if confidence <= 0.45:
        return {
            "decision": "REAL",
            "risk_level": "LOW",
            "grade": "LOW",
            "message": "AI 분석 결과, 이미지에서 조작 흔적이 발견되지 않았습니다."
        }

    return {
        "decision": "REVIEW_REQUIRED",
        "risk_level": "MEDIUM",
        "grade": "MEDIUM",
        "message": "AI 분석만으로는 판단이 어려워 추가 검증이 필요합니다."
    }

def analyze_image(request: Request, job_uuid: str, s3_key: str) -> dict:
    model = request.app.state.models.get("image")
    if model is None:
        raise AppError(Errors.MODEL_NOT_LOADED)

    try:
        image_bytes = s3_client.download_bytes(s3_key)
    except ClientError as e:
        raise AppError(Errors.INFERENCE_FAILED, details=f"S3 download failed: {str(e)}")

    confidence = predict_xception(image_bytes)
    decision_info = interpret_deepfake(confidence)

    result_payload = {
        "job_uuid": job_uuid,
        "task": "deepfake_image",
        "input_s3_key": s3_key,
        "output": {
            "decision": decision_info["decision"],     
            "confidence": confidence,
            "risk_level": decision_info["risk_level"],
            "grade": decision_info["grade"],
            "message": decision_info["message"],
        },
        "meta": {
            "model": "xception",
            "weight": settings.IMAGE_WEIGHT_PATH.split("/")[-1],
            "generated_at": _utc_now_iso(),
            "request_id": getattr(request.state, "request_id", None),
        }
    }

    result_s3_key = _make_result_key(job_uuid, "deepfake_image")
    try:
        s3_client.upload_json(result_s3_key, result_payload)
    except ClientError as e:
        raise AppError(Errors.INFERENCE_FAILED, details=f"S3 upload failed: {str(e)}")
    
    spring_client = ImageSpringClient()

    callback_payload = {
        "jobUuid": job_uuid,
        "decision": decision_info["decision"],
        "riskLevel": decision_info["risk_level"],
        "message": decision_info["message"],
        "confidence": confidence,
        "rawResult": result_payload,
    }

    spring_client.send_deepfake_result(callback_payload)

    return {
        "task": "deepfake_image",
        "decision": decision_info["decision"],
        "confidence": confidence,
        "risk_level": decision_info["risk_level"],
        "grade": decision_info["grade"],
        "message": decision_info["message"],
        "result_s3_key": result_s3_key,
    }

