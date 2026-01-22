# app/domains/image/service.py
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch
from fastapi import Request
from botocore.exceptions import ClientError

from app.common.exceptions import AppError
from app.common.errors import Errors
from app.core.s3 import s3_client
from app.core.config import settings

from app.domains.image.deepfake.predictor import predict_xception, load_xception_model
from app.domains.image.deepfake.preprocess import preprocess_image
from app.domains.image.spring_client import ImageSpringClient

from app.domains.image.deepfake.visualize import (
    BBox,
    decode_image_bytes,
    encode_jpg,
    encode_png,
    choose_bbox_fallback,
    crop_by_bbox,
    gradcam_heatmap,
    overlay_heatmap_on_bgr,
    draw_bbox,
    detect_face_bboxes_mtcnn, 
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_result_key(job_uuid: str, task: str) -> str:
    # image/result/{YYYY}/{MM}/{DD}/{jobUuid}/{task}.json
    dt = datetime.now(timezone.utc)
    return f"image/result/{dt:%Y/%m/%d}/{job_uuid}/{task}.json"


def _make_artifact_key(job_uuid: str, artifact_type: str, ext: str) -> str:
    # image/processed/{YYYY}/{MM}/{DD}/{jobUuid}/{artifact_type}.{ext}
    dt = datetime.now(timezone.utc)
    return f"image/processed/{dt:%Y/%m/%d}/{job_uuid}/{artifact_type}.{ext}"


def _get_device() -> torch.device:
    if getattr(settings, "USE_GPU", False) and torch.cuda.is_available():
        gpu_id = getattr(settings, "GPU_DEVICE_ID", 0)
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


_DEVICE = _get_device()


def interpret_deepfake(confidence: float) -> dict:
    # (임시 룰) 나중에 모델/캘리브레이션에 맞춰 조정 가능
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
    model_loaded = request.app.state.models.get("image")
    if model_loaded is None:
        raise AppError(Errors.MODEL_NOT_LOADED)

    # 1) S3에서 원본 이미지 다운로드
    try:
        image_bytes = s3_client.download_bytes(s3_key)
    except ClientError as e:
        raise AppError(Errors.INFERENCE_FAILED, details=f"S3 download failed: {str(e)}")

    # 2) 딥페이크 예측 (확률)
    confidence = predict_xception(image_bytes)
    decision_info = interpret_deepfake(confidence)

    # 3) 결과 JSON S3 업로드 (원본 결과 보관)
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
            "weight": getattr(settings, "IMAGE_WEIGHT_PATH", "").split("/")[-1],
            "generated_at": _utc_now_iso(),
            "request_id": getattr(request.state, "request_id", None),
        }
    }

    result_s3_key = _make_result_key(job_uuid, "deepfake_image")
    try:
        s3_client.upload_json(result_s3_key, result_payload)
    except ClientError as e:
        raise AppError(Errors.INFERENCE_FAILED, details=f"S3 upload failed: {str(e)}")

        
    artifacts: List[Dict[str, Any]] = []
    artifact_warnings: List[str] = []

    try:
        img_bgr = decode_image_bytes(image_bytes)

        # 0) 얼굴 검출 (multi-face)
        faces = detect_face_bboxes_mtcnn(img_bgr, _DEVICE, min_conf=0.90)

        # fallback
        if not faces:
            bbox = choose_bbox_fallback(img_bgr)
            faces = [bbox]
            artifact_warnings.append("face_detect_failed: fallback_to_full_image")

    except Exception as e:
        # decode 자체가 실패하면 이후 전부 의미 없음
        artifact_warnings.append(f"decode_failed: {e}")
        faces = []


    # face crop
    for i, bbox in enumerate(faces):
        # (1) bbox overlay 이미지
        try:
            bbox_bgr = draw_bbox(img_bgr, bbox)
            bbox_jpg = encode_jpg(bbox_bgr)
            bbox_key = _make_artifact_key(job_uuid, f"bbox_{i}", "jpg")
            s3_client.upload_bytes(bbox_key, bbox_jpg, "image/jpeg")

            artifacts.append({
                "artifactStage": "processed",
                "artifactType": "bbox",
                "s3Key": bbox_key,
                "meta": {"bbox": bbox.to_dict(), "index": i}
            })
        except Exception as e:
            artifact_warnings.append(f"bbox_failed_{i}: {e}")

        # (2) face crop
        try:
            crop_bgr = crop_by_bbox(img_bgr, bbox)
            crop_jpg = encode_jpg(crop_bgr)
            face_crop_key = _make_artifact_key(job_uuid, f"face_crop_{i}", "jpg")
            s3_client.upload_bytes(face_crop_key, crop_jpg, "image/jpeg")

            artifacts.append({
                "artifactStage": "processed",
                "artifactType": "face_crop",
                "s3Key": face_crop_key,
                "meta": {"bbox": bbox.to_dict(), "index": i}
            })
        except Exception as e:
            artifact_warnings.append(f"face_crop_failed_{i}: {e}")

        # (3) face별 Grad-CAM (face crop 기준으로 heatmap 만들고, crop 위에 overlay)
        try:
            # crop을 bytes로 다시 encode해서 preprocess에 넣는 방식(가장 간단/안전)
            crop_bgr = crop_by_bbox(img_bgr, bbox)
            crop_jpg = encode_jpg(crop_bgr)  # bytes

            x = preprocess_image(crop_jpg).to(_DEVICE)
            x = x.clone().detach().requires_grad_(True)

            model = load_xception_model()
            hm = gradcam_heatmap(model, x)  # (H,W) = preprocess size

            # overlay는 "crop 이미지" 위에 해야 shape mismatch가 없음
            overlay_bgr = overlay_heatmap_on_bgr(crop_bgr, hm, alpha=0.45)
            overlay_png = encode_png(overlay_bgr)

            heatmap_key = _make_artifact_key(job_uuid, f"heatmap_{i}", "png")
            s3_client.upload_bytes(heatmap_key, overlay_png, "image/png")

            artifacts.append({
                "artifactStage": "result",
                "artifactType": "heatmap",
                "s3Key": heatmap_key,
                "meta": {"method": "gradcam", "bbox": bbox.to_dict(), "index": i}
            })
        except Exception as e:
            artifact_warnings.append(f"heatmap_failed_{i}: {e}")


    # 5) Spring 콜백 (결과 저장)
    spring_client = ImageSpringClient()

    callback_payload = {
        "jobUuid": job_uuid,
        "decision": decision_info["decision"],
        "riskLevel": decision_info["risk_level"],
        "message": decision_info["message"],
        "confidence": confidence,
        "rawResult": result_payload,      # 원본 결과 JSON
        "resultS3Key": result_s3_key,     # 결과 JSON S3 key
        "artifacts": artifacts,           # bbox/face_crop/heatmap
        "warnings": artifact_warnings,    # 시각화 실패 시 경고
    }

    spring_client.send_deepfake_result(callback_payload)

    # 6) FastAPI 응답
    return {
        "task": "deepfake_image",
        "decision": decision_info["decision"],
        "confidence": confidence,
        "risk_level": decision_info["risk_level"],
        "grade": decision_info["grade"],
        "message": decision_info["message"],
        "result_s3_key": result_s3_key,
        "artifacts": artifacts,
        "warnings": artifact_warnings,
    }
