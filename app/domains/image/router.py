# app/domains/image/router.py
from fastapi import APIRouter, UploadFile, File, Request
from app.common.responses import ok, ApiResponse
from app.domains.image.schemas import ImageAnalyzeResponse
from app.domains.image.service import analyze_image
from app.core.s3 import s3_client

router = APIRouter()

@router.post("/analyze", response_model=ApiResponse[ImageAnalyzeResponse])
async def analyze(request: Request, file: UploadFile = File(...)):
    rid = request.state.request_id
    content = await file.read()

    result = analyze_image(request, content)
    return ok(result, request_id=rid)

@router.post("/test-s3")
async def test_s3():
    s3_client.upload_bytes(
        key="test/api/from-fastapi.txt",
        data=b"uploaded from api",
        content_type="text/plain",
    )
    return {"ok": True}