from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from app.common.responses import ok, ApiResponse
from app.domains.image.service import analyze_image

router = APIRouter()

class ImageAnalyzeRequest(BaseModel):
    job_uuid: str = Field(..., description="job uuid from Spring")
    s3_key: str = Field(..., description="S3 object key for image file")

@router.post("/analyze", response_model=ApiResponse[dict])
async def analyze(request: Request, body: ImageAnalyzeRequest):
    rid = request.state.request_id
    result = analyze_image(
        request=request,
        job_uuid=body.job_uuid,
        s3_key=body.s3_key,
    )
    return ok(result, request_id=rid)