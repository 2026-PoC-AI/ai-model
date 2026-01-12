from fastapi import APIRouter, Request
from app.common.responses import ok, ApiResponse

router = APIRouter()

@router.get("/health", response_model=ApiResponse[dict])
def health(request: Request):
    rid = request.state.request_id
    models = getattr(request.app.state, "models", {})
    status = {k: (models.get(k) is not None) for k in ["audio", "video", "image", "text"]}
    return ok({"status": "ok", "models": status}, request_id=rid)
