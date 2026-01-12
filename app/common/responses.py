# app/common/responses.py
from typing import Any, Generic, Optional, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[dict] = None
    request_id: Optional[str] = None

def ok(data: Any = None, request_id: str | None = None) -> ApiResponse[Any]:
    return ApiResponse(success=True, data=data, error=None, request_id=request_id)

def fail(code: str, message: str, http_status: int, request_id: str | None, details: Any = None) -> ApiResponse[Any]:
    return ApiResponse(
        success=False,
        data=None,
        request_id=request_id,
        error={
            "code": code,
            "message": message,
            "http_status": http_status,
            "details": details,
        },
    )
