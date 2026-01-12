# app/common/errors.py
from dataclasses import dataclass

@dataclass(frozen=True)
class ErrorDef:
    code: str
    message: str
    http_status: int

class Errors:
    INTERNAL = ErrorDef("COMMON_INTERNAL_ERROR", "Internal server error.", 500)
    VALIDATION = ErrorDef("COMMON_VALIDATION_ERROR", "Validation error.", 422)

    MODEL_NOT_LOADED = ErrorDef("MODEL_NOT_LOADED", "Model is not loaded.", 503)
    INFERENCE_FAILED = ErrorDef("INFERENCE_FAILED", "Inference failed.", 500)
