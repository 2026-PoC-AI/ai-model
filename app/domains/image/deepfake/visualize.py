# app/domains/image/deepfake/visualize.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from facenet_pytorch import MTCNN

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

    def clamp(self, W: int, H: int) -> "BBox":
        x = max(0, min(self.x, W - 1))
        y = max(0, min(self.y, H - 1))
        w = max(1, min(self.w, W - x))
        h = max(1, min(self.h, H - y))
        return BBox(x, y, w, h)

    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """bytes -> BGR(H,W,3)"""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img


def encode_jpg(img_bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Failed to encode jpg")
    return buf.tobytes()


def encode_png(img_bgr_or_gray: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr_or_gray)
    if not ok:
        raise ValueError("Failed to encode png")
    return buf.tobytes()


def choose_bbox_fallback(img_bgr: np.ndarray) -> BBox:
    """MVP: 얼굴검출 없으면 '전체 이미지'를 bbox로 사용"""
    H, W = img_bgr.shape[:2]
    return BBox(0, 0, W, H)


def crop_by_bbox(img_bgr: np.ndarray, bbox: BBox) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    bb = bbox.clamp(W, H)
    return img_bgr[bb.y:bb.y + bb.h, bb.x:bb.x + bb.w].copy()


def _find_last_conv_layer(model: nn.Module) -> nn.Module:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise ValueError("No Conv2d layer found for Grad-CAM")
    return last


@torch.no_grad()
def _normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    hm = hm - hm.min()
    denom = (hm.max() + 1e-8)
    hm = hm / denom
    return hm


def gradcam_heatmap(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: Optional[nn.Module] = None,
) -> np.ndarray:

    model.eval()

    if target_layer is None:
        target_layer = _find_last_conv_layer(model)

    activations = None
    gradients = None

    def fwd_hook(_, __, output):
        nonlocal activations
        activations = output

    def bwd_hook(_, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    # forward
    logits = model(input_tensor)

    # (1) forward 직후 디버깅
    print("[GradCAM] input requires_grad:", input_tensor.requires_grad)
    print("[GradCAM] input shape:", tuple(input_tensor.shape))
    try:
        print("[GradCAM] logits shape:", tuple(logits.shape))
        print("[GradCAM] logits sample:", logits.detach().flatten()[:5].cpu().numpy())
    except Exception as e:
        print("[GradCAM] logits debug failed:", e)

    print("[GradCAM] target_layer:", target_layer)
    print("[GradCAM] target_layer type:", type(target_layer))

    # score 잡기
    if hasattr(logits, "ndim") and logits.ndim > 1:
        score = logits[:, 0].sum()
    else:
        score = logits.squeeze()

    model.zero_grad(set_to_none=True)

    # backward ( 1번만)
    score.backward()

    print("[GradCAM] activations captured:", activations is not None)
    print("[GradCAM] gradients captured:", gradients is not None)
    if activations is not None:
        try:
            print("[GradCAM] activations shape:", tuple(activations.shape))
        except Exception as e:
            print("[GradCAM] activations shape debug failed:", e)
    if gradients is not None:
        try:
            print("[GradCAM] gradients shape:", tuple(gradients.shape))
        except Exception as e:
            print("[GradCAM] gradients shape debug failed:", e)

    h1.remove()
    h2.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM hooks failed")

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1)
    cam = F.relu(cam)

    cam_np = cam.detach().cpu().numpy()[0]
    cam_np = _normalize_heatmap(cam_np)

    H, W = input_tensor.shape[2:]
    cam_np = cv2.resize(cam_np, (W, H))

    return cam_np.astype(np.float32)

def overlay_heatmap_on_bgr(
    base_bgr: np.ndarray,
    heatmap_01: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    base_bgr: (H,W,3) 원본 이미지
    heatmap_01: (h,w) [0,1] Grad-CAM 결과
    """

    H, W = base_bgr.shape[:2]

    # (1) heatmap을 원본 이미지 크기로 리사이즈
    if heatmap_01.shape[0] != H or heatmap_01.shape[1] != W:
        heatmap_01 = cv2.resize(
            heatmap_01,
            (W, H),
            interpolation=cv2.INTER_LINEAR
        )

    # (2) uint8 변환
    hm_u8 = np.clip(heatmap_01 * 255, 0, 255).astype(np.uint8)

    # (3) 컬러맵
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)

    # (4) overlay
    out = cv2.addWeighted(base_bgr, 1.0, hm_color, alpha, 0)

    return out


def draw_bbox(img_bgr: np.ndarray, bbox: BBox) -> np.ndarray:
    out = img_bgr.copy()
    H, W = out.shape[:2]
    bb = bbox.clamp(W, H)
    cv2.rectangle(out, (bb.x, bb.y), (bb.x + bb.w, bb.y + bb.h), (0, 255, 0), 2)
    return out

# --- Face detection (MTCNN) ---
_mtcnn = None

def _get_mtcnn(device: torch.device):
    global _mtcnn
    if _mtcnn is None:
        # keep_all=True => 여러 얼굴 박스 반환
        _mtcnn = MTCNN(keep_all=True, device=device)
    return _mtcnn

def detect_face_bboxes_mtcnn(img_bgr: np.ndarray, device: torch.device, min_conf: float = 0.90) -> list[BBox]:
    """
    returns: List[BBox] (confidence 필터 적용)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mtcnn = _get_mtcnn(device)
    boxes, probs = mtcnn.detect(img_rgb)

    if boxes is None or probs is None:
        return []

    out: list[BBox] = []
    for (x1, y1, x2, y2), p in zip(boxes, probs):
        if p is None or float(p) < min_conf:
            continue
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
        out.append(BBox(
            x=int(x1),
            y=int(y1),
            w=int(x2 - x1),
            h=int(y2 - y1),
        ))
    return out