# app/domains/image/deepfake/preprocess.py
import io
from PIL import Image
import torch
import torchvision.transforms as T

_transform = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(image)
    return tensor.unsqueeze(0)  # (1, C, H, W)