from .base import DeepfakeDetector
from .xception import XceptionNet
from .efficientnet import EfficientNetB4
from .ensemble import EnsembleModel

__all__ = [
    'DeepfakeDetector',
    'XceptionNet', 
    'EfficientNetB4',
    'EnsembleModel'
]

print("Model state_dict keys (first 5):")
for key in list(self.xception.state_dict().keys())[:5]:
    print(f"  Model: {key}")

print("Checkpoint state_dict keys (first 5):")
for key in list(state_dict.keys())[:5]:
    print(f"  Checkpoint: {key}")