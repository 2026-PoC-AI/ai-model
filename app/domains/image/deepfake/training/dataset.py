# dataset.py
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXT = (".jpg", ".jpeg", ".png")


class KoDFFrameDataset(Dataset):
    def __init__(
        self,
        real_root: str,
        fake_root: str,
        transform=None,
        max_per_class: int = 2000,
        seed: int = 42,
    ):
        self.transform = transform
        random.seed(seed)

        real_images = [p for p in Path(real_root).rglob("*.jpg")]
        fake_images = [p for p in Path(fake_root).rglob("*.jpg")]

        print(f"[RAW] real={len(real_images)}, fake={len(fake_images)}")

        if max_per_class:
            real_images = random.sample(real_images, min(max_per_class, len(real_images)))
            fake_images = random.sample(fake_images, min(max_per_class, len(fake_images)))

        self.samples = [(p, 0) for p in real_images] + [(p, 1) for p in fake_images]
        random.shuffle(self.samples)

        print(f"[USED] real={len(real_images)}, fake={len(fake_images)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_transform(train: bool = True):
    if train:
        return T.Compose([
            T.Resize((299, 299)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
