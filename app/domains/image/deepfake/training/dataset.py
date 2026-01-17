# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXT = (".jpg", ".jpeg", ".png")


class KoDFImageDataset(Dataset):

    def __init__(self, real_dir: str, fake_dir: str, transform=None):
        self.samples = []
        self.transform = transform

        # REAL = 0
        for root, _, files in os.walk(real_dir):
            for fname in files:
                if fname.lower().endswith(IMG_EXT):
                    self.samples.append(
                        (os.path.join(root, fname), 0)
                    )

        # FAKE = 1
        for root, _, files in os.walk(fake_dir):
            for fname in files:
                if fname.lower().endswith(IMG_EXT):
                    self.samples.append(
                        (os.path.join(root, fname), 1)
                    )

        if len(self.samples) == 0:
            raise RuntimeError("No images found in KoDF dataset")

        print(f"[KoDFDataset] Loaded {len(self.samples)} images")

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
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
            ),
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
