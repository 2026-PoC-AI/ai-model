import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score

from dataset import KoDFFrameDataset, build_transform
from xception_model import XceptionNet

# =====================
# CONFIG
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REAL_ROOT = "/content/drive/MyDrive/Colab Notebooks/Fake Huters - 헌터x걸즈/image/KoDF/real_frames"
FAKE_ROOT = "/content/drive/MyDrive/Colab Notebooks/Fake Huters - 헌터x걸즈/image/KoDF/fake_frames"

SAVE_PATH = "/content/drive/MyDrive/Colab Notebooks/Fake Huters - 헌터x걸즈/image/output/custom_xception.pth"

MAX_PER_CLASS = 5000
BATCH_SIZE = 32
EPOCHS = 2
NUM_WORKERS = 2


def evaluate(model, loader):
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            prob = torch.sigmoid(model(x)).squeeze(1)
            y_true.extend(y.tolist())
            y_prob.extend(prob.tolist())

    return roc_auc_score(y_true, y_prob)


def train():
    full_ds = KoDFFrameDataset(
        REAL_ROOT,
        FAKE_ROOT,
        transform=None,
        max_per_class=MAX_PER_CLASS,
    )

    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_ds.dataset.transform = build_transform(train=True)
    val_ds.dataset.transform = build_transform(train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = XceptionNet(pretrained=True).to(DEVICE)
    print("🚀 Model initialized")

    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for i, (x, y) in enumerate(train_loader):
            if epoch == 0 and i == 0:
                print("✅ First batch loaded")

            x = x.to(DEVICE, non_blocking=True)
            y = y.float().unsqueeze(1).to(DEVICE)

            loss = criterion(model(x), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_auc = evaluate(model, val_loader)

        print(
            f"[Epoch {epoch+1}] "
            f"loss={total_loss/len(train_loader):.4f} "
            f"val_auc={val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"🔥 Best model saved (AUC={best_auc:.4f})")

    print("✅ Training completed")


if __name__ == "__main__":
    train()
