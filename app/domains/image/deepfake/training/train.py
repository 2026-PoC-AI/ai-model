import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score

from app.domains.image.deepfake.xception_model import XceptionNet
from dataset import KoDFImageDataset, build_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = (
    "/content/drive/MyDrive/Colab Notebooks/"
    "Fake Huters - 헌터x걸즈/image/kodf"
)

REAL_DIR = f"{BASE_DIR}/real"
FAKE_DIR = f"{BASE_DIR}/fake"

SAVE_PATH = (
    "/content/drive/MyDrive/Colab Notebooks/"
    "Fake Huters - 헌터x걸즈/image/output/custom_xception.pth"
)

BATCH_SIZE = 16
EPOCHS = 10
PATIENCE = 3


def evaluate(model, loader):
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logit = model(x)
            prob = torch.sigmoid(logit).squeeze(1)

            y_true.extend(y.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())

    acc = (
        (torch.tensor(y_prob) > 0.5) ==
        torch.tensor(y_true)
    ).float().mean().item()

    auc = roc_auc_score(y_true, y_prob)
    return acc, auc


def train():
    full_ds = KoDFImageDataset(
        real_dir=REAL_DIR,
        fake_dir=FAKE_DIR,
        transform=build_transform(train=True),
    )

    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size

    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )
    
    model = XceptionNet(pretrained=True).to(DEVICE)

    for p in model.backbone.parameters():
        p.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=3e-4,
    )

    best_auc = 0.0
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.float().unsqueeze(1).to(DEVICE)

            logit = model(x)
            loss = criterion(logit, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc, auc = evaluate(model, val_loader)

        print(
            f"[Epoch {epoch + 1}] "
            f"loss={total_loss / len(train_loader):.4f} "
            f"val_acc={acc:.4f} val_auc={auc:.4f}"
        )

        if auc > best_auc:
            best_auc = auc
            wait = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"🔥 Best model saved (AUC={best_auc:.4f})")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("⏹ Early stopping")
                break


if __name__ == "__main__":
    train()