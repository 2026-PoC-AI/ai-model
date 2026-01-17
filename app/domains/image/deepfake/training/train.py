import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from app.domains.image.deepfake.xception_model import XceptionNet
from app.domains.image.deepfake.training.dataset import (
    FFPPImageDataset,
    build_transform,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "/content/drive/MyDrive/Colab Notebooks/Fake Huters - 헌터x걸즈/image/ffpp/train"
VAL_DIR   = "/content/drive/MyDrive/Colab Notebooks/Fake Huters - 헌터x걸즈/image/ffpp/val"
SAVE_PATH = "/content/drive/MyDrive/Colab Notebooks/Fake Huters - 헌터x걸즈/image/output/custom_xception.pth"


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

    acc = ((torch.tensor(y_prob) > 0.5) == torch.tensor(y_true)).float().mean().item()
    auc = roc_auc_score(y_true, y_prob)

    return acc, auc


def train():
    train_ds = FFPPImageDataset(TRAIN_DIR, build_transform(True))
    val_ds   = FFPPImageDataset(VAL_DIR, build_transform(False))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = XceptionNet(pretrained=True).to(DEVICE)

    for p in model.backbone.parameters():
        p.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=3e-4
    )

    best_auc = 0.0
    patience, wait = 3, 0

    for epoch in range(10):
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
            f"[Epoch {epoch+1}] "
            f"loss={total_loss/len(train_loader):.4f} "
            f"val_acc={acc:.4f} val_auc={auc:.4f}"
        )

        if auc > best_auc:
            best_auc = auc
            wait = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"🔥 Best model saved (AUC={best_auc:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print("⏹ Early stopping")
                break


if __name__ == "__main__":
    train()
