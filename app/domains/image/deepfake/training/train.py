import torch
from torch import nn
from torch.utils.data import DataLoader
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
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            prob = torch.sigmoid(model(x))
            pred = (prob > 0.5).long().squeeze(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


def train():
    train_ds = FFPPImageDataset(
        TRAIN_DIR,
        transform=build_transform(train=True),
    )
    val_ds = FFPPImageDataset(
        VAL_DIR,
        transform=build_transform(train=False),
    )

    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False
    )

    model = XceptionNet(pretrained=True).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_acc = 0.0
    patience = 3
    wait = 0
    MAX_EPOCHS = 30

    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.float().unsqueeze(1).to(DEVICE)

            logit = model(x)
            loss = criterion(logit, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        acc = evaluate(model, val_loader)

        print(
            f"[Epoch {epoch+1}] "
            f"loss={running_loss/len(train_loader):.4f} "
            f"val_acc={acc:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            wait = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"🔥 Best model saved (acc={best_acc:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print("⏹ Early stopping triggered")
                break


if __name__ == "__main__":
    train()
