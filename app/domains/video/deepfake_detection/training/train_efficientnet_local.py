import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import json
from tqdm import tqdm
from pathlib import Path
import time

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        real_dir = os.path.join(data_dir, split, 'real')
        fake_dir = os.path.join(data_dir, split, 'fake')
        
        self.real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.jpg')]
        self.fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.jpg')]
        
        self.images = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        
        print(f"{split} dataset: {len(self.real_images)} real, {len(self.fake_images)} fake")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class EfficientNetB4(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super(EfficientNetB4, self).__init__()
        
        print(f"Initializing EfficientNet-B4 (pretrained={pretrained})...")
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained)
        
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, save_dir='./weights/efficientnet'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_auc': []
        }
        self.best_val_acc = 0.0
        self.start_time = time.time()
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
        
        return epoch_loss, metrics
    
    def save_checkpoint(self, filename, epoch, metrics):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Saved: {filename}")
    
    def train(self, num_epochs=20, patience=10):
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}")
        print(f"Patience: {patience}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print("-" * 60)
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_metrics = self.validate()
            
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics['auc'])
            
            elapsed = time.time() - self.start_time
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            print(f"Val AUC: {val_metrics['auc']:.4f}")
            print(f"Confusion Matrix:\n{val_metrics['confusion_matrix']}")
            print(f"Elapsed time: {elapsed/60:.1f} min")
            
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint('efficientnet_best.pth', epoch, val_metrics)
                print(f"New Best Model! Accuracy: {self.best_val_acc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{patience}")
            
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'efficientnet_epoch_{epoch+1}.pth', epoch, val_metrics)
            
            if patience_counter >= patience:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break
        
        with open(os.path.join(self.save_dir, 'efficientnet_training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Best Val Accuracy: {self.best_val_acc:.4f}")
        print(f"Total time: {total_time/3600:.2f} hours")

if __name__ == '__main__':
    # 데이터 경로 수정
    data_dir = '../data/processed_efficientnet'
    
    # 데이터셋 생성
    train_dataset = DeepfakeDataset(data_dir, 'train', get_transforms('train'))
    val_dataset = DeepfakeDataset(data_dir, 'val', get_transforms('val'))
    
    # 데이터 로더 (CPU면 num_workers=0, GPU면 4)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # 모델 초기화 (pretrained=True)
    model = EfficientNetB4(num_classes=2, pretrained=True, dropout=0.5).to(device)
    print("Model initialized with pretrained weights\n")
    
    # Trainer 초기화 및 학습
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.train(num_epochs=20, patience=10)