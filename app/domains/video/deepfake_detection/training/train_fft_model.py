import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import sys
from datetime import datetime

# metrics import 시도
try:
    from .metrics import calculate_metrics
except ImportError:
    try:
        from metrics import calculate_metrics
    except ImportError:
        # sklearn으로 대체
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        
        def calculate_metrics(y_true, y_pred):
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred, average='binary'),
                'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
                'auc': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
            }

class FFTDeepfakeDataset(Dataset):
    """FFT 전처리된 데이터셋 - 배치 파일을 on-demand로 로드"""
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # 배치 파일 목록
        self.batch_files = sorted(self.data_dir.glob(f'{split}_batch_*.npz'))
        
        if not self.batch_files:
            raise ValueError(f"{split} 배치 파일을 찾을 수 없습니다: {data_dir}")
        
        print(f"{split} 데이터 초기화 중... ({len(self.batch_files)}개 배치)")
        
        # 각 배치의 샘플 인덱스 매핑
        self.batch_indices = []
        self.cumulative_sizes = [0]
        
        for batch_file in self.batch_files:
            batch = np.load(batch_file)
            batch_size = len(batch['labels'])
            self.batch_indices.append(batch_file)
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + batch_size)
        
        self.total_size = self.cumulative_sizes[-1]
        
        print(f"{split} 데이터 초기화 완료: {self.total_size}개 샘플")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        # 어느 배치에 속하는지 찾기
        batch_idx = 0
        for i in range(len(self.cumulative_sizes) - 1):
            if self.cumulative_sizes[i] <= idx < self.cumulative_sizes[i + 1]:
                batch_idx = i
                break
        
        # 배치 내 인덱스
        local_idx = idx - self.cumulative_sizes[batch_idx]
        
        # 배치 파일 로드
        batch_file = self.batch_indices[batch_idx]
        batch = np.load(batch_file)
        
        image = batch['data'][local_idx]
        label = batch['labels'][local_idx]
        
        # 정규화
        image = image.astype(np.float32) / 255.0
        
        # 채널 순서 변경 (H, W, C) -> (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)


class FFTClassifier(nn.Module):
    """FFT 스펙트럼 기반 딥페이크 분류기"""
    def __init__(self, num_classes=2, dropout=0.5):
        super(FFTClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 14 * 14, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        import torch.nn.functional as F
        
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """1 에포크 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        if torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    # 메트릭 계산
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    
    return epoch_loss, epoch_acc, metrics


def validate(model, val_loader, criterion, device):
    """검증"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    # 메트릭 계산
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    
    return epoch_loss, epoch_acc, metrics


def plot_training_history(history, save_path):
    """학습 히스토리 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(history['train_f1'], label='Train F1', marker='o')
    axes[1, 0].plot(history['val_f1'], label='Val F1', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Training and Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # AUC
    axes[1, 1].plot(history['train_auc'], label='Train AUC', marker='o')
    axes[1, 1].plot(history['val_auc'], label='Val AUC', marker='s')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].set_title('Training and Validation AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"학습 히스토리 저장: {save_path}")


def main():
    # 하이퍼파라미터
    config = {
        'batch_size': 64,  # 배치 크기 늘리기
        'num_epochs': 5,   # 에포크 줄이기
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'dropout': 0.5,
        'num_workers': 0,
        'use_subset': True,  # 추가
        'subset_ratio': 0.1  # Train 데이터의 10%만 사용
    }
    
    # 경로 설정
    current_file = Path(__file__)
    data_root = current_file.parent.parent.parent / 'data'
    fft_data_path = data_root / 'fft_processed'
    weights_dir = current_file.parent.parent / 'weights' / 'fft'
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n사용 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "="*60)
    print("FFT 모델 학습 시작")
    print("="*60)
    print(f"설정: {json.dumps(config, indent=2)}")
    
    # 데이터셋 로드
    print("\n데이터셋 로드 중...")
    train_dataset = FFTDeepfakeDataset(fft_data_path, split='train')
    val_dataset = FFTDeepfakeDataset(fft_data_path, split='val')
    test_dataset = FFTDeepfakeDataset(fft_data_path, split='test')
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nDataLoader 준비 완료")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # 모델 생성
    model = FFTClassifier(num_classes=2, dropout=config['dropout']).to(device)
    
    # 파라미터 수 확인
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n모델 파라미터:")
    print(f"  총 파라미터: {total_params:,}")
    print(f"  학습 가능 파라미터: {trainable_params:,}")
    
    # 옵티마이저 및 스케줄러
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 학습 히스토리
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []
    }
    
    best_val_f1 = 0.0
    best_epoch = 0
    
    # 학습 시작
    print("\n" + "="*60)
    print("학습 시작")
    print("="*60)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch [{epoch+1}/{config['num_epochs']}]")
        
        # 학습
        train_loss, train_acc, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        # 검증
        val_loss, val_acc, val_metrics = validate(
            model, val_loader, criterion, device
        )
        
        # 스케줄러 업데이트
        scheduler.step(val_loss)
        
        # 히스토리 저장
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_metrics['f1'])
        history['train_auc'].append(train_metrics['auc'])
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        # 결과 출력
        print(f"\nTrain - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
                f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
                f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        # 최고 성능 모델 저장
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            
            checkpoint_path = weights_dir / 'fft_best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['f1'],
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"✓ 최고 성능 모델 저장! (Val F1: {val_metrics['f1']:.4f})")
    
    print("\n" + "="*60)
    print("학습 완료!")
    print("="*60)
    print(f"최고 성능: Epoch {best_epoch}, Val F1: {best_val_f1:.4f}")
    
    # 학습 히스토리 시각화
    plot_path = weights_dir / 'training_history.png'
    plot_training_history(history, plot_path)
    
    # 히스토리 JSON 저장
    history_path = weights_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"학습 히스토리 저장: {history_path}")
    
    # 최고 모델로 테스트
    print("\n" + "="*60)
    print("테스트 데이터 평가")
    print("="*60)
    
    checkpoint = torch.load(weights_dir / 'fft_best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"\nTest 결과:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    
    # 테스트 결과 저장
    test_results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_metrics': test_metrics,
        'best_epoch': best_epoch,
        'config': config
    }
    
    results_path = weights_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n테스트 결과 저장: {results_path}")


if __name__ == '__main__':
    main()