#  CNN-LSTM - 비디오 시퀀스
import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from datetime import datetime
import os

from models.cnn_lstm import get_model
from utils.data_loader import get_data_loaders

class VideoTrainer:
    """
    비디오 시퀀스 기반 딥페이크 탐지 모델 학습 클래스
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*50}")
        print(f"Video Trainer Initialization")
        print(f"{'='*50}")
        print(f"Device: {self.device}")
        print(f"Model: {config['model_name']}")
        
        # 경로 설정
        project_root = Path(__file__).parent.parent
        self.data_dir = project_root.parent / 'data' / 'processed' / 'frames_16'
        self.splits_file = project_root.parent / 'data' / 'splits' / 'splits.json'
        
        # DataLoader 생성
        print("\nLoading data...")
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            str(self.data_dir),
            str(self.splits_file),
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            image_size=config['image_size']
        )
        
        # 모델 생성
        print("\nCreating model...")
        self.model = get_model(
            model_name=config['model_name'],
            num_classes=2,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            pretrained=config['pretrained']
        )
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # 손실 함수 및 옵티마이저
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # 학습 기록
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        self.best_val_acc = 0.0
        
        # 체크포인트 디렉토리
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self):
        """
        1 에포크 학습
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for frames, labels in pbar:
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(frames)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # 통계
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """
        검증
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for frames, labels in pbar:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{running_loss/len(pbar):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, train_loss, val_loss, val_acc, is_best=False):
        """
        체크포인트 저장
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': self.config
        }
        
        # 에포크별 체크포인트
        if epoch % self.config['save_interval'] == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Best 모델 저장
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path} (acc: {val_acc:.2f}%)")
    
    def train(self):
        """
        전체 학습 프로세스
        """
        print(f"\n{'='*50}")
        print("Starting Training")
        print(f"{'='*50}")
        print(f"Epochs: {self.config['num_epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        
        start_time = datetime.now()
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.config['num_epochs']}")
            print(f"{'='*50}")
            
            # 학습
            train_loss, train_acc = self.train_epoch()
            
            # 검증
            val_loss, val_acc = self.validate()
            
            # 학습률 조정
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 결과 출력
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # History 저장
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 체크포인트 저장
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            self.save_checkpoint(epoch, train_loss, val_loss, val_acc, is_best=is_best)
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        # Training history 저장
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*50}")
        print("Training Completed!")
        print(f"{'='*50}")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Total training time: {training_time}")
        print(f"History saved: {history_path}")
        print(f"{'='*50}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train video-based deepfake detection model')
    
    parser.add_argument('--model', type=str, default='resnet_lstm',
                        choices=['cnn_lstm', 'resnet_lstm'],
                        help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Config 생성
    config = {
        'model_name': args.model,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': 0.0001,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'pretrained': True,
        'image_size': 224,
        'device': args.device,
        'num_workers': args.num_workers,
        'save_interval': args.save_interval,
        'checkpoint_dir': './checkpoints/frames_16'
    }
    
    # Trainer 생성 및 학습
    trainer = VideoTrainer(config)
    trainer.train()