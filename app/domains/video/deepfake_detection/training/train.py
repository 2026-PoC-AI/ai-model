import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import json
from datetime import datetime

from models.xception import XceptionNet
from preprocessing.dataset import DeepfakeDataset, get_transforms
from training.config import TrainingConfig
from training.metrics import MetricsCalculator

class Trainer:
    """
    딥페이크 탐지 모델 학습 클래스
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # 모델 초기화
        self.model = XceptionNet(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            dropout=config.dropout
        ).to(self.device)
        
        print(f"Model initialized: {config.model_name}")
        print(f"Device: {self.device}")
        
        # 데이터 로더
        self.train_loader = self._create_dataloader('train')
        self.val_loader = self._create_dataloader('val')
        
        # 손실 함수
        self.criterion = nn.CrossEntropyLoss()
        
        # 옵티마이저
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 스케줄러
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.min_lr
        )
        
        # 평가 지표 계산기
        self.metrics_calculator = MetricsCalculator()
        
        # 학습 기록
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_metrics': []
        }
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def _create_dataloader(self, split):
        """
        데이터 로더 생성 (클래스 불균형 처리 포함)
        """
        dataset = DeepfakeDataset(
            data_dir=self.config.data_dir,
            split=split,
            transform=get_transforms(split)
        )
        
        if split == 'train' and self.config.use_weighted_sampler:
            # 클래스별 샘플 수 계산
            labels = [sample[1] for sample in dataset.samples]
            class_counts = np.bincount(labels)
            
            print(f"Class distribution: Real={class_counts[0]}, Fake={class_counts[1]}")
            
            # 가중치 계산
            class_weights = 1.0 / class_counts
            sample_weights = [class_weights[label] for label in labels]
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                sampler=sampler,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=(split == 'train'),
                num_workers=self.config.num_workers,
                pin_memory=True
            )
    
    def train_epoch(self):
        """
        1 에폭 학습
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        return avg_loss, accuracy
    
    def validate(self):
        """
        검증 수행
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = self.metrics_calculator.calculate_metrics(
            all_labels, all_preds, all_probs
        )
        
        return avg_loss, metrics
    
    def train(self):
        """
        전체 학습 프로세스
        """
        print(f"\nStarting training")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Total epochs: {self.config.num_epochs}\n")
        
        start_time = datetime.now()
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print("-" * 50)
            
            # 학습
            train_loss, train_acc = self.train_epoch()
            
            # 검증
            val_loss, val_metrics = self.validate()
            
            # 스케줄러 업데이트
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 결과 출력
            print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            self.metrics_calculator.print_metrics(val_metrics, prefix='Val ')
            print(f"Learning Rate: {current_lr:.6f}")
            
            # 기록 저장
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_metrics'].append(val_metrics)
            
            # 최고 성능 모델 저장
            if val_metrics['accuracy'] > self.best_val_acc + self.config.min_delta:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
                self.patience_counter = 0
                print(f"\nBest model saved (accuracy: {val_metrics['accuracy']:.4f})")
            else:
                self.patience_counter += 1
            
            # 주기적 저장
            if (epoch + 1) % self.config.save_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_metrics)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print(f"\n{'='*50}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Total training time: {training_time}")
        print(f"{'='*50}\n")
        
        # 학습 기록 저장
        self.save_training_history()
    
    def save_checkpoint(self, filename, epoch, metrics):
        """
        체크포인트 저장
        """
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': {
                'model_name': self.config.model_name,
                'num_classes': self.config.num_classes,
                'dropout': self.config.dropout
            }
        }
        
        filepath = os.path.join(self.config.save_dir, filename)
        torch.save(checkpoint, filepath)
    
    def save_training_history(self):
        """
        학습 기록 저장
        """
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # confusion matrix를 리스트로 변환
        history_to_save = {
            'train_loss': self.train_history['train_loss'],
            'train_acc': self.train_history['train_acc'],
            'val_loss': self.train_history['val_loss'],
            'val_metrics': []
        }
        
        for metrics in self.train_history['val_metrics']:
            metrics_copy = metrics.copy()
            if 'confusion_matrix' in metrics_copy:
                metrics_copy['confusion_matrix'] = metrics_copy['confusion_matrix'].tolist()
            history_to_save['val_metrics'].append(metrics_copy)
        
        filepath = os.path.join(self.config.save_dir, 'training_history.json')
        with open(filepath, 'w') as f:
            json.dump(history_to_save, f, indent=2)
        
        print(f"Training history saved to {filepath}")

if __name__ == '__main__':
    config = TrainingConfig()
    trainer = Trainer(config)
    trainer.train()