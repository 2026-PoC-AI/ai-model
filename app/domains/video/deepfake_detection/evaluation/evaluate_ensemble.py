import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from models.xception import XceptionNet
from models.efficientnet import EfficientNetB4
from models.ensemble import EnsembleModel
from preprocessing.dataset import DeepfakeDataset, get_transforms
from training.metrics import MetricsCalculator
from tqdm import tqdm

def load_model(model_path, model_type='xception', device='cpu'):
    """모델 로드"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    if model_type == 'xception':
        model = XceptionNet(
            num_classes=config.get('num_classes', 2),
            pretrained=False,
            dropout=config.get('dropout', 0.5)
        )
    else:
        model = EfficientNetB4(
            num_classes=config.get('num_classes', 2),
            pretrained=False,
            dropout=config.get('dropout', 0.5)
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def evaluate_ensemble(xception_path, efficientnet_path, 
                        data_dir='../data/processed',
                        weights=None, 
                        ensemble_method='soft_voting',
                        device='cpu'):
    """앙상블 모델 평가"""
    
    # 모델 로드
    print("Loading models...")
    xception_model = load_model(xception_path, 'xception', device)
    efficientnet_model = load_model(efficientnet_path, 'efficientnet', device)
    
    # 앙상블 모델 생성
    ensemble = EnsembleModel(
        xception_model,
        efficientnet_model,
        weights=weights,
        ensemble_method=ensemble_method
    )
    ensemble.to(device)
    ensemble.eval()
    
    # 검증 데이터 로드
    print("\nLoading validation data...")
    val_dataset = DeepfakeDataset(
        data_dir=data_dir,
        split='val',
        transform=get_transforms('val')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )
    
    # 평가
    print(f"\nEvaluating ensemble ({ensemble_method})...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            # 앙상블 예측
            probs = ensemble(images)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # 성능 계산
    metrics_calculator = MetricsCalculator()
    metrics = metrics_calculator.calculate_metrics(all_labels, all_preds, all_probs)
    
    return metrics

if __name__ == "__main__":
    device = 'cpu'
    
    # 모델 경로
    xception_path = 'weights/xception/xception_best_20260116.pth'
    efficientnet_path = 'weights/efficientnet/efficientnet_best.pth'
    
    print("="*60)
    print("Ensemble Model Evaluation")
    print("="*60)
    
    # 1. Soft Voting (동일 가중치)
    print("\n[1] Soft Voting (Equal weights: 0.5, 0.5)")
    metrics = evaluate_ensemble(
        xception_path, efficientnet_path,
        weights=[0.5, 0.5],
        ensemble_method='soft_voting',
        device=device
    )
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    # 2. Soft Voting (XceptionNet 우선)
    print("\n" + "="*60)
    print("[2] Soft Voting (XceptionNet priority: 0.6, 0.4)")
    metrics = evaluate_ensemble(
        xception_path, efficientnet_path,
        weights=[0.6, 0.4],
        ensemble_method='soft_voting',
        device=device
    )
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    # 3. Hard Voting
    print("\n" + "="*60)
    print("[3] Hard Voting")
    metrics = evaluate_ensemble(
        xception_path, efficientnet_path,
        ensemble_method='hard_voting',
        device=device
    )
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)