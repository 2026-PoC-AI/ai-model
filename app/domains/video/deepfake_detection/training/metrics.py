import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

class MetricsCalculator:
    """
    평가 지표 계산 클래스
    """
    @staticmethod
    def calculate_metrics(labels, preds, probs):
        """
        모든 평가 지표 계산
        
        Args:
            labels: 실제 레이블
            preds: 예측 레이블
            probs: 예측 확률 (fake 클래스)
            
        Returns:
            dict: 평가 지표들
        """
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary'
        )
        
        try:
            auc = roc_auc_score(labels, probs)
        except:
            auc = 0.0
        
        cm = confusion_matrix(labels, preds)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm
        }
    
    @staticmethod
    def print_metrics(metrics, prefix=''):
        """
        평가 지표 출력
        """
        print(f"{prefix}Accuracy: {metrics['accuracy']:.4f}")
        print(f"{prefix}Precision: {metrics['precision']:.4f}")
        print(f"{prefix}Recall: {metrics['recall']:.4f}")
        print(f"{prefix}F1: {metrics['f1']:.4f}")
        print(f"{prefix}AUC: {metrics['auc']:.4f}")
        
        if 'confusion_matrix' in metrics:
            print(f"{prefix}Confusion Matrix:")
            print(metrics['confusion_matrix'])