import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembleModel(nn.Module):
    """
    XceptionNet과 EfficientNet-B4의 앙상블 모델
    """
    def __init__(self, xception_model, efficientnet_model, 
                    weights=None, ensemble_method='soft_voting'):
        """
        Args:
            xception_model: XceptionNet 모델
            efficientnet_model: EfficientNet-B4 모델
            weights: 각 모델의 가중치 [w1, w2] (soft_voting 시 사용)
            ensemble_method: 'soft_voting', 'hard_voting', 'weighted_voting'
        """
        super(EnsembleModel, self).__init__()
        
        self.xception = xception_model
        self.efficientnet = efficientnet_model
        self.ensemble_method = ensemble_method
        
        # 가중치 설정
        if weights is None:
            self.weights = [0.5, 0.5]  # 기본값: 동일 가중치
        else:
            self.weights = weights
        
        # 평가 모드로 설정
        self.xception.eval()
        self.efficientnet.eval()
    
    def forward(self, x):
        """
        Args:
            x: 입력 이미지 텐서 (B, C, H, W)
        
        Returns:
            앙상블 예측 결과 (B, num_classes)
        """
        with torch.no_grad():
            # 각 모델의 예측
            xception_out = self.xception(x)
            efficientnet_out = self.efficientnet(x)
        
        if self.ensemble_method == 'soft_voting':
            # 확률 평균 (가중치 적용)
            xception_probs = F.softmax(xception_out, dim=1)
            efficientnet_probs = F.softmax(efficientnet_out, dim=1)
            
            ensemble_probs = (self.weights[0] * xception_probs + 
                            self.weights[1] * efficientnet_probs)
            return ensemble_probs
        
        elif self.ensemble_method == 'hard_voting':
            # 클래스 예측 후 다수결
            xception_preds = torch.argmax(xception_out, dim=1)
            efficientnet_preds = torch.argmax(efficientnet_out, dim=1)
            
            # 두 예측이 같으면 그대로, 다르면 xception 우선
            ensemble_preds = torch.where(
                xception_preds == efficientnet_preds,
                xception_preds,
                xception_preds  # 동점일 때 xception 우선
            )
            
            # 원-핫 인코딩으로 변환
            num_classes = xception_out.size(1)
            ensemble_probs = F.one_hot(ensemble_preds, num_classes).float()
            return ensemble_probs
        
        elif self.ensemble_method == 'weighted_voting':
            # 가중치 적용한 로짓 평균
            ensemble_logits = (self.weights[0] * xception_out + 
                             self.weights[1] * efficientnet_out)
            return F.softmax(ensemble_logits, dim=1)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")