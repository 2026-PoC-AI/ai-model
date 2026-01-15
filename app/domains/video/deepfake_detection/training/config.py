import torch
from datetime import datetime

class TrainingConfig:
    """
    학습 설정 클래스
    """
    def __init__(
        self,
        model_name='xception',
        data_dir='data/processed',
        batch_size=8,
        num_epochs=20,
        learning_rate=0.0001,
        device=None,
        num_workers=0,
        save_dir='weights'
    ):
        # 데이터 경로
        self.data_dir = data_dir
        
        # 모델 설정
        self.model_name = model_name
        self.num_classes = 2
        self.pretrained = True
        self.dropout = 0.5
        
        # 학습 하이퍼파라미터
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = 1e-5
        
        # 옵티마이저
        self.optimizer = 'adam'
        
        # 스케줄러
        self.scheduler = 'cosine'
        self.min_lr = 1e-6
        
        # 체크포인트 (날짜 포함)
        self.save_dir = save_dir
        self.save_frequency = 5
        
        # 날짜 형식 (YYYYMMDD)
        self.date_str = datetime.now().strftime('%Y%m%d')
        
        # 장치
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.num_workers = num_workers
        
        # Early stopping
        self.patience = 5
        self.min_delta = 0.001
        
        # 클래스 불균형
        self.use_weighted_sampler = True