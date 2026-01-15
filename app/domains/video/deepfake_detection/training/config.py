import torch

class TrainingConfig:
    """
    학습 설정 클래스
    """
    # 데이터 경로 (상대경로)
    data_dir = '../data/processed'
    
    # 모델 설정
    model_name = 'xception'
    num_classes = 2
    pretrained = True
    dropout = 0.5
    
    # 학습 하이퍼파라미터 (소량 데이터용)
    batch_size = 8  # 작게
    num_epochs = 20  # 적게
    learning_rate = 0.0001  # 작게 (pretrained 모델이라)
    weight_decay = 1e-5
    
    # 옵티마이저
    optimizer = 'adam'
    
    # 스케줄러
    scheduler = 'cosine'
    min_lr = 1e-6
    
    # 체크포인트
    save_dir = './weights'
    save_frequency = 5
    
    # 장치
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 0  # Windows에서 오류 방지
    
    # Early stopping (소량 데이터라 빨리 수렴)
    patience = 5
    min_delta = 0.001
    
    # 클래스 불균형
    use_weighted_sampler = True