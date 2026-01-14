import torch

class TrainingConfig:
    """
    학습 설정 클래스
    """
    # 데이터 경로
    data_dir = './data/processed'
    
    # 모델 설정
    model_name = 'xception'
    num_classes = 2
    pretrained = True
    dropout = 0.5
    
    # 학습 하이퍼파라미터
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    weight_decay = 1e-5
    
    # 옵티마이저 설정
    optimizer = 'adam'
    
    # 스케줄러 설정
    scheduler = 'cosine'
    min_lr = 1e-6
    
    # 체크포인트
    save_dir = './models/weights'
    save_frequency = 5
    
    # 장치 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4
    
    # Early stopping
    patience = 10
    min_delta = 0.001
    
    # 클래스 불균형 처리
    use_weighted_sampler = True