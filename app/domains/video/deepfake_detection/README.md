# Deepfake Detection - XceptionNet Baseline

## 데이터셋 준비

### 1. Celeb-DF v2 다운로드

1. 신청: https://github.com/yuezunli/celeb-deepfakeforensics
2. 승인 후 Google Drive 링크 받기
3. 다운로드:
   - Celeb-real/ → `data/raw/celeb-df/Celeb-real/`
   - Celeb-synthesis/ → `data/raw/celeb-df/Celeb-synthesis/`

### 2. 데이터 구조
```
data/
├── raw/
│   └── celeb-df/
│       ├── Celeb-real/
│       ├── YouTube-real/
│       └── Celeb-synthesis/
└── processed/
    ├── train/
    │   ├── real/
    │   └── fake/
    └── val/
        ├── real/
        └── fake/
```

### 3. 전처리
```bash
cd ~/Desktop/ai-model

# PYTHONPATH 설정해서 실행
PYTHONPATH=app/domains/video/deepfake_detection python app/domains/video/deepfake_detection/preprocessing/prepare_data_simple.py

PYTHONPATH=app/domains/video/deepfake_detection python app/domains/video/deepfake_detection/preprocessing/prepare_data.py
```

## 주의사항

- `data/` 디렉토리는 Git에 포함되지 않습니다
- 각자 데이터셋을 다운로드해야 합니다