import numpy as np

def aggregate_predictions(predictions, method='mean'):
    """
    프레임별 예측을 비디오 수준으로 집계
    
    Args:
        predictions: 프레임별 fake 확률 리스트
        method: 집계 방식
            - 'mean': 평균
            - 'max': 최대값
            - 'topk': 상위 k개 평균 (k=5)
    
    Returns:
        집계된 fake 확률
    """
    if len(predictions) == 0:
        return 0.5
    
    predictions = np.array(predictions)
    
    if method == 'mean':
        return np.mean(predictions)
    elif method == 'max':
        return np.max(predictions)
    elif method == 'topk':
        k = min(5, len(predictions))
        top_k_preds = np.sort(predictions)[-k:]
        return np.mean(top_k_preds)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")