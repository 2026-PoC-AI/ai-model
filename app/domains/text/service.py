# app/domains/text/fake_news/service.py
from .predictor import FakeNewsPredictor

def get_evidence_sentences(text: str, predictor: FakeNewsPredictor, top_n: int = 2):
    # 문장 단위로 쪼개서 각 문장의 '의심 점수' 계산
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
    if not sentences: return [text]
    
    scored_sentences = []
    for sent in sentences:
        prob = predictor.predict_proba(sent)
        scored_sentences.append((sent, prob))
    
    # 점수 높은 문장 상위 N개
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored_sentences[:top_n]]

def analyze_fake_news(text: str, predictor: FakeNewsPredictor):
    prob = predictor.predict_proba(text)
    score = round(prob * 100, 2)
    label = 1 if prob >= 0.5 else 0
    
    # 신뢰 등급 산출
    if score >= 70: level = "HIGH"
    elif score >= 40: level = "MID"
    else: level = "LOW"
    
    # 증거 데이터 추출
    keywords = predictor.get_top_features(text)
    sentences = get_evidence_sentences(text, predictor)
    
    # 사용자 메시지 구성
    msg = f"분석 결과 가짜뉴스 확률이 {score}%로 의심됩니다."
    if label == 1:
        msg += f" 특히 '{', '.join(keywords)}'와 같은 표현이 주요 의심 근거입니다."

    return {
        "score": score,
        "label": label,
        "level": level,
        "evidence": {"keywords": keywords, "sentences": sentences},
        "message": msg
    }