# app/domains/video/model.py
from app.domains.video.schemas import VideoAnalyzeResponse, VideoEvidence

class VideoModel:
    def predict(self, video_bytes: bytes) -> VideoAnalyzeResponse:
        return VideoAnalyzeResponse(
            risk_score=78,
            grade="HIGH",
            evidence=[
                VideoEvidence(frame_index=12, score=0.86, reason="Face region inconsistency (dummy)"),
                VideoEvidence(frame_index=58, score=0.81, reason="Temporal artifact pattern (dummy)"),
            ],
            warnings=["dummy-video-warning"],
        )

def load_video_model() -> VideoModel:
    return VideoModel()
