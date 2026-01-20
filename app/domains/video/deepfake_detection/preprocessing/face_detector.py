import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch

class FaceDetector:
    """
    얼굴 검출 및 정렬 클래스
    MTCNN을 사용해 얼굴을 검출하고 랜드마크 기반으로 정렬
    """
    def __init__(self, device=None):
        # device가 None이면 자동 선택
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.mtcnn = MTCNN(
            keep_all=False,
            device=self.device,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            post_process=False
        )
    
    def detect_and_align(self, image):
        """
        이미지에서 얼굴을 검출하고 정렬
        
        Args:
            image: BGR 포맷의 numpy 배열
            
        Returns:
            aligned_face: 정렬된 얼굴 이미지 (RGB)
            box: 얼굴 바운딩 박스
        """
        # BGR에서 RGB로 변환
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 얼굴 검출
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        
        if boxes is None:
            return None, None
        
        # 가장 신뢰도 높은 얼굴 선택
        max_idx = np.argmax(probs)
        box = boxes[max_idx]
        landmark = landmarks[max_idx]
        
        # 얼굴 정렬
        aligned_face = self.align_face(image, landmark)
        
        return aligned_face, box
    
    def align_face(self, image, landmarks, target_size=(299, 299)):
        """
        랜드마크 기반 얼굴 정렬
        
        Args:
            image: RGB 이미지
            landmarks: 5개의 랜드마크 좌표 [left_eye, right_eye, nose, left_mouth, right_mouth]
            target_size: 출력 이미지 크기
            
        Returns:
            정렬되고 크롭된 얼굴 이미지
        """
        # 눈 좌표 추출
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        # 회전 각도 계산
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # 두 눈 사이 중심점
        eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                        (left_eye[1] + right_eye[1]) / 2)
        
        # 회전 변환 행렬
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
        
        # 이미지 회전
        rotated = cv2.warpAffine(
            image, M, (image.shape[1], image.shape[0]),
            flags=cv2.INTER_CUBIC
        )
        
        # 눈 사이 거리 기반 크롭 (3.0으로 조정)
        eye_distance = np.linalg.norm(right_eye - left_eye)
        crop_size = int(eye_distance * 3.0)
        
        x_center, y_center = int(eyes_center[0]), int(eyes_center[1])
        x1 = max(0, x_center - crop_size)
        y1 = max(0, y_center - int(crop_size * 1.2))
        x2 = min(rotated.shape[1], x_center + crop_size)
        y2 = min(rotated.shape[0], y_center + int(crop_size * 1.4))
        
        cropped = rotated[y1:y2, x1:x2]
        
        # 목표 크기로 리사이즈
        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_CUBIC)
        
        return resized