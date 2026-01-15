from models.base import DeepfakeDetector

class XceptionNet(DeepfakeDetector):
    """
    Xception 기반 딥페이크 탐지 모델
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super(XceptionNet, self).__init__(
            backbone_name='xception',
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )