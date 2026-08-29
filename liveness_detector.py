"""
Liveness Detection Engine using a lightweight Convolutional Neural Network (MobileNetV2).
Evaluates whether a captured frame is a live human eye or a presentation attack (photo/screen).
"""

import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


class IrisLivenessModel(nn.Module):
    """MobileNetV2 backbone modified for binary classification (Live vs. Spoof)."""
    
    def __init__(self, pretrained: bool = True) -> None:
        super(IrisLivenessModel, self).__init__()
        # Load lightweight MobileNetV2
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)
        
        # Replace classification head for 2 classes: 0 = Spoof, 1 = Live
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class LivenessDetector:
    """Inference engine for real-time liveness verification."""

    def __init__(self, model_path: str = "models/liveness_mobilenet.pth") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = IrisLivenessModel(pretrained=True)
        
        # Load custom weights if available
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logging.info(f"Loaded custom liveness weights from {model_path}")
        except FileNotFoundError:
            logging.warning(f"Custom weights '{model_path}' not found. Operating with pre-trained feature structure.")

        self.model.to(self.device)
        self.model.eval()

        # Transformation pipeline matching MobileNet expectations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, frame: np.ndarray, confidence_threshold: float = 0.50) -> tuple[bool, float]:
        """
        Evaluates a frame to determine if it is a live eye.

        Args:
            frame (np.ndarray): Input OpenCV BGR image frame.
            confidence_threshold (float): Minimum confidence required to pass.

        Returns:
            tuple[bool, float]: (is_live, confidence_score)
        """
        if frame is None or frame.size == 0:
            return False, 0.0

        # Convert OpenCV BGR to PIL RGB Image
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # Preprocess tensor
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
            # Index 1 = Live
            live_confidence = float(probabilities[1].item())

        is_live = live_confidence >= confidence_threshold
        logging.info(f"Liveness Check -> Live Confidence: {live_confidence:.2f} | Result: {'LIVE' if is_live else 'SPOOF'}")

        return is_live, live_confidence


if __name__ == "__main__":
    # Self-test using a blank frame
    test_detector = LivenessDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    status, score = test_detector.predict(dummy_frame)
    print(f"\n[Test Verification] Liveness Module initialized successfully! Output: status={status}, score={score:.2f}")