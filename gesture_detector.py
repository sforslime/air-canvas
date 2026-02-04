"""MediaPipe-based hand landmark tracking and gesture recognition (Tasks API)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

Point = Tuple[int, int]

# Constants for landmarks
INDEX_FINGER_TIP = 8
INDEX_FINGER_PIP = 6
MIDDLE_FINGER_TIP = 12
MIDDLE_FINGER_PIP = 10
RING_FINGER_TIP = 16
RING_FINGER_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

@dataclass
class GestureResult:
    gesture: str
    cursor: Optional[Point]
    landmarks: Optional[Any] # List[NormalizedLandmark]
    finger_states: Dict[str, bool]


class GestureDetector:
    """Encapsulates MediaPipe Hands (Tasks API) and lightweight gesture heuristics."""

    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
    ) -> None:
        model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        if not os.path.exists(model_path):
             raise RuntimeError(f"Model not found at {model_path}. Please download hand_landmarker.task")
             
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=tracking_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.landmarker.close()

    def process(self, frame: np.ndarray) -> GestureResult:
        """Run the MediaPipe graph and return gesture metadata."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        result = self.landmarker.detect(mp_image)
        
        # result.hand_landmarks is a list of lists of landmarks (one list per hand)
        landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
        
        h, w = frame.shape[:2]
        cursor = None
        finger_states: Dict[str, bool] = {"index": False, "middle": False, "ring": False, "pinky": False}

        if landmarks is not None:
            cursor = self._landmark_to_point(landmarks, INDEX_FINGER_TIP, w, h)
            finger_states = self._get_finger_states(landmarks)

        gesture = self._interpret_gesture(finger_states)
        return GestureResult(gesture=gesture, cursor=cursor, landmarks=landmarks, finger_states=finger_states)

    def draw_hand_annotations(
        self,
        frame: np.ndarray,
        landmarks: List[Any],
    ) -> None:
        """Overlay the landmark skeleton on the webcam feed."""
        h, w = frame.shape[:2]
        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            cv2.line(frame, 
                     (int(start.x * w), int(start.y * h)), 
                     (int(end.x * w), int(end.y * h)), 
                     (255, 255, 255), 2)
        
        # Draw points
        for lm in landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 0, 255), -1)

    def _get_finger_states(self, landmarks: List[Any]) -> Dict[str, bool]:
        tips = {"index": INDEX_FINGER_TIP, "middle": MIDDLE_FINGER_TIP, "ring": RING_FINGER_TIP, "pinky": PINKY_TIP}
        pips = {"index": INDEX_FINGER_PIP, "middle": MIDDLE_FINGER_PIP, "ring": RING_FINGER_PIP, "pinky": PINKY_PIP}
        states: Dict[str, bool] = {}
        for name in tips:
            tip = landmarks[tips[name]]
            pip = landmarks[pips[name]]
            states[name] = tip.y < pip.y
        return states

    def _interpret_gesture(self, finger_states: Dict[str, bool]) -> str:
        idx = finger_states["index"]
        mid = finger_states["middle"]
        ring = finger_states["ring"]
        pinky = finger_states["pinky"]

        if idx and mid and ring and pinky:
            return "clear"
        if not idx and not mid and not ring and not pinky:
            return "fist"
        if idx and not mid and not ring and not pinky:
            return "draw"
        return "idle"

    @staticmethod
    def _landmark_to_point(
        landmarks: List[Any],
        index: int,
        width: int,
        height: int,
    ) -> Point:
        lm = landmarks[index]
        return int(lm.x * width), int(lm.y * height)
