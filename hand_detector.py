from typing import List, Tuple

import cv2

try:
    import mediapipe.python.solutions as mp_solutions
    MP_HANDS = mp_solutions.hands
    MP_DRAW = mp_solutions.drawing_utils
except ImportError:
    import mediapipe as mp
    MP_HANDS = mp.solutions.hands
    MP_DRAW = mp.solutions.drawing_utils

from config import Config
from utils import normalize_point


class HandDetector:
    """Hand detection using MediaPipe Hands with landmark extraction."""

    def __init__(self) -> None:
        self.mp_hands = MP_HANDS
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE,
        )
        self.mp_draw = MP_DRAW

    def process_frame(self, frame: "cv2.VideoCapture") -> Tuple[List[dict], "cv2.VideoCapture"]:
        """Process the frame and return detected hands with normalized landmarks."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        hands_data = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for handedness, landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                handedness_label = handedness.classification[0].label.upper()
                score = handedness.classification[0].score
                hands_data.append(
                    {
                        "handedness": handedness_label,
                        "score": score,
                        "landmarks": landmarks,
                    }
                )
        return hands_data, results

    def draw_hand(self, frame: "cv2.VideoCapture", hand_info: dict) -> None:
        """Draw hand landmarks, connections, and a labeled bounding box."""
        landmarks = hand_info["landmarks"]
        image_height, image_width = frame.shape[:2]
        self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

        points = [normalize_point(lm.x, lm.y, image_width, image_height) for lm in landmarks.landmark]
        x_coordinates = [point[0] for point in points]
        y_coordinates = [point[1] for point in points]
        x_min, x_max = min(x_coordinates), max(x_coordinates)
        y_min, y_max = min(y_coordinates), max(y_coordinates)

        for tip_index in (4, 8, 12, 16, 20):
            tip = points[tip_index]
            cv2.circle(frame, tip, 12, Config.UI_ACCENT_COLOR, 2, cv2.LINE_AA)
            cv2.circle(frame, tip, 6, Config.UI_TEXT_COLOR, cv2.FILLED)

        cv2.rectangle(
            frame,
            (x_min - 15, y_min - 15),
            (x_max + 15, y_max + 15),
            Config.UI_PRIMARY_COLOR,
            2,
        )

        label_text = f"{hand_info['handedness']} HAND ({hand_info['score'] * 100:.0f}%)"
        cv2.putText(
            frame,
            label_text,
            (x_min - 10, y_min - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            Config.UI_ACCENT_COLOR,
            2,
            cv2.LINE_AA,
        )
