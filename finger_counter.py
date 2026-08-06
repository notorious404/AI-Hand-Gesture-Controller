from typing import Dict, List, Tuple

from config import Config
from utils import normalize_point


class FingerCounter:
    """Count extended fingers per hand using MediaPipe landmarks."""

    def __init__(self) -> None:
        self.finger_tip_ids = [4, 8, 12, 16, 20]
        self.finger_pip_ids = [3, 6, 10, 14, 18]

    def count_fingers(self, hand_landmarks: "mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList",
                      handedness: str,
                      frame_size: Tuple[int, int]) -> int:
        """Count fingers for one hand, using handedness for thumb detection."""
        if not hand_landmarks:
            return 0

        image_width, image_height = frame_size
        landmark_points: List[Tuple[int, int]] = [
            normalize_point(lm.x, lm.y, image_width, image_height)
            for lm in hand_landmarks.landmark
        ]

        extended = 0
        thumb_tip_x = landmark_points[4][0]
        thumb_ip_x = landmark_points[3][0]
        if handedness == "RIGHT":
            if thumb_tip_x > thumb_ip_x:
                extended += 1
        else:
            if thumb_tip_x < thumb_ip_x:
                extended += 1

        for tip_id, pip_id in zip(self.finger_tip_ids[1:], self.finger_pip_ids[1:]):
            if landmark_points[tip_id][1] < landmark_points[pip_id][1]:
                extended += 1

        return extended

    def count_all_hands(self, hands_data: List[dict], frame_size: Tuple[int, int]) -> Dict[str, int]:
        """Return finger counts grouped by handedness and total count."""
        counts = {"LEFT": 0, "RIGHT": 0, "TOTAL": 0}
        for hand_info in hands_data:
            handedness = hand_info["handedness"]
            count = self.count_fingers(hand_info["landmarks"], handedness, frame_size)
            counts[handedness] = count
            counts["TOTAL"] += count
        return counts
