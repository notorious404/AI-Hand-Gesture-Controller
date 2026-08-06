from typing import Dict, List, Tuple

from utils import get_distance, normalize_point


class GestureRecognizer:
    """Rule-based hand gesture recognition using landmark positions."""

    def __init__(self) -> None:
        self.gesture_rules = [
            (self._is_open_palm, "Open Palm"),
            (self._is_closed_fist, "Closed Fist"),
            (self._is_thumbs_up, "Thumbs Up"),
            (self._is_thumbs_down, "Thumbs Down"),
            (self._is_peace_sign, "Peace Sign"),
            (self._is_rock_sign, "Rock Sign"),
            (self._is_ok_sign, "OK Sign"),
            (self._is_pointing_up, "Pointing Up"),
            (self._is_pointing_left, "Pointing Left"),
            (self._is_pointing_right, "Pointing Right"),
            (self._is_call_me, "Call Me Gesture"),
            (self._is_three_fingers, "Three Fingers"),
            (self._is_four_fingers, "Four Fingers"),
            (self._is_five_fingers, "Five Fingers"),
        ]

    def recognize(
        self,
        hand_landmarks: "mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList",
        frame_size: Tuple[int, int],
        handedness: str = "RIGHT",
    ) -> str:
        """Return a gesture name when a rule matches the hand pose."""
        if not hand_landmarks:
            return "No Gesture"

        landmark_points = self._map_landmarks_to_points(hand_landmarks, frame_size)
        for rule, label in self.gesture_rules:
            if rule(landmark_points, handedness):
                return label
        return "Unknown"

    def _map_landmarks_to_points(self, landmarks, frame_size: Tuple[int, int]) -> List[Tuple[int, int]]:
        width, height = frame_size
        return [normalize_point(lm.x, lm.y, width, height) for lm in landmarks.landmark]

    def _hand_size(self, points: List[Tuple[int, int]]) -> float:
        return max(get_distance(points[0], points[9]), 1.0)

    def _is_finger_extended(self, tip: Tuple[int, int], pip: Tuple[int, int]) -> bool:
        return tip[1] < pip[1]

    def _is_finger_bent(self, tip: Tuple[int, int], pip: Tuple[int, int]) -> bool:
        return tip[1] > pip[1]

    def _is_open_palm(self, points: List[Tuple[int, int]], handedness: str = "RIGHT") -> bool:
        hand_size = self._hand_size(points)
        fingers_extended = all(
            self._is_finger_extended(points[tip], points[pip])
            for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
        )
        thumb_extended = abs(points[4][0] - points[3][0]) > hand_size * 0.25
        if handedness == "RIGHT":
            thumb_extended = thumb_extended and points[4][0] > points[3][0]
        else:
            thumb_extended = thumb_extended and points[4][0] < points[3][0]
        return fingers_extended and thumb_extended

    def _is_closed_fist(self, points: List[Tuple[int, int]], handedness: str = "RIGHT") -> bool:
        hand_size = self._hand_size(points)
        return (
            all(self._is_finger_bent(points[tip], points[pip]) for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))
            or all(get_distance(points[tip], points[0]) < hand_size * 0.55 for tip in (8, 12, 16, 20))
        )

    def _is_thumbs_up(self, points: List[Tuple[int, int]], handedness: str = "RIGHT") -> bool:
        return points[4][1] < points[3][1] and all(
            self._is_finger_bent(points[tip], points[pip])
            for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
        )

    def _is_thumbs_down(self, points: List[Tuple[int, int]], handedness: str = "RIGHT") -> bool:
        return points[4][1] > points[3][1] and all(
            self._is_finger_bent(points[tip], points[pip])
            for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
        )

    def _is_peace_sign(self, points: List[Tuple[int, int]], handedness: str = "RIGHT") -> bool:
        return (
            self._is_finger_extended(points[8], points[6])
            and self._is_finger_extended(points[12], points[10])
            and self._is_finger_bent(points[16], points[14])
            and self._is_finger_bent(points[20], points[18])
        )

    def _is_rock_sign(self, points: List[Tuple[int, int]], handedness: str = "RIGHT") -> bool:
        return (
            self._is_finger_bent(points[8], points[6])
            and self._is_finger_bent(points[12], points[10])
            and self._is_finger_extended(points[16], points[14])
            and self._is_finger_extended(points[20], points[18])
        )

    def _is_ok_sign(self, points: List[Tuple[int, int]], handedness: str = "RIGHT") -> bool:
        hand_size = self._hand_size(points)
        distance = get_distance(points[4], points[8])
        return distance < hand_size * 0.35 and self._is_finger_extended(points[12], points[10]) and self._is_finger_extended(points[16], points[14])

    def _is_pointing_up(self, points: List[Tuple[int, int]], handedness: str = "RIGHT") -> bool:
        return (
            self._is_finger_extended(points[8], points[6])
            and self._is_finger_bent(points[12], points[10])
            and self._is_finger_bent(points[16], points[14])
            and self._is_finger_bent(points[20], points[18])
        )

    def _is_pointing_left(self, points: List[Tuple[int, int]], handedness: str = "RIGHT") -> bool:
        return points[8][0] < points[6][0] and self._is_finger_bent(points[12], points[10])

    def _is_pointing_right(self, points: List[Tuple[int, int]], handedness: str = "RIGHT") -> bool:
        return points[8][0] > points[6][0] and self._is_finger_bent(points[12], points[10])

    def _is_call_me(self, points: List[Tuple[int, int]], handedness: str = "RIGHT") -> bool:
        thumb_extended = abs(points[4][0] - points[3][0]) > 40
        if handedness == "RIGHT":
            thumb_extended = thumb_extended and points[4][0] > points[3][0]
        else:
            thumb_extended = thumb_extended and points[4][0] < points[3][0]
        pinky_extended = self._is_finger_extended(points[20], points[18])
        return thumb_extended and pinky_extended and all(
            self._is_finger_bent(points[tip], points[pip])
            for tip, pip in ((8, 6), (12, 10), (16, 14))
        )

    def _is_three_fingers(self, points: List[Tuple[int, int]], handedness: str = "RIGHT") -> bool:
        return (
            self._is_finger_extended(points[8], points[6])
            and self._is_finger_extended(points[12], points[10])
            and self._is_finger_extended(points[16], points[14])
            and self._is_finger_bent(points[20], points[18])
        )

    def _is_four_fingers(self, points: List[Tuple[int, int]], handedness: str = "RIGHT") -> bool:
        return (
            all(self._is_finger_extended(points[tip], points[pip]) for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))
            and self._is_finger_bent(points[4], points[3])
        )

    def _is_five_fingers(self, points: List[Tuple[int, int]], handedness: str = "RIGHT") -> bool:
        return all(self._is_finger_extended(points[tip], points[pip]) for tip, pip in ((4, 3), (8, 6), (12, 10), (16, 14), (20, 18)))
