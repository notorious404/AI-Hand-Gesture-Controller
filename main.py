from typing import List, Tuple

import cv2
import numpy as np
import time
import warnings

from config import Config
from gesture_recognizer import GestureRecognizer
from hand_detector import HandDetector
from finger_counter import FingerCounter
from volume_controller import VolumeController
from action_manager import ActionManager
from notification_manager import NotificationManager
from utils import calculate_fps, get_distance


warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated",
    category=UserWarning,
    module=r"google\.protobuf\.symbol_database",
)

class AIHandGestureController:
    """Main application class for the AI Hand Gesture Controller."""

    def __init__(self) -> None:
        self.capture = cv2.VideoCapture(Config.CAMERA_ID, cv2.CAP_DSHOW)
        self._configure_camera()
        self.hand_detector = HandDetector()
        self.gesture_recognizer = GestureRecognizer()
        self.finger_counter = FingerCounter()
        self.volume_controller = VolumeController()
        self.notification_manager = NotificationManager()
        self.action_manager = ActionManager(self.volume_controller, self.notification_manager)
        self.previous_frame_time = 0.0
        self.show_help = False
        self.volume_enabled = True
        self.finger_counter_enabled = True
        self.gesture_enabled = True
        self.camera_capture_enabled = True
        self.screenshot_enabled = True
        self.last_gesture_label = "No Gesture"
        self.min_volume_distance = 30
        self.max_volume_distance = 200
        self.help_overlay = [line for line in Config.HELP_TEXT]
        self.window_name = "AI Hand Gesture Controller"

    def _configure_camera(self) -> None:
        """Set up capture properties for camera resolution and error handling."""
        if not self.capture.isOpened():
            raise RuntimeError("Unable to open webcam. Please check your camera settings.")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        self.capture.set(cv2.CAP_PROP_FPS, Config.MAX_FPS)

    def _update_status(self, frame: "cv2.VideoCapture", text: str, position: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        """Overlay status text on the frame."""
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    def _draw_status_cards(self, frame: "cv2.VideoCapture", fps: float, counts: dict, gesture_label: str, confidence: float) -> None:
        """Draw the information panel UI on the video feed."""
        height, width = frame.shape[:2]

        self._draw_card(frame, (20, 20, 420, 100), "Session", [
            f"FPS: {int(fps)}",
            f"Gesture: {gesture_label}",
            f"Left: {counts.get('LEFT', 0)}  Right: {counts.get('RIGHT', 0)}",
            f"Total Fingers: {counts.get('TOTAL', 0)}",
        ])
        self._draw_card(frame, (width - 430, 20, width - 20, 100), "Details", [
            f"Confidence: {int(confidence * 100)}%",
            f"Active: {self.action_manager.active_feature}",
            f"Status: {self.action_manager.status_badge}",
            f"Volume: {self.volume_controller.volume_status()}",
        ])

        controls = [
            "Q - Quit",
            "R - Reset",
            "V - Toggle Volume",
            "G - Toggle Gesture",
            "F - Toggle Counter",
            "C - Toggle Camera Capture",
            "S - Toggle Screenshot",
            "H - Toggle Help",
        ]
        self._draw_card(frame, (20, 120, 420, 230), "Controls", controls)

    def _draw_card(self, frame: "cv2.VideoCapture", rect: Tuple[int, int, int, int], title: str, lines: List[str]) -> None:
        x1, y1, x2, y2 = rect
        # Render title and lines only — remove filled box/border per user request
        cv2.putText(frame, title, (x1 + 10, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, Config.UI_PRIMARY_COLOR, 2, cv2.LINE_AA)
        for idx, line in enumerate(lines, start=1):
            cv2.putText(frame, line, (x1 + 12, y1 + 24 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, Config.UI_TEXT_COLOR, 1, cv2.LINE_AA)

    def _draw_help_overlay(self, frame: "cv2.VideoCapture") -> None:
        """Draw help instructions on the frame."""
        width = frame.shape[1]
        start_y = frame.shape[0] - 170
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, start_y - 20), (width - 10, frame.shape[0] - 10), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        for idx, line in enumerate(self.help_overlay):
            self._update_status(frame, line, (20, start_y + idx * 30), Config.UI_TEXT_COLOR)

    def _render_hand_labels(self, frame: "cv2.VideoCapture", hands_data: List[dict]) -> float:
        """Draw landmarks and collect the maximum confidence for all detected hands."""
        max_confidence = 0.0
        for hand_info in hands_data:
            self.hand_detector.draw_hand(frame, hand_info)
            max_confidence = max(max_confidence, hand_info["score"])
        return max_confidence

    def _intentional_volume_gesture(self, gesture_label: str) -> bool:
        """Only allow volume changes when the gesture clearly indicates one hand and index/thumb interaction."""
        # Treat 'Five Fingers' as equivalent to 'Open Palm' for volume control
        return gesture_label in {"Open Palm", "Five Fingers", "Pointing Up", "Pointing Left", "Pointing Right", "OK Sign"}

    def _process_volume_control(self, frame: "cv2.VideoCapture", hand_info: dict) -> None:
        """Map thumb-index distance to system volume and draw volume UI elements."""
        width, height = frame.shape[1], frame.shape[0]
        points = [self._normalize_landmark_point(lm, width, height) for lm in hand_info["landmarks"].landmark]
        thumb_tip = points[4]
        index_tip = points[8]
        distance = get_distance(thumb_tip, index_tip)
        normalized_volume = self.volume_controller.map_distance_to_volume(distance, self.min_volume_distance, self.max_volume_distance)
        self.volume_controller.set_volume(normalized_volume)
        self.volume_controller.render_volume_ui(frame, f"Volume {self.volume_controller.volume_status()}", thumb_tip, index_tip)
        self._update_status(frame, f"Finger Dist: {int(distance)} px", (thumb_tip[0] - 50, thumb_tip[1] - 20), Config.UI_ACCENT_COLOR)

    def _normalize_landmark_point(self, lm, width: int, height: int) -> Tuple[int, int]:
        return int(lm.x * width), int(lm.y * height)

    def _reset_state(self) -> None:
        """Restore default application state."""
        self.volume_enabled = True
        self.gesture_enabled = True
        self.finger_counter_enabled = True
        self.last_gesture_label = "No Gesture"

    def run(self) -> None:
        """Main application loop for camera capture, detection, and UI updates."""
        while True:
            success, frame = self.capture.read()
            if not success or frame is None:
                fallback_frame = frame if frame is not None else np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype="uint8")
                frame = cv2.flip(fallback_frame, 1)
                cv2.putText(
                    frame,
                    "Camera unavailable. Reconnecting...",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    Config.UI_WARN_COLOR,
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(self.window_name, frame)
                if cv2.waitKey(1000) & 0xFF == ord("q"):
                    break
                continue

            frame = cv2.flip(frame, 1)
            hands_data, results = self.hand_detector.process_frame(frame)
            counts = self.finger_counter.count_all_hands(hands_data, (frame.shape[1], frame.shape[0])) if self.finger_counter_enabled else {"LEFT": 0, "RIGHT": 0, "TOTAL": 0}
            gesture_label = "No Gesture"
            confidence = 0.0

            if hands_data:
                confidence = self._render_hand_labels(frame, hands_data)
                if self.gesture_enabled:
                    for hand_info in hands_data:
                        handedness = hand_info["handedness"]
                        hand_info["gesture"] = self.gesture_recognizer.recognize(
                            hand_info["landmarks"],
                            (frame.shape[1], frame.shape[0]),
                            handedness,
                        )
                        hand_info["frame_capture"] = frame.copy()
                gesture_label = hands_data[0].get("gesture", "No Gesture")
                if self.finger_counter_enabled:
                    counts = self.finger_counter.count_all_hands(hands_data, (frame.shape[1], frame.shape[0]))
                if self.volume_enabled and self.volume_controller.volume_api_available and len(hands_data) == 1:
                    if self._intentional_volume_gesture(gesture_label):
                        self._process_volume_control(frame, hands_data[0])
                self.last_gesture_label = gesture_label
                self.notification_manager.log_action(f"Gesture Detected: {gesture_label}")
            else:
                self._update_status(frame, "No hands detected", (25, 250), Config.UI_WARN_COLOR)

            current_time = time.time()
            fps = calculate_fps(self.previous_frame_time, current_time)
            self.previous_frame_time = current_time
            self.action_manager.update(frame, hands_data, current_time, self.volume_enabled, self.camera_capture_enabled, self.screenshot_enabled)
            self._draw_status_cards(frame, fps, counts, self.last_gesture_label, confidence)
            self.notification_manager.render(frame)
            if self.show_help:
                self._draw_help_overlay(frame)

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                self._reset_state()
            if key == ord("v"):
                self.volume_enabled = not self.volume_enabled
            if key == ord("g"):
                self.gesture_enabled = not self.gesture_enabled
            if key == ord("f"):
                self.finger_counter_enabled = not self.finger_counter_enabled
            if key == ord("c"):
                self.camera_capture_enabled = not self.camera_capture_enabled
            if key == ord("s"):
                self.screenshot_enabled = not self.screenshot_enabled
            if key == ord("h"):
                self.show_help = not self.show_help

        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = AIHandGestureController()
    app.run()
