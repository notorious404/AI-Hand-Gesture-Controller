from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np

from config import Config
from notification_manager import NotificationManager
from utils import calculate_angle, clamp, get_centroid, get_distance
from volume_controller import VolumeController


@dataclass
class HandGestureState:
    label_history: Deque[str] = field(default_factory=lambda: deque(maxlen=Config.GESTURE_STABILITY_FRAMES))
    confirmed_label: str = "No Gesture"
    last_confirmed_label: str = "No Gesture"
    stable_since: float = 0.0
    action_ready: bool = False
    cooldown_until: float = 0.0
    rotation_history: Deque[Tuple[Tuple[int, int], float]] = field(default_factory=lambda: deque(maxlen=Config.ROTATION_HISTORY))
    rotation_direction: Optional[str] = None
    rotation_progress: float = 0.0
    last_rotation_notification: float = 0.0
    active_feature: str = "Idle"
    status_badge: str = "🔴 Lost Hand"


class ActionManager:
    """Central action coordinator for gesture-driven controls."""

    def __init__(self, volume_controller: VolumeController, notification_manager: NotificationManager) -> None:
        self.volume_controller = volume_controller
        self.notification_manager = notification_manager
        self.states: Dict[str, HandGestureState] = {
            "LEFT": HandGestureState(),
            "RIGHT": HandGestureState(),
        }
        self.active_feature: str = "Idle"
        self.status_badge: str = "🔴 Lost"

    def update(
        self,
        frame: np.ndarray,
        hands_data: List[dict],
        current_time: float,
        volume_enabled: bool,
        camera_capture_enabled: bool,
        screenshot_enabled: bool,
    ) -> None:
        self.active_feature = "Tracking" if hands_data else "Searching"
        self.status_badge = "🟢 Tracking" if hands_data else "🟡 Searching"

        if hands_data:
            for hand_info in hands_data:
                self._process_gesture_state(hand_info, current_time, camera_capture_enabled, screenshot_enabled)
            if volume_enabled and Config.ENABLE_VOLUME_ROTATION:
                self._process_volume_rotation(frame, hands_data, current_time)
        else:
            self._reset_rotation_targets()

        self._render_frame_overlays(frame)

    def _process_gesture_state(
        self,
        hand_info: dict,
        current_time: float,
        camera_capture_enabled: bool,
        screenshot_enabled: bool,
    ) -> None:
        hand_label = hand_info["handedness"]
        state = self.states[hand_label]
        gesture_label = hand_info.get("gesture", "No Gesture")
        state.label_history.append(gesture_label)

        stable_label = self._get_stable_label(state)
        if stable_label != state.confirmed_label:
            # preserve the previous confirmed label for transition checks
            state.last_confirmed_label = state.confirmed_label
            state.confirmed_label = stable_label
            state.stable_since = current_time
            state.rotation_progress = 0.0
            # do not forcibly clear action_ready here; let _update_state_actions manage readiness
            # this ensures transitions like Open Palm -> Closed Fist can trigger actions

        self._update_state_actions(hand_label, hand_info, state, current_time, camera_capture_enabled, screenshot_enabled)

    def _get_stable_label(self, state: HandGestureState) -> str:
        if not state.label_history:
            return state.confirmed_label
        counted = Counter(state.label_history)
        label, count = counted.most_common(1)[0]
        if count / len(state.label_history) >= Config.GESTURE_STABILITY_RATIO:
            return label
        return state.confirmed_label

    def _update_state_actions(
        self,
        hand_label: str,
        hand_info: dict,
        state: HandGestureState,
        current_time: float,
        camera_capture_enabled: bool,
        screenshot_enabled: bool,
    ) -> None:
        if current_time < state.cooldown_until:
            state.active_feature = "Cooldown"
            state.status_badge = "🟡 Cooldown"
            return

        if state.confirmed_label == "Open Palm" and current_time - state.stable_since >= Config.ACTION_STABILITY_TIME:
            state.action_ready = True
            state.active_feature = "Gesture Ready"
            state.status_badge = "🟢 Ready"
            state.rotation_progress = min(1.0, (current_time - state.stable_since) / Config.ACTION_STABILITY_TIME)
            return

        if state.confirmed_label == "Closed Fist" and state.last_confirmed_label == "Open Palm" and state.action_ready:
            if hand_label == "RIGHT" and camera_capture_enabled and Config.ENABLE_CAMERA_CAPTURE:
                self._capture_photo(hand_info, current_time)
            elif hand_label == "LEFT" and screenshot_enabled and Config.ENABLE_SCREENSHOT:
                self._take_screenshot(current_time)
            state.action_ready = False
            state.cooldown_until = current_time + (Config.PHOTO_COOLDOWN if hand_label == "RIGHT" else Config.SCREENSHOT_COOLDOWN)
            state.active_feature = "Triggered"
            state.status_badge = "🟢 Action Triggered"
            state.rotation_progress = 0.0
            return

        if state.confirmed_label != "Open Palm":
            state.action_ready = state.action_ready and state.last_confirmed_label == "Open Palm"

        state.active_feature = "Tracking"
        state.status_badge = "🟢 Tracking"
        if state.confirmed_label != "Open Palm":
            state.rotation_progress = 0.0

    def _process_volume_rotation(self, frame: np.ndarray, hands_data: List[dict], current_time: float) -> None:
        selected_hand = self._select_rotation_hand(hands_data)
        if not selected_hand:
            return

        hand_label = selected_hand["handedness"]
        state = self.states[hand_label]
        point = self._get_index_tip(selected_hand, frame)
        if point is None:
            return

        state.rotation_history.append((point, current_time))
        rotation_result = self._detect_circular_motion(state)
        if rotation_result:
            direction, angle_degrees, angular_velocity = rotation_result
            delta = (angle_degrees / 360.0) * Config.ROTATION_SENSITIVITY
            if direction == "Clockwise":
                self.volume_controller.set_volume(self.volume_controller.get_volume() + delta)
                text = f"🔊 Volume Increasing {self.volume_controller.volume_status()}"
                self.active_feature = "Volume Rotation"
                self.status_badge = "🟢 Volume"
            else:
                self.volume_controller.set_volume(self.volume_controller.get_volume() - delta)
                text = f"🔊 Volume Decreasing {self.volume_controller.volume_status()}"
                self.active_feature = "Volume Rotation"
                self.status_badge = "🟢 Volume"

            if current_time - state.last_rotation_notification > Config.ROTATION_NOTIFICATION_INTERVAL:
                self.notification_manager.add_notification(text, duration=1.4)
                state.last_rotation_notification = current_time

            state.rotation_direction = direction
            state.rotation_progress = min(1.0, angle_degrees / 360.0)
            state.rotation_history.clear()
        else:
            # Fallback: small, incremental rotation detection for smoother BMW-like knob behavior
            points = [p for p, _ in state.rotation_history]
            if len(points) >= 3:
                angles = [calculate_angle(p, get_centroid(points)) for p in points]
                small_total = 0.0
                for a_prev, a_next in zip(angles, angles[1:]):
                    delta_ang = ((a_next - a_prev + 180) % 360) - 180
                    small_total += delta_ang
                # apply small rotation sensitivity
                if abs(small_total) > 6.0:
                    small_delta = (small_total / 360.0) * (Config.ROTATION_SENSITIVITY * 1.2)
                    if small_total < 0:
                        # Clockwise
                        self.volume_controller.set_volume(self.volume_controller.get_volume() + abs(small_delta))
                        text = f"🔊 Volume Increasing {self.volume_controller.volume_status()}"
                    else:
                        self.volume_controller.set_volume(self.volume_controller.get_volume() - abs(small_delta))
                        text = f"🔊 Volume Decreasing {self.volume_controller.volume_status()}"
                    if current_time - state.last_rotation_notification > Config.ROTATION_NOTIFICATION_INTERVAL:
                        self.notification_manager.add_notification(text, duration=0.9)
                        state.last_rotation_notification = current_time
                    state.rotation_direction = "Clockwise" if small_total < 0 else "Counter-Clockwise"
                    state.rotation_progress = min(1.0, abs(small_total) / 360.0)
                    # keep recent points to continue smoothing
                    if len(state.rotation_history) > max(6, Config.ROTATION_HISTORY // 2):
                        state.rotation_history.popleft()
                else:
                    state.rotation_direction = None
                    state.active_feature = "Tracking"
                    self.status_badge = "🟢 Tracking"
            else:
                state.rotation_direction = None
                state.active_feature = "Tracking"
                self.status_badge = "🟢 Tracking"

        self._draw_rotation_guide(frame, point, state)

    def _select_rotation_hand(self, hands_data: List[dict]) -> Optional[dict]:
        for hand_info in hands_data:
            if hand_info["handedness"] == "RIGHT":
                return hand_info
        return hands_data[0] if hands_data else None

    def _get_index_tip(self, hand_info: dict, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        landmarks = hand_info.get("landmarks")
        if not landmarks:
            return None
        height, width = frame.shape[:2]
        index_landmark = landmarks.landmark[8]
        return int(index_landmark.x * width), int(index_landmark.y * height)

    def _detect_circular_motion(self, state: HandGestureState) -> Optional[Tuple[str, float, float]]:
        if len(state.rotation_history) < Config.ROTATION_HISTORY:
            return None

        points = [point for point, _ in state.rotation_history]
        center = get_centroid(points)
        radii = [get_distance(point, center) for point in points]
        average_radius = sum(radii) / len(radii)
        if average_radius < Config.ROTATION_MIN_RADIUS:
            return None

        if max(radii) - min(radii) > average_radius * Config.ROTATION_RADIUS_VARIATION:
            return None

        angles = [calculate_angle(point, center) for point in points]
        total_rotation = 0.0
        for previous_angle, next_angle in zip(angles, angles[1:]):
            delta = ((next_angle - previous_angle + 180) % 360) - 180
            total_rotation += delta

        abs_rotation = abs(total_rotation)
        if abs_rotation < Config.ROTATION_MIN_ANGLE_DEGREES:
            return None

        duration = state.rotation_history[-1][1] - state.rotation_history[0][1]
        angular_velocity = abs_rotation / max(duration, 1e-3)
        if angular_velocity < Config.ROTATION_MIN_ANGULAR_VELOCITY:
            return None

        direction = "Clockwise" if total_rotation < 0 else "Counter-Clockwise"
        return direction, abs_rotation, angular_velocity

    def _capture_photo(self, hand_info: dict, current_time: float) -> None:
        Path(Config.CAPTURE_DIR).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        file_path = Path(Config.CAPTURE_DIR) / f"capture_{timestamp}.jpg"
        frame = hand_info.get("frame_capture")
        if frame is None:
            return
        cv2.imwrite(str(file_path), frame)
        thumbnail = cv2.resize(frame, (120, 90))
        self.notification_manager.add_notification("📸 Photo Captured", thumbnail=thumbnail)
        self._play_shutter_sound()

    def _take_screenshot(self, current_time: float) -> None:
        Path(Config.SCREENSHOT_DIR).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        file_path = Path(Config.SCREENSHOT_DIR) / f"screenshot_{timestamp}.png"
        with mss.mss() as sct:
            screenshot = np.array(sct.grab(sct.monitors[0]))
        screenshot = cv2.cvtColor(screenshot[..., :3], cv2.COLOR_BGRA2BGR)
        cv2.imwrite(str(file_path), screenshot)
        thumbnail = cv2.resize(screenshot, (120, 90))
        self.notification_manager.add_notification("🖥 Screenshot Saved", thumbnail=thumbnail)

    def _play_shutter_sound(self) -> None:
        try:
            import winsound

            winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
        except Exception:
            pass

    def _draw_rotation_guide(self, frame: np.ndarray, point: Tuple[int, int], state: HandGestureState) -> None:
        if point is None:
            return
        radius = 60
        cv2.circle(frame, point, radius, Config.UI_SECONDARY_COLOR, 2, cv2.LINE_AA)
        if state.rotation_direction:
            color = Config.UI_ACCENT_COLOR
            text = f"{state.rotation_direction}"
            cv2.putText(frame, text, (point[0] - 80, point[1] - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            progress_angle = int(360 * state.rotation_progress)
            cv2.ellipse(frame, point, (radius + 10, radius + 10), -90, 0, progress_angle, color, 3)

    def _render_frame_overlays(self, frame: np.ndarray) -> None:
        for hand_label, state in self.states.items():
            if state.confirmed_label == "Open Palm" and state.action_ready:
                self._draw_progress(frame, state)

    def _draw_progress(self, frame: np.ndarray, state: HandGestureState) -> None:
        center = (frame.shape[1] - 160, frame.shape[0] - 120)
        radius = 40
        cv2.circle(frame, center, radius, Config.UI_SECONDARY_COLOR, 2, cv2.LINE_AA)
        progress_angle = int(360 * state.rotation_progress)
        cv2.ellipse(frame, center, (radius, radius), -90, 0, progress_angle, Config.UI_ACCENT_COLOR, 4)
        cv2.putText(frame, f"Ready {int(state.rotation_progress * 100)}%", (center[0] - 70, center[1] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.UI_TEXT_COLOR, 2, cv2.LINE_AA)

    def _reset_rotation_targets(self) -> None:
        for state in self.states.values():
            state.rotation_history.clear()
            state.rotation_direction = None
            state.rotation_progress = 0.0

    def get_status(self) -> Tuple[str, str]:
        return self.active_feature, self.status_badge
