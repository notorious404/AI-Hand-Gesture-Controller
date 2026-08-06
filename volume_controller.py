from typing import Tuple

import comtypes
from ctypes import POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from config import Config
from utils import clamp, format_percentage


class VolumeController:
    """Windows system volume controller using pycaw."""

    def __init__(self) -> None:
        self.volume_api_available = False
        self.volume_interface = None
        self.min_volume = 0.0
        self.max_volume = 1.0
        self.current_volume = 0.0
        self.volume_history = []
        self._initialize_audio_interface()

    def _initialize_audio_interface(self) -> None:
        """Attempt to initialize the Windows audio interface for volume control."""
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume_interface = comtypes.cast(interface, POINTER(IAudioEndpointVolume))
            self.min_volume, self.max_volume, _ = self.volume_interface.GetVolumeRange()
            self.current_volume = self.get_volume()
            self.volume_api_available = True
        except Exception:
            self.volume_api_available = False

    def get_volume(self) -> float:
        """Read the current system volume level in normalized 0.0 to 1.0 scale."""
        if self.volume_interface is None:
            return 0.0
        db_value = self.volume_interface.GetMasterVolumeLevel()
        normalized = clamp((db_value - self.min_volume) / (self.max_volume - self.min_volume), 0.0, 1.0)
        return normalized

    def set_volume(self, normalized_volume: float) -> None:
        """Set the system volume after smoothing and clamping the target level."""
        if not self.volume_api_available or self.volume_interface is None:
            return

        normalized = clamp(normalized_volume, 0.0, 1.0)
        self.volume_history.append(normalized)
        if len(self.volume_history) > Config.VOLUME_SMOOTHING:
            self.volume_history.pop(0)

        target = sum(self.volume_history) / len(self.volume_history)
        db_value = self.min_volume + target * (self.max_volume - self.min_volume)
        self.volume_interface.SetMasterVolumeLevel(db_value, None)
        self.current_volume = target

    def volume_status(self) -> str:
        """Return a formatted volume status string for display."""
        return format_percentage(self.current_volume * 100)

    def map_distance_to_volume(self, distance: float, min_distance: float, max_distance: float) -> float:
        """Map distance between thumb and index finger to normalized volume."""
        if distance <= min_distance:
            return 0.0
        if distance >= max_distance:
            return 1.0
        return (distance - min_distance) / (max_distance - min_distance) * Config.VOLUME_SENSITIVITY

    def render_volume_ui(self, frame, volume_text: str, start_point: Tuple[int, int], end_point: Tuple[int, int]) -> None:
        """Draw volume details and UI accents on the video frame."""
        cv2 = __import__("cv2")
        cv2.line(frame, start_point, end_point, Config.UI_ACCENT_COLOR, 3)
        cv2.circle(frame, start_point, 10, Config.UI_ACCENT_COLOR, cv2.FILLED)
        cv2.circle(frame, end_point, 10, Config.UI_ACCENT_COLOR, cv2.FILLED)
        volume_bar_x, volume_bar_y = 30, frame.shape[0] - 120
        volume_bar_height = 200
        volume_fill = int(volume_bar_height * self.current_volume)

        cv2.rectangle(frame, (volume_bar_x, volume_bar_y), (volume_bar_x + 30, volume_bar_y + volume_bar_height), Config.UI_SECONDARY_COLOR, 2)
        cv2.rectangle(frame, (volume_bar_x, volume_bar_y + volume_bar_height - volume_fill), (volume_bar_x + 30, volume_bar_y + volume_bar_height), Config.UI_ACCENT_COLOR, cv2.FILLED)
        cv2.putText(frame, volume_text, (volume_bar_x + 40, volume_bar_y + volume_bar_height - volume_fill - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.UI_ACCENT_COLOR, 2)
