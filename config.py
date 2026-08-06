from typing import Tuple


class Config:
    """Central configuration for AI Hand Gesture Controller."""

    CAMERA_ID: int = 0
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    MAX_FPS: int = 30

    MIN_DETECTION_CONFIDENCE: float = 0.7
    MIN_TRACKING_CONFIDENCE: float = 0.7

    GESTURE_DISTANCE_THRESHOLD: float = 0.2
    VOLUME_SENSITIVITY: float = 1.0
    VOLUME_SMOOTHING: int = 5

    ENABLE_VOLUME_ROTATION: bool = True
    ENABLE_SCREENSHOT: bool = True
    ENABLE_CAMERA_CAPTURE: bool = True
    PHOTO_COOLDOWN: float = 2.0
    SCREENSHOT_COOLDOWN: float = 2.0
    ROTATION_SENSITIVITY: float = 0.08
    ROTATION_HISTORY: int = 12
    ROTATION_MIN_RADIUS: float = 30.0
    ROTATION_MIN_ANGLE_DEGREES: float = 140.0
    ROTATION_MIN_ANGULAR_VELOCITY: float = 110.0
    ROTATION_RADIUS_VARIATION: float = 0.35
    ROTATION_NOTIFICATION_INTERVAL: float = 0.6
    GESTURE_STABILITY_FRAMES: int = 8
    GESTURE_STABILITY_RATIO: float = 0.75
    ACTION_STABILITY_TIME: float = 0.35
    ACTION_LOG_LINES: int = 6
    NOTIFICATION_DURATION: float = 2.2
    LOG_DIR: str = "logs"
    CAPTURE_DIR: str = "captures"
    SCREENSHOT_DIR: str = "screenshots"

    UI_PRIMARY_COLOR: Tuple[int, int, int] = (41, 128, 185)
    UI_SECONDARY_COLOR: Tuple[int, int, int] = (236, 240, 241)
    UI_ACCENT_COLOR: Tuple[int, int, int] = (46, 204, 113)
    UI_WARN_COLOR: Tuple[int, int, int] = (231, 76, 60)
    UI_TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)
    UI_BACKGROUND_ALPHA: float = 0.65

    HELP_TEXT: Tuple[str, ...] = (
        "Q - Quit",
        "R - Reset",
        "V - Toggle Volume Control",
        "G - Toggle Gesture Detection",
        "F - Toggle Finger Counter",
        "C - Toggle Camera Capture",
        "S - Toggle Screenshot Capture",
        "H - Toggle Help",
    )

    DEFAULT_HAND_LANDMARKS: int = 21
    HAND_CONNECTIONS = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    )
