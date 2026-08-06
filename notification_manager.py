from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from config import Config
from utils import clamp


@dataclass
class Notification:
    text: str
    color: Tuple[int, int, int]
    start_time: float
    duration: float
    thumbnail: Optional[np.ndarray] = None
    progress: float = 0.0


class NotificationManager:
    """Manage on-screen notifications and action log rendering."""

    def __init__(self) -> None:
        self.notifications: List[Notification] = []
        self.log_lines: Deque[str] = deque(maxlen=Config.ACTION_LOG_LINES)
        self._ensure_log_directory()
        self.logger = self._configure_logger()

    def _ensure_log_directory(self) -> None:
        Path(Config.LOG_DIR).mkdir(parents=True, exist_ok=True)

    def _configure_logger(self) -> logging.Logger:
        logger = logging.getLogger("AIHandGestureController")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            log_file = Path(Config.LOG_DIR) / "app.log"
            handler = logging.FileHandler(log_file, encoding="utf-8")
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
            logger.addHandler(handler)
        return logger

    def add_notification(
        self,
        text: str,
        color: Tuple[int, int, int] = Config.UI_ACCENT_COLOR,
        duration: float = Config.NOTIFICATION_DURATION,
        thumbnail: Optional[np.ndarray] = None,
        progress: float = 0.0,
    ) -> None:
        notification = Notification(
            text=text,
            color=color,
            start_time=time.time(),
            duration=duration,
            thumbnail=thumbnail,
            progress=progress,
        )
        self.notifications.append(notification)
        self.log_action(text)

    def log_action(self, message: str) -> None:
        self.logger.info(message)
        timestamp = time.strftime("%H:%M:%S")
        self.log_lines.appendleft(f"[{timestamp}] {message}")

    def render(self, frame: np.ndarray) -> None:
        self._render_action_log(frame)
        self._render_notifications(frame)

    def _render_action_log(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        card_top = height - 220
        card_left = 10
        card_right = 420
        card_bottom = height - 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (card_left, card_top), (card_right, card_bottom), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (card_left, card_top), (card_right, card_bottom), Config.UI_PRIMARY_COLOR, 2)
        cv2.putText(frame, "Recent Actions", (card_left + 18, card_top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.UI_TEXT_COLOR, 2, cv2.LINE_AA)
        for index, line in enumerate(list(self.log_lines)[: Config.ACTION_LOG_LINES], start=1):
            cv2.putText(
                frame,
                line,
                (card_left + 18, card_top + 30 + index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                Config.UI_SECONDARY_COLOR,
                1,
                cv2.LINE_AA,
            )

    def _render_notifications(self, frame: np.ndarray) -> None:
        current_time = time.time()
        valid_notifications: List[Notification] = []
        for notification in self.notifications:
            elapsed = current_time - notification.start_time
            if elapsed > notification.duration:
                continue
            valid_notifications.append(notification)
            self._draw_notification(frame, notification, elapsed)
        self.notifications = valid_notifications

    def _draw_notification(self, frame: np.ndarray, notification: Notification, elapsed: float) -> None:
        width = frame.shape[1]
        y_offset = int(20 + len(self.notifications) * 70)
        alpha = min(1.0, elapsed * 2.0)
        overlay = frame.copy()
        card_start = (width - 380, y_offset)
        card_end = (width - 10, y_offset + 60)
        cv2.rectangle(overlay, card_start, card_end, notification.color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.9 * alpha, frame, 1 - 0.9 * alpha, 0, frame)
        cv2.rectangle(frame, card_start, card_end, Config.UI_SECONDARY_COLOR, 1)
        text_location = (card_start[0] + 18, card_start[1] + 35)
        cv2.putText(frame, notification.text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.65, Config.UI_TEXT_COLOR, 2, cv2.LINE_AA)
        if notification.thumbnail is not None:
            thumbnail = notification.thumbnail.copy()
            thumbnail_height = card_end[1] - card_start[1] - 16
            thumbnail_width = int(thumbnail.shape[1] * thumbnail_height / thumbnail.shape[0])
            thumbnail = cv2.resize(thumbnail, (thumbnail_width, thumbnail_height))
            x = card_start[0] + 18
            y = card_start[1] + 10
            frame[y : y + thumbnail.shape[0], x : x + thumbnail.shape[1]] = thumbnail
            cv2.rectangle(frame, (x - 2, y - 2), (x + thumbnail.shape[1] + 2, y + thumbnail.shape[0] + 2), Config.UI_SECONDARY_COLOR, 1)
