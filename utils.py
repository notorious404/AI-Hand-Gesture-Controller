import math
import time
from typing import List, Tuple


def calculate_fps(previous_time: float, current_time: float) -> float:
    """Calculate frame per second based on previous and current timestamps."""
    if current_time == previous_time:
        return 0.0
    return 1.0 / (current_time - previous_time)


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a value between a minimum and maximum range."""
    return max(minimum, min(value, maximum))


def lerp(value: float, source_range: Tuple[float, float], target_range: Tuple[float, float]) -> float:
    """Linearly interpolate a value from one range to another."""
    (src_min, src_max), (tgt_min, tgt_max) = (source_range, target_range)
    if src_max == src_min:
        return tgt_min
    normalized = (value - src_min) / (src_max - src_min)
    return tgt_min + normalized * (tgt_max - tgt_min)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two values safely without ZeroDivisionError."""
    return numerator / denominator if denominator else default


def format_percentage(value: float) -> str:
    """Format a float value as a rounded percentage string."""
    return f"{int(clamp(value, 0, 100))}%"


def get_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """Return the Euclidean distance between two 2D points."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def calculate_angle(point: Tuple[int, int], center: Tuple[int, int]) -> float:
    """Return the angle of a point relative to a center in degrees."""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return math.degrees(math.atan2(dy, dx)) % 360


def get_centroid(points: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Return the centroid of a list of points."""
    if not points:
        return 0, 0
    x = sum(point[0] for point in points) / len(points)
    y = sum(point[1] for point in points) / len(points)
    return int(x), int(y)


def normalize_point(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    """Convert normalized landmarks to pixel coordinates."""
    return int(x * width), int(y * height)
