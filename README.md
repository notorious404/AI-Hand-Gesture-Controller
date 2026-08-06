# AI Hand Gesture Controller

AI Hand Gesture Controller is a professional Python desktop application that detects hand gestures in real time using a webcam. It counts fingers, recognizes common gestures, and can control Windows volume using thumb-index distance.

## Features

- Real-time webcam support with HD resolution selection
- MediaPipe-powered hand detection for one or two hands
- Independent finger counts for left and right hands
- Rule-based gesture recognition for common hand poses
- Windows volume control using thumb-index distance
- Smooth UI overlay with FPS, gesture labels, confidence, and controls
- Circular volume control via index finger rotation
- Gesture-based webcam photo capture and desktop screenshot capture
- Animated action notifications, logging, and modern cards
- Keyboard shortcuts for live feature toggles and help display
- Graceful error handling for webcam and audio interface issues

## Installation

1. Install Python 3.11 or newer.
2. Open a terminal in the project folder.
3. Create an isolated environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```
> Note: `mediapipe` is pinned to `0.10.13` for compatibility with the current hand tracking implementation.
5. Run the application:

```bash
python main.py
```

## Requirements

- Python 3.11+
- Windows OS for volume control via `pycaw`
- Webcam

## Screenshots

> Add your screenshots to the `screenshots/` folder and reference them here.

## How Finger Counting Works

The application uses MediaPipe hand landmarks and handedness classification to identify extended fingers. Thumb detection is handled separately for left and right hands so counts remain accurate when both hands are visible.

## How Gesture Recognition Works

Gesture recognition is implemented with rule-based functions that inspect landmark positions. This makes the system fast and simple to extend with future machine learning classification.

## How Volume Control Works

The application supports two volume control modes:

- Thumb-index distance control for fast manual volume adjustment.
- Circular index-finger rotation control for smooth gesture-based volume changes.

When the right index finger traces a stable circular path, the system detects clockwise or counter-clockwise rotation and updates the Windows master volume gradually.

## Project Structure

```
AI-Hand-Gesture-Controller/

├── main.py
├── action_manager.py
├── notification_manager.py
├── hand_detector.py
├── gesture_recognizer.py
├── finger_counter.py
├── volume_controller.py
├── config.py
├── utils.py
├── requirements.txt
├── README.md
├── assets/
├── captures/
├── screenshots/
├── logs/
└── models/
```

New modules:

- `action_manager.py` — coordinates volume rotation, screenshot, and camera capture triggers.
- `notification_manager.py` — renders slide-in alerts, logs actions, and shows thumbnails.

## Keyboard Shortcuts

- `Q` → Quit
- `R` → Reset application state
- `V` → Toggle volume control
- `G` → Toggle gesture detection
- `F` → Toggle finger counter
- `C` → Toggle camera photo capture gesture
- `S` → Toggle screenshot gesture
- `H` → Toggle help overlay

## Troubleshooting

- If the webcam is unavailable, ensure no other app is using it and try again.
- If volume control does not work, verify `pycaw` is installed and you are on Windows.
- If the frame rate is low, reduce resolution in `config.py`.

## Future Improvements

- Add mouse control and virtual drawing gestures
- Add presentation control and media shortcuts
- Add brightness control and smart home integration
- Add AI-based gesture classification and sign language recognition
- Add screenshot capture by gesture
