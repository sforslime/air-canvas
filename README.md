# Air Canvas – Hand Gesture Drawing App

Air Canvas turns your webcam into an invisible paintbrush. Wave your hand in mid-air and watch smooth digital ink appear in real time, powered by OpenCV for video capture and MediaPipe Hands for landmark tracking.

## Features
- Real-time hand landmark detection with finger skeleton overlays
- Gesture-controlled drawing: point to draw, make a fist to pause, show an open palm to clear
- Smooth, anti-aliased strokes and a live cursor indicator for precise control
- Dual-window display: augmented webcam feed plus a dedicated canvas view
- Graceful handling of webcam availability and keyboard interrupt exits

## Requirements
- Python 3.10 or newer
- Webcam or USB camera compatible with OpenCV
- pip for dependency management

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate  # On macOS/Linux use: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the App
```bash
python main.py
```
Press `q` in the Air Canvas window to exit.

## Gesture Mapping
| Gesture | How | Action |
| --- | --- | --- |
| Draw | Extend the index finger while folding the rest | Paint continuous strokes |
| Pause | Form a fist (all fingers folded) | Lift the digital pen without clearing the canvas |
| Clear | Display an open palm (all fingers extended) | Reset the canvas to start over |

## Tips & Troubleshooting
- Ensure good, even lighting so MediaPipe can detect landmarks reliably.
- If the wrong gesture triggers, keep your hand within the frame and slow down transitions between poses.
- Strokes follow the index finger tip; move the whole hand rather than bending only the finger for smoother curves.
- If the webcam cannot be opened, close other apps that may be using it and rerun `python main.py`.

## Folder Structure
```
air-canvas/
├── main.py              # Application entry point & UI loop
├── gesture_detector.py  # MediaPipe Hands wrapper + gesture logic
├── canvas.py            # Canvas class for persistent drawing
├── utils.py             # Helper utilities for rendering overlays
├── requirements.txt     # Runtime dependencies
└── README.md            # Project guide
```

Add your own screenshots or GIFs of the running app directly to this README to showcase results.

## Fork & Extend the Project
1. Visit the GitHub repository and click **Fork** to copy it to your account.
2. Clone your fork locally: `git clone https://github.com/<your-username>/air-canvas.git`.
3. Add the original project as an upstream remote for easy syncing:
	```bash
	git remote add upstream https://github.com/sforslime/air-canvas.git
	git fetch upstream
	git checkout main
	git merge upstream/main
	```
4. Create a feature branch (`git checkout -b feature/new-gesture`), implement changes, and push to your fork.
5. Open a pull request against `sforslime/air-canvas` to share improvements.

Issues and enhancements are welcome—keep the gestures flowing!
