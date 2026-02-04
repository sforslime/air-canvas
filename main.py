"""Entry point for the Air Canvas interactive drawing experience."""
from __future__ import annotations

import sys
from typing import Dict

import cv2

from canvas import DrawingCanvas
from gesture_detector import GestureDetector
from utils import GestureFilter, PointSmoother, blend_frames, draw_cursor, put_multiline_text

GESTURE_COLORS: Dict[str, tuple[int, int, int]] = {
    "draw": (0, 255, 0),
    "fist": (0, 0, 255),
    "clear": (0, 255, 255),
    "idle": (255, 255, 255),
}


def run() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to access the webcam. Make sure a camera is connected.")
        sys.exit(1)

    detector = GestureDetector()
    canvas: DrawingCanvas | None = None
    prev_point = None
    smoother = PointSmoother(momentum=0.75)
    gesture_filter = GestureFilter(confirm_frames=3, default="idle")
    last_gesture = "idle"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or cannot read from webcam. Exiting.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            if canvas is None:
                canvas = DrawingCanvas(width, height)
            else:
                canvas.resize_if_needed(width, height)

            result = detector.process(frame)
            if result.landmarks is not None:
                detector.draw_hand_annotations(frame, result.landmarks)

            cursor = smoother.update(result.cursor)
            gesture = gesture_filter.update(result.gesture)

            if gesture == "draw" and cursor is not None:
                if prev_point is None:
                    prev_point = cursor
                canvas.draw_line(prev_point, cursor)
                prev_point = cursor
            else:
                prev_point = None

            if gesture == "clear" and last_gesture != "clear" and canvas is not None:
                canvas.clear()

            last_gesture = gesture

            overlay = blend_frames(frame, canvas.get_image()) if canvas else frame
            cursor_color = GESTURE_COLORS.get(gesture, (255, 255, 255))
            draw_cursor(overlay, cursor, cursor_color)
            put_multiline_text(
                overlay,
                [
                    f"Gesture: {gesture.upper()}",
                    "Controls: INDEX=draw, FIST=stop, PALM=clear",
                    "Press 'q' to quit",
                ],
            )

            cv2.imshow("Air Canvas", overlay)
            if canvas is not None:
                cv2.imshow("Canvas", canvas.get_image())

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
