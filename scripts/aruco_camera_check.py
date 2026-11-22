"""Live webcam check for the robot's ArUco markers: outlines, ids, pixel centres and a frame rate.

Read-only: it opens the camera and a window, and nothing else. No robot link, no file is written.

    python scripts/aruco_camera_check.py        (press Q to quit)

The marker dictionary is not chosen here. It is the project dictionary (``bot.dictionary`` in
``config/arena_overlay.yaml``), built by ``control_center.backend.marker_settings``, so this script, the legacy window
and the Command Centre always agree. The dictionary in use is printed at start-up.

Until 22 November 2026 this file was ``april_test.py`` in the repository root. The on-screen wording ("Tags: N",
"No tags detected") is unchanged because the test playbook (CAM-1, CAM-2) refers to it.
"""

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control_center.backend.marker_settings import build_detector, project_dictionary  # noqa: E402

# Camera to open. 1 is the Logitech C525 on the development laptop (0 is the built-in camera).
CAMERA_INDEX = 1
WINDOW = "ArUco marker check"


def main() -> int:
    print("Marker dictionary:", project_dictionary())
    detector = build_detector(project_dictionary())

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return 1

    print("ArUco marker check running.")
    print("Press Q to quit.")

    previous_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Marker detection works on grayscale images.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for tag_corners, tag_id in zip(corners, ids.flatten()):
                points = tag_corners.reshape((4, 2))
                center_x = int(points[:, 0].mean())
                center_y = int(points[:, 1].mean())

                cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)
                cv2.putText(frame, f"ID {tag_id} ({center_x}, {center_y})", (center_x + 10, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, f"Tags: {len(ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No tags detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
