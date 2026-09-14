import cv2
import time

from app.video.capture import VideoCaptureSession
from app.video.mediapipe_processor import MediaPipeProcessor


def main():

    print("Starting MediaPipe visual test...")
    print("A camera window should open.")
    print("Press Q to quit.")

    face_count = 0
    total_frames = 0

    with VideoCaptureSession(
        source=0,
        duration_seconds=60,
        target_fps=30,
    ) as camera:

        with MediaPipeProcessor(
            max_num_faces=2,
            refine_landmarks=True,
        ) as processor:

            while True:

                # Get one frame
                frame = next(
                    camera.frames(),
                    None,
                )

                if frame is None:
                    print("Camera finished.")
                    break

                total_frames += 1

                if not frame.valid or frame.rgb_frame is None:
                    continue

                # --------------------------------------------------
                # MediaPipe
                # --------------------------------------------------

                result = processor.process_frame(
                    frame.rgb_frame,
                    timestamp_ms=int(
                        frame.timestamp * 1000
                    ),
                )

                if result.face_detected:
                    face_count += 1

                # --------------------------------------------------
                # Convert RGB -> BGR for OpenCV display
                # --------------------------------------------------

                display = cv2.cvtColor(
                    frame.rgb_frame,
                    cv2.COLOR_RGB2BGR,
                )

                # --------------------------------------------------
                # Text
                # --------------------------------------------------

                if result.face_detected:

                    cv2.putText(
                        display,
                        "FACE DETECTED",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        display,
                        f"Faces: {result.num_faces}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    if result.ear is not None:

                        cv2.putText(
                            display,
                            f"EAR: {result.ear:.3f}",
                            (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                    if result.gaze_horizontal is not None:

                        cv2.putText(
                            display,
                            (
                                f"Gaze: "
                                f"{result.gaze_horizontal:.2f}, "
                                f"{result.gaze_vertical:.2f}"
                            ),
                            (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                else:

                    cv2.putText(
                        display,
                        "NO FACE DETECTED",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

                # --------------------------------------------------
                # Show camera
                # --------------------------------------------------

                cv2.imshow(
                    "MindBloom - MediaPipe Test",
                    display,
                )

                # --------------------------------------------------
                # Quit with Q
                # --------------------------------------------------

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

    cv2.destroyAllWindows()

    print()
    print("===================================")
    print("MEDIAPIPE VISUAL TEST COMPLETE")
    print("===================================")
    print(f"Total frames: {total_frames}")
    print(f"Frames with face: {face_count}")


if __name__ == "__main__":
    main()