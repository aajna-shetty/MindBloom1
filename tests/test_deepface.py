import sys
import os

# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import cv2

from app.video.deepface_processor import DeepFaceProcessor


print("Starting DeepFace visual emotion test...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera could not open")
    exit()


processor = DeepFaceProcessor()

print("✅ Camera opened")
print("Press Q to quit")


frame_count = 0

current_emotion = "Waiting..."
current_scores = {}


while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Frame capture failed")
        break


    frame_count += 1


    # DeepFace is expensive
    # Run once every 30 frames
    if frame_count % 30 == 0:

        print(f"Analyzing frame {frame_count}...")

        try:

            result = processor.analyze_frame(frame)


            if result.success:

                current_emotion = result.dominant_emotion

                current_scores = result.emotion_scores


                print(
                    "Emotion:",
                    current_emotion
                )


            else:

                current_emotion = "No face"


        except Exception as e:

            print(
                "DeepFace error:",
                e
            )

            current_emotion = "Error"


    # ==============================
    # DRAW RESULTS ON CAMERA WINDOW
    # ==============================


    cv2.putText(
        frame,
        f"Emotion: {current_emotion}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )


    y = 80


    for emotion, score in current_scores.items():

        text = (
            f"{emotion}: "
            f"{float(score)*100:.1f}%"
        )


        cv2.putText(
            frame,
            text,
            (20,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        y += 30



    cv2.imshow(
        "MindBloom - Emotion Detection",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cap.release()
cv2.destroyAllWindows()


print("Camera closed")