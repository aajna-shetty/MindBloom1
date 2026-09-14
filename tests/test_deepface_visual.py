import sys
import os
from collections import deque, Counter

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


# ==============================
# EMOTION MEMORY
# ==============================

emotion_history = deque(maxlen=10)

current_emotion = "Waiting..."
current_scores = {}



while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Frame capture failed")
        break


    frame_count += 1



    # DeepFace is expensive
    # Run every 15 frames

    if frame_count % 15 == 0:


        print(
            f"Analyzing frame {frame_count}"
        )


        try:

            result = processor.analyze_frame(frame)


            if result.success:


                scores = {
                    k: float(v)
                    for k, v in result.emotion_scores.items()
                }


                current_scores = scores



                # Highest probability emotion

                predicted_emotion = max(
                    scores,
                    key=scores.get
                )


                confidence = scores[predicted_emotion]



                # Ignore weak predictions

                if confidence >= 0.40:

                    emotion_history.append(
                        predicted_emotion
                    )


                # Temporal smoothing

                if len(emotion_history) > 0:

                    current_emotion = Counter(
                        emotion_history
                    ).most_common(1)[0][0]


                else:

                    current_emotion = "Uncertain"



                print(
                    "Raw:",
                    predicted_emotion,
                    confidence
                )


                print(
                    "Smoothed:",
                    current_emotion
                )



            else:

                current_emotion = "No face"
                current_scores = {}



        except Exception as e:

            print(
                "DeepFace error:",
                e
            )

            current_emotion = "Error"



    # ==============================
    # DRAW ON CAMERA
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


    for emotion, score in sorted(
        current_scores.items(),
        key=lambda x: x[1],
        reverse=True
    ):


        cv2.putText(
            frame,
            f"{emotion}: {score*100:.1f}%",
            (20,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )


        y += 30



    cv2.imshow(
        "MindBloom - DeepFace Emotion",
        frame
    )



    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cap.release()

cv2.destroyAllWindows()


print("Camera closed")