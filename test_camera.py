import cv2
import time

print("Starting camera test...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ CAMERA COULD NOT BE OPENED")
    raise SystemExit

print("✅ Camera opened successfully")
print("Press Q to quit")

start = time.time()

while True:
    ret, frame = camera.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    cv2.imshow("MindBloom Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("✅ Camera released")
print(f"Camera test duration: {time.time() - start:.2f} seconds")
