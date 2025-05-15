import cv2
from ultralytics import YOLO

model = YOLO('best.pt')

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Nope")
    exit()

while True:
    ret, frame = camera.read()
    if not ret:
        print("Fejl")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
