import cv2
import time
from ultralytics import YOLO

# Eğitilmiş modeli yükle
model = YOLO("/home/harunrk/Sensors/Examples/runs/detect/train2/weights/best.pt")

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Kamera bulunamadı!")
    exit()

cv2.namedWindow("Lepton Stream", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Lepton Stream", 800, 600)

while True:
    time.sleep(0.01)  
    ret, frame = cap.read()
    if not ret:
        print("Frame alınamadı")
        break

    frame8 = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    bigger = cv2.resize(frame8, (320, 240), interpolation=cv2.INTER_LINEAR)

    results = model(bigger)[0]

    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        if conf >= 0.65: 
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(bigger, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(bigger, f'{int(conf*100)}%', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    display = cv2.rotate(bigger, cv2.ROTATE_90_CLOCKWISE)

    cv2.imshow("Lepton Stream", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
