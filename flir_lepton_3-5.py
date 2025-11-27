# This script captures and displays video from a FLIR Lepton 3.5 thermal camera using OpenCV.

import cv2
import time

cap = cv2.VideoCapture(2) 
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("Lepton Stream", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame alınamadı")
        break

    frame8 = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # display = cv2.rotate(frame8, cv2.ROTATE_90_CLOCKWISE)

    cv2.imshow("Lepton Stream", frame8)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
