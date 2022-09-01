import cv2
import mediapipe as mp
import time
import FaceDetectionModule as FDM

cap = cv2.VideoCapture(0)
detector = FDM.FaceDetection()
ptime = 0
while True:
    success, img = cap.read()
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    img = detector.faceDetect(img)

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (36, 23, 21), 3)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
