import time
import cv2
import mediapipe as mp

cap=cv2.VideoCapture(0)
ptime=0
mpFace= mp.solutions.face_detection
mpDraw= mp.solutions.drawing_utils
faceDetection = mpFace.FaceDetection(min_detection_confidence=0.6)


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)

    if results.detections:
        for id,det in enumerate(results.detections):
            # mpDraw.draw_detection(img,det)
            # print(id,det)
            # print(det.score)
            # print(det.location_data.relative_bounding_box)
            bbox= det.location_data.relative_bounding_box
            h, w, c = img.shape
            bb= int(bbox.xmin * w), int(bbox.ymin * h),\
                int(bbox.width * w), int(bbox.height * h)
            cv2.rectangle(img, bb, (255,219,50), 2)
            cv2.putText(img, f'{int(det.score[0]*100)}%',
                        (bb[0], bb[1]-20), cv2.FONT_HERSHEY_PLAIN,
                        2, (230,199,0), 3)



    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime

    cv2.putText(img,f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (36,23,21), 3)
    cv2.imshow('Image',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
