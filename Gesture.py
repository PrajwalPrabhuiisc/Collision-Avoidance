import cv2
import time
import os
import handtrack as htm

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
fourcc = cv2.VideoWriter_fourcc(*'WMV1')
out = cv2.VideoWriter('Gesture.avi', fourcc, 20.0, (640, 480))
folderpath = 'Finger Images'
mylist = os.listdir(folderpath)
overlaylist = []
for impath in mylist:
    image = cv2.imread(f'{folderpath}/{impath}')
    image = cv2.resize(image, (150, 180))
    overlaylist.append(image)
prevt = 0
tipids = [4, 8, 12, 16, 20]
detector = htm.handDetector(detectioncon=0.75)
while True:
    ret, frame = cap.read()
    frame = detector.findhands(frame)
    lmlist = detector.findposition(frame, draw=False)
    if len(lmlist) != 0:
        fingers = []
        if lmlist[tipids[0]][1] > lmlist[tipids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if lmlist[tipids[id]][2] < lmlist[tipids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalfinger = fingers.count(1)
        print(totalfinger)
        h, w, c = overlaylist[totalfinger - 1].shape
        frame[0:h, 0:w] = overlaylist[totalfinger - 1]
        cv2.rectangle(frame, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, str(totalfinger), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 255), 20)
    currt = time.time()
    fps = 1 / (currt - prevt)
    prevt = currt
    cv2.putText(frame, f'FPS:- {int(fps)}', (450, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    out.write(frame)
    cv2.imshow('Detected image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
