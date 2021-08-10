import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture('2.mp4')
prevt = 0
mypose = mp.solutions.pose
pose = mypose.Pose()
mpdraw = mp.solutions.drawing_utils
while True:
    _, frame = cap.read()
    currt = time.time()
    fps = 1 / (currt - prevt)
    prevt = currt
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpdraw.draw_landmarks(frame, results.pose_landmarks, mypose.POSE_CONNECTIONS)
    cv2.putText(frame, f'FPS:- {int(fps)}', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Mediapipe Pose Bare Min', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
