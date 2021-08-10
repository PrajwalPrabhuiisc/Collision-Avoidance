import cv2
import mediapipe as mp
import time

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
prevt = 0
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        _, frame = cap.read()
        currt = time.time()
        fps = 1 / (currt - prevt)
        prevt = currt
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
        cv2.putText(image, f'FPS:- {int(fps)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        out.write(image)
        cv2.imshow('MediaPipe Face Detection', image)
        if cv2.waitKey(1) == ord('q'):
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()
