import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxhands=2, detectioncon=0.5, trackcon=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.detectioncon = detectioncon
        self.trackcon = trackcon
        self.myHands = mp.solutions.hands
        self.hands = self.myHands.Hands(self.mode, self.maxhands, self.detectioncon, self.trackcon)
        self.mp_drawing = mp.solutions.drawing_utils

    def findhands(self, frame, draw=True):
        imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(frame, handlms, self.myHands.HAND_CONNECTIONS)
        return frame

    def findposition(self, frame, handno=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 255, 0), cv2.FILLED)
        return lmlist


def main():
    fourcc = cv2.VideoWriter_fourcc(*'WMV1')
    out = cv2.VideoWriter('Handtrack.avi', fourcc, 20.0, (640, 480))
    prevt = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        _, frame = cap.read()
        frame = detector.findhands(frame)
        lmlist = detector.findposition(frame)
        if len(lmlist) != 0:
            print(lmlist[4])
        currt = time.time()
        fps = 1 / (currt - prevt)
        prevt = currt
        cv2.putText(frame, f'FPS:- {int(fps)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        out.write(frame)
        cv2.imshow('MediaPipe Hand Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
