import cv2
import numpy as np
from keras.models import load_model
import time
model = load_model("./model2-007.model")

results = {0: 'Please wear mask', 1: 'mask is worn'}
GR_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
rect_size = 4
cap = cv2.VideoCapture('http://10.163.240.21:8080/video')
haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
prevt = 0
# cap.set(cv2.CAP_PROP_FPS, 5)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('Mask Detection.avi', fourcc, 20.0, (640, 480))
while True:
    (rval, im) = cap.read()
    im = cv2.flip(im, 1, 1)
    currt = time.time()
    fps = 1 / (currt - prevt)
    prevt = currt
    cv2.putText(im, f'FPS:- {int(fps)}', (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f]

        face_img = im[y:y + h, x:x + w]
        rerect_sized = cv2.resize(face_img, (150, 150))
        normalized = rerect_sized / 255.0
        reshaped = np.reshape(normalized, (1, 150, 150, 3))
        reshaped = np.vstack([reshaped])
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(im, (x, y), (x + w, y + h), GR_dict[label], 2)
        cv2.rectangle(im, (x, y - 40), (x + w, y), GR_dict[label], -1)
        cv2.putText(im, results[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow('LIVE', im)
    # out.write(im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
