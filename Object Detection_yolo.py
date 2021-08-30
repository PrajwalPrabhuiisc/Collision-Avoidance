import cv2
import numpy as np
import time

wT, hT = 320, 320
confthreshold = 0.5
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Object Detection.avi', fourcc, 5.0, (640, 480))

def datasetlist(class_file):
    class_name = []
    with open(class_file, 'rt') as f:
        class_name = f.read().rstrip('\n').split('\n')
    return class_name


class_file = 'coco.names'
class_names = datasetlist(class_file)
# print(class_names)
model_config = "yolov3-320.cfg"
model_weights = "yolov3.weights"
prevt = 0
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 20)
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
nmsthreshold = 0.2


def findObjects(outputs, image):
    h, w, c = image.shape
    bbox = []
    clId = []
    confidence = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classID = np.argmax(scores)
            confs = scores[classID]
            if confs > confthreshold:
                wt, ht = int(det[2] * w), int(det[3] * h)
                x, y = int((det[0] * w) - wt / 2), int((det[1] * w) - ht / 2)
                bbox.append([x, y, wt, ht])
                clId.append(classID)
                confidence.append(float(confs))
    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confidence, confthreshold, nmsthreshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(image, f'{class_names[clId[i]].upper()} {int(confidence[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    _, frame = cap.read()
    currt = time.time()
    fps = 1 / (currt - prevt)
    prevt = currt
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (wT, hT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    output = net.forward(outputNames)
    # print(output[0].shape)
    # print(output[1].shape)
    # print(output[2].shape)
    findObjects(output, frame)

    cv2.putText(frame, f'FPS:- {int(fps)}', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Yolo-object detection', frame)
    out.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()

