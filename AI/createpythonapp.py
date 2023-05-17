import cv2 as cv
import numpy as np
import time
cap = cv.VideoCapture(0)
whT = 220
confThreshold = 0.4
nmsThreshold = 0.2

#### LOAD MODEL
## Coco Names
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('n').split('n')
# print(classNames[0])
## Model Files
modelConfiguration = "yolov3_testing.cfg"
modelWeights = "yolov3_training_cio.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
prev_timeframe=0
new_timeframe=0

def findObjects(outputs, img,fps):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv.rectangle(img, (x, y), (x + w, y + h), (25, 28, 255), 2)
        cv.putText(img, f'{"Cio"} {int(confs[i] * 100)}%',
                   (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv.putText(img, f'{fps}FPS',(20,20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

while True:
    success, img = cap.read()
    new_timeframe=time.time()
    fps = 1 /(new_timeframe-prev_timeframe)
    prev_timeframe=new_timeframe
    fps=int(fps)
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs, img,fps)

    cv.imshow('Image', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# Destroy all the windows
cv.destroyAllWindows()