import cv2 as cv
import cv2 as cvKV
import numpy as np
import time
import socket
import struct
import keyboard
import pickle
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


HOST='192.168.100.7'
PORT=8485

PIHOST = '192.168.100.41'  # Connect to localhost
PIPORT = 1234

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn,addr=s.accept()

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))

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

    while len(data) < payload_size:
        data += conn.recv(4096)
        if not data:
            cvKV.destroyAllWindows()
            conn, addr = s.accept()
            continue
        # receive image row data form client socket
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    # unpack image using pickle
    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cvKV.imdecode(frame, cvKV.IMREAD_COLOR)

    img = frame.copy()
    # success, img = cap.read()
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