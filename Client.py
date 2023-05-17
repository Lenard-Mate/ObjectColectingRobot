import cv2

import socket
import struct
import pickle
import imutils

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.100.7', 8485))
cam = cv2.VideoCapture(0)


while True:
    ret, frame = cam.read()
    print(frame)
    frame = imutils.resize(frame, width=640,height=480)
    frame = cv2.flip(frame, 180)
    result, image = cv2.imencode('.png', frame)
    data = pickle.dumps(image, 0)
    size = len(data)


    client_socket.sendall(struct.pack(">L", size) + data)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()