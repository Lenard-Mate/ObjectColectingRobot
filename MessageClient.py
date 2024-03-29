import socket

import keyboard

HOST = '192.168.100.41'  # Connect to localhost
PORT = 1234

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    while True:
        if keyboard.is_pressed('w'):
            s.sendall(b'w')
        elif keyboard.is_pressed('d'):
            s.sendall(b'd')
        elif keyboard.is_pressed('a'):
            s.sendall(b'a')
        elif keyboard.is_pressed('s'):
            s.sendall(b's')
        else:
            s.sendall(b'k')
        data = s.recv(1024)
        print(f"Received: {data.decode()}")
