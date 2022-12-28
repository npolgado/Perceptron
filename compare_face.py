#!usr/bin/env python3
import time

import cv2
import numpy as np

from perceptron import *

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

RESIZE_FACTOR = 512
TRAIN = False

try:
    p = Perceptron(np.zeros((RESIZE_FACTOR, RESIZE_FACTOR)))

    st = time.monotonic()
    is_face = False
    count = 0

    while True:  
        count += 1
        dt = float(time.monotonic() - st)
        if float(time.monotonic() - st) > 10:
            fps = count / 60
            print(f"---------------- FPS = {fps} ----------------")
            count = 0
            st = time.monotonic()

        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        is_face = len(faces)

        if not is_face:
            resized_image = cv2.resize(gray, (RESIZE_FACTOR, RESIZE_FACTOR))
            normalized_image = resized_image / 255.0
            
            guess = bool(p.predict(normalized_image) > p.bias)

            # if guess == True and TRAIN:
            #     print("\t\tIncorrect...")
            #     p.train(normalized_image, 0.0)
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face = gray[y:y+h, x:x+w]

                resized_image = cv2.resize(face, (RESIZE_FACTOR, RESIZE_FACTOR))
                normalized_image = resized_image / 255.0

                guess = bool(p.predict(normalized_image) > p.bias)
                
                # if guess == True and TRAIN:
                #     print("\t\tIncorrect...")
                #     p.train(normalized_image, 0.0)

        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("exiting")
            video_capture.release()
            cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("exiting")
    video_capture.release()
    cv2.destroyAllWindows()
