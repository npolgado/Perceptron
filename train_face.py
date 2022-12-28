#!usr/bin/env python3
import os
import time

import cv2
import numpy as np

from perceptron import *

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

RESIZE_FACTOR = 512
TRAIN = True

try:
    p = Perceptron(np.zeros((RESIZE_FACTOR, RESIZE_FACTOR)))
    st = time.monotonic()
    is_face = False
    root = os.path.dirname(__file__)
    training_folder = os.path.join(root, "faces")
    training_folder_truth = os.path.join(root, "my_face")
    num_incorrect_passes = 1
    num_truth_passes = 25
    while True:  
        if float(time.monotonic() - st) > 30:
            print("\t\tTRAINING ON _NOT_YOUR_FACE_")
            for i in range(num_incorrect_passes):
                for root, dirs, files in os.walk('./faces'):
                    for file in files:

                        im = cv2.imread(os.path.join(training_folder, file))
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

                        faces = face_cascade.detectMultiScale(im, 1.3, 5)
                        bool_faces = len(faces)
                        for (x, y, w, h) in faces:
                            face = im[y:y+h, x:x+w]

                            resized_image = cv2.resize(face, (RESIZE_FACTOR, RESIZE_FACTOR))
                            normalized_image = resized_image / 255.0
                            
                            if TRAIN:
                                p.train(normalized_image, 0.0)
                            else:
                                if p.predict(normalized_image) > p.bias:
                                    print("False!")
                                else:
                                    print("True!")
                        if not bool_faces:
                            resized_image = cv2.resize(im, (RESIZE_FACTOR, RESIZE_FACTOR))
                            normalized_image = resized_image / 255.0
                            p.train(normalized_image, 0.0)

            print("\t\tTRAINING ON _YOUR_FACE_")
            for i in range(num_truth_passes):
                for root, dirs, files in os.walk('./my_face'):
                    for file in files:

                        im = cv2.imread(os.path.join(training_folder_truth, file))
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

                        faces = face_cascade.detectMultiScale(im, 1.3, 5)
                        bool_faces = len(faces)
                        for (x, y, w, h) in faces:
                            face = im[y:y+h, x:x+w]

                            resized_image = cv2.resize(face, (RESIZE_FACTOR, RESIZE_FACTOR))
                            normalized_image = resized_image / 255.0
                            
                            if TRAIN:
                                p.train(normalized_image, 1.0)
                            else:
                                if p.predict(normalized_image) > p.bias:
                                    print("False!")
                                else:
                                    print("True!")
                        if not bool_faces:
                            resized_image = cv2.resize(im, (RESIZE_FACTOR, RESIZE_FACTOR))
                            normalized_image = resized_image / 255.0
                            p.train(normalized_image, 0.0)

            print("\t\tRESETING...\n")
            st = time.monotonic()
            p.accuracy = 0
            p.correct_guesses = 0
            p.wrong_guesses = 0

        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        is_face = len(faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = gray[y:y+h, x:x+w]

            resized_image = cv2.resize(face, (RESIZE_FACTOR, RESIZE_FACTOR))
            normalized_image = resized_image / 255.0

            if TRAIN:
                p.train(normalized_image, 1.0)
            else:
                if p.predict(normalized_image) > p.bias:
                    print("False!")
                else:
                    print("True!")

        if not is_face:
            resized_image = cv2.resize(gray, (RESIZE_FACTOR, RESIZE_FACTOR))
            normalized_image = resized_image / 255.0

            if TRAIN:
                pass
                # print(f"guessing: {bool(p.predict(normalized_image) > p.bias)}")
            else:
                if p.predict(normalized_image) > p.bias:
                    print("False!")
                else:
                    print("True!")

        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("exiting")
            p.save_weights()
            video_capture.release()
            cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("exiting")
    p.save_weights()
    video_capture.release()
    cv2.destroyAllWindows()
