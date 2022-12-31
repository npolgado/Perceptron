#!usr/bin/env python3
import os
import re
import sys
import time

import cv2
import mss
import numpy as np

from perceptron import *

SCALAR = 1
RESIZE_FACTOR = 1024

TRAIN = True

LEARNING_NO_FACE = 0.0
LEARNING_WRONG_FACE = 1.0
LEARNING_CORRECT_FACE = 1.0

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a mss.MSS object and get the width and height of the primary monitor
mss_obj = mss.mss()

num_mon = len(mss_obj.monitors)
chose_monitor = False

monitor_dims = []
for i in range(num_mon):
    monitor_dims.append((mss_obj.monitors[i]["width"], mss_obj.monitors[i]["height"]))

print(mss_obj.monitors)
print(monitor_dims)

# choose monitor 
try:
    curr_mon = int(input(f"which of the {num_mon} monitor(s)? (0, 1, 2, 3, ...)"))
    assert(type(curr_mon) == int)
    chose_monitor = True
except Exception as e:
    print(f"ERROR: bad input\n{e}\n\nusing 0...")
    curr_mon = 0
    chose_monitor = False

shape_full_screens = monitor_dims[0]
shape_full_screens_scaled = (int(shape_full_screens[0]), int(shape_full_screens[1]))

# print(monitor_dims)
# print(shape_full_screens_scaled)

width = mss_obj.monitors[curr_mon]["width"]
height = mss_obj.monitors[curr_mon]["height"]
scaled = (int(width / SCALAR), int(height / SCALAR))

def close_training():
    global p
    p.save_weights()
    cv2.destroyAllWindows()
    sys.exit()

try:
    p = Perceptron(np.zeros((RESIZE_FACTOR, RESIZE_FACTOR)))

    st = time.monotonic()
    st_batch = time.monotonic()

    is_face = False
    
    root = os.path.dirname(__file__)
    training_folder = os.path.join(root, "training")
    training_folder_ground = os.path.join(training_folder, "ground")
    training_folder_truth = os.path.join(training_folder, "truth")

    while True:
        dt = float(time.monotonic() - st)
        dt_batch = float(time.monotonic() - st_batch)

        # Capture the desktop and convert it to a cv2 image
        with mss_obj as sct:
            sct_img = np.array(sct.grab(sct.monitors[curr_mon]))
            img = cv2.resize(sct_img, (width, height), interpolation=cv2.INTER_LINEAR)
            # frame = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_draw = cv2.resize(gray, (RESIZE_FACTOR, RESIZE_FACTOR), interpolation=cv2.INTER_LINEAR)
            
        # get your frame
        faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.05, minNeighbors=5, minSize=(150,150))
        is_face = len(faces)


        if not bool(is_face):
            # resize to train
            resized_image = cv2.resize(gray, (RESIZE_FACTOR, RESIZE_FACTOR))

            normalized_image = resized_image / 255.0

            if TRAIN:
                p.train(normalized_image, 0.0, learning_rate=LEARNING_NO_FACE)
            else: 
                guess = bool(p.predict(normalized_image) > p.bias)

        else:
            for (x, y, w, h) in faces:

                # draw face on frames
                cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # extract face
                face = gray[y:y+h, x:x+w]

                resized_image = cv2.resize(face, (RESIZE_FACTOR, RESIZE_FACTOR))

                normalized_image = resized_image / 255.0

                if TRAIN:
                    p.train(normalized_image, 0.0, learning_rate=LEARNING_WRONG_FACE)
                else:
                    guess = bool(p.predict(normalized_image) > p.bias)

        # Display the image
        cv2.imshow("Desktop", gray)

        if dt > 30 and TRAIN:
            print("\n\t\tTRAINING ON _YOUR_FACE_\n")
            for root, dirs, files in os.walk(training_folder_truth):
                np.random.shuffle(files)
                for file in files:

                    im = cv2.imread(os.path.join(training_folder_truth, file))
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

                    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.05, minNeighbors=5, minSize=(200,200))
                    bool_faces = len(faces)

                    if not bool_faces:
                        resized_image = cv2.resize(gray, (RESIZE_FACTOR, RESIZE_FACTOR))
                        normalized_image = resized_image / 255.0
                        p.predict(normalized_image)
                    
                        if TRAIN: 
                            p.train(normalized_image, 0.0, learning_rate=LEARNING_NO_FACE)
                        else: 
                            guess = bool(p.predict(normalized_image) > p.bias)

                        cv2.imshow('Desktop', normalized_image)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            close_training()

                    else:
                        for (x, y, w, h) in faces:
                            face = gray[y:y+h, x:x+w]

                            resized_image = cv2.resize(face, (RESIZE_FACTOR, RESIZE_FACTOR))
                            normalized_image = resized_image / 255.0
                            
                            if TRAIN: 
                                p.train(normalized_image, 1.0, learning_rate=LEARNING_CORRECT_FACE)
                            else: 
                                guess = bool(p.predict(normalized_image) > p.bias)
                            
                            cv2.imshow('Desktop', normalized_image)

                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                close_training()

            print("\n\t\tRESETING STATS...\n")
            p.accuracy = 0
            p.correct_guesses = 0
            p.wrong_guesses = 0
            st = time.monotonic()
            print("\n\t\tSWITCHING TO LEARNING MODE, SMILE :)\n")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("exiting")
            p.save_weights()
            cv2.destroyAllWindows()
            sys.exit()

except KeyboardInterrupt:
    print("exiting")
    p.save_weights()
    cv2.destroyAllWindows()
    sys.exit()
