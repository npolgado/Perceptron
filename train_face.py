#!usr/bin/env python3
import os
import time

import cv2
import numpy as np

from perceptron import *

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

RESIZE_FACTOR = 1024
TRAIN = True
LEARNING_NO_FACE = 0.05
LEARNING_WRONG_FACE = 1.0
LEARNING_CORRECT_FACE = 1.0

TRAIN_DATASET_TIMING = 60
TRAINING_BATCH_INTERVAL = int(TRAIN_DATASET_TIMING / 20)

try:
    p = Perceptron(np.zeros((RESIZE_FACTOR, RESIZE_FACTOR)))

    st = time.monotonic()
    st_batch = time.monotonic()

    is_face = False
    
    root = os.path.dirname(__file__)
    training_folder = os.path.join(root, "training")
    training_folder_ground = os.path.join(training_folder, "ground")
    training_folder_truth = os.path.join(training_folder, "truth")
    
    num_incorrect_passes = 1
    num_webcam_passes = 2
    num_truth_passes = 1

    all_facial_images = []
    
    while True:  
        dt = float(time.monotonic() - st)
        dt_batch = float(time.monotonic() - st_batch)

        # get webcam frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # get your frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        is_face = len(faces)

        if not is_face:
            resized_image = cv2.resize(gray, (RESIZE_FACTOR, RESIZE_FACTOR))
            normalized_image = resized_image / 255.0

            if TRAIN:
                # guess = bool(p.predict(normalized_image) > p.bias)
                p.train(normalized_image, 0.0, learning_rate=LEARNING_NO_FACE)
            else: 
                guess = bool(p.predict(normalized_image) > p.bias)

        else:
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face = gray[y:y+h, x:x+w]
                
                resized_image = cv2.resize(face, (RESIZE_FACTOR, RESIZE_FACTOR))
                normalized_image = resized_image / 255.0
                all_facial_images.append(normalized_image)
                
                if TRAIN:
                    vis_weights = np.array(p.weights)
                    # vis_weights = vis_weights * 10
                    vis_weights = vis_weights + 100
                    vis = cv2.resize(
                        vis_weights,
                        (face.shape[1], face.shape[0]),
                        interpolation = cv2.INTER_LINEAR
                    )
                    # print(np.shape(vis))
                    gray[y:y+face.shape[1], x:x+face.shape[0]] = vis

            if dt_batch > TRAINING_BATCH_INTERVAL and TRAIN:
                for i in range(num_webcam_passes):
                    np.random.shuffle(all_facial_images)
                    for j in all_facial_images:
                        p.train(j, 1.0, learning_rate=LEARNING_CORRECT_FACE)
                all_facial_images = []
                st_batch = time.monotonic()
            else:
                guess = bool(p.predict(normalized_image) > p.bias)

        cv2.imshow('Webcam', gray)

        if dt > TRAIN_DATASET_TIMING and TRAIN:
            print("\t\tSWITCHING TO IMAGE TRAINING...\n")
            print("\t\tTRAINING ON _NOT_YOUR_FACE_")
            for root, dirs, files in os.walk(training_folder_ground):
                for i in range(num_incorrect_passes):
                    np.random.shuffle(files)
                    for file in files:

                        im = cv2.imread(os.path.join(training_folder_ground, file))
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

                        faces = face_cascade.detectMultiScale(im, 1.3, 5)
                        bool_faces = len(faces)

                        if not bool_faces:
                            resized_image = cv2.resize(im, (RESIZE_FACTOR, RESIZE_FACTOR))
                            normalized_image = resized_image / 255.0
                            p.predict(normalized_image)

                            if TRAIN: 
                                p.train(normalized_image, 0.0, learning_rate=LEARNING_NO_FACE)
                            else: 
                                guess = bool(p.predict(normalized_image) > p.bias)
                        
                        else:
                            for (x, y, w, h) in faces:
                                face = im[y:y+h, x:x+w]

                                resized_image = cv2.resize(face, (RESIZE_FACTOR, RESIZE_FACTOR))
                                normalized_image = resized_image / 255.0
                                
                                if TRAIN: 
                                    p.train(normalized_image, 0.0, learning_rate=LEARNING_WRONG_FACE)
                                else: 
                                    guess = bool(p.predict(normalized_image) > p.bias)

            print("\t\tTRAINING ON _YOUR_FACE_")
            for root, dirs, files in os.walk(training_folder_truth):
                for i in range(num_truth_passes):
                    np.random.shuffle(files)
                    for file in files:

                        im = cv2.imread(os.path.join(training_folder_truth, file))
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

                        faces = face_cascade.detectMultiScale(im, 1.3, 5)
                        bool_faces = len(faces)

                        if not bool_faces:
                            resized_image = cv2.resize(im, (RESIZE_FACTOR, RESIZE_FACTOR))
                            normalized_image = resized_image / 255.0
                            p.predict(normalized_image)
                        
                            if TRAIN: 
                                p.train(normalized_image, 0.0, learning_rate=LEARNING_NO_FACE)
                            else: 
                                guess = bool(p.predict(normalized_image) > p.bias)
                        
                        else:
                            for (x, y, w, h) in faces:
                                face = im[y:y+h, x:x+w]

                                resized_image = cv2.resize(face, (RESIZE_FACTOR, RESIZE_FACTOR))
                                normalized_image = resized_image / 255.0
                                
                                if TRAIN: 
                                    p.train(normalized_image, 1.0, learning_rate=LEARNING_CORRECT_FACE)
                                else: 
                                    guess = bool(p.predict(normalized_image) > p.bias)

            print("\t\tRESETING STATS...\n")
            p.accuracy = 0
            p.correct_guesses = 0
            p.wrong_guesses = 0
            st = time.monotonic()
            print("\t\tSWITCHING TO LEARNING MODE, SMILE :)")
        
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
