#!usr/bin/env python3
import os
import time

import cv2
import numpy as np

from perceptron import *

RESIZE_FACTOR = 1024
SIZE_X = 1280
SIZE_Y = 720
TRAIN = True
LEARNING_NO_FACE = 1.0
LEARNING_WRONG_FACE = 1.0
LEARNING_CORRECT_FACE = 1.0

TRAIN_DATASET_TIMING = 5
TRAINING_BATCH_INTERVAL = int(TRAIN_DATASET_TIMING / 5)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(3, SIZE_X)
video_capture.set(4, SIZE_Y)

try:
    p = Perceptron(np.zeros((SIZE_X, SIZE_Y)))

    st = time.monotonic()
    st_batch = time.monotonic()

    is_face = False
    
    root = os.path.dirname(__file__)
    training_folder = os.path.join(root, "training")
    training_folder_ground = os.path.join(training_folder, "ground")
    training_folder_truth = os.path.join(training_folder, "truth")
    
    num_incorrect_passes = 1
    num_webcam_passes = 1
    num_truth_passes = 1

    all_facial_images = []
    
    while True:  
        dt = float(time.monotonic() - st)
        dt_batch = float(time.monotonic() - st_batch)

        # get webcam frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if np.shape(frame) != (SIZE_X, SIZE_Y):
            train_im = cv2.resize(
                gray,
                (SIZE_X, SIZE_Y),
                interpolation=cv2.INTER_LINEAR
            )
            bool_guess = bool(p.predict(gray) > p.bias)
            # p.train(gray, 1.0, learning_rate=LEARNING_CORRECT_FACE)
        else:
            bool_guess = bool(p.predict(gray) > p.bias)
            # p.train(gray, 1.0, learning_rate=LEARNING_CORRECT_FACE)

        vis_weights = p.weights.T

        # Normalize the heatmap to the range [0, 255]
        heatmap = cv2.normalize(vis_weights, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Apply a colormap to the heatmap
        colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        result = cv2.addWeighted(frame, 0.7, colormap, 0.3, 0)

        '''
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

                # draw face on frames
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face = gray[y:y+h, x:x+w]
                
                resized_image = cv2.resize(face, (RESIZE_FACTOR, RESIZE_FACTOR))
                normalized_image = resized_image / 255.0

                # add to bulk array
                all_facial_images.append(normalized_image)
                
                if TRAIN:
                    # add weights to gray frame
                    vis_weights = np.array(p.weights)

                    # Get the minimum and maximum values in the array
                    min_val = vis_weights.min()
                    max_val = vis_weights.max()

                    # Map the values in the array to the range 0 to 255
                    mapped_data = np.interp(vis_weights, (min_val, max_val), (0, 255))
                    
                    # Resize to face on frame
                    vis = cv2.resize(
                        mapped_data,
                        (face.shape[1], face.shape[0]),
                        interpolation = cv2.INTER_LINEAR
                    )
                    
                    # insert into grayscale frame
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
        '''

        cv2.imshow('Webcam', result)

        if dt > TRAIN_DATASET_TIMING and TRAIN:
            print("\t\tSWITCHING TO IMAGE TRAINING...\n")
            print("\t\tTRAINING ON _NOT_YOUR_FACE_")
            for root, dirs, files in os.walk(training_folder_ground):
                for i in range(num_incorrect_passes):
                    np.random.shuffle(files)
                    for file in files:

                        im = cv2.imread(os.path.join(training_folder_ground, file))
                        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                        resized_image = cv2.resize(im_gray, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_LINEAR)
                        normalized_image = resized_image / 255.0
                        
                        faces = face_cascade.detectMultiScale(im, 1.3, 5)
                        bool_faces = len(faces)
                        
                        if bool_faces:
                            p.train(normalized_image, 0.0, learning_rate=LEARNING_WRONG_FACE)
                        else: 
                            p.train(normalized_image, 0.0, learning_rate=LEARNING_NO_FACE)
                        
                        '''
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
                        '''

            print("\t\tTRAINING ON _YOUR_FACE_")
            for root, dirs, files in os.walk(training_folder_truth):
                for i in range(num_truth_passes):
                    np.random.shuffle(files)
                    for file in files:

                        im = cv2.imread(os.path.join(training_folder_truth, file))
                        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

                        if np.shape(im_gray)[1] > np.shape(im_gray)[0]: 
                            resized_image = cv2.resize(im_gray, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_LINEAR)
                        else:
                            resized_image = cv2.resize(im_gray.T, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_LINEAR)
                        
                        normalized_image = resized_image / 255.0

                        faces = face_cascade.detectMultiScale(im, 1.3, 5)
                        bool_faces = len(faces)
                                                
                        if bool_faces:
                            p.train(normalized_image, 1.0, learning_rate=LEARNING_CORRECT_FACE)
                        else: 
                            p.train(normalized_image, 1.0, learning_rate=LEARNING_CORRECT_FACE)
                        
                        '''
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
                        '''

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
