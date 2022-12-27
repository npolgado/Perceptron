#!usr/env/python3
import cv2
import time
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

RESIZE_FACTOR = 256
ROUNDING_AMOUNT = 5
TRAIN = True

class Perceptron:
    
    def __init__(self, learning_rate=0.1, weights=None, bias=0):
        """
        Initialize the perceptron with the given learning rate and weights
        """
        self.learning_rate = learning_rate

        self.save_path = os.path.dirname(__file__)
        self.load_weights(weights)

        self.bias = bias
        self.adjusted = 0
        self.accuracy = 0
        self.loss = 0
        self.correct_guesses = 0
        self.wrong_guesses = 0

        self.max_weight = 0
        self.min_weight = 10000
        print(f"weights: {self.weights.shape}")

    def load_weights(self, weights):
        path = filedialog.askopenfilename()
        if os.path.exists(path):
            self.weights = np.loadtxt(path)
            print(self.weights)
            self.save_path = path
        else:
            self.weights = weights
            self.save_path = None

    def add_inputs(self, input):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] += input[i][j]

    def subtract_inputs(self, input):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] -= input[i][j]

    def train(self, inputs, label):
        """
        Train the perceptron weights based on the given inputs and labels
        """
        prediction = self.predict(inputs)
        print_pred = round(prediction, ROUNDING_AMOUNT)

        bool_prediction = bool(0)
        
        if prediction > self.bias:
            bool_prediction = False

            if bool_prediction == bool(label):
                self.correct_guesses += 1
            else:
                self.wrong_guesses += 1
                self.subtract_inputs(inputs)
                self.adjusted += 1
                
        else:
            bool_prediction = True

            if bool_prediction == bool(label):
                self.correct_guesses += 1
            else:
                self.wrong_guesses += 1
                self.add_inputs(inputs)
                self.adjusted += 1

        self.accuracy = round(float((self.correct_guesses / (self.wrong_guesses + self.correct_guesses))*100), ROUNDING_AMOUNT)
        self.loss = round(float(100 - self.accuracy), ROUNDING_AMOUNT)
            
    def predict(self, inputs):
        """
        Make a prediction based on the given inputs
        """
        output = float(0)
        for i in range(len(inputs)):
            for j in range(len(inputs[i])):
                output += float(inputs[i][j] * self.weights[i][j])
        
        print(f"{self.adjusted}\t| +: {self.accuracy} %\t| -: {self.loss} %\t| {round(output, ROUNDING_AMOUNT)}\t| {bool(output < self.bias)}")
        return output

    def save_weights(self):
        date = str(datetime.datetime.now().date())

        max = 0
        min = 100000
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                val = self.weights[i][j]
                if val > max: max = val
                elif val < min: min = val
        
        root = os.path.dirname(__file__)
        weights_folder = os.path.join(root, "weights")
        fig_path = os.path.join(weights_folder, f"{date}.png")
        raw_path = os.path.join(weights_folder, f"{date}.txt")

        if self.save_path:
            if os.path.exists(self.save_path):
                raw_path = self.save_path
                fig_path = os.path.join(os.path.split(raw_path)[0], f"{date}.png")

        plt.imshow(self.weights, cmap='viridis', vmin=min, vmax=max)
        plt.colorbar()

        plt.savefig(fig_path)
        np.savetxt(raw_path, self.weights, fmt='%.3f')

try:
    p = Perceptron(0.1, np.random.rand(RESIZE_FACTOR, RESIZE_FACTOR))
    st = time.monotonic()
    is_face = False
    root = os.path.dirname(__file__)
    training_folder = os.path.join(root, "faces")
    while True:  
        if float(time.monotonic() - st) > 30:
            for root, dirs, files in os.walk('./faces'):
                for file in files:

                    im = cv2.imread(os.path.join(training_folder, file))
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

                    faces = face_cascade.detectMultiScale(im, 1.3, 5)

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
            st = time.monotonic()
            p.accuracy = 0
            p.correct_guesses = 0
            p.wrong_guesses = 0
            p.adjusted = 0

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
                p.train(normalized_image, 0.0)
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
