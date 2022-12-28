#!usr/bin/env python3
import datetime
import os
import sys
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np

ROUNDING_AMOUNT = 5

'''
IF THE PERCEPTRON OUTPUT IS 

'''


class Perceptron:
    def __init__(self, weights=None, bias=20):
        """
        Initialize the perceptron with the given learning rate and weights
        """
        self.save_path = os.path.dirname(__file__)
        self.load_weights(weights)
        self.shape = self.weights.shape

        self.bias = bias

        self.correct_guesses = 0
        self.wrong_guesses = 0
        self.adjusted = 0
        self.accuracy = 0
        self.loss = 0

        self.max_weight = 0
        self.min_weight = 10000
        print(f"weights: {self.shape}")

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
            bool_prediction = True

            if bool_prediction == bool(label):
                self.correct_guesses += 1
            else:
                print("Output should have been False. Subtracting Image...")
                self.wrong_guesses += 1
                self.subtract_inputs(inputs)
                self.adjusted += 1
                
        else:
            bool_prediction = False

            if bool_prediction == bool(label):
                self.correct_guesses += 1
            else:
                print("Output should have been True. Adding Image...")
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
        
        print(f"{self.adjusted}\t| +: {self.accuracy} %\t| -: {self.loss} %\t| {round(output, ROUNDING_AMOUNT)}\t| {bool(output > self.bias)}")
        return output

    def save_weights(self):
        date = str(datetime.datetime.now().date())
        date = date.split("-")[1] + "_" + date.split("-")[2]

        max = 0
        min = 100000
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                val = self.weights[i][j]
                if val > max: max = val
                elif val < min: min = val
        
        root = os.path.dirname(__file__)
        weights_folder = os.path.join(root, "weights")
        visuals_folder = os.path.join(weights_folder, "images")

        if not os.path.exists(weights_folder):
            print("creating weights folder")
            os.mkdir(weights_folder)

        if not os.path.exists(visuals_folder):
            print("creating images folder")
            os.mkdir(visuals_folder)
        
        fig_path = os.path.join(visuals_folder, f"{date}_{int(self.accuracy)}_{self.adjusted}_render.png")
        raw_path = os.path.join(weights_folder, f"{date}_{int(self.accuracy)}_{self.adjusted}_raw.txt")

        if self.save_path:
            if os.path.exists(self.save_path):
                weights_file_name = os.path.split(self.save_path)[1].split(".")[0]
                raw_path = self.save_path
                fig_path = os.path.join(visuals_folder, f"{weights_file_name}.png")

        plt.imshow(self.weights, cmap='viridis', vmin=min, vmax=max)
        plt.colorbar()

        plt.savefig(fig_path)
        np.savetxt(raw_path, self.weights, fmt='%.3f')
