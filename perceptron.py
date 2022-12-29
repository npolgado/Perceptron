#!usr/bin/env python3
import datetime
import os
import sys
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np

ROUNDING_AMOUNT = 5

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

    def add_inputs(self, input, learning_rate=1.0):
        # for i in range(len(self.weights)):
        #     for j in range(len(self.weights[i])):
        #         self.weights[i][j] += input[i][j]
        new_inputs = input * learning_rate
        self.weights = np.add(self.weights, new_inputs)

    def subtract_inputs(self, input, learning_rate=1.0):
        # for i in range(len(self.weights)):
        #     for j in range(len(self.weights[i])):
        #         self.weights[i][j] -= input[i][j]
        new_inputs = input * learning_rate
        self.weights = np.subtract(self.weights, new_inputs)

    def train(self, inputs, label, learning_rate=1.0):
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
                self.wrong_guesses += (1 * learning_rate)
                self.subtract_inputs(inputs, learning_rate)
                self.adjusted += (1 * learning_rate)
                
        else:
            bool_prediction = False

            if bool_prediction == bool(label):
                self.correct_guesses += 1
            else:
                print("Output should have been True. Adding Image...")
                self.wrong_guesses += (1 * learning_rate)
                self.add_inputs(inputs, learning_rate)
                self.adjusted += (1 * learning_rate)

        self.accuracy = round(float((self.correct_guesses / (self.wrong_guesses + self.correct_guesses))*100), ROUNDING_AMOUNT)
        self.loss = round(float(100 - self.accuracy), ROUNDING_AMOUNT)
            
    def predict(self, inputs):
        """
        Make a prediction based on the given inputs
        """
        output = np.dot(inputs, self.weights)
        output = np.sum(output)

        print(f"{self.adjusted}\t| +: {self.accuracy} %\t| -: {self.loss} %\t| {round(output, ROUNDING_AMOUNT)}\t| {bool(output > self.bias)}")
        return output

    def find_valid_fig_path(self, fig_path):
        '''recursive function to index visualizations. validity is based on the file path existing'''

        if os.path.exists(fig_path):
            # already a visual for this path

            fig_path_split = os.path.split(fig_path)
            fig_root = fig_path_split[0]
            fig_name_split = fig_path_split[1].split("-")
            # print(fig_path)
            # print(fig_name_split)

            if bool(int(len(fig_name_split) - 1)):
                # (already indexed a vis!) more than one item in the split
                fig_new_name = str(str(int(fig_name_split[0]) + 1) + "-" + str(fig_name_split[1]))
                new_fig_path = os.path.join(fig_root, fig_new_name)

                # recursive call to see if new_fig is valid, if not increment the index
                new_fig_path = self.find_valid_fig_path(new_fig_path)               
                    
            else:
                # (new indexable vis!) only one item in the split
                fig_new_name = "0-" + str(fig_name_split[0])
                new_fig_path = os.path.join(fig_root, fig_new_name)

                # recursive call to see if new_fig is valid, if not increment the index
                new_fig_path = self.find_valid_fig_path(new_fig_path)
            
            # if the recursion returns a valid path, we return it
            return new_fig_path
        else:
            # if it was valid to begin with, its safe to write to
            return fig_path

    def save_weights(self):
        '''
        TODO: add error handling for file already exiting for png files
        '''
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
        
        fig_path = os.path.join(visuals_folder, f"{date}_RES{int(self.shape[0])}_ADJ{int(self.adjusted)}_render.png")
        raw_path = os.path.join(weights_folder, f"{date}_RES{int(self.shape[0])}_ADJ{int(self.adjusted)}_raw.txt")

        if self.save_path:
            # weights path existings, loaded from weights this session
            if os.path.exists(self.save_path):
                # weights are valid path

                weights_file_name = os.path.split(self.save_path)[1].split("_raw.")[0]
                raw_path = self.save_path
                fig_path = os.path.join(visuals_folder, f"{weights_file_name}_render.png")

        fig_path = self.find_valid_fig_path(fig_path)
        print(fig_path)

        plt.imshow(self.weights, cmap='viridis', vmin=min, vmax=max)
        plt.colorbar()

        plt.savefig(fig_path)
        np.savetxt(raw_path, self.weights, fmt='%.3f')
