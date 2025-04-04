#-------------------------------------------------------------------------
# AUTHOR: Maayan Israel
# FILENAME: perceptron.py
# SPECIFICATION: This program implements a Perceptron and MLP classifier to classify the digits from the Optdigits dataset.
# FOR: CS 4210- Assignment #3
#-----------------------------------------------------------*/

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier 
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

highest_perceptron_accuracy = 0.0
highest_mlp_accuracy = 0.0

for learning_rate in n: 

    for isPerceptron in r: 

        for isShuffled in r: 

            if isPerceptron:
              clf = Perceptron(eta0=learning_rate, shuffle=isShuffled, max_iter=10000)    
            else:
              clf = MLPClassifier(activation='logistic', learning_rate_init=learning_rate, hidden_layer_sizes=25, shuffle=isShuffled, max_iter=10000) 

            clf.fit(X_training, y_training)

            number_of_correct_predictions = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                result = clf.predict([x_testSample])
                if result[0] == y_testSample:
                    number_of_correct_predictions += 1

            accuracy = number_of_correct_predictions / 1797

            if isPerceptron and accuracy > highest_perceptron_accuracy:
                highest_perceptron_accuracy = accuracy
                print(f"Highest Perceptron accuracy so far: {accuracy:.4f}, Parameters: learning rate={learning_rate}, shuffle={isShuffled}")
            elif accuracy > highest_mlp_accuracy:
                highest_mlp_accuracy = accuracy
                print(f"Highest MLP accuracy so far: {accuracy:.4f}, Parameters: learning rate={learning_rate}, shuffle={isShuffled}")
