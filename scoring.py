from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import gzip
import pickle
import os
import glob
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


# Function for model scoring
def score_model():
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    pickle_file = os.path.join(model_path, 'trainedmodel.pkl')
    test_file = os.path.join(test_data_path, 'testdata.csv')
    infile = open(pickle_file, 'rb')
    model = pickle.load(infile)
    test_data = pd.read_csv(test_file)
    X = test_data[['lastmonth_activity', 'lastyear_activity',
                   'number_of_employees']].values.reshape(-1, 3)
    y = test_data[['exited']].values.reshape(-1, 1)

    predicted = model.predict(X)
    f1score = metrics.f1_score(predicted, y)
    file1 = open(os.path.join(model_path, "latestscore.txt"), 'w')
    file1.write(str(f1score))
    return f1score


if __name__ == '__main__':
    score_model()
