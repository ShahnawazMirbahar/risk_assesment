import json
from sklearn.linear_model import LogisticRegression
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


# Function for training the model
def train_model():

    # use this logistic regression for training
    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)

    # fit the logistic regression to your data
    data = pd.read_csv(f"{dataset_csv_path}/finaldata.csv")
    df = pd.DataFrame(data, columns=['corporation', 'lastmonth_activity',
                                     'lastyear_activity', 'number_of_employees', 'exited'])
    X = df[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y = df['exited']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='auto', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)
    model.fit(X_train, y_train)

# write the trained model to your workspace in a file called trainedmodel.pkl
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    pkl_filename = os.path.join(model_path, "trainedmodel.pkl")
    with open(pkl_filename, 'wb') as file:

        pickle.dump(model, file)


if __name__ == '__main__':
    train_model()
