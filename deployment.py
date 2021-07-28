from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil


# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])


# function for deployment


def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    if not os.path.exists(prod_deployment_path):
        os.mkdir(prod_deployment_path)

    pickle_original = os.path.join(model_path, 'trainedmodel.pkl')
    pickle_target = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    shutil.copyfile(pickle_original, pickle_target)

    score_original = os.path.join(model_path, 'latestscore.txt')
    score_target = os.path.join(prod_deployment_path, 'latestscore.txt')
    shutil.copyfile(score_original, score_target)

    ingested_original = os.path.join(dataset_csv_path, 'ingestedfiles.txt')
    ingested_target = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    shutil.copyfile(ingested_original, ingested_target)


if __name__ == '__main__':
    store_model_into_pickle()
