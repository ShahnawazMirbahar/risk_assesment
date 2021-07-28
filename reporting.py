import pickle
from diagnostics import*
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])


# Function for reporting
def score_model():
    test_file = os.path.join(test_data_path, 'testdata.csv')
    test_data = pd.read_csv(test_file)
    y_act = test_data['exited'].tolist()
    y_pred = model_predictions(test_data).tolist()
    matrix = confusion_matrix(y_act, y_pred)
    classes = ["0", "1"]
    df_matrix = pd.DataFrame(matrix, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    df_plot = sn.heatmap(df_matrix, annot=True)
    df_plot.figure.savefig(os.path.join(
        output_model_path, "confusionmatrix.png"))

    print(matrix)

    print(y_act)
    print(y_pred)
# calculate a confusion matrix using the test data and the deployed model
# write the confusion matrix to the workspace


if __name__ == '__main__':
    score_model()
