import pandas as pd
import numpy as np
import timeit
import glob
import os
import subprocess
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
test_file = os.path.join(test_data_path, 'testdata.csv')
test_data = pd.read_csv(test_file)

# Function to get model predictions


def model_predictions(dataset):
    # read the deployed model and a test dataset, calculate predictions
    pickle_file = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    #test_file = os.path.join(test_data_path, 'testdata.csv')
    infile = open(pickle_file, 'rb')
    model = pickle.load(infile)
    #test_data = pd.read_csv(test_file)
    # print(test_data.shape)
    # print(test_data.head())
    # print(test_data.describe())
    X = dataset[['lastmonth_activity', 'lastyear_activity',
                 'number_of_employees']].values.reshape(-1, 3)
    y = dataset[['exited']].values.reshape(-1, 1)
    y_pred = model.predict(X)
    print(y_pred)

    return y_pred   # return value should be a list containing all predictions

# Function to get summary statistics


def dataframe_summary():
    # calculate summary statistics here
    test_file = os.path.join(dataset_csv_path, 'finaldata.csv')
    data = pd.read_csv(test_file)
    data_mean = data[["lastmonth_activity", "lastyear_activity",
                      "number_of_employees", "exited"]].mean()
    # print(data.describe())
    # print(data_mean)
    data_median = data[["lastmonth_activity", "lastyear_activity",
                        "number_of_employees", "exited"]].median()

    # print(data_median)

    data_std = data[["lastmonth_activity", "lastyear_activity",
                     "number_of_employees", "exited"]].std()

    # print(data_std)
    data_summary = []
    data_summary.append(data_mean)
    data_summary.append(data_median)
    data_summary.append(data_std)
    data_summary = np.array(data_summary).tolist()
    print(data_summary)

    return data_summary  # return value should be a list containing all summary statistics

# Function to `get timings


def missing_data():
    data_file = os.path.join(dataset_csv_path, 'finaldata.csv')
    df1 = pd.read_csv(data_file)
    element_value = df1.shape[0]
    percentage1 = (df1['corporation']. isna().sum())*100/element_value
    percentage2 = (df1['lastmonth_activity']. isna().sum())*100/element_value
    percentage3 = (df1['lastyear_activity']. isna().sum())*100/element_value
    percentage4 = (df1['number_of_employees']. isna().sum())*100/element_value
    percentage5 = (df1['exited']. isna().sum())*100/element_value
    missing_percent = []
    missing_percent.append(percentage1)
    missing_percent.append(percentage2)
    missing_percent.append(percentage3)
    missing_percent.append(percentage4)
    missing_percent.append(percentage5)
    print(missing_percent)

    return missing_percent


def execution_time():
    # calculate timing of training.py and ingestion.py
    timer = []
    starttime1 = timeit.default_timer()
    os.system('ingestion.py')
    timing1 = timeit.default_timer() - starttime1
    timer.append(timing1)
    starttime2 = timeit.default_timer()
    os.system('training.py')
    timing2 = timeit.default_timer() - starttime2
    timer.append(timing2)
    print(timer)
    return timer  # return a list of 2 timing values in seconds

# Function to check dependencies


def outdated_packages_list():
    installed = subprocess.check_output(
        ['pip', 'list', '-o', '--format', 'columns'])
    with open('installed.txt', 'wb') as f:
        f.write(installed)
    df1 = pd.read_csv('requirements.txt', sep="==",
                      header=None, engine='python')
    df1.rename(columns={0: 'Package', 1: 'Version'}, inplace=True)
    df2 = pd.read_fwf('installed.txt', header=None,
                      engine='python')
    df2 = df2.drop([0, 1])

    df2.rename(columns={0: 'Package', 1: 'Version',
                        2: 'Latest', 3: 'Type'}, inplace=True)
    del df2['Type']
    # df2.drop(columns=0)
    # df2 = pd.DataFrame(df2)
    # print(df1.head())
    # print(df2)
    df3 = df1.merge(df2, how="left", on=['Package', 'Version'])
    # df3.fillna(value=df3['Version'])
    df3['Latest'] = df3['Latest'].fillna(df3['Version'])
    print(df3)
    return df3
    # print(df2)
    # df3 = df1.merge(df2, how="left", on="Package")
    # print(df3)
# get a list of


if __name__ == '__main__':
    model_predictions(test_data)
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()
    # print(prod_deployment_path)
