from flask import Flask, session, jsonify, request
from diagnostics import*
from scoring import*
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
# import diagnosis
# import predict_exited_from_saved_model
import json
import os


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])

prediction_model = None


# Prediction Endpoint
@app.route("/prediction", methods=['GET', 'POST', 'OPTIONS'])
def predict():
    # call the prediction function you created in Step 3
    location = request.args.get('location')
    df = pd.read_csv(location)
    prediction = model_predictions(df)
    print(prediction)

    print(location)

    return str(prediction)  # add return value for prediction outputs

# Scoring Endpoint


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    # check the score of the deployed model
    df2 = score_model()
    return str(df2)  # add return value (a single F1 score number)

# Summary Statistics Endpoint


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():

    # check means, medians, and modes for each column
    df3 = dataframe_summary()
    return str(df3)  # return a list of all calculated summary statistics

# Diagnostics Endpoint


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def percent():
    # check timing and percent NA values
    df1 = str(execution_time())
    df2 = str(missing_data())
    df3 = str(outdated_packages_list())
    deliver = df1 + '\n' + df2 + '\n' + df3
    print(deliver)

    return deliver  # add return value for all diagnostics


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
