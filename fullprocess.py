import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import json
import subprocess
from app import app
#import apicalls
import os.path
import os
import glob
import pandas as pd

# Check and read new data
with open('config.json', 'r') as f:
    config = json.load(f)

input_path = config['input_folder_path']
model_path = config['output_model_path']
deployment_path = config['prod_deployment_path']
# first, read ingestedfiles.txt
text_file1 = os.path.join(deployment_path, 'ingestedfiles.txt')
f = open(text_file1, "r")
new_data = f.read()
# print(new_data)
# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
source_files = glob.glob(os.path.join(input_path, "data*.csv"))
print(source_files)
data_file = os.path.basename(str(source_files))
print(data_file)
# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if data_file in new_data:
    print('exit')
    exit()
else:
    print('False')
    ingestion.merge_multiple_dataframe()
# if result == False:
# Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
score_file1 = os.path.join(deployment_path, 'latestscore.txt')
score = open(score_file1, "r")
latest_score = score.read()
print(latest_score)
training.train_model()
scoring.score_model()
score_file2 = os.path.join(model_path, 'latestscore.txt')
score2 = open(score_file2, "r")
new_score = score2.read()
# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
if new_score > latest_score:
    model_drift = True
    print(new_score + '>' + latest_score)
    print('true')
else:
    model_drift = False
    exit()
# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
##################Diagnostics and reporting
if model_drift == True:
    deployment.store_model_into_pickle()
    # run diagnostics.py and reporting.py for the re-deployed model
    reporting.score_model()
    #os.system('python diagnostics.py')
    subprocess.Popen('python apicalls.py')
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
exit()
