import requests
import json
import os
import subprocess
import time

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"

with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])

# Call each API endpoint and store the responses
# put an API call here
response1 = requests.get(
    'http://127.0.0.1:8000/prediction?location=' + test_data_path + '/testdata.csv').content
# put an API call here
response2 = requests.get('http://127.0.0.1:8000/scoring').content
# put an API call here
response3 = requests.get('http://127.0.0.1:8000/summarystats').content
# put an API call here
response4 = requests.get('http://127.0.0.1:8000/diagnostics').content

response1 = response1.decode('UTF-8')
response2 = response2.decode('UTF-8')
response3 = response3.decode('UTF-8')
response4 = response4.decode('UTF-8')
# combine all API responses
responses = response1+'\n' + response2+'\n'+response3+'\n' + response4

# write the responses to your workspace
file1 = open(os.path.join(output_model_path, "apireturns.txt"), 'w')
file1.write(str(responses))
