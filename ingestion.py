import pandas as pd
import numpy as np
import os.path
import os
import glob
import json
from datetime import datetime


# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


# Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    path = input_folder_path
    save_path = output_folder_path
    file_name = "ingestedfiles.txt"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    completeName = os.path.join(save_path, file_name)
    f = open(completeName, "a")
    all_files = glob.glob(os.path.join(path, "data*.csv"))
    f.write(str(all_files))
    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)
    df_merged.drop_duplicates(subset=None, inplace=True)
    finaldata_output_path = os.path.join(output_folder_path, "finaldata.csv")
    # print(finaldata_output_path)
    if os.path.exists(finaldata_output_path):
        df1 = pd.read_csv(finaldata_output_path)
        df2 = df_merged
        df3 = pd.concat([df1, df2], ignore_index=True)
        df3.drop_duplicates(subset=None, inplace=True)
        print(df3)
        df3.to_csv(os.path.join(output_folder_path,
                                "finaldata.csv"), index=False)

    else:
        df_merged.to_csv(os.path.join(output_folder_path,
                                      "finaldata.csv"), index=False)


if __name__ == '__main__':
    merge_multiple_dataframe()
