# In this script we will load the txt file and preprocess it to a pandas dataframe.
import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from preprocessing.constants import dates_regex
from preprocessing.utils import extract_dates, extract_name, exteact_message

class Preprocessor:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self):
        with open(self.file_path, 'r') as f:
            data_lines = f.readlines()
        return data_lines

    def preprocess_data(self, data_lines):
        data = {}
        name = ''
        for i, line in enumerate(data_lines):
            prev_name = name
            date, hour, new_line = extract_dates(line, dates_regex)
            name, new_line = extract_name(new_line)
            message = exteact_message(new_line)
            if name=='':
                if (date=='') & (hour==''):
                    name = prev_name

            data[i] = {'date':date, 'hour':hour, 'name':name, 'message':message}
            
        data = pd.DataFrame(data).T

        return data

    def run(self):
        data_lines = self.load_data()
        data = self.preprocess_data(data_lines)
        return data

    def save_dataframe(self, data, file_path: str):
        data.to_csv(file_path, index=False)

if __name__ == '__main__':
    preprocessor = Preprocessor('/workspaces/group_members_classification/data/raw/us_trip.txt')
    data = preprocessor.run()
    preprocessor.save_dataframe(data, '/workspaces/group_members_classification/data/processed/prepared_raw_data/us_trip.csv')