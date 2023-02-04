# In this script we will load preprocessed data, preprocessit to make it ready for machine translation.
import os
import sys
import re
import pandas as pd
import emoji

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from preprocessing.constants import is_english_regex, special_tokens
from config import config

class Preprocessor:
    """
    This class is responsible for loading the preprocessed data and preprocess it to make it ready for machine translation.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self):
        data = pd.read_csv(self.file_path)
        return data

    def preprocess_data(self, data):
        """
        preprocess the data to make it ready for machine translation.
        params: data: pd.DataFrame
        """
        data['date'] = data['date'].fillna(method='ffill')
        data['hour'] = data['hour'].fillna(method='ffill') 
        data['name'] = data['name'].fillna('')
        data['name'] = data['name'].apply(lambda x: config['names_dictionary'][x] if x in config['names_dictionary'] else '')
        data['is_message'] = data.name.apply(lambda x: x != '') & data.message.apply(lambda x: x != ' <Media omitted>')
        data['emoji_in_msg'] = data.message.apply(lambda x: self._is_contains_emoji(x))
        data['message_contains_english'] = data.message.apply(lambda x: re.search(is_english_regex, x)).isnull() == False

        processed_data = data[
            data.is_message & 
            (data.emoji_in_msg == False) & 
            (data.message_contains_english == False)
            ].drop(['is_message', 'emoji_in_msg', 'message_contains_english'], axis=1).reset_index(drop=True)

        processed_data = self._join_messages(processed_data)

        return processed_data

    def _is_contains_emoji(self, s):
        return len([c for c in s if c in emoji.EMOJI_DATA]) > 0

    def _join_messages(self, df):
        """
        Join messages that are from the same person and in the same time.
        params: df: pd.DataFrame
        """
        df = self.__add_to_join_next_message_column(df)
        columns_to_index = {df.columns[i]: i for i in range(len(df.columns))}
        df_copy = df.copy()
        df_length = df_copy.shape[0]
        i = 0
        while i < df_length - 1:
            while df_copy.iloc[i,columns_to_index['to_join_next_message']]:
                message = df_copy.iloc[i,columns_to_index['message']]
                message = message + '. ' if message[-1] not in special_tokens else message + ' '
                df_copy.iloc[i+1,columns_to_index['message']] =message + df_copy.iloc[i+1 ,columns_to_index['message']]
                df_copy = df_copy.drop(df_copy.iloc[i].name)
                df_length -= 1
            i += 1
        return df_copy.reset_index(drop=True)

    def __date_and_hour_to_datetime(self, date, hour):
        return pd.to_datetime(date + ' ' + hour)

    def __add_to_join_next_message_column(self, df):
        df['timtestamp'] = df.apply(lambda x: self.__date_and_hour_to_datetime(x.date, x.hour), axis=1)
        df['message_after_less_than_a_minute'] = (df['timtestamp'].diff().dt.total_seconds() <= 60)[1:].reset_index(drop=True)
        df['same_name_in_next_message'] = df['name'][1:].reset_index(drop=True) == df['name'][:-1]
        df['to_join_next_message'] = df['message_after_less_than_a_minute'] & df['same_name_in_next_message']
        df = df.drop(['date', 'hour', 'message_after_less_than_a_minute', 'same_name_in_next_message'], axis=1)

        return df

    def run(self):
        """
        Load the preprocess data and preprocess it to make it ready for machine translation.
        """
        data_preprocessed = self.load_data()
        data_ready_for_machine_translation = self.preprocess_data(data_preprocessed)
        return data_ready_for_machine_translation

    def save_dataframe(self, data, file_path: str):
        data.to_csv(file_path, index=False)

if __name__ == '__main__':
    preprocessor = Preprocessor('/workspaces/group_members_classification/data/processed/prepared_raw_data/us_trip.csv')
    data = preprocessor.run()
    preprocessor.save_dataframe(data, '/workspaces/group_members_classification/data/processed/prepared_for_machine_translation/us_trip.csv')
    