import pandas as pd
from tqdm import tqdm
from transformers import pipeline

class MachineTranslation:
    """
    In this class we get the DataFrame ready for machine translation.
    We will convert the message column to a list of sentences, and translate it to English.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self):
        data = pd.read_csv(self.file_path)
        return data

    def translate_data(self, data):
        """
        Translate the data to English.
        params: data: pd.DataFrame
        """
        translated_sentences = []
        pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-he-en", max_length=1000)
        list_of_sentences = data['message'].tolist()
        for sentence_id in tqdm(range(len(list_of_sentences))):
            
            translated_sentences.append(pipe(list_of_sentences[sentence_id])[0]['translation_text'])
        # tmp_data = data[:100].copy()
        data['translated_message'] = translated_sentences
        return data

    def run(self):
        """
        Load the data and translate it to English.
        """
        data_ready_for_machine_translation = self.load_data()
        translated_data = self.translate_data(data_ready_for_machine_translation)

        return translated_data
    
    def save_dataframe(self, translated_data, file_path: str):
        translated_data.to_csv(file_path, index=False)

if __name__ == '__main__':
    machine_translation = MachineTranslation('/workspaces/group_members_classification/data/processed/prepared_for_machine_translation/us_trip.csv')
    data = machine_translation.run()
    machine_translation.save_dataframe(data, '/workspaces/group_members_classification/data/processed/prepared_for_classifiaction/us_trip2.csv')
    