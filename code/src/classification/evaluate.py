import os
import sys
import pandas as pd
import numpy as np
import pickle
import torch
from transformers import BertForSequenceClassification

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from classification.data_preprocessing import DataPreprocessing as ClassificationDataPreprocessor
from classification.model_preprocessing import ModelPreprocessing as ClassificationModelPreprocessor

from classification.utils import get_predicted_name

class Evaluator:
    """
    In this class we will evaluate the model.
    We will load the test data and the model, calculate the avaluation metrics and will do an error analysis
    """
    def __init__(self, global_data_path:str ,datasets:list, model_path:str, model_infernce_path:str ,names_to_classify: list = []):
        self.gloabel_data_path = global_data_path
        self.datasets_names = datasets
        self.model_path = model_path
        self.model_inference_path= model_infernce_path
        self.name_to_classify = names_to_classify
    
    def get_data_preprocessed_for_model(self):
        """
        Use the data preprocessing class and model preprocessing class to get the data ready for the model.
        We need that to get the model predictions.
        """
        preprocessor = ClassificationDataPreprocessor(
            self.gloabel_data_path, self.datasets_names, self.name_to_classify
            )
        loaded_datasets = preprocessor.load_data()
        dataset = preprocessor.concat_datasets(loaded_datasets)
        dataset = preprocessor.filter_dataset_with_names_to_classify(dataset)
        dataset = preprocessor.drop_na_from_dataset(dataset)
        text, labels, index_to_names_dict = preprocessor.prepare_text_and_labels(dataset)
        
        return text, labels, index_to_names_dict, dataset
    
    def get_model_prepocessed(self, text: np.array, labels: np.array, index_to_names_dict:dict):
        """
        preprocess the data model for the evaluation.
        Args:
            text (np.array): text to classify
            labels (np.array): labels for the text
            index_to_names_dict (dict): dictionary of the index to names
            
        """
        preprocessor = ClassificationModelPreprocessor(text, labels, index_to_names_dict)
        token_ids, attention_masks = preprocessor.get_tokenized_text()  
        
        return token_ids, attention_masks
    
    def load_model(self, num_of_labels: int):
        """
        Load the model from the model path.
        Args:
            num_of_labels (int): number of labels in the model
        """
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels = num_of_labels,
                output_attentions = False,
                    output_hidden_states = False)
        path = f"{self.model_path}/model.pt"

        model.load_state_dict(torch.load(path))
        
        with open(f"{self.model_path}/names_mapping.pickle", 'rb') as handle:
            names_mapping = pickle.load(handle)
        return model, names_mapping
    
    def get_model_predictions(self, model, names_mapping, token_ids, attention_masks):
        """
        Use the data and model to get the model predictions - probability for each label.
        Args:
            model (torch model): model to use
            names_mapping (dict): mapping between the names and the labels
            token_ids (torch tensor): token ids
            attention_masks (torch tensor): attention masks
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            output = model(token_ids.to(device), token_type_ids = None, attention_mask = attention_masks.to(device))
        probs = torch.softmax(output.logits, dim=1)
        predictions = pd.DataFrame(probs, columns=[f"{name}_prob" for name in names_mapping.values()])
        return predictions

    def get_predictions_with_data(self, predictions, dataset, names_mapping):
        """
        Concat the predictions with the data.
        add the predicted name and the real name.
        Args:
            predictions (pd dataframe): predictions
            dataset (pd dataframe): data
            names_mapping (dict): mapping between the names and the labels
        """
        predictions_with_data = pd.concat([dataset, predictions], axis=1)
        predictions_with_data['preds'] = predictions_with_data.apply(lambda row: get_predicted_name(row, names_mapping), axis=1)
        predictions_with_data['name'] = predictions_with_data['name'].apply(lambda x: names_mapping[x])
        return predictions_with_data
    
    def save_dataframe(self, predictions_with_data):
        
        model_date = self.model_path.split('/')[-2]
        inference_path = os.path.join(self.model_inference_path, model_date)
        os.mkdir(inference_path)
        inference_path_with_file_name = os.path.join(inference_path, 'predictions.csv')
        predictions_with_data.to_csv(inference_path_with_file_name, index=False)
        
    def run(self):
        """
        run the evaluation.
        """
        text, labels, index_to_names_dict, dataset = self.get_data_preprocessed_for_model()
        token_ids, attention_masks = self.get_model_prepocessed(text, labels, index_to_names_dict)
        model, names_mapping = self.load_model(len(config['names_to_classify']))
        predictions = self.get_model_predictions(model, names_mapping, token_ids, attention_masks)
        predictions = self.get_predictions_with_data(predictions, dataset, names_mapping)
        self.save_dataframe(predictions)
        
if __name__ == '__main__':
    from config import config
    evaluator = Evaluator(
        config['prepared_for_classification_path'], 
        config['test_file_names'], 
        config['model_path'], 
        config['model_inference_path'],
        config['names_to_classify'])
    text, labels, index_to_names_dict, dataset = evaluator.get_data_preprocessed_for_model()
    token_ids, attention_masks = evaluator.get_model_prepocessed(text, labels, index_to_names_dict)
    model, names_mapping = evaluator.load_model(len(config['names_to_classify']))
    predictions = evaluator.get_model_predictions(model, names_mapping, token_ids, attention_masks)
    predictions = evaluator.get_predictions_with_data(predictions, dataset, names_mapping)
    evaluator.save_dataframe(predictions)