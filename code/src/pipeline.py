#TODO: change all prints to logger

import os
import sys
import pickle
import logging
import numpy as np
import torch
from datetime import datetime

from transformers import BertForSequenceClassification

from config import config
from preprocessing.raw_data_preprocessing import Preprocessor as RawDataPreprocessor
from preprocessing.data_preprocessing import Preprocessor as DataPreprocessor
from machine_translation.machine_translation import MachineTranslation
from classification.data_preprocessing import DataPreprocessing as ClassificationDataPreprocessor
from classification. model_preprocessing import ModelPreprocessing as ClassificationModelPreprocessor
from classification.train import Train

sys.path.append("/workspaces/group_members_classification/")

# Configure the logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    handlers=[logging.FileHandler('logger.log', mode='w'),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

class Pipeline:
    """
    This class is responsible for running all pipeline.
    From taking the raw txt files - to the text classification
    """
    def __init__(self, config):
        self.data_paths = self.get_data_paths(config)
        logger.debug(f"Pipeline initialized with the following data files: {self.data_paths}")


    def get_data_paths(self, config):
        data_paths = []
        for file_name in config['raw_data_file_names']:
            data_paths.append(os.path.join(config['raw_data_path'],  file_name + '.txt'))
        return data_paths

    def load_data_csv(self):
        logger.debug(f"Loading data from csv")
        for file_name in config['raw_data_file_names']:
            logger.debug(f"Loading {file_name} from csv")
            preprocessor = RawDataPreprocessor(os.path.join(config['raw_data_path'],  file_name + '.txt'))
            data = preprocessor.run()
            preprocessor.save_dataframe(data, os.path.join(config['prepared_raw_data_path'],  file_name + '.csv'))
            logger.debug(f"{file_name} loaded and saved as csv")
            
    def prepare_data_for_machine_translation(self):
        logger.debug(f"Data preprocessing started")
        for file_name in config['raw_data_file_names']:
            logger.debug(f"Data preprocessing started for {file_name}")
            preprocessor = DataPreprocessor(os.path.join(config['prepared_raw_data_path'],  file_name + '.csv'))
            data = preprocessor.run()
            preprocessor.save_dataframe(data, os.path.join(config['prepared_for_machine_translation_path'],  file_name + '.csv'))
            logger.debug(f"{file_name} preprocessed and saved for machine translation")
            
    def prepare_data_for_classification(self):
        logger.debug(f"Machine translation started")
        for file_name in config['raw_data_file_names']:
            logger.debug(f"Machine translation started for {file_name}")
            machine_translation = MachineTranslation(os.path.join(config['prepared_for_machine_translation_path'],  file_name + '.csv'))
            data = machine_translation.run()
            machine_translation.save_dataframe(data, os.path.join(config['prepared_for_classification_path'],  file_name + '.csv'))
            logger.debug(f"{file_name} is translated and saved for classification")
            
    def classification_data_processing(self):
        logger.debug(f"Classification data processing started")
        preprocessor = ClassificationDataPreprocessor(
            config['prepared_for_classification_path'], config['train_file_names'], config['names_to_classify']
            )
        text, labels, index_to_names_dict = preprocessor.run()
        logger.debug(f"Classification data processing done")
        logger.debug(f"The names to classify are: {index_to_names_dict}")
        logger.debug(f"The total length of the data is: {len(text)}")
        unique, counts = np.unique(labels, return_counts=True)
        logger.debug(f"labels value counts {np.asarray((unique, counts)).T}")
        return text, labels, index_to_names_dict
     
    def classification_model_processing(self, text, labels, index_to_names_dict):
        logger.debug(f"Classification model processing started")
        preprocessor = ClassificationModelPreprocessor(text, labels, index_to_names_dict)
        train_dataloader, validation_dataloader = preprocessor.run()
        logger.debug(f"Classification model processing done")
        logger.debug(f"The train_dataloader length is: {len(train_dataloader.dataset)}")
        logger.debug(f"The validation_dataloader length is: {len(validation_dataloader.dataset)}")
        return train_dataloader, validation_dataloader
    
    def train_classification_model(self, train_dataloader, validation_dataloader, num_labels):
        logger.debug(f"Classification model training started")
        trainer = Train(train_dataloader, validation_dataloader, num_labels)
        trainer.run()
        logger.debug(f"Classification model training done")
        return trainer.model
            
    def save_model(self, model:BertForSequenceClassification, index_to_names_dict: dict, model_path: str):
        """Save the model and name mapping to restore original names

        Args:
            model (BertForSequenceClassification): trained model
            index_to_names_dict (dict): names mapping - from index to name
            model_path (str): path to save the model
        """
        logger.debug(f"Saving model started")
        # Get the current date 
        now = datetime.now()
        current_date = now.strftime("%Y_%m_%d_%H_%M_%S")

        # Create a new directory
        model_path = os.path.join(model_path, current_date)
        os.mkdir(model_path)
        
        print(f"Saving model to {model_path}")
        # save the model within the new directory
        torch.save(model.state_dict(), f"{model_path}/model.pt")
        with open(f'{model_path}/names_mapping.pickle', 'wb') as handle:
            pickle.dump(index_to_names_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f"model and names mapping saved to {model_path}")
        


if __name__ == '__main__':
    pipeline = Pipeline(config)  
    pipeline.load_data_csv()
    pipeline.prepare_data_for_machine_translation()
    pipeline.prepare_data_for_classification()
    text, labels, index_to_names_dict = pipeline.classification_data_processing() 
    train_dataloader, validation_dataloader = pipeline.classification_model_processing(
        text, labels, index_to_names_dict
        )
    model = pipeline.train_classification_model(train_dataloader, validation_dataloader, len(index_to_names_dict))
    pipeline.save_model(model, index_to_names_dict, config['classification_model_path'])
