import os
import sys
import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from classification.constants import pretrained_model_name, batch_size, val_ratio

class ModelPreprocessing:
    """
    We use the outputs from the DataPreprocessing class to prepare the data for the model.
    The data include the text, the labels and index-names mapping dictionary.
    """
    def __init__(self, text: list, labels: list, index_to_names_dict: dict):
        self.text = text
        self.labels = labels
        self.index_to_names_dict = index_to_names_dict
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, do_lower_case = True)
        
    def get_tokenized_text(self):
        """Tokenize the text
        """
        token_ids = []
        attention_masks = []

        for sample in self.text:
            
            encoding_dict = self._tokenize_sample(sample, self.tokenizer)
            token_ids.append(encoding_dict['input_ids']) 
            attention_masks.append(encoding_dict['attention_mask'])

        token_ids = torch.cat(token_ids, dim = 0)
        attention_masks = torch.cat(attention_masks, dim = 0)
        
        return token_ids, attention_masks
  
        
    def _tokenize_sample(self, input_text:str, tokenizer:BertTokenizer):
        """
        Tokenize sample (a sentence in our case) with tokenizer.
        Args:
            input_text (str): sample to tokenize
            tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase): tokenizer to use
        Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
            - input_ids: list of token ids
            - token_type_ids: list of token type ids
            - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
        """
        
        return tokenizer.encode_plus(
                                input_text,
                                add_special_tokens = True,
                                max_length = 64,
                                padding='max_length',
                                return_attention_mask = True,
                                return_tensors = 'pt',
                                truncation=True
                        )

    def train_validation_split(self, token_ids:list, attention_masks:list):
        """Split the data to train and validation sets

        Args:
            token_ids (list): the tokenized text converted to ids
            attention_masks (list): the paddings for the text
        """
        labels = torch.tensor(self.labels) 
        # Indices of the train and validation splits stratified by labels
        train_idx, val_idx = train_test_split(
            np.arange(len(labels)),
            test_size = val_ratio,
            shuffle = True,
            stratify = labels)

        # Train and validation sets
        train_set = TensorDataset(token_ids[train_idx], 
                                attention_masks[train_idx], 
                                labels[train_idx])

        val_set = TensorDataset(token_ids[val_idx], 
                                attention_masks[val_idx], 
                                labels[val_idx])

        # Prepare DataLoader
        train_dataloader = DataLoader(
                    train_set,
                    sampler = RandomSampler(train_set),
                    batch_size = batch_size
                )

        validation_dataloader = DataLoader(
                    val_set,
                    sampler = SequentialSampler(val_set),
                    batch_size = batch_size
                )
        return train_dataloader, validation_dataloader
        
    def run(self):
        """Run the preprocessing
        """
        token_ids, attention_masks = self.get_tokenized_text()
        train_dataloader, validation_dataloader = self.train_validation_split(token_ids, attention_masks)
        return train_dataloader, validation_dataloader
    
if __name__ == '__main__':
    from data_preprocessing import DataPreprocessing
    from constants import datasets
    data_preprocessing = DataPreprocessing(datasets)
    text, labels, index_to_names_dict = data_preprocessing.run()
    print("done data preprocessing")
    model_preprocessing = ModelPreprocessing(text, labels, index_to_names_dict)
    train_dataloader, validation_dataloader = model_preprocessing.run()
    print("done model preprocessing")