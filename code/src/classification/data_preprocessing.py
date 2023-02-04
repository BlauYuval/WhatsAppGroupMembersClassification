import pandas as pd

class DataPreprocessing:
    """
    In this class we will preprocess the data for the model.
    """
    def __init__(self, global_data_path:str ,datasets:list, names_to_classify: list = []):
        self.gloabel_data_path = global_data_path
        self.datasets_names = datasets
        self.names_to_classify = names_to_classify
        
    def load_data(self):
        """load the data from the data paths.
        Returns:
            dict: dictionary of the datasets.
        """
        datasets = {}
        for dataset in self.datasets_names:
            datasets[dataset] = pd.read_csv(f'{self.gloabel_data_path}/{dataset}.csv')
        return datasets
        
    def concat_datasets(self, datasets: dict):
        """Concatenate the datasets into one dataset

        Args:
            datasets (dict): pandas dataframes
        """
        dataset = pd.concat(datasets)
        return dataset
    
    def filter_dataset_with_names_to_classify(self,dataset: pd.DataFrame):
        """Filter the dataset with the names to classify

        Args:
            dataset (pd.DataFrame): pandas dataframe
        """
        if self.names_to_classify:
            dataset = dataset[dataset['name'].isin(self.names_to_classify)]
        return dataset
    
    def drop_na_from_dataset(self, dataset: pd.DataFrame):
        """Drop the nan values from the dataset
        """
        dataset = dataset.dropna().reset_index(drop = True)
        return dataset
        
    def prepare_text_and_labels(self, dataset: pd.DataFrame):
        """Prepare the text and labels for the model

        Args:
            dataset (pd.DataFrame): pandas dataframe
        """
        if  len(self.names_to_classify) == 0:
            self.names_to_classify = dataset['name'].unique()
        names_to_index_dict = {self.names_to_classify[i]: i for i in range(len(self.names_to_classify))}
        index_to_names_dict = {i: self.names_to_classify[i] for i in range(len(self.names_to_classify))}
        dataset['name'] = dataset['name'].apply(lambda n: names_to_index_dict[n])
        text = dataset['translated_message'].values
        labels = dataset['name'].values
        return text, labels, index_to_names_dict
    
    def run(self):
        """Run the preprocessing
        """
        loaded_datasets = self.load_data()
        dataset = self.concat_datasets(loaded_datasets)
        dataset = self.filter_dataset_with_names_to_classify(dataset)
        dataset = self.drop_na_from_dataset(dataset)
        text, labels, index_to_names_dict = self.prepare_text_and_labels(dataset)
        return text, labels, index_to_names_dict
        
    
        

        
if __name__ == '__main__':
    from constants import datasets
    model_preprocessing = DataPreprocessing(datasets)
    loaded_datasets = model_preprocessing.load_data()
    dataset = model_preprocessing.concat_datasets(loaded_datasets)
    dataset = model_preprocessing.filter_dataset_with_names_to_classify(dataset)
    text, labels, index_to_names_dict = model_preprocessing.prepare_text_and_labels(dataset)
    print(index_to_names_dict)
    print(text[:10])
    print(labels[:10])