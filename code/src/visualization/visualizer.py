import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from visualization.mask_names import mask

class Visulaizer:
    """
    In this class we take the predictions with data table and visualize it.
    """
    def __init__(self, model_inference_path:str):
        self.model_inference_path = model_inference_path
        
    def load_predictions(self):
        """
        Load the predictions from the model inference path.
        """
        predictions = pd.read_csv(os.path.join(self.model_inference_path, 'predictions.csv'))
            
        return predictions
    
    def _mask_names(self, text, lower=True):
        for name in mask:
            if lower:
                text = text.replace(name.lower(), mask[name].lower())
            else:
                text = text.replace(name, mask[name])
        return text
    
    def mask_names_in_predictions(self, predictions:pd.DataFrame):
        """
        Mask the names in the predictions.
        """
        predictions.columns = [self._mask_names(col, lower=True) for col in predictions.columns]
        predictions['translated_message'] = predictions['translated_message'].apply(lambda text: self._mask_names(text, lower=False))
        predictions['name'] = predictions['name'].apply(lambda text: self._mask_names(text, lower=True))
        predictions['preds'] = predictions['preds'].apply(lambda text: self._mask_names(text, lower=True))
        
        return predictions
    
    def get_predictions_names(self, predictions:pd.DataFrame):
        """
        Get the names of the predictions.
        """
        names = predictions['name'].unique()
        
        return names
    
    def create_text_dict_for_visualization(self, predictions:pd.DataFrame, names:np.array):
        """
        Create a dictionary of the text for the visualization.
        """
        texts_dict = {}
        for true_name in names:
            texts_dict[true_name] = {}
            for preds_name in names:
                if true_name == preds_name:
                    
                    messages = predictions[
                        (predictions['name'] == true_name) & 
                        (predictions['preds'] == preds_name)
                        ].sort_values(f'{true_name}_prob', ascending=False).head(7).translated_message.apply(lambda x:x if len(x)<=50 else f"{x[:30]}...").tolist()
                    
                else:
                    
                    messages_df = predictions[
                        (predictions['name'] == true_name) &
                        (predictions['preds'] == preds_name)
                    ].copy()
                    if len(messages_df) < 5:
                        messages = messages_df.translated_message.apply(lambda x:x if len(x)<=50 else f"{x[:50]}...").tolist()
                    else:
                        messages = messages_df.sample(3).translated_message.apply(lambda x:x if len(x)<=50 else f"{x[:50]}...").tolist()
                texts_dict[true_name][preds_name] = '\n'.join(messages)
                
                    
        return texts_dict
    
    def get_confusion_matrix(self, predictions:pd.DataFrame, names:np.array):
        """
        Create the confusion matrix of the predictions.     
        assign the columns and index names to the confusion matrix.   
        Args:
            predictions: predictions dataframe
            names: names of the predictions
        Returns:
            confusion_matrix_df: confusion matrix dataframe
        """
        confusion_matrix_df = pd.DataFrame(
            confusion_matrix(
                predictions['name'], 
                predictions['preds'], 
                labels=names),
            columns=[f"{name}_pred" for name in names],
            index=[f"{name}_true" for name in names])
        
        return confusion_matrix_df
        
    def _get_index_to_names_dict(self, predictions:pd.DataFrame):
        """
        This function returns a dictionary of the index to the name.
        This is used for the visualization indexing.
        Args:
            predictions: predictions dataframe
        Returns:
            index_to_names_dict: dictionary of the index to the name
        """
        index_to_names_dict = {i[0]: i[1] for i in predictions[['idx_name', 'name']].value_counts().index.tolist()}
        return index_to_names_dict
    
    def plot_confusion_matrix(self,
                              cm, 
                              texts_dict,
                            classes,
                            index_to_names_dict,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix with text examples for each class.
        Args:
            cm: confusion matrix
            index_to_names_dict: dict of index to name
            title: title of the plot
            cmap: color map
        """
        fig, ax = plt.subplots(figsize=(11,11))

        ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks, classes, rotation=45)
        ax.set_yticks(tick_marks, classes)
        thresh = cm.max() / 1.5,

        classes_length = len(classes)
        for i in range(classes_length):
            for j in range(classes_length):
                true_name = index_to_names_dict[j]
                pred_name = index_to_names_dict[i]
                ax.text(i, j, f"Actual name is {true_name}\n Predicted Name is {pred_name}\n \n Text Examples:\n",
                    verticalalignment="bottom",
                    horizontalalignment="center",
                    fontsize=10,
                    color="white" if cm[i,j] > thresh else "black")
                ax.text(i, j, f"{texts_dict[true_name][pred_name]}",
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontsize=8,
                    color="white" if cm[i,j] > thresh else "black")

        plt.tight_layout()
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        
        circles = [mpatches.Rectangle((-0.5+i,-0.5+i), width=1,height=1, fill=False, color='black', linewidth=5) for i in range(len(classes))]
        for circle in circles:
            ax.add_patch(circle)

        plt.savefig(
            os.path.join(self.model_inference_path, 'confusion_matrix.png'), 
            dpi=300, bbox_inches='tight')
        
    def run(self, is_mask_names:bool=False):
        """
        Run the visualization step.
        In this step we use the predictions data and create confusion matrix and text examples for each class.
        """
        predictions = self.load_predictions()
        if is_mask_names:
            predictions = self.mask_names_in_predictions(predictions)
        names = self.get_predictions_names(predictions)
        text_examples_for_each_class = self.create_text_dict_for_visualization(predictions, names)
        confusion_matrix_df = self.get_confusion_matrix(predictions, names)
        self.plot_confusion_matrix(
            confusion_matrix_df.values,
            text_examples_for_each_class, 
            names,
            self._get_index_to_names_dict(predictions))
        
if __name__ == '__main__':
    from config import config
    path = os.path.join(config['model_inference_path'], config['model_path'].split('/')[-2])
    vis = Visulaizer(path)
    vis.run(is_mask_names=True)