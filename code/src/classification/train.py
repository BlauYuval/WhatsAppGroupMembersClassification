import os
import sys
from tqdm import trange
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from classification.constants import lr, eps, pretrained_model_name, epochs
from classification.utils import b_metrics

class Train:
    """
    In this class we will train the model.
    We will use the pretrained model and fine tune it on our data.
    We also have the train_dataloader, validation_dataloader from the ModelPreprocessing class.
    """
    def __init__(self, train_dataloader:DataLoader, validation_dataloader:DataLoader, num_of_labels:int) -> None:
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        # Load the BertForSequenceClassification model
        self.model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels = num_of_labels,
            output_attentions = False,
            output_hidden_states = False)
        # Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                    lr = lr,
                                    eps = eps
                                    )
        
    def run(self):
        """Tain the model
        """
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf

        for _ in trange(epochs, desc = 'Epoch'):
            
            # ========== Training ==========
            
            # Set model to training mode
            self.model.train()
            
            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(self.train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                self.optimizer.zero_grad()
                # Forward pass
                train_output = self.model(b_input_ids, 
                                    token_type_ids = None, 
                                    attention_mask = b_input_mask, 
                                    labels = b_labels)
                # Backward pass
                train_output.loss.backward()
                self.optimizer.step()
                # Update tracking variables
                tr_loss += train_output.loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            # ========== Validation ==========

            # Set model to evaluation mode
            self.model.eval()

            # Tracking variables 
            val_accuracy = []
            val_precision = []
            val_recall = []
            val_specificity = []

            for batch in self.validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    # Forward pass
                    eval_output = self.model(b_input_ids, 
                                        token_type_ids = None, 
                                        attention_mask = b_input_mask)
                logits = eval_output.logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                # Calculate validation metrics
                b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
                val_accuracy.append(b_accuracy)
                # Update precision only when (tp + fp) !=0; ignore nan
                if b_precision != 'nan': val_precision.append(b_precision)
                # Update recall only when (tp + fn) !=0; ignore nan
                if b_recall != 'nan': val_recall.append(b_recall)
                # Update specificity only when (tn + fp) !=0; ignore nan
                if b_specificity != 'nan': val_specificity.append(b_specificity)

            print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
            print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
            print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
            print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
            print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')
            
        return self.model

        
        