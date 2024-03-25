import warnings
from datasets import SpamDataset
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import NLLLoss, CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm 

warnings.filterwarnings("ignore")

class RoBERTaEncoder:
    def __init__(self, device="cpu", size="base"):
        self.device = device
        print("using device:", device)

        self.model = torch.hub.load("pytorch/fairseq", "roberta." + size)
        self.model.register_classification_head("spam_classifier", num_classes=2)
        self.model.train()  # Set the model to evaluation mode

        self.model.to(self.device)

    def encode_text(self, text):
        tokens = self.model.encode(text)
        tokens = tokens[:512]
        tokens = tokens.to(self.device)  
        with torch.no_grad(): 
            last_layer_features = self.model.extract_features(tokens, return_all_hiddens=False)[0, 0, :]
        return last_layer_features

    def predict_spam(self, text):
        self.model.train()
        all_tokens = []
        for t in text:
            tokens = self.model.encode(t)
            tokens = tokens[:512]
            all_tokens.append(tokens)

        padded_tokens = pad_sequence(all_tokens, batch_first=True, padding_value=1)
        tokens = padded_tokens.to(self.device)  
        logprobs = self.model.predict("spam_classifier", tokens)
        return logprobs

    def save_weights(self, filepath):
        self.model.to('cpu')
        torch.save(self.model.state_dict(), filepath)

    def load_weights(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))

    def pad_lists(self,input_lists):
        # Find the length of the longest list
        max_length = max(len(lst) for lst in input_lists)
        
        # Iterate through each list and pad with 1s if necessary
        for lst in input_lists:
            while len(lst) < max_length:
                lst.append(1)
        
        return input_lists

