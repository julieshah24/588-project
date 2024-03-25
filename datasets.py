import re
import torch
from torch.utils.data import Dataset
import pandas as pd

import pandas as pd
import torch
from torch.utils.data import Dataset

class SpamDataset(Dataset):
    def __init__(self, csv_file, sort_by_date=False, num_bins=10):
        self.data = pd.read_csv(csv_file)
        self.data['Spam/Ham'] = (self.data['Spam/Ham'] == 'spam').astype(int)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        if sort_by_date:
            self.data = self.data.sort_values(by='Date')
        
        if num_bins is not None and sort_by_date:
            num_bins = max(int(num_bins), 1)
            bin_labels = [i for i in range(num_bins)]
            self.data['Time Bin'] = pd.cut(self.data['Date'], bins=num_bins, labels=bin_labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        message_id = self.data.iloc[idx, 0]
        subject = self.clean_text(str(self.data.iloc[idx, 1]))
        message = self.clean_text(str(self.data.iloc[idx, 2]))

        spam_ham = self.data.iloc[idx, 3]
        date = self.data.iloc[idx, 4]
        time_bin = self.data.iloc[idx].get('Time Bin', 'N/A')  
        
        sample = {
            'Message ID': message_id,
            'Subject': subject,
            'Message': message,
            'Spam/Ham': spam_ham,
            'Date': date,
            'Time Bin': time_bin
        }

        return sample
    
    def clean_text(self, text):
        return text
    
    def  spam_collate_fn(self, batch):
        batch_dict = {
            'Message ID': [],
            'Subject': [],
            'Message': [],
            'Spam/Ham': [],
            'Date': [],
            'Time Bin': []
        }
        for sample in batch:
            for key, value in sample.items():
                batch_dict[key].append(value)
        
        return batch_dict
