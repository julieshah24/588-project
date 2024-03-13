import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class SpamDataset(Dataset):
    def __init__(self, csv_file, transform=None, sort_by_date=False, shuffle=False):
        self.data = pd.read_csv(csv_file)        

        self.data['Spam/Ham'] = (self.data['Spam/Ham'] == 'spam').astype(int)
        self.data['Date'] = pd.to_datetime(self.data['Date'])

        if sort_by_date:
            self.data = self.data.sort_values(by='Date')

        if shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        message_id = self.data.iloc[idx, 0]
        subject = str(self.data.iloc[idx, 1])
        message = str(self.data.iloc[idx, 2])
        spam_ham = self.data.iloc[idx, 3]
        date = self.data.iloc[idx, 4]
        
        sample = {'Message ID': message_id, 'Subject': subject, 'Message': message, 'Spam/Ham': spam_ham, 'Date': date}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    @staticmethod
    def collate_fn(batch):
        collated_batch = {}
        for key in batch[0].keys():
            collated_batch[key] = [sample[key] for sample in batch]
        return collated_batch

dataset = SpamDataset('enron_spam_data.csv', sort_by_date=True, shuffle=True)

# Initialize data loader with collate function
data_loader = DataLoader(dataset, batch_size=3, collate_fn=SpamDataset.collate_fn)

# Iterate over data loader
for batch in data_loader:
    print(batch)
