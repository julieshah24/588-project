import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from text_encoders import RoBERTaEncoder
from datasets import SpamDataset
import torch
from tqdm import tqdm 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = RoBERTaEncoder(device=device)

dataset = SpamDataset('enron_spam_data.csv', sort_by_date=True)

encoded_features = []
time_bins = []
spam_label = []

for data in tqdm(dataset):
    message = data['Subject']
    spam_label.append(data['Spam/Ham'])
    feature_vector = encoder.encode(message).squeeze().detach().cpu().numpy() 
    encoded_features.append(feature_vector)
    time_bins.append(data['Time Bin'])

encoded_features = np.array(encoded_features)


##############
#### TSNE ####
##############
tsne = TSNE(n_components=2, random_state=42)
tsne_features = tsne.fit_transform(encoded_features)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=spam_label, cmap='spring')
plt.colorbar(scatter, label='Time Bin')
plt.title('t-SNE visualization of RoBERTa Features with spam/ham Coloring')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()
plt.savefig('enron_roberta_tsne_only_spam.png')

##############
#### PCA #####
##############

pca = PCA(n_components=2)
pca_features = pca.fit_transform(encoded_features)

plt.figure(figsize=(8, 6))
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=spam_label, cmap='spring', alpha=0.7)
plt.title('PCA visualization of RoBERTa Features with spam/ham Coloring')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Class')
plt.show()