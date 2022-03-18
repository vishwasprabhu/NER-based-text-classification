import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd


# if a word is not in your vocabulary use len(vocabulary) as the encoding
class NERDataset(Dataset):
    def __init__(self, df_enc):
        self.df = df_enc

    def __len__(self):
        """ Length of the dataset """
        L = len(self.df)-4
        return L

    def __getitem__(self, idx):
        """ returns x[idx], y[idx] for this dataset
        
        x[idx] should be a numpy array of shape (5,)
        """
        x, y = np.array(self.df['word'].values[idx:idx+5]), np.array(self.df['label'].values[idx+2])
        return x, y 


def label_encoding(cat_arr):
   """ Given a numpy array of strings returns a dictionary with label encodings.

   First take the array of unique values and sort them (as strings). 
   """
   sorted_list = sorted(set(list(cat_arr.astype('str'))))
   vocab2index = {word:i for i,word in enumerate(sorted_list)}
   return vocab2index


def dataset_encoding(df, vocab2index, label2index):
    """Apply vocab2index to the word column and label2index to the label column

    Replace columns "word" and "label" with the corresponding encoding.
    If a word is not in the vocabulary give it the index V=(len(vocab2index))
    """
    V = len(vocab2index)
    df_enc = df.copy()
    df_enc['word'] = df_enc['word'].map(vocab2index).fillna(len(vocab2index)).astype(int)
    df_enc['label'] = df_enc['label'].map(label2index).fillna(len(label2index)).astype(int)
    return df_enc


class NERModel(nn.Module):
    def __init__(self, vocab_size, n_class, emb_size=50, seed=3):
        """
        Initialize an embedding layer and a linear layer
        """
        super(NERModel, self).__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.linear = nn.Linear(5*emb_size,n_class)
        
    def forward(self, x):
        """Apply the model to x
        
        1. x is a (N,5). Lookup embeddings for x
        2. reshape the embeddings (or concatenate) such that x is N, 5*emb_size 
           .flatten works
        3. Apply a linear layer
        """
        x = self.emb(x)
        x = x.flatten(1,2)
        x = self.linear(x)
        return x

def get_optimizer(model, lr = 0.01, wd = 0.0):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optim

def train_model(model, optimizer, train_dl, valid_dl, epochs=10):
    for i in range(epochs):
        model.train()
        losses = []
        for x, y in train_dl:
            y_hat = model(x)
            loss = F.cross_entropy(y_hat,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = np.mean(losses)
        valid_loss, valid_acc = valid_metrics(model, valid_dl)
        print("train loss  %.3f val loss %.3f and accuracy %.3f" % (
            train_loss, valid_loss, valid_acc))

def valid_metrics(model, valid_dl):
    losses = []
    y_preds = []
    ys = []
    model.eval()
    for x, y in valid_dl:
        y_hat = model(x)
        loss = F.cross_entropy(y_hat,y)
        y_prob = torch.log_softmax(y_hat,dim=1)
        ys.append(y.numpy())
        losses.append(loss)
        y_pred = torch.argmax(y_prob, dim=1)
        y_preds.append(y_pred)
    val_loss = np.mean(losses)
    val_acc = accuracy_score(np.concatenate(ys), np.concatenate(y_preds))
    return val_loss, val_acc

