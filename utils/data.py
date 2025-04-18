# utils/data.py
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
from collections import defaultdict

class HackerNewsDataset(Dataset):
    def __init__(self, df, word2vec_model):
        self.df = df.reset_index(drop=True)
        self.embedding = word2vec_model.embedding
        self.vocab = word2vec_model.vocab
        self.dim = word2vec_model.embedding_dim

        # Encode domain
        domains = sorted(set(df["domain"]))
        self.domain2idx = {d: i for i, d in enumerate(domains)}
        self.domain_embed = torch.nn.Embedding(len(domains), 8)

        # Normalize other structured fields
        self.max_title_len = df["title_length"].max()
        self.hours = df["hour"].astype(float) / 23.0
        self.weekdays = df["weekday"].astype(float) / 6.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = [self.vocab[w] for w in row["title"].split() if w in self.vocab]
        if tokens:
            vecs = self.embedding(torch.tensor(tokens))
            title_emb = torch.mean(vecs, dim=0)
        else:
            title_emb = torch.zeros(self.dim)

        domain_idx = torch.tensor(self.domain2idx.get(row["domain"], 0))
        domain_vec = self.domain_embed(domain_idx)

        # Normalize + cast
        title_len = torch.tensor(row["title_length"] / self.max_title_len)
        hour = torch.tensor(self.hours[idx])
        weekday = torch.tensor(self.weekdays[idx])

        # Structured features
        features = torch.cat([domain_vec, title_len.unsqueeze(0), hour.unsqueeze(0), weekday.unsqueeze(0)])

        # Final input vector = [title embedding + structured features]
        final_input = torch.cat([title_emb, features])

        return final_input, torch.tensor(row["score"], dtype=torch.float32)

def train_test_split_data(df):
    return train_test_split(df, test_size=0.2, random_state=42)
