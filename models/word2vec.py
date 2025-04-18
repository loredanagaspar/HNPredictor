import torch
import torch.nn as nn
from utils.tokenizer import build_tokenizer
import json
import os
import pickle
import time


class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, corpus, window_size=2):
        self.pairs = []
        for sent in corpus:
            for i, center in enumerate(sent):
                for j in range(max(i - window_size, 0), min(i + window_size + 1, len(sent))):
                    if i != j:
                        self.pairs.append((center, sent[j]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.out = nn.Linear(embed_dim, vocab_size)
        self.embedding_dim = embed_dim
        self.vocab = None

    def forward(self, x):
        emb = self.embedding(x)
        return self.out(emb)


def train_word2vec(path=None, vector_size=100, window=2, epochs=5, vocab_out="hn_vocab.json"):
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            raw = f.read()
    else:
        from datasets import load_dataset
        print(f"[{time.ctime()}] 📥 Downloading 'text8' from Hugging Face...")
        ds = load_dataset("afmck/text8-chunked1024", split='train')
        raw = "\n".join(ds["text"])

    print(f"[{time.ctime()}] 🧠 Building tokenizer from corpus...")
    word2idx, id2word = build_tokenizer()

    with open("data/corpus.pkl", "rb") as f:
        indexed = pickle.load(f)

    tokens = [word2idx.get(w, 0) for w in indexed]
    windowed = [tokens[i:i + 20] for i in range(0, len(tokens) - 20, 20)]

    dataset = SkipGramDataset(windowed, window_size=window)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.ctime()}] ✅ Using device: {device}")

    model = SkipGramModel(len(word2idx), vector_size).to(device)
    model.vocab = word2idx

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        for center, context in dataloader:
            center = center.to(device)
            context = context.to(device
            pred = model(center)
            loss = criterion(pred, context)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())
     
       

    save_vocab(word2idx, vocab_out)
    return model


def save_vocab(vocab: dict, path: str):
    with open(path, 'w') as f:
        json.dump(vocab, f)


def load_vocab(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)
