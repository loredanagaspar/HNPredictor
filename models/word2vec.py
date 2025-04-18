import torch
import torch.nn as nn
from utils.tokenizer import word_tokenize
import json
import os


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


def build_vocab(text):
    tokens = set(w for line in text for w in line)
    word2idx = {w: idx for idx, w in enumerate(sorted(tokens))}
    return word2idx


def train_word2vec(path=None, vector_size=100, window=2, epochs=5, vocab_out="hn_vocab.json"):
    # Load text from file OR HuggingFace dataset fallback
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            raw = f.read()
    else:
        from datasets import load_dataset
        print("Downloading 'text8' from Hugging Face...")
        ds = load_dataset("afmck/text8-chunked1024", split='train')
        raw = "\n".join(ds["text"])

    # Tokenize and build vocab
    sentences = [word_tokenize(sent) for sent in raw.split('.') if sent]
    word2idx = build_vocab(sentences)
    indexed = [[word2idx[w] for w in sent if w in word2idx]
               for sent in sentences]
    dataset = SkipGramDataset(indexed, window_size=window)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True)

    model = SkipGramModel(len(word2idx), vector_size)
    model.vocab = word2idx
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for center, context in dataloader:
            pred = model(center)
            loss = criterion(pred, context)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    save_vocab(word2idx, vocab_out)
    return model


def save_vocab(vocab: dict, path: str):
    with open(path, 'w') as f:
        json.dump(vocab, f)


def load_vocab(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)
