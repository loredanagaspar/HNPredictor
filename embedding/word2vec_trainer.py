# /embedding/word2vec_torch.py

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from typing import List, Tuple
import random


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_word_idx):
        emb = self.embedding(center_word_idx)
        out = self.output(emb)
        return out


def build_vocab(token_lists: List[List[str]], min_count: int = 5) -> Tuple[dict, dict]:
    word_freq = Counter(word for tokens in token_lists for word in tokens)
    vocab = {word: i for i, (word, freq) in enumerate(
        word_freq.items()) if freq >= min_count}
    index2word = {i: w for w, i in vocab.items()}
    return vocab, index2word


def generate_skipgram_data(token_lists: List[List[str]], vocab: dict, window_size: int = 2) -> List[Tuple[int, int]]:
    data = []
    for tokens in token_lists:
        indexed = [vocab[word] for word in tokens if word in vocab]
        for i, center in enumerate(indexed):
            for j in range(max(i - window_size, 0), min(i + window_size + 1, len(indexed))):
                if i != j:
                    context = indexed[j]
                    data.append((center, context))
    return data


def train_skipgram_model(pairs: List[Tuple[int, int]], vocab_size: int, embedding_dim: int = 100, epochs: int = 5) -> SkipGramModel:
    model = SkipGramModel(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(pairs)
        for center, context in pairs:
            center_tensor = torch.tensor([center])
            target_tensor = torch.tensor([context])

            output = model(center_tensor)
            loss = loss_fn(output, target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    return model
