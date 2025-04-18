# hn_upvote_predictor/main.py

from utils.db import fetch_data
from utils.data import HackerNewsDataset, train_test_split_data
from models.word2vec import train_word2vec, save_vocab
from models.regressor import train_model
import torch
from torch.utils.data import DataLoader


def main():
    print("Fetching data...")
    df = fetch_data()

    print("Training Word2Vec (Skip-gram)...")
    w2v_model = train_word2vec("data/text8", vocab_out="hn_vocab.json")

    print("Creating datasets...")
    train_df, test_df = train_test_split_data(df)
    train_set = HackerNewsDataset(train_df, w2v_model)
    test_set = HackerNewsDataset(test_df, w2v_model)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    print("Training regression model...")
    model = train_model(train_loader, w2v_model.embedding_dim, val_loader=test_loader)

    torch.save(model.state_dict(), "upvote_regressor.pt")
    torch.save(w2v_model.state_dict(), "hn_word2vec.pt")
    save_vocab(w2v_model.vocab, "hn_vocab.json")

    print("✅ Training complete.")


if __name__ == '__main__':
    main()
