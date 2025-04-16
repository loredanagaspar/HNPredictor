import pandas as pd
from pipeline.tokenizer import clean_and_tokenize
from embedding.word2vec_trainer import build_vocab, generate_skipgram_data, train_skipgram_model

if __name__ == "__main__":
    print("📄 Loading data from CSV...")
    df = pd.read_csv("hn_title_modeling_dataset.csv")

    print("✅ Data loaded:", df.shape)
    print("🧾 Columns:", df.columns.tolist())
    print(df.head(3))

# Tokenize titles
df["tokens"] = df["title"].astype(str).apply(clean_and_tokenize)
print("🗣️ Tokenized titles:", df["tokens"].head(3))

# Save the DataFrame with tokens to a new CSV file
df.to_csv("hn_title_modeling_dataset_with_tokens.csv", index=False)
print("✅ Tokens saved to 'hn_title_modeling_dataset_with_tokens.csv'")

token_lists = df["tokens"].tolist()
vocab, index2word = build_vocab(token_lists, min_count=5)
pairs = generate_skipgram_data(token_lists, vocab, window_size=2)

print("🧠 Training Skip-gram model...")
model = train_skipgram_model(pairs, vocab_size=len(vocab), embedding_dim=100)
torch.save(model.state_dict(), "skipgram_hn.pt")
