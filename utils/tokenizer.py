import collections
import requests
import pickle
import os

CORPUS_URL = "https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8"
CORPUS_FILE = "data/text8"
CORPUS_PROCESSED = "data/corpus.pkl"
VOCAB2ID_FILE = "data/tkn_words_to_ids.pkl"
ID2VOCAB_FILE = "data/tkn_ids_to_words.pkl"


def download_text8():
    if not os.path.exists(CORPUS_FILE):
        print("📦 Downloading text8 corpus...")
        r = requests.get(CORPUS_URL)
        os.makedirs("data", exist_ok=True)
        with open(CORPUS_FILE, 'wb') as f:
            f.write(r.content)


def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()
    stats = collections.Counter(words)
    words = [word for word in words if stats[word] > 5]
    return words


def create_lookup_tables(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    word_counts = collections.Counter(words)
    vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
    int_to_vocab = {ii + 1: word for ii, word in enumerate(vocab)}
    int_to_vocab[0] = '<PAD>'
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab


def build_tokenizer():
    download_text8()

    print("🧹 Preprocessing text8...")
    with open(CORPUS_FILE, 'r') as f:
        raw = f.read()

    words = preprocess(raw)
    with open(CORPUS_PROCESSED, 'wb') as f:
        pickle.dump(words, f)

    vocab2id, id2vocab = create_lookup_tables(words)

    with open(VOCAB2ID_FILE, 'wb') as f:
        pickle.dump(vocab2id, f)

    with open(ID2VOCAB_FILE, 'wb') as f:
        pickle.dump(id2vocab, f)

    print(f"✅ Preprocessed {len(words):,} tokens, vocab size: {len(vocab2id):,}")
    return vocab2id, id2vocab


def word_tokenize(text: str, vocab2id: dict[str, int]) -> list[int]:
    words = text.lower().split()
    return [vocab2id.get(w, 0) for w in words]  # 0 = <PAD>
