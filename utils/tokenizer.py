import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize as nltk_word_tokenize

def word_tokenize(text):
    return nltk_word_tokenize(text.lower())
