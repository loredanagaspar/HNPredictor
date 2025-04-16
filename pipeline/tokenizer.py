import re
from typing import List


def clean_and_tokenize(text: str) -> List[str]:
    """
    Cleans and tokenizes the input text.
    - Converts to lowercase
    - Removes punctuation and special characters
    - Splits into words (tokens)

    Args:
        text (str): The input text to clean and tokenize.

    Returns:
        List[str]: A list of cleaned and tokenized words.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Split into words (tokens)
    tokens = text.split()

    return tokens
