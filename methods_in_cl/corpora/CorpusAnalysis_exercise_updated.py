import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, WordNetLemmatizer
import matplotlib.pyplot as plt
from collections import Counter
import chardet
import string
import re
from typing import Dict, List, Tuple, Union
import sys


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")


def detect_file_encoding(file_path: str) -> str:
    """Detect the encoding of a file using chardet."""
    with open(file_path, "rb") as f:
        raw_data = f.read(10000)  # Read the first 10,000 bytes
        result = chardet.detect(raw_data)
        return result["encoding"]


def load_corpus(file_path: str) -> List[str]:
    """Load the content of the file with fallback encodings."""

    # Detect encoding
    encoding = detect_file_encoding(file_path=file_path)
    print(f"Detected encoding for {file_path}: {encoding}")

    try:
        # Try opening with the detected encoding
        with open(file_path, "r", encoding=encoding) as f:
            lines = f.readlines()
    except (UnicodeDecodeError, LookupError):
        print(
            f"Failed to decode {file_path} with detected encoding '{encoding}'. Trying fallback encoding 'utf-8'."
        )
        try:
            # Fallback to utf-8
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            print(
                f"Failed to decode {file_path} with 'utf-8'. Trying fallback encoding 'latin-1'."
            )
            try:
                # Fallback to latin-1
                with open(file_path, "r", encoding="latin-1") as f:
                    lines = f.readlines()
            except UnicodeDecodeError as e:
                print(f"Failed to decode {file_path} with all encodings. Error: {e}")
                return None
    return lines


def tokenize_sentences(lines: List[str]) -> Tuple[List[str], List[str]]:
    """
    Tokenize the lines into sentences and words.

    Parameters:
    lines (List[str]): A list of strings where each string is a line of text.

    Returns:
    Tuple[List[str], List[str]]:
        - A list of sentences extracted from the input lines.
        - A list of tokens (words) extracted from the sentences.
    """
    sentences = []
    tokens = []
    """ADD CODE HERE"""

    return sentences, tokens


def process_tokens(tokens: List[str]) -> Tuple[List[str], List[str]]:
    """
    Process tokens to compute lemmas and POS tags, removing punctuation.

    Parameters:
    tokens (list of str): A list of tokens (words) to be processed.

    Returns:
    tuple: A tuple containing two lists:
        - lemmas (list of str): A list of lemmatized words.
        - pos_tags (list of str): A list of part-of-speech tags corresponding to the lemmas.
    """
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Regular expression for filtering out punctuation
    punctuation_regex = re.compile(f"[{re.escape(string.punctuation)}]")

    """Get list of lemmas and part-of-speech tags, do not include punctuation"""
    lemmas = []
    pos_tags = []
    """ ADD CODE HERE HERE """

    return lemmas, pos_tags


def compute_statistics(
    sentences: List[str], tokens: List[str], lemmas: List[str], pos_tags: List[str]
) -> Dict[str, Union[Counter, int, float]]:
    """Compute various statistics for the corpus.

    Parameters:
    sentences (list): A list of sentences in the corpus.
    tokens (list): A list of tokens in the corpus.
    lemmas (list): A list of lemmas corresponding to the tokens.
    pos_tags (list): A list of part-of-speech tags corresponding to the tokens.

    Returns:
    dict: A dictionary containing the following statistics:
        - num_sentences (int): The total number of sentences.
        - num_tokens (int): The total number of tokens.
        - num_types (int): The total number of unique types.
        - types_tokens_ratio (float): The ratio of unique types to total tokens.
        - avg_sentence_length (float): The average length of sentences.
        - min_sentence_length (int): The length of the shortest sentence.
        - max_sentence_length (int): The length of the longest sentence.
        - sentence_lengths (list): A list of lengths of each sentence.
        - lemma_distribution (Counter): A distribution of lemmas.
        - pos_distribution (Counter): A distribution of part-of-speech tags.
    """

    """ ADD CODE HERE: calculations (remove all the None/-1 placeholders)"""
    sentence_lengths = []
    num_sentences = None
    num_tokens = None
    num_types = None

    types_tokens_ratio = -1
    avg_sentence_length = -1
    min_sentence_length = None
    max_sentence_length = None

    return {
        "num_sentences": num_sentences,
        "num_tokens": num_tokens,
        "num_types": num_types,
        "types_tokens_ratio": types_tokens_ratio,
        "avg_sentence_length": avg_sentence_length,
        "min_sentence_length": min_sentence_length,
        "max_sentence_length": max_sentence_length,
        "sentence_lengths": sentence_lengths,
        "lemma_distribution": Counter(lemmas),
        "pos_distribution": Counter(pos_tags),
    }


def visualize_statistics(
    stats: Dict[str, Union[Counter, int, float]], corpus_name: str
):
    """Visualize distributions and statistics.

    Parameters:
    stats (dict): A dictionary containing statistical data for the corpus,
            including sentence lengths, lemmas, and POS tags.
    corpus_name (str): The name of the corpus being analyzed.
    """

    # Plot for Sentence length distribution
    """ ADD CODE HERE """

    # Plot for Top 20 lemmas
    """ ADD CODE HERE """

    # Plot for POS tag distribution
    """ ADD CODE HERE """


def analyze_corpus(file_path: str, corpus_name: str):
    """Main function to analyze the corpus.

    Parameters:
    file_path (str): The path to the corpus file to be analyzed.
    corpus_name (str): The name of the corpus for display purposes.
    """

    lines = load_corpus(file_path=file_path)
    if lines is None:
        return

    sentences, tokens = tokenize_sentences(lines=lines)
    lemmas, pos_tags = process_tokens(tokens=tokens)
    stats = compute_statistics(
        sentences=sentences, tokens=tokens, lemmas=lemmas, pos_tags=pos_tags
    )

    # Print statistics
    print(f"Corpus: {corpus_name}")
    print(f"Number of sentences: {stats['num_sentences']}")
    print(f"Number of tokens: {stats['num_tokens']}")
    print(f"Number of types (lemmas): {stats['num_types']}")
    print(f"Types/Tokens Ratio: {stats['types_tokens_ratio']:.4f}")
    print(f"Average Sentence Length: {stats['avg_sentence_length']:.2f}")
    print(f"Min Sentence Length: {stats['min_sentence_length']}")
    print(f"Max Sentence Length: {stats['max_sentence_length']}")

    visualize_statistics(stats, corpus_name)


def process_corpora(txt_files):
    """Process and analyze multiple .txt files.

    Parameters:
    txt_files (list of str): A list of paths to the .txt files to be processed.
    """
    for txt_path in txt_files:
        print(f"Processing {txt_path}...")
        corpus_name = os.path.basename(txt_path)
        analyze_corpus(file_path=txt_path, corpus_name=corpus_name)


if __name__ == "__main__":

    # Example usage to process all files at once

    corpus_files = [
        "MovieCorpus.txt",
        "ACL_partial_abstract_corpus.txt",
        "BNCSplitWordsCorpus.txt",
        "english-brown.txt",
        "TwitterLowerAsciiCorpus.txt",
    ]

    process_corpora(corpus_files)

    # Example step-by-step usage for one corpus only

    corpus_file = "MovieCorpus.txt"
    corpus_lines = load_corpus(corpus_file)
    sentences, tokens = tokenize_sentences(corpus_lines)
    lemmas, pos_tags = process_tokens(tokens)
    stats = compute_statistics(sentences, tokens, lemmas, pos_tags)
    visualize_statistics(stats, corpus_file.replace(".txt", ""))
