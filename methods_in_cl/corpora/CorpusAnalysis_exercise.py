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
import sys


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


def detect_file_encoding(file_path):
    """Detect the encoding of a file."""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # Read the first 10,000 bytes
        result = chardet.detect(raw_data)
        return result['encoding']

def load_corpus(file_path):
    """Load the content of the file with fallback encodings."""
    # Detect encoding using chardet
    encoding = detect_file_encoding(file_path)
    print(f"Detected encoding for {file_path}: {encoding}")
    
    try:
        # Try opening with the detected encoding
        with open(file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()
    except (UnicodeDecodeError, LookupError):
        print(f"Failed to decode {file_path} with detected encoding '{encoding}'. Trying fallback encoding 'utf-8'.")
        try:
            # Fallback to utf-8
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            print(f"Failed to decode {file_path} with 'utf-8'. Trying fallback encoding 'latin-1'.")
            try:
                # Fallback to latin-1
                with open(file_path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
            except UnicodeDecodeError as e:
                print(f"Failed to decode {file_path} with all encodings. Error: {e}")
                return None
    return lines


def tokenize_sentences(lines):
    """Tokenize the lines into sentences and words."""
    sentences = []
    tokens = []
    """Add Code"""
    # I need to tokenize the sentences line by line
    # And then I need to tokenize the words in each sentence
    # I think this takes two loops but why can't I do it in one?
    for line in lines:
        sentences.extend(nltk.sent_tokenize(line))
        for word in nltk.word_tokenize(line):
            tokens.extend(nltk.word_tokenize(word))
    
    # print(f"Sentences: {sentences[:10]}")
    print(f"Tokens: {tokens[:10]}")
    return sentences, tokens

def process_tokens(tokens):
    """Process tokens to compute lemmas and POS tags, removing punctuation."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Regular expression for filtering out punctuation
    punctuation_regex = re.compile(f"[{re.escape(string.punctuation)}]")
    
    """Get list of lemmas and part-of-speech tags, do not include punctuation"""
    lemmas = []
    pos_tags = []
    for word in tokens:
        if word.lower() not in stop_words and not punctuation_regex.fullmatch(word):
            lemmas.append(lemmatizer.lemmatize(word))
        if not punctuation_regex.fullmatch(word):
            pos_tags.append(pos_tag([word])[0][1])
    
    return lemmas, pos_tags


def compute_statistics(sentences, tokens, lemmas, pos_tags):
    """Compute various statistics for the corpus."""
    """Add Code: calculations"""
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
    num_sentences = len(sentences)
    num_tokens = len(tokens)
    num_types = len(set(lemmas))
    
    types_tokens_ratio = num_types / num_tokens
    avg_sentence_length = sum(sentence_lengths) / num_sentences
    min_sentence_length = min(sentence_lengths)
    max_sentence_length = max(sentence_lengths)
    
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
        "pos_distribution": Counter(pos_tags)
    }



def visualize_statistics(stats, corpus_name):
    """Visualize distributions and statistics."""
    
    # Plot for Sentence length distribution
    plt.hist(stats['sentence_lengths'], bins=20)
    plt.title(f"Sentence length distribution for {corpus_name}")
    plt.xlabel("Sentence length")
    plt.ylabel("Frequency")
    plt.show()

    # Plot for Top 20 lemmas
    top_20_lemmas = stats['lemma_distribution'].most_common(20)
    if top_20_lemmas:
        lemmas, counts = zip(*top_20_lemmas)
        plt.bar(lemmas, counts)
        plt.title(f"Top 20 lemmas for {corpus_name}")
        plt.xlabel("Lemma")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha='right')
        plt.show()

    # Plot for POS tag distribution
    plt.bar(stats['pos_distribution'].keys(), stats['pos_distribution'].values())
    plt.title(f"POS tag distribution for {corpus_name}")
    plt.xlabel("POS tag")
    plt.ylabel("Frequency")
    plt.show()


def analyze_corpus(file_path, corpus_name):
    """Main function to analyze the corpus."""
    lines = load_corpus(file_path)
    if lines is None:
        return

    sentences, tokens = tokenize_sentences(lines)
    lemmas, pos_tags = process_tokens(tokens)
    stats = compute_statistics(sentences, tokens, lemmas, pos_tags)

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
    """Process and analyze multiple .txt files."""
    for txt_path in txt_files:
        print(f"Processing {txt_path}...")
        corpus_name = os.path.basename(txt_path)
        analyze_corpus(txt_path, corpus_name)


if __name__ == "__main__":
    # Option 1: use command-line arguments (commented out)
    # if len(sys.argv) < 2:
    #     print("Usage: python CorpusAnalysis_exercise.py <path_to_txt_file1> <path_to_txt_file2> ...")
    #     sys.exit(1)
    # txt_files = sys.argv[1:]

    # Option 2: automatically load all .txt files from the local corpus_files directory
    corpus_dir = os.path.join(os.path.dirname(__file__), "corpus_files")
    txt_files = [
        os.path.join(corpus_dir, fname)
        for fname in os.listdir(corpus_dir)
        if fname.lower().endswith(".txt")
    ]
    process_corpora(txt_files)
