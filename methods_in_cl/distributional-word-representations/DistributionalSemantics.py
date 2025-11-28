from itertools import combinations
import math
import os
import re
import string
from typing import Tuple, Optional, List, Dict, Set

from nltk import pos_tag, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.linalg import svd

from CorpusAnalysis_solution import (
    load_corpus,
)  # solution file from previous corpus analysis assignment


def tokenize_sentences(lines: List[str]) -> Tuple[List[str], List[str]]:
    """
    Tokenize the lines into sentences and words.

    Parameters:
    lines (List[str]): A list of strings where each string is a line of text.

    Returns:
    Tuple[List[List[str]]:
        - A list of sentences which are tokenized into tokens (words)
    """
    tokenized_sentences = []

    for line in lines:
        line_sentences = sent_tokenize(line)
        for sentence in line_sentences:
            tokenized_sentences.append(word_tokenize(sentence))

    return tokenized_sentences


def process_tokens(
    tokenized_sentences: List[List[str]],
) -> Tuple[List[List[str]], Dict[str, int]]:
    """
    Process tokens to compute lemmas and POS tags, removing punctuation and stopwords.

    This function takes a list of tokenized sentences, removes punctuation and stopwords,
    and applies lemmatization to each token. It returns the processed sentences along with
    a count of each unique lemma found in the input.

    Parameters:
    tokenized_sentences (List[List[str]]): A list of tokenized sentences, where each sentence
        is represented as a list of strings (tokens).

    Returns:
    Tuple[List[List[str]], Dict[str, int]]:
        - lemmatized_tokenized_sentences (List[List[str]]): A list of tokenized sentences
          after removing punctuation and stopwords, and applying lemmatization.
        - lemma_counts (Dict[str, int]): A dictionary where each key is a noun lemma and the
          corresponding value is the count of that lemma in the corpus.
    """
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Regular expression for filtering out punctuation
    punctuation_regex = re.compile(f"[{re.escape(string.punctuation)}]+")

    lemmatized_tokenized_sentences = []
    lemma_counts = {}

    for sentence in tokenized_sentences:
        lemmatized_tokenized_sentences.append([])
        for token in sentence:
            # skip punctuation and stop words
            if not punctuation_regex.fullmatch(token) or token.lower() in stop_words:
                # add lemma to new sentence version
                lemma = lemmatizer.lemmatize(token.lower())
                lemmatized_tokenized_sentences[-1].append(lemma)
                # get POS
                pos_is_noun = pos_tag([token])[0][1].startswith("NN")
                # if necessary initialize lemma as key
                if lemma not in lemma_counts.keys() and pos_is_noun:
                    lemma_counts[lemma] = 0
                # add nouns to lemma count
                if pos_is_noun:
                    lemma_counts[lemma] += 1

    return lemmatized_tokenized_sentences, lemma_counts


def create_cooccurrence_matrix(
    W: Set[str], C: List[List[str]], N: int = 5
) -> List[List[int]]:
    """
    Creates a co-occurrence matrix for given words and contexts.

    Parameters:
    W (Set[str]): A set of words for which the co-occurrence matrix is to be created.
    C (List[List[str]]): A corpus as a list of sentences,
        where each sentence is a list of pre-processed tokens.
    N (int, default=5): The window length (symmetric, on both sides)

    Returns:
    M (List[List[int]]): A 2D list representing the co-occurrence matrix M,
                     where the element at [i][j] indicates the number of times
                     word i co-occurs with word j in the provided contexts.
    """
    pass

    word_index = {}
    coocs = {}
    j = 0

    # go over all sentences and the words in them,
    # fill up word_index and coocs
    """ ADD CODE HERE """

    # initialize the co-occurrence matrix M with zeros
    M = []  # placeholder
    """ ADD CODE HERE """

    # now fill the co-occurrence matrix M
    """ ADD CODE HERE """

    return M


def apply_svd(M, n_dims: Optional[int] = None) -> List[List[float]]:
    """
    Apply Singular Value Decomposition (SVD) to the co-occurrence matrix.

    Parameters:
    M (List[List[int]]): The co-occurrence matrix to decompose.
    n_dims (Optional[int], default=None): The number of singular values and vectors to compute.
        If None, all singular values and vectors are computed (n_dims = len(M[0]))

    Returns:
    M_reduced (List[List[float]]): Matrix of singular vectors of size len(M) x n_dims.
    """
    # get singular value decomposition
    M_svd, _, _ = svd(M, full_matrices=False)

    # if n_dims is specified, reduce the number of dimensions
    if n_dims is not None:
        M_svd = M_svd[:, :n_dims]

    # convert to list
    M_svd = M_svd.tolist()

    return M_svd


def calculate_cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Cosine similarity is a measure of similarity between two non-zero vectors
    of an inner product space. It is defined as the cosine of the angle
    between the two vectors, which can be computed using the dot product
    and the magnitudes (norms) of the vectors.

    Parameters:
    a (List[float]): The first vector.
    b (List[float]): The second vector.

    Returns:
    float: The cosine similarity between the two vectors. Returns 0.0 if
           either vector is zero.
    """

    cos_sim = 0.0  # placeholder
    """ ADD CODE HERE """

    return cos_sim


if __name__ == "__main__":
    corpora = [c for c in os.listdir() if c.endswith(".txt")]

    stop_words = set(stopwords.words("english"))

    for corpus in corpora:

        print(f"\n\n------------- {corpus.upper()} -------------\n\n")

        # preprocessing
        print("Processing corpus:", corpus)
        corpus_lines = load_corpus(os.path.join("data/corpora", corpus))
        print("Done loading.")
        tokenized_sentences = tokenize_sentences(corpus_lines)
        print("Done tokenizing.")
        lemmatized_sentences, lemma_counts = process_tokens(tokenized_sentences)
        print("Done preprocessing tokens.")
        # select target words (noun lemmas with at least 50 occurrences)
        target_words = {}  # placeholder
        """ ADD CODE HERE """

        M = create_cooccurrence_matrix(
            W=target_words, S=stop_words, C=lemmatized_sentences, N=5
        )
        print("Done creating co-occurrence matrix.")

        # apply SVD to reduce dimensionality to 100
        """ ADD CODE HERE """

        # calculate all pairwise cosine similarities
        """ ADD CODE HERE """

        # print the 10 most similar and 10 most dissimilar word pairs
        """ ADD CODE HERE """
