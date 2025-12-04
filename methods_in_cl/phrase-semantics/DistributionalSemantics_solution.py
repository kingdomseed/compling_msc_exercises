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


def tokenize_sentences(lines: List[str]) -> List[List[str]]:
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
        - lemma_counts (Dict[str, int]): A dictionary where each key is a lemma and the
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
            if (
                not punctuation_regex.fullmatch(token)
                and token.lower() not in stop_words
            ):
                # add lemma to new sentence version
                lemma = lemmatizer.lemmatize(token.lower())
                lemmatized_tokenized_sentences[-1].append(lemma)
                # get POS
                pos_is_noun = pos_tag([token])[0][1].startswith("NN")
                # if necessary initialize lemma as key
                if lemma not in lemma_counts.keys() and pos_is_noun:
                    lemma_counts[lemma] = 0
                # increment lemma count
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

    # go over all sentences and the words in them
    for sentence in C:
        for i, word in enumerate(sentence):
            # if the current word is a target word
            if word in W:
                t_word = word
                # if this is the first time encounterig the target word, initialize its co-occurrence dict
                if t_word not in coocs.keys():
                    coocs[t_word] = {}
                # adapt window size to sentence boundaries
                l = max(0, i - N)
                r = min(len(sentence), i + N + 1)
                # go over all context words in the window
                for k in range(l, r):
                    c_word = sentence[k]
                    # if the context word is not the target word
                    if k != i:
                        # make sure the word has a column index
                        if c_word not in word_index.keys():
                            word_index[c_word] = j
                            j += 1
                        # update co-occurrence count
                        if c_word not in coocs[t_word].keys():
                            coocs[t_word][c_word] = 1
                        else:
                            coocs[t_word][c_word] += 1

    # initialize the co-occurrence matrix M with zeros
    M = []
    for t_word in W:
        M.append([])
        for c_word in word_index.keys():
            M[-1].append(0)

    # now fill the co-occurrence matrix M
    for i, t_word in enumerate(W):
        for c_word in coocs[t_word].keys():
            j = word_index[c_word]
            M[i][j] = coocs[t_word][c_word]

    return M


def apply_svd(M: List[List[int]], n_dims: Optional[int] = None) -> List[List[float]]:
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
    # dot product between a and b
    dot = 0.0
    for a_i, b_i in zip(a, b):
        dot += a_i * b_i

    # norm of a
    norm_a = 0.0
    for a_i in a:
        norm_a += a_i * a_i
    norm_a = math.sqrt(norm_a)

    # norm of b
    norm_b = 0.0
    for b_i in b:
        norm_b += b_i * b_i
    norm_b = math.sqrt(norm_b)

    # avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0

    cos_sim = dot / (norm_a * norm_b)

    return cos_sim


if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_dir = os.path.join(script_dir, "corpus_files")
    
    corpora = [os.path.join(corpus_dir, "ACL_partial_abstract_corpus.txt")]

    stop_words = set(stopwords.words("english"))

    for corpus in corpora:
        print(f"\n\n------------- {corpus.upper()} -------------\n\n")
        print("Processing corpus:", corpus)
        corpus_lines = load_corpus(corpus)
        if corpus_lines is None:
            print(f"Skipping {corpus}: no data returned from load_corpus.")
            continue
        print("Done loading.")
        tokenized_sentences = tokenize_sentences(corpus_lines)
        print("Done tokenizing.")
        lemmatized_sentences, lemma_counts = process_tokens(tokenized_sentences)
        print("Done preprocessing tokens.")
        target_words = {w for w, w_count in lemma_counts.items() if w_count >= 50}
        M = create_cooccurrence_matrix(W=target_words, C=lemmatized_sentences, N=5)
        print(f"Co-occurrence matrix size: {len(M)} x {len(M[0])}")
        print("Done creating co-occurrence matrix.")

        M_reduced = apply_svd(M, n_dims=100)
        print("Done reducing dimensions.")

        pairs = combinations(range(len(M_reduced)), 2)
        cos_sims = []
        for a, b in pairs:
            if a != b:
                cos_sim = calculate_cosine_similarity(M_reduced[a], M_reduced[b])
                cos_sims.append((list(target_words)[a], list(target_words)[b], cos_sim))
        print("Calculated cosine similarities.")
        cos_sims = sorted(cos_sims, key=lambda x: x[2], reverse=True)

        print("\nTop 10 most similar word pairs based on cosine similarity:\n")
        for i in range(10):
            print(
                f"Cosine similarity between '{cos_sims[i][0]}' and '{cos_sims[i][1]}': {cos_sims[i][2]:.3f}"
            )
        print("\nTop 10 most dissimilar word pairs based on cosine similarity:\n")
        for i in range(1, 11):
            print(
                f"Cosine similarity between '{cos_sims[-i][0]}' and '{cos_sims[-i][1]}': {cos_sims[-i][2]:.3f}"
            )
