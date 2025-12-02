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
from scipy.spatial.distance import cosine

from CorpusAnalysis_solution import (
    load_corpus,
)  # solution file from previous corpus analysis assignment


def tokenize_sentences(lines: List[str]) -> List[List[str]]:
    """
    Tokenize the lines into sentences and words.

    Parameters:
    lines (List[str]): A list of strings where each string is a line of text.

    Returns:
    List[List[str]:
        - A list of sentences which are tokenized into tokens (words)
        - A nested list of tokens (words) extracted from the sentences.
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
            if not punctuation_regex.fullmatch(token) and token.lower() not in stop_words:
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
    """ MY CODE HERE """
    for sentence in C:
        # sentence-tokenize line
        for i, word_token in enumerate(sentence):
            if word_token in W:
                if word_token not in coocs:
                    coocs[word_token] = {}
                l = max(0, i - N)
                r = min(len(sentence), i + N + 1) # r is not included
                for k in range(l, r): 
                    context_word = sentence[k]
                    # if the context word is not the target word itself
                    if context_word != word_token: 
                        # make sure the word has a column index
                        if context_word not in word_index:
                            word_index[context_word] = j
                            j += 1
                        # update co-occurrence count
                        if context_word not in coocs[word_token]:
                            coocs[word_token][context_word] = 1
                        else: 
                            coocs[word_token][context_word] += 1
                        # fills the two dictionaries we need for our co-occurrence matrix

    # initialize the co-occurrence matrix M with zeros
    """ MY CODE HERE """
    M = [[0] * len(word_index) for _ in range(len(W))]
    # or
    # M = []
    # for target_word in W:
    #     M.append([])
    #     for context_word in word_index:
    #         M[-1].append(0)
    

    # now fill the co-occurrence matrix M
    # For each target word (row i), look up its context words in coocs,
    # find the column index j for each context word, and set M[i][j] to the count.
    """ MY CODE HERE """
    for i, t_word in enumerate(W):
        for c_word in coocs[t_word].keys():
            j = word_index[c_word]
            M[i][j] = coocs[t_word][c_word]
    #for i, wt in enumerate(W):
    #    for context_word, c in coocs[wt].items():
    #        j = word_index[context_word]
    #        M[i][j] = c

    # print(f"Columns: {len(M)}, Rows: {len(M[0])}")

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

    print(f"SVD applied. New dimensions: {len(M_svd)} x {len(M_svd[0])}")
    # print columns and rows for the 8x8 matrix with their content
    for i, row in enumerate(M_svd):
        if i < 8:
            print(f"Row {i}: {row}")

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

    cos_sim = 1 - cosine(a, b)


    return float(cos_sim)


if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_dir = os.path.join(script_dir, "corpus_files")
    
    # corpora = [c for c in os.listdir(corpus_dir) if c.endswith(".txt")]
    corpora = ["ACL_partial_abstract_corpus.txt"]  # Just test with one corpus

    stop_words = set(stopwords.words("english"))

    for corpus in corpora:

        print(f"\n\n------------- {corpus.upper()} -------------\n\n")

        # preprocessing
        print("Processing corpus:", corpus)
        corpus_lines = load_corpus(os.path.join(corpus_dir, corpus)) or []
        print("Done loading.")
        tokenized_sentences = tokenize_sentences(corpus_lines)
        print("Done tokenizing.")
        lemmatized_sentences, lemma_counts = process_tokens(tokenized_sentences)
        print("Done preprocessing tokens.")
        # select target words (noun lemmas with at least 15 occurrences)
        target_words = {}  # placeholder create a loop and use a lemma count threshold
        """ MY CODE HERE """
        if len(target_words) == 0:
            print("No target words selected yet, selecting now...")
        
        for lemma, count in lemma_counts.items():
            if count >= 50:
                target_words[lemma] = count
        target_words = set(target_words.keys())
        print(f"Selected {len(target_words)} target words.")

        M = create_cooccurrence_matrix(
            W=target_words, C=lemmatized_sentences, N=5
        )
        print(f"Done creating co-occurrence matrix. M: {M[0][:10]}")

        # apply SVD to reduce dimensionality to 100
        M_reduced = apply_svd(M, n_dims=100)

        # calculate all pairwise cosine similarities
        all_sims = []
        for row_i, row_j in combinations(range(len(M_reduced)), 2):
            vec_i = M_reduced[row_i]
            vec_j = M_reduced[row_j]
            cos_sim = calculate_cosine_similarity(vec_i, vec_j)
            all_sims.append((cos_sim, (row_i, row_j)))

        # sort by cosine similarity (tuples sort by first element by default)
        all_sims.sort()

        # 10 most dissimilar (smallest cosine)
        ten_most_dissimilar = all_sims[:10]

        # 10 most similar (largest cosine, reversed for descending order)
        ten_most_similar = all_sims[-10:][::-1]

        # optionally print all pairwise similarities
       # for cos_sim, (row_i, row_j) in all_sims:
            # print(
                # f"Cosine similarity between word {row_i} and word {row_j}: {cos_sim:.5f}"
            # )

        # print the 10 most similar and 10 most dissimilar word pairs
        print("\nTen most similar word pairs:")
        for sim, (i, j) in ten_most_similar:
            print(f"Words {i} and {j} with cosine similarity {sim:.5f}")
        print("\nTen most dissimilar word pairs:")
        for sim, (i, j) in ten_most_dissimilar:
            print(f"Words {i} and {j} with cosine similarity {sim:.5f}")