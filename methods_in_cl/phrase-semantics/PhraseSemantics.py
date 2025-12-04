import itertools
from typing import Dict, List, Tuple

from DistributionalSemantics_solution import calculate_cosine_similarity


# - - - - - - - - - - - load embeddings - - - - - - - - - - -


def load_embeddings_from_file(emb_file: str) -> Dict[str, List[float]]:
    """
    Load word embeddings from a specified file.

    The function reads a file containing word embeddings, where each line
    consists of a word followed by its corresponding vector representation.
    The word and vector are separated by whitespace.

    Args:
        emb_file (str): The path to the file containing the embeddings.
                        Defaults to 'w2v_top2kGN.txt'.

    Returns:
        Dict[str, List[float]]: A dictionary mapping words to their
                                 corresponding (numeric!) vector representations.
    """

    embeddings = {}

    with open(emb_file, "r") as f:
        # each line is string of word followed by vector values
        for line in f.readlines():
            # split at whitespace
            parts = line.strip().split()
            # first part is the word
            word = parts[0]
            # convert each vector value from string to float
            vector = [float(vi) for vi in parts[1:]]
            # save embedding to dict
            embeddings[word] = vector

    return embeddings


# - - - - - - - - - - - vector arithmetics - - - - - - - - - - -


def compute_analogy_vector(
    embeddings: Dict[str, List[float]],
    base_term: str,
    subtract_term: str,
    add_term: str,
) -> List[float]:
    """
    Find the word that completes the analogy based on the provided embeddings.

    This function takes a list of word embeddings and computes a new vector
    by performing vector arithmetic of the following form:

        base term - subtract term +  add term

    It then finds the word whose embedding is most similar to the resulting
    vector, excluding the terms used in the analogy.

    Args:
        embeddings (Dict[str, List[float]]): A dictionary mapping words to their
                                corresponding vector representations.
        base_term (str): The base term.
        subtract_term (str): The term to be subtracted.
        add_term (str): The term to be added.

    Returns:
        List[float]: The result vector obtained from the vector arithmetic.
    """
    # Retrieve embeddings for the specified terms

    # --- MY CODE HERE ---
    base_embedding = embeddings[base_term]
    subtract_embedding = embeddings[subtract_term]
    add_embedding = embeddings[add_term]

    result_vector = []

    # for each dimension in the vector, compute the analogy operation

    # --- MY CODE HERE ---
    for i in range(len(base_embedding)):
        result_vector.append(base_embedding[i] - subtract_embedding[i] + add_embedding[i])

    return result_vector


def compute_similarity_ranking(
    v: List[float], embeddings: Dict[str, List[float]]
) -> List[Tuple[str, float]]:
    """
    Compute a similarity ranking of all words in the embeddings based on their
    cosine similarity to the provided vector.

    Args:
        v (List[float]): The vector to compare against.
        embeddings (Dict[str, List[float]]): A dictionary where keys are words and
                                              values are their corresponding embeddings.
    Returns:
        List[Tuple[str, float]]: A list of tuples where each tuple contains a word
                                    and its cosine similarity to the vector `v`,
                                    sorted in non-ascending order of similarity.
    """
    similarity_ranking = []

    # --- YOUR CODE HERE ---
    for word, embedding in embeddings.items():
        similarity = calculate_cosine_similarity(v, embedding)
        similarity_ranking.append((word, similarity))

    # sort the ranking list by similarity in non-ascending order
    similarity_ranking.sort(key=lambda x: x[1], reverse=True)

    return similarity_ranking


def report_analogy_results(
    analogy_words: List[Tuple[str, str, str]], embeddings: Dict[str, List[float]]
) -> None:
    """
    Reports the results of analogy computations based on provided analogy words and embeddings.

    This function takes a list of analogy words, where each analogy is represented as a tuple
    containing a base term, a term to subtract, and a term to add. It computes the resulting
    vector for each analogy and finds the most similar term from the embeddings. The results
    are printed in a human-readable format.

    Args:
        analogy_words (List[Tuple[str, str, str]]): A list of tuples, each containing three strings
            representing the base term, the term to subtract, and the term to add.
        embeddings (Dict[str, List[float]]): A dictionary mapping terms to their corresponding
            vector embeddings.

    Returns:
        None: This function does not return any value; it prints the results directly.
    """

    # --- YOUR CODE HERE ---
    for base, subtract, add in analogy_words:
        result = compute_analogy_vector(embeddings, base, subtract, add)
        most_similar = compute_similarity_ranking(result, embeddings)
        if add or subtract or base == most_similar[0][0]:
            print(f"{base} - {subtract} + {add} => {most_similar[1][0]}")
        else:
            print(f"{base} - {subtract} + {add} => {most_similar[0][0]}")


# - - - - - - - - - - - phrase embeddings - - - - - - - - - - -


def check_target_words(
    target_words: Dict[str, List[str]], embeddings: Dict[str, List[float]]
) -> Dict[str, List[str]]:
    """
    Checks the target words against the provided embeddings and removes any words
    that do not have corresponding embeddings.

    Args:
        target_words (Dict[str, List[str]]): A dictionary where keys are categories
                                              and values are lists of target words.
        embeddings (Dict[str, List[float]]): A dictionary where keys are words and
                                              values are their corresponding embeddings.

    Returns:
        Dict[str, List[str]]: The updated dictionary with target words that have no
                               embedding removed.
    """

    for words in target_words.values():
        for word in words:
            # if a word has an embedding, remove it from the dict
            if word not in embeddings.keys():
                print(f"Missing embedding for word '{word}', removing from list.")
                words.remove(word)

    return target_words


def compose_phrase(
    embeddings: Dict[str, List[float]], word_1: str, word_2: str, comp_method: str
) -> List[float]:
    """
    Compose a phrase embedding from two word embeddings using the specified method.

    Args:
        embeddings (Dict[str, List[float]]): A dictionary where keys are words and
                                              values are their corresponding embeddings.
        word_1 (str): The first constituent.
        word_2 (str): The second constituent.
        comp_method (str): The composition method to use ('addition', 'multiplication', or 'concatenation').

    Returns:
        List[float]: The composed phrase embedding.
    """
    # get embeddings for both words
    w1_embedding = embeddings[word_1]
    w2_embedding = embeddings[word_2]

    p_embedding = []

    # --- MY CODE HERE ---

    if comp_method == "addition":
        for vector_1, vector_2 in zip(w1_embedding, w2_embedding):
            p_embedding.append(vector_1 + vector_2)
    elif comp_method == "multiplication":
        for vector_1, vector_2 in zip(w1_embedding, w2_embedding):
            p_embedding.append(vector_1 * vector_2)
    elif comp_method == "concatenation":
        p_embedding = w1_embedding + w2_embedding
    else:
        raise ValueError(f"Unknown composition method: {comp_method}")


    return p_embedding


def calculate_phrase_embeddings(
    target_phrases: List[Tuple[str, str]], embeddings: Dict[str, List[float]]
) -> Dict[str, Dict[str, List[float]]]:
    """
    Calculate phrase embeddings for a list of target phrases using different composition methods.

    Args:
        target_phrases (List[Tuple[str, str]]): A list of tuples where each tuple
                                                 contains two constituents forming a phrase.
        embeddings (Dict[str, List[float]]): A dictionary where keys are words and
                                              values are their corresponding embeddings.
    Returns:

        Dict[str, Dict[str, List[float]]]: A nested dictionary where the outer keys
                                             are phrases, the inner keys are composition
                                             methods, and the values are the corresponding
                                             phrase embeddings.
    """
    # --- MY CODE HERE ---
    phrase_embeddings = {}
    for phrase in target_phrases:
        addition = compose_phrase(embeddings, phrase[0], phrase[1], "addition")
        multiplication = compose_phrase(embeddings, phrase[0], phrase[1], "multiplication")
        concatenation = compose_phrase(embeddings, phrase[0], phrase[1], "concatenation")
        phrase_embeddings[phrase] = {
            "addition": addition,
            "multiplication": multiplication,
            "concatenation": concatenation,
        }
    return phrase_embeddings


def report_semantic_similarity(
    embeddings: Dict[str, List[float]],
    phrase_embeddings: Dict[str, Dict[str, List[float]]],
) -> None:
    """
    Reports the semantic similarity between phrases and their constituent words
    using various composition methods.

    Args:
        embeddings (Dict[str, List[float]]): A dictionary where keys are words
            and values are their corresponding embeddings (list of floats).
        phrase_embeddings (Dict[str, Dict[str, List[float]]]): A dictionary where
            keys are phrases (strings) and values are dictionaries. Each inner
            dictionary has composition method names as keys and their corresponding
            embeddings (list of floats) as values.

    Returns:
        None: This function prints the similarity scores to the console.
    """

    # --- YOUR CODE HERE ---
    for phrase, method_dict in phrase_embeddings.items():
        for method, phrase_vec in method_dict.items():
            similarity_first_word = calculate_cosine_similarity(phrase_vec, embeddings[phrase[0]])
            similarity_second_word = calculate_cosine_similarity(phrase_vec, embeddings[phrase[1]])
            print(f"{phrase} - {method} => {similarity_first_word}, {similarity_second_word}")



def report_nearest_neighbors(
    embeddings: Dict[str, List[float]],
    phrase_embeddings: Dict[str, Dict[str, List[float]]],
) -> None:
    """
    Reports the nearest neighbors for given phrases based on their embeddings.

    This function computes the similarity ranking of phrase embeddings against a set of word embeddings
    and prints the top 5 nearest neighbors for each phrase using different composition methods.

    Args:
        embeddings (Dict[str, List[float]]): A dictionary mapping words to their embeddings.
        phrase_embeddings (Dict[str, Dict[str, List[float]]]): A dictionary mapping phrases to their
            composition methods and corresponding embeddings.

    Returns:
        None: This function does not return any value. It prints the results directly.
    """

    # --- YOUR CODE HERE ---
    for phrase, method_dict in phrase_embeddings.items():
        for method, phrase_vec in method_dict.items():
            ranking = compute_similarity_ranking(phrase_vec, embeddings)
            print(f"{phrase} - {method} => {ranking[:5]}")

if __name__ == "__main__":
    # define analogies for vector arithmetics
    analogy_words = [
        ("man", "woman", "he"),
        ("her", "him", "she"),
        ("father", "mother", "son"),
        ("students", "student", "school"),
        ("buying", "buy", "pay"),
        ("older", "old", "good"),
        ("fully", "full", "final"),
    ]

    # define target words for phrase semantics
    target_words = {
        "N": ["car", "child", "world", "business"],
        "J": ["big", "small", "red", "bad"],
        "V": ["drive", "see"],
    }

    for embedding_file in ["w2v_neg300_top2kGN.txt", "GloVe_42B_300_top2kGN.txt"]:

        print("-" * 100)
        print(f"\n--- Using embeddings from file: {embedding_file} ---\n")
        print("-" * 100)

        embeddings = load_embeddings_from_file(embedding_file)

        # ---------- Vector Arithmetics ----------
        report_analogy_results(analogy_words=analogy_words, embeddings=embeddings)

        # ---------- Phrase Semantics ----------

        # create phrases
        target_words = check_target_words(target_words, embeddings)
        target_phrases = list(itertools.product(target_words["J"], target_words["N"]))
        target_phrases += list(itertools.product(target_words["V"], target_words["N"]))

        phrase_embeddings = calculate_phrase_embeddings(
            target_phrases=target_phrases, embeddings=embeddings
        )

        # semantic similarity to constituents
        report_semantic_similarity(
            embeddings=embeddings, phrase_embeddings=phrase_embeddings
        )
        # nearest neighbors
        report_nearest_neighbors(
            embeddings=embeddings, phrase_embeddings=phrase_embeddings
        )
