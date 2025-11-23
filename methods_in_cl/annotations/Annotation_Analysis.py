from itertools import combinations  # to generate all annotator pairs
from typing import List, Set, Any, Dict  # typing hints


def calculate_cohens_kappa(A_x: List[int], A_y: List[int], K: Set[Any]) -> float:
    """
    Calculate Cohen's Kappa statistic for two raters.

    Cohen's Kappa is a measure of inter-rater agreement for categorical items.
    It takes into account the agreement occurring by chance.

    Parameters:
    A_x (List[int]): A list of integer ratings from the first rater.
    M_y (List[int]): A list of integer ratings from the second rater.
    K (Set[Any]): A set of all possible category values.

    Returns:
    float: The Cohen's Kappa statistic, which ranges from -1 (complete disagreement)
        to 1 (complete agreement), with 0 indicating random agreement.
    """
    # ensure all annotators rated the same (number of) instances
    assert len(A_x) == len(A_y), "Both annotators must have rated the same instances."
    assert not any(A_xj is None for A_xj in A_x), "Annotator x has missing ratings."
    assert not any(A_yj is None for A_yj in A_y), "Annotator y has missing ratings."

    J = list(range(len(A_x)))  # list of instances (indexes)
    len_J = len(J)  # number of instances

    # calculate agreement A_xy
    A_xy = sum([A_x[j] == A_y[j] for j in J]) / len_J

    # Percentage of times Annotator X choose K
    # List version requires index fiddling with k-1--not extensible easily:
    # P_x = [(sum([A_x[j] == k for j in J]) / len_J) for k in K]

    P_x = {k: sum([A_x[j] == k for j in J]) / len_J for k in K}

    # Percentage of times Annotator Y choose K
    P_y = {k: sum([A_y[j] == k for j in J]) / len_J for k in K}

    # calculate chance agreement E_xy -- when P_x and P_y are lists
    # you need something like this:
    # E_xy = sum([P_x[k-1] * P_y[k-1] for k in K])
    # Otherwise with a dictionary this works:
    E_xy = sum([P_x[k] * P_y[k] for k in K])

    # print results
    print(f"Agreement: {A_xy:.3f}")
    print(f"Expected agreement: {E_xy:.3f}")

    # calculate Cohen's kappa
    cohens_kappa = (A_xy - E_xy) / (1 - E_xy)  # placeholder

    print(f"Cohen's Kappa: {cohens_kappa:.3f}")
    return cohens_kappa


# note that this function can be useful, but there are also many valid solutions without it
def create_N_kj(A: List[List[int]], K: Set[Any]) -> Dict[Any, List[int]]:
    """
    Create a dictionary that counts number of ratings per instance j for each category k.

    Parameters:
    A (List[List[int]]): The annotation matrix where rows represent annotators and columns represent instances.
    K (Set[Any]): The set of categories

    Returns:
    Dict[Any, List[int]]: A dictionary where each key is category from K and the value is a list of counts,
          where each count corresponds to the number of annotations of that category for each instance in A.
    """
    N_kj = {}

    J = list(range(len(A[0])))

    for k in K:
        # add one dictionary entry for each class so you can call them by class value, not index
        N_kj[k] = []
        # create one value for each instance j
        for j in J:
            n_kj = 0
            # count how many annotators choose A
            for i in range(len(A)):
                if A[i][j] == k:
                    n_kj += 1
            N_kj[k].append(n_kj)

    return N_kj


def calculate_fleiss_kappa(A: List[List[int]], K: Set[Any]) -> float:
    """
    Calculate Fleiss' Kappato assess the reliability of
    agreement between two or more annotators.

    Parameters:
    A (List[List[int]]): A matrix of integer ratings from N raters (rows)
        and |J| instances (columns).
    K (Set[Any]): A set of all possible category values.

    Returns:
    float: The Fleiss' Kappa statistic, which ranges from -1 (complete disagreement)
        to 1 (complete agreement), with 0 indicating random agreement.
    """
    # ensure all annotators rated the same (number of) instances
    assert (
        len(
            set(
                [
                    len([a_ij for a_ij in A_i if a_ij])
                    for A_i in list(map(list, zip(*A)))
                ]
            )
        )
        == 1
    ), "All instances must have the same number of annotations"
    for i, A_i in enumerate(A):
        assert not any(
            A_ij is None for A_ij in A_i
        ), "Annotator {i} has missing ratings."

    J = list(range(len(A[0])))  # list of instances
    len_J = len(J)  # number of instances
    n = len(A)  # number of annotators

    # create counts per class and instance
    N_jk = create_N_kj(A=A, K=K)

    # calculate agreement score
    P = None  # placeholder

    # ----- YOUR CODE HERE ------

    # calculate the chance agreement
    P_e = None  # placeholder

    # ----- YOUR CODE HERE ------

    # calculate Fleiss' kappa
    fleiss_kappa = None  # placeholder

    # ----- YOUR CODE HERE ------

    return fleiss_kappa


def get_biases(A: List[List[int]], K: Set[Any]) -> Dict[Any, List[List[float]]]:
    """
    Calculate annotation biases for a set of annotators and classes.

    Parameters:
    A (List[List[int]]): A matrix of integer ratings from N raters (rows)
        and |J| instances (columns).
    K (Set[Any]): A set of all possible category values.

    Returns:
    Dict[Any, List[List[float]]]: A dictionary containing lists of bias metrics for each class.
          The biases to choose from are:
          - "diff": Differences between the average frequency and the individual frequency.
          - "rat": Ratios of the average frequency to the individual frequency.
          - "com": Complements of the individual frequency.
          - "inv": Inverses of the individual frequency.
    """
    # calculate classwise annotation frequency per annotator
    Freq_i = {}
    # calculate classwise annotation frequency
    Freq = {}

    # ----- YOUR CODE HERE ------

    # define all biases
    W = {
        "diff": [],
        "rat": [],
        "com": [],
        "inv": [],
    }

    # ----- YOUR CODE HERE ------

    return W


def aggregate_annotations(A: List[List[int]], K: Set[Any], W: List[List[float]]):
    """
    Aggregate annotations using different bias correction methods.

    Parameters:
    A (List[List[int]]): A matrix of integer ratings from N raters (rows)
        and |J| instances (columns).
    K (Set[Any]): A set of all possible category values.
    W: The nested list of bias correction weigths

    Returns:
    List[List[Any]]: A list of lists containing the aggregated label(s) for each instance.
        Each sublist contains one label or - in case of ties - multiple labels.
    """

    J = list(range(len(A[0])))

    F_w = []

    # ----- YOUR CODE HERE ------

    # get actual labels
    labels = []

    # ----- YOUR CODE HERE ------


if __name__ == "__main__":

    A = [
        [
            1,
            1,
            1,
            3,
            3,
            2,
            3,
            3,
            1,
            2,
            2,
            2,
            1,
            3,
            1,
        ],
        [
            1,
            2,
            1,
            2,
            3,
            2,
            3,
            3,
            1,
            1,
            2,
            2,
            1,
            3,
            1,
        ],
        [
            2,
            1,
            1,
            3,
            3,
            2,
            3,
            3,
            2,
            2,
            2,
            2,
            2,
            3,
            1,
        ],
    ]

    # Cohen's kappa
    pairs = combinations(range(len(A)), 2)
    for n_x, n_y in pairs:
        cohens_kappa = calculate_cohens_kappa(A[n_x], A[n_y], K={1, 2, 3})
        if cohens_kappa is not None:
            print(
                f"Agreement between annotator {n_x} and annotator {n_y}: {cohens_kappa:.3f}\n"
            )

    # Fleiss' kappa
    fleiss_kappa = calculate_fleiss_kappa(A, K={1, 2, 3})
    if fleiss_kappa is not None:
        print(f"Fleiss' kappa for all annotators: {fleiss_kappa:.3f}")

    # Annotation aggregation
    W = get_biases(A=A, K={1, 2, 3})

    for bcr in W.keys():
        label_indices, labels = aggregate_annotations(A=A, K={1, 2, 3}, W=W["rat"])
        print(f"Aggregated labels with {bcr} bias correction: {labels}")
