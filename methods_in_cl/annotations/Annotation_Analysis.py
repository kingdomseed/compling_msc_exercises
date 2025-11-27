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

    # create counts per class and instance the key is the category
    # the values represent how many annotators chose category k
    # for instance j N_jk is a dictionary of lists
    # the key is the category the value is a list of counts
    N_jk = create_N_kj(A=A, K=K)

    # 1) Expected agreement by chance (P_e)
    # Count how many times each category is used overall across all items and raters.
    # Then turn those counts into proportions (p_k). If raters guessed with these
    # overall proportions, the chance agreement is the sum of p_k squared.
    totals_per_k = {k: sum(N_jk[k]) for k in K}
    p_k = {k: totals_per_k[k] / (n * len_J) for k in K}
    P_e = sum(pk ** 2 for pk in p_k.values()) # returns an iterable over the values of the dict (no keys)

    # 2) Observed agreement (P)
    # For each item j: look at how many raters chose each category for that item.
    # If a category was chosen n_jk times, it contributes n_jk * (n_jk - 1)
    # to the "pairwise agreement" on that item. Normalize by n*(n-1), then
    # average over all items.
    P_js = []
    for j in J:
        same_label_pairs = sum(N_jk[k][j] * (N_jk[k][j] - 1) for k in K)
        P_j = same_label_pairs / (n * (n - 1))
        P_js.append(P_j)
    P = sum(P_js) / len_J

    # 3) Fleiss' kappa: how much better than chance the observed agreement is.
    fleiss_kappa = (P - P_e) / (1 - P_e) if (1 - P_e) != 0 else 0.0

    print(f"Fleiss' Kappa: {fleiss_kappa:.3f}")
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
    Freq_i = {a: {k: 0 for k in K} for a in range(len(A))}

    annotator_number = 0
    for annotator_ratings in A:
        for category in annotator_ratings:
            Freq_i[annotator_number][category] += 1
        annotator_number += 1

    # calculate classwise annotation frequency
    Freq = {k: 0 for k in K}
    for annotator_number in Freq_i:
        # Get this annotator's category counts
        annotator_counts = Freq_i[annotator_number]

        for category in annotator_counts:
            # Add this annotator's count for this category to the total
            Freq[category] += annotator_counts[category]

    # Convert Freq_i counts to frequencies
    total_per_annotator = len(A[0])  # How many instances each annotator rated
    total_all_annotators = len(A) * len(A[0])  # Total annotations across all annotators

    # Convert Freq_i (counts) to frequencies by dividing by total per annotator
    for annotator_number in Freq_i:
        for category in Freq_i[annotator_number]:
            Freq_i[annotator_number][category] = Freq_i[annotator_number][category] / total_per_annotator

    # Convert Freq (counts) to frequencies by dividing by total across all annotators
    for category in Freq:
        Freq[category] = Freq[category] / total_all_annotators

    # define all biases
    W = {
        "diff": [],
        "rat": [],
        "com": [],
        "inv": [],
    }

    # ----- YOUR CODE HERE ------
    # STEP 1: Calculate bias weights for each annotator
    # Purpose: Create correction weights to handle annotator biases
    # Loop through each annotator (0, 1, 2, etc.)
    for annotator_number in range(len(A)):
        # Create empty lists to store weights for this annotator
        # Each list will have one weight per category in K
        diff_weights = []
        rat_weights = []
        com_weights = []
        inv_weights = []
        
        # STEP 2: For each category, calculate 4 different bias correction weights
        for category in K:
            # Get how often THIS annotator used this category (as a frequency 0-1)
            freq_i = Freq_i[annotator_number][category]
            # Get how often ALL annotators used this category on average (as a frequency 0-1)
            freq_avg = Freq[category]
            
            # BIAS TYPE 1 - "diff": Difference-based BCR
            # Formula: wi_k = 1 + Freq(k) - Freq_i(k)
            # If annotator uses category less than average, weight > 1 (upweight)
            # If annotator uses category more than average, weight < 1 (downweight)
            diff_weights.append(1 + freq_avg - freq_i)
            
            # BIAS TYPE 2 - "rat": Ratio of average to individual
            # If annotator underuses category, ratio > 1 (upweight)
            # If annotator overuses category, ratio < 1 (downweight)
            if freq_i > 0:
                rat_weights.append(freq_avg / freq_i)
            else:
                rat_weights.append(1.0)  # neutral weight if annotator never used this category
            
            # BIAS TYPE 3 - "com": Complement-based BCR
            # Formula: wi_k = 1 + 1/|K| - Freq_i(k)
            # Gives more weight to categories the annotator rarely uses
            com_weights.append(1 + (1 / len(K)) - freq_i)
            
            # BIAS TYPE 4 - "inv": Inverse of frequency
            # Heavily weights categories the annotator rarely uses
            if freq_i > 0:
                inv_weights.append(1 / freq_i)
            else:
                inv_weights.append(1.0)  # neutral weight if annotator never used this category
        
        # STEP 3: Store this annotator's weights in the W dictionary
        # W["diff"] will be a list of lists: [[annotator0_weights], [annotator1_weights], ...]
        W["diff"].append(diff_weights)
        W["rat"].append(rat_weights)
        W["com"].append(com_weights)
        W["inv"].append(inv_weights)

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
    # STEP 1: Calculate weighted scores for each instance
    # Purpose: Combine all annotators' votes using bias correction weights
    for j in J:
        # Create a dictionary to track scores for each possible category
        # Start all scores at 0.0
        category_scores = {k: 0.0 for k in K}
        
        # STEP 2: Go through each annotator's vote for this instance
        for i in range(len(A)):
            # Get what category this annotator chose for instance j
            # Example: If A[0][5] = 2, annotator 0 chose category 2 for instance 5
            chosen_category = A[i][j]
            
            # STEP 3: Find the weight for this annotator and category
            # K is a set, so convert to sorted list for consistent indexing
            k_list = sorted(list(K))
            category_index = k_list.index(chosen_category)
            
            # Get the bias correction weight for this annotator and category
            # W[i] = all weights for annotator i
            # W[i][category_index] = weight for this specific category
            weight = W[i][category_index]
            
            # STEP 4: Add the weighted vote to this category's score
            # This is like a weighted voting system
            category_scores[chosen_category] += weight
        
        # STEP 5: Store the scores for this instance
        # F_w will be a list of dictionaries, one per instance
        F_w.append(category_scores)

    # get actual labels
    labels = []

    # ----- YOUR CODE HERE ------
    # STEP 6: Convert weighted scores to final labels
    # Purpose: Pick the category with the highest score for each instance
    for j in J:
        # Get the weighted scores for this instance
        # Example: {1: 2.5, 2: 1.8, 3: 0.7} means category 1 has highest score
        scores = F_w[j]
        
        # STEP 7: Find the maximum score value
        max_score = max(scores.values())
        
        # STEP 8: Get all categories that have this maximum score
        # Usually just one, but if there's a tie, include all tied categories
        # Example: If scores = {1: 2.5, 2: 2.5, 3: 1.0}, both 1 and 2 are returned
        best_categories = [k for k in scores if scores[k] == max_score]
        
        # STEP 9: Add the winning category/categories to the final labels list
        # labels[j] will be a list like [1] or [1, 2] if tied
        labels.append(best_categories)
    
    # STEP 10: Return both the weighted scores and the final aggregated labels
    # F_w = detailed scores for analysis
    # labels = final decision for each instance
    return F_w, labels


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
