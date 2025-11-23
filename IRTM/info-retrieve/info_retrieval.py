
# 1)
# def index(filename) takes the path to the file
# as an argument and puts all documents into a
# non-positional inverted index.
# Index should consist of a dictionary and postings lists.
# Each entry of the dictionary should contain three values:
# the normalized term, the size of the postings list, and
# the pointer to the postings list.
# The data structure should be prepared to store the postings
# list separately from the dictionary, therefore do not just put a
# List data structure as a value into a Map/Tree/Dictionary.
# Put the postings lists into a data structure that you could
# store elsewhere.
# For instance, use a separate id-to-value mapping for that.
# Normalize tokens and terms according to your discretion
# You can filter Tweets if you think that are not relevant
# Describe your decisions in your submission
# The postings list itself consists of postings which contain
# each a document id
# and a pointer to the next postings.
# For the dictionary, you can use hashing methods like a
# dictionary in Python or a tree structure. For Postings Lists
# you can either implement the lists from scratch or use existing
# data structures like lists in Python.
def index(filename):
    dictionary = {}
    postings_list = []

    # Read CSV (columns: date, tweet_id, username, display_name, text)
    # For each tweet:
    #   - Extract and normalize tokens
    #   - For each unique term in tweet:
    #       - If term not in dictionary:
    #           - Create new dictionary entry
    #           - Create new postings list
    #       - Add tweet_id to term's postings list

    # Sort all postings lists by doc_id


# 2)
# def query(term) where term is one term as a string. It should
# return a postings list for that term

# 3)
# query(term1, term2) where you assume that both terms are
# connected with logical AND. Implement the intersection
# algorithm discussed in the lecture for two postings lists:
# INTERSECT(p1, p2)
# 1  answer ← ⟨⟩
# 2  while p1 ≠ NIL and p2 ≠ NIL
# 3    do if docID(p1) = docID(p2)
# 4      then ADD(answer, docID(p1))
# 5        p1 ← next(p1)
# 6        p2 ← next(p2)
# 7      else if docID(p1) < docID(p2)
# 8        then p1 ← next(p1)
# 9      else p2 ← next(p2)
# 10 return answer
# Do not access the lists array-style. Instead use an
# iterator (in Python: listiter = iter(listname); next(listiter).

# 4)
# Query your index for the information need "show me tweets of people
# who talk about the side effects of malaria vaccines". Provide us
# with your query and a subset of results. The results should be minimally
# represented by the Tweet-ID and optionally also the Tweet text.