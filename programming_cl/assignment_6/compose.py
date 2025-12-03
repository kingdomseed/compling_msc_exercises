def word2dict(word):
    wordasdict ={}
    for letter in word:
        if letter in wordasdict:
            wordasdict[letter] += 1
        else:
            wordasdict[letter] = 1
    return wordasdict
def can_be_composed(word1, word2):
    word1asdict = word2dict(word1)
    word2asdict = word2dict(word2)
    for letter in word1asdict:
        # if there are not enough of a letter in word 2 (in other words,
        # if word1 has more of a letter than word2) return False
        # we need a 0 as a default value in case the letter is not in word2
        # in order to return a valid comparison between two ints
        # otherwise we would be comparing None to an int
        if word1asdict[letter] > word2asdict.get(letter, 0):
            return False

    return True

# returns number of occurrences of each letter
# in a given word along with that letter
# is the key, occurrences is the value

# Test cases
# print(can_be_composed("python", "pantyhose"))
# print(can_be_composed("python", "ponytail"))
# print(can_be_composed("code", "docent"))
# print(can_be_composed("messy", "reassembly"))
# print(can_be_composed("computational", "alphanumeric"))
# print(can_be_composed("linguistics", "criminologists"))