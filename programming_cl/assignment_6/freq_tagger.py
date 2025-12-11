
def most_common(freq_dict: dict[str, int]) -> str:
    # If dictionary is empty, return unknown
    if not freq_dict:
        return "UNK"
    
    # max() iterates over keys, compares by their values (via lambda), returns the key with highest value
    return max(freq_dict, key=lambda k: freq_dict.get(k, 0))        

def read_conllu(file_path) -> list[tuple[str, str]]:
    conllu = open(file_path, 'r', encoding='utf-8')
    word_pos_tag_list = []
    print ("Reading conllu file...")
    for line in conllu:
        split_text = line.split()
        if len(split_text) < 4:
            continue
        tuplized = (split_text[1], split_text[3])
        word_pos_tag_list.append(tuplized)

    return word_pos_tag_list

def train_and_tag(train_file, test_words) -> list[str]:
    word_tag_tuples = read_conllu(train_file)
    # Iterate over the word_tag_tuples and compare to Test-words
    # Sentence so that you can tag all of the words in test_words
    # and return a list of POS tags for test_words
    # 1st find out if the word exists in the training data
    # 2nd if it exists, find the most common tag for that word
    # 3rd if it does not exist, return "UNK" tag for that word
    final_tag_list = []
    for word in test_words:
        if word in dict(word_tag_tuples):
            # Find all tags for that word
            tags = [tag for w, tag in word_tag_tuples if w.lower() == word.lower()]
            # Create frequency dictionary
            freq_dict = {}
            for tag in tags:
                if tag in freq_dict:
                    freq_dict[tag] += 1
                else:
                    freq_dict[tag] = 1
            # Get most common tag
            most_common_tag = most_common(freq_dict)
            final_tag_list.append(most_common_tag)
        else:
            final_tag_list.append("UNK")

    return final_tag_list



# xdata_set = read_conllu("programming_cl/assignment_6/small_trainconllu.sec")

most_common_tag_test_1 = most_common({'NOUN': 10, 'VERB': 5, 'DET': 15, 'ADP': 7})
print("Most common tag test 1 (expected 'DET'):", most_common_tag_test_1)

print(train_and_tag("/Users/jholt/development/CL_Python/programming_cl/assignment_6/en_goldconllu.sec", ["The", "citizens", "hate", "weapons", "but", "love", "Christmas"]))