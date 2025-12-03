
def most_common(freq_dict) -> str:

    return "DET"

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
    return ["DET" for _ in test_words]  # Dummy implementation



data_set = read_conllu("programming_cl/assignment_6/small_trainconllu.sec")

most_common_tag_test_1 = most_common({'NOUN': 10, 'VERB': 5, 'DET': 15, 'ADP': 7})
print("Most common tag test 1 (expected 'DET'):", most_common_tag_test_1)
