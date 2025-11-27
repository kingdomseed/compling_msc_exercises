
# Hint: basic components only; look at how Bubble Sort works

def sort(words: list[str]):
    for current_index in range(len(words)):
        for previous_index in range(len(words) - 1):
            previous_element = words[previous_index]
            next_element = words[current_index]
            if previous_element > next_element:
                words[previous_index], words[current_index] = words[current_index], words[previous_index]
