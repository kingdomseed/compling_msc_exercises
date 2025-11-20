word = input("Enter a word: ")
first_word = ""
last_word = ""
while True:
    if word.lower() == "done":
        print (first_word, 'comes first in the dictionary')
        print (last_word, 'comes last in the dictionary')
        break
    if(word < first_word) or (first_word == ""):
        first_word = word
    if(word > last_word) or (last_word == ""):
        last_word = word
    word = input("Enter a word: ")