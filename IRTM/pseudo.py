import re

def tokenize(text: str) -> list[str]:
    """Splits the input text into tokens."""
    tokenized_list = []
    for token in re.split(r'\W+', text):
        tokenized_list.append(token)

    return tokenized_list

text_test = "O’Really isn’t in S."
print(tokenize(text_test))  # Example usage

with open("/Users/jholt/jupyter_nb/IRTM/wiki-en-flower.txt", "r") as file:
    text = file.read()
result = tokenize(text)
print(result[:100])  # Print first 100 tokens from the file

