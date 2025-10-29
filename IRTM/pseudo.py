import re
import time

def tokenize(text: str) -> list[str]:
    """Splits the input text into tokens."""
    tokenized_list: list[str] = []
    for token in re.split(r'\W+', text):
        tokenized_list.append(token)

    return tokenized_list

# Load the file once
with open("/Users/jholt/jupyter_nb/IRTM/wiki-en-flower.txt", "r") as file:
    original_text = file.read()

# Test with multiples of the original
multipliers = [1, 2, 4, 8]

print("Testing O(n) complexity:\n")
for mult in multipliers:
    test_text = original_text * mult
    
    start = time.time()
    result = tokenize(test_text)
    end = time.time()
    
    elapsed = end - start
    print(f"Multiplier: {mult:2d}x | Size: {len(test_text):8d} chars | Time: {elapsed:.6f} sec | Tokens: {len(result)}")

