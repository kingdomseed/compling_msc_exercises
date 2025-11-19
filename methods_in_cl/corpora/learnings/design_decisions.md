# Token Processing Design Decisions

This note summarizes key design choices when implementing a token-processing step like `process_tokens` (tokenization → stopword filtering → lemmatization → POS tagging).

Use it as a checklist when you have to write this logic yourself.

---

## 1. What goes into `tokens`?

**Question:** Are your `tokens` raw strings, or tuples (e.g., `(word, tag)`)?

- If **strings** (e.g., `"cats"`):
  - You still need to run a POS tagger (e.g., `pos_tag(tokens)`).
  - A loop like `for word in tokens:` treats `word` as a single string.
- If **tuples** `(word, tag)`: 
  - Some earlier step already did POS tagging.
  - You then unpack in the loop: `for word, tag in tagged_tokens:`.

**Implication:** Be clear on the type of each element in `tokens` before you start writing loops.

---

## 2. Stopwords: include or exclude (and where)?

**Decisions:**

- **A. Exclude stopwords from everything**
  - Skip them before lemmatization and POS tagging.
  - `lemmas` and `pos_tags` only contain content words.
  - POS distribution reflects only content-word categories.

- **B. Exclude stopwords from lemmas, but keep POS tags**
  - Lemmas list focuses on content words.
  - POS tags list includes function words (e.g., `DT`, `IN`, `PRP`).
  - POS distribution reflects overall syntactic patterns.

- **C. Include stopwords everywhere**
  - Lemmas and POS tags both include stopwords.
  - Useful if you care about frequency of function words (e.g., stylistic analysis).

**Checklist:**
- Do my research questions care about function words?
- Do I want lemma-based stats (types, TTR) to ignore stopwords?

---

## 3. Casing: when to lowercase

**Library behavior:**

- `stopwords.words('english')` are lowercase.
- `WordNetLemmatizer` does **not** lowercase for you.
- `pos_tag` can use capitalization (e.g., `NNP` vs `NN`).

**Common pattern:**

```python
lower = word.lower()
if lower not in stop_words:
    lemmas.append(lemmatizer.lemmatize(lower))
# For POS tagging, you may keep original `word` to preserve case
pos_tags.append(pos_tag([word])[0][1])
```

**Decisions:**

- Lemmas / vocabulary:
  - **Case-insensitive**: treat `"Dog"` and `"dog"` as the same type → lowercase before lemmatizing.
  - **Case-sensitive**: keep them separate → do not lowercase.

- Stopword filtering:
  - If you want `"The"` to match `"the"` in the stopword list, lowercase before checking.

---

## 4. POS tagging: once vs per-token

### Option 1: Tag whole sequence once (recommended)

```python
from nltk import pos_tag

 tagged_tokens = pos_tag(tokens)  # [ (word, tag), ... ]
for word, tag in tagged_tokens:
    ...
```

- More efficient (one call, not many).
- Tagger can use context (neighbors) to choose tags.

### Option 2: Tag each token in isolation

```python
for word in tokens:
    tag = pos_tag([word])[0][1]
    ...
```

- Simpler to understand locally.
- Slower and less context-aware.

**Checklist:**
- Is simplicity more important than efficiency/context here?
- For small exercises, either is acceptable; for real corpora, prefer tagging once.

---

## 5. Lemmatization: noun-only vs POS-aware

**Default behavior:**

```python
lemmatizer.lemmatize(word)  # assumes noun if no POS provided
```

- Good for plural nouns → singular (e.g., `"cats"` → `"cat"`).
- Weak for verbs / adjectives (e.g., `"running"` stays `"running"`).

**POS-aware lemmatization:**

1. Get POS tags (e.g., using `pos_tag`).
2. Map tagset (e.g., Penn Treebank) to WordNet POS:
   - tags starting with `N` → `'n'`
   - tags starting with `V` → `'v'`
   - tags starting with `J` → `'a'`
   - tags starting with `R` → `'r'`
3. Call:

```python
lemmatizer.lemmatize(word, wn_pos)
```

**Trade-off:**

- **Noun-only:** fewer lines, easier to write; less linguistically precise.
- **POS-aware:** more code and decisions; better normalization across verbs/adjectives.

---

## 6. Punctuation handling

**Key question:** How do you detect punctuation tokens?

- Using a regex over `string.punctuation`:

```python
punctuation_regex = re.compile(f"[{re.escape(string.punctuation)}]")
```

- Do you want to:
  - Skip tokens that are **exactly** punctuation: `fullmatch(token)`.
  - Skip tokens that **contain** punctuation anywhere: `search` or `match`.

**Checklist:**
- Am I accidentally dropping things like `"don't"` or `"U.S."` if I use a too-broad regex?

---

## 7. Alignment between `lemmas` and `pos_tags`

If you return **two separate lists**, think about alignment:

- Option A: Same length, same order, 1 lemma ↔ 1 POS tag.
- Option B: Lemmas skip some tokens (e.g., stopwords), POS tags do not.

**If A:**
- Keep all filtering rules consistent across both lists.

**If B:**
- Be clear in comments and later code that lengths differ.
- Document what each list represents.

---

## 8. Minimal checklist when implementing `process_tokens`

1. **What is in `tokens`?** Strings or tuples?
2. **Stopwords:** remove them from lemmas only, POS only, both, or neither?
3. **Casing:** should lemmas and stopword checks be case-insensitive?
4. **POS tagging:** call `pos_tag` once on the whole list or per token?
5. **Lemmatization:** noun-only or POS-aware?
6. **Punctuation:** how exactly are punctuation tokens detected and filtered?
7. **Return shape:** do `lemmas` and `pos_tags` have the same length and alignment?

Use this as a prompt the next time you design a token-processing pipeline.

---

## 9. References

**NLTK documentation and guides:**
- Stopwords corpus and usage: https://www.nltk.org/book/ch02.html
- POS tagging (concepts and `pos_tag`): https://www.nltk.org/book/ch05.html
- WordNet and `WordNetLemmatizer`: https://www.nltk.org/howto/wordnet.html
- Tokenization (`word_tokenize`, `sent_tokenize`): https://www.nltk.org/api/nltk.tokenize.html
- NLTK taggers API reference: https://www.nltk.org/api/nltk.tag.html

**Python standard library docs:**
- `re` (regular expressions): https://docs.python.org/3/library/re.html
- `string.punctuation`: https://docs.python.org/3/library/string.html#string.punctuation
