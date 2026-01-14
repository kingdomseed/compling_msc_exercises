from constituents import Phrase, Token

'''
Construct the parse tree for the sentence "The cat chases the mouse"
Returns:
    A Phrase object representing the parse tree
'''
# Create the tokens
t_the = Token("DT", "the")
t_The = Token("DT", "The")
t_cat = Token("NN", "cat")
punct = Token("PUNCT", ".")
t_chases = Token("VB", "chases")
t_mouse = Token("NN", "mouse")

# Create the noun phrases
p_np1 = Phrase("NP", [t_The, t_cat])
p_np2 = Phrase("NP", [t_the, t_mouse])

# Create the verb phrase
p_vp = Phrase("VP", [t_chases, p_np2])

# Create the complete sentence phrase
sent = Phrase("S", [p_np1, p_vp, punct])