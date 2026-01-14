from constituents import Phrase, Token

'''
Construct the parse tree for the sentence "The cat chases the mouse"
Returns:
    A Phrase object representing the parse tree
'''
# Create the tokens
t_the = Token("DT", "The")
t_cat = Token("NN", "cat")
t_chases = Token("VBD", "chases")
t_mouse = Token("NN", "mouse")

# Create the noun phrases
p_np1 = Phrase("NP", [t_the, t_cat])
p_np2 = Phrase("NP", [t_the, t_mouse])

# Create the verb phrase
p_vp = Phrase("VP", [t_chases, p_np2])

# Create the complete sentence phrase
sent = Phrase("S", [p_np1, p_vp])

# The base is a list of two strings 
# which is just a Token class
print(str(t_the))
print(str(p_vp))