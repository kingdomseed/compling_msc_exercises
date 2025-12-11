class Token:
    def __init__(self, pos, word):
        '''
        Construct a token
        Args:
            pos: The part of speech, as a string.  i.e. NN, JJ, etc.
            word: The token's text, as a string
        '''
        self.pos = pos
        self.word = word

class Phrase:
    def __init__(self, phrase_type, children):
        '''
        Construct a phrase
        Args:
            phrase_type: The phrases type, as a string.  i.e. NP, VP, ADJP, etc.
            children: The phrase's children --- a list.  Each child is
                either a token (a leaf node) or another Phrase (an interior node)
        '''
        self.phrase_type = phrase_type
        self.children = children
