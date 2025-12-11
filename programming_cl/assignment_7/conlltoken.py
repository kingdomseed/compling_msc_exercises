class ConLLToken:
    def __init__(self, form, lemma, pos, morph):
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.morph = morph

    def __str__(self):
        return f"{self.form},{self.lemma},{self.pos},{self.morph}"

    def is_punctuation(self):
        if self.pos is "PUNCT":
            return True
        else: 
            return False
    def is_inflected(self):
        if self.lemma.lower() == self.form.lower():
            return True
        else: 
            return False
    def get_person(self):
        for feature in self.morph.split("|"):
            if feature.startswith("Person"):
                return feature.split("=")[1]
        return None
    
