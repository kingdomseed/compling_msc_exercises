from conlltoken import ConLLToken


class ConLLUTokenBuilder:
    def __init__(self):
        super().__init__()
    
    def buildToken(self, line):
        fields = line.strip().split()
        return ConLLToken(fields[1], fields[2], fields[3], fields[5])

class ConLL09TokenBuilder(ConLLUTokenBuilder):
    def __init__(self):
        super().__init__()
    
    def buildToken(self, line):
        fields = line.strip().split()
        return ConLLToken(fields[1], fields[2], fields[4], fields[6])
    