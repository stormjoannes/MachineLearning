import P1.PerceptronLayer as PerceptronLayer

class PerceptronNetwork(object):
    def __init__(self, layers=[PerceptronLayer]) -> None:
        self.lrs = layers
        self.outp = []

    def feed_forward(self, inp: [float]):
        """"
        Iedere layer pakt als input de output van de vorige layer. Deze word
        """
        for i in self.lrs:
            outputLayer = self.outp.append(i.activatieLayer(inp))
        nextInp = outputLayer
        self.outp = nextInp
        return nextInp

    def __str__(self) -> str:
        """
        Formatten van variabelen om ze duidelijk te returnen en te printen.
        """
        return f"Aantal layers: {self.layers} \nOutput: {self.outp}"