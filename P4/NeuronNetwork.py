import P3.NeuronLayer as NeuronLayer

class PerceptronNetwork(object):
    def __init__(self, layers=[NeuronLayer]) -> None:
        self.neuronLayers = layers
        self.outp = []
        self.errors = []

    def feed_forward(self, inp: [float]):
        """"
        Iedere layer pakt als input de output van de vorige layer.
        """
        for i in self.neuronLayers:
            outputLayer = self.outp.append(i.activatieLayer(inp))
        nextInp = outputLayer
        self.outp = nextInp
        return nextInp

    def setError(self):
        for layer in self.neuronLayers:
            self.errors.append(layer.setError())

    def calcLoss(self):
        pass

    def __str__(self) -> str:
        """
        Formatten van variabelen om ze duidelijk te returnen en te printen.
        """
        return f"Aantal layers: {self.neuronLayers} \nOutput: {self.outp}"