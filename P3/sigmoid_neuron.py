import math

class Neuron(object):
    def __init__(self, threshold: float, weights: [float], bias: float) -> None:
        self.th = threshold
        self.w = weights
        self.b = bias
        self.outp = 0

    def activatieNeuron(self, inp: [float]):
        """
        Functie verwerken van de weights en de bias optellen, sigmoid doen va de output en dit returnen.
        """
        for i in range(len(inp)):
            self.outp += inp[i] * self.w[i]
        self.outp += self.b
        self.outp = 1 / (1 + math.exp(-self.outp))

        return self.outp

    def __str__(self) -> str:
        """
        Formatten van variabelen om ze duidelijk te returnen en te printen.
        """
        return f'Uitvoer:  \nWeights: {self.w} \nBias {self.b} \nOutput: {self.outp}'