import math

class Neuron(object):
    def __init__(self, weights: [float], bias: float) -> None:
        self.weights = weights
        self.bias = bias
        self.outp = 0

    def activatieNeuron(self, inp: [float]):
        """
        Functie verwerken van de weights en de bias optellen, sigmoid doen va de output en dit returnen.
        """
        for i in range(len(inp)):
            self.outp += inp[i] * self.weights[i]
        self.outp += self.bias
        self.sigmoid()

        return self.outp

    def sigmoid(self):
        self.outp = 1 / (1 + math.exp(-self.outp))

    def __str__(self) -> str:
        """
        Formatten van variabelen om ze duidelijk te returnen en te printen.
        """
        return f'Uitvoer:  \nWeights: {self.weights} \nBias {self.bias} \nOutput: {self.outp}'

testAndPort = Neuron(weights=[0.5, 0.5], bias=-1.5)
output = []
print(round(testAndPort.activatieNeuron([0, 1])), 'ja')