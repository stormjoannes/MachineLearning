import math

class Neuron(object):
    def __init__(self, weights: [float], bias: float, learningRate: float) -> None:
        self.weights = weights
        self.bias = bias
        self.outp = 0
        self.errors = []
        self.neuronError = 0
        self.gradient = 0
        self.learningRate = learningRate
        self.hiddenError = 0

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

    def setError(self, target: float):
        """
        Berekenen error.
        """
        afgelInp = self.outp * (1 - self.outp)
        self.neuronError = afgelInp * -(target - self.outp)

        return self.neuronError
        # tweedeMacht = [self.errors[i] ** 2 for i in range(len(self.errors))]
        # return sum(tweedeMacht) / len(tweedeMacht)

    def setGradient(self, outpNeuron: float):
        # deltaWeights = []
        self.gradient = outpNeuron * self.neuronError

        # for i in self.weights:
        deltaWeights = self.learningRate * outpNeuron * self.neuronError
        # deltaWeight = self.weights[i]
        # deltaWeights.append(deltaWeight)
        deltaBias = self.learningRate * self.neuronError

        return deltaWeights, deltaBias

    def update(self, input: [float], target: float, outpNeuron: float):
        self.setError(target)
        deltas = self.setGradient(outpNeuron)
        for i in self.weights:
            self.weights[i] = self.weights[i] - deltas[0]
        self.bias = self.bias - deltas[1]

    def calcHiddenError(self, hidOutp: float):
        afgelInp = hidOutp * (1 - hidOutp)
        for i in self.weights
            sum van lijst met alle errors
        self.hiddenError = afgelInp * self.epsilonDing * self.weights * self.neuronError

    def __str__(self) -> str:
        """
        Formatten van variabelen om ze duidelijk te returnen en te printen.
        """
        return f'Uitvoer:  \nWeights: {self.weights} \nBias {self.bias} \nOutput: {self.outp}'

# testAndPort = Neuron(weights=[0.5, 0.5], bias=-1.5)
# output = []
print(round(testAndPort.activatieNeuron([0, 1])), 'ja')