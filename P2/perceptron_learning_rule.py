class Perceptron(object):
    def __init__(self, threshold: float, weights: [float], bias: float, learningRate: float) -> None:
        self.threshold = threshold
        self.weights = weights
        self.bias = bias
        self.learningRate = learningRate
        self.errors = []

    def processInput(self, inp: [float]):
        """
        Functie verwerken van de weights en de bias optellen.
        """
        outp = 0
        for i in range(len(inp)):
            outp += inp[i] * self.weights[i]
        outp += self.bias
        return self.activatiePercep(outp)

    def activatiePercep(self, outp: float):
        """
        Akkoord geven als output boven threshold is.
        """
        return 1 if outp >= self.threshold else 0

    def update(self, input: [float], expected: float):
        """
        Berekenen van nieuwe weights en bias, error updaten.
        """
        error = expected - self.processInput(input)
        self.errors.append(error)

        for i in range(len(self.weights)):
            deltaWeights = self.learningRate * error * input[i]
            self.weights[i] = self.weights[i] + deltaWeights

        deltaBias = self.learningRate * error
        self.bias = self.bias + deltaBias

    def error(self):
        """
        Berekenen mean squared error.
        """
        tweedeMacht = [self.errors[i]**2 for i in range(len(self.errors))]
        return sum(tweedeMacht) / len(tweedeMacht)

    def __str__(self) -> str:
        """
        Formatten van variabelen om ze duidelijk te returnen en te printen.
        """
        return f'Uitvoer:  \nWeights: {self.weights} \nBias {self.bias}'