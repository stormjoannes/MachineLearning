class Perceptron(object):
    def __init__(self, threshold: float, weights: [float], bias: float, learningRate: float) -> None:
        self.threshold = threshold
        self.weights = weights
        self.bias = bias
        self.outp = 0
        self.learningRate = learningRate
        self.errors = []

    def processInput(self, inp: [float]):
        """
        Functie verwerken van de weights en de bias optellen.
        """
        for i in range(len(inp)):
            self.outp += inp[i] * self.weights[i]
        self.outp += self.bias
        return self.activatiePercep()

    def activatiePercep(self):
        """
        Akkoord geven als output boven threshold is.
        """
        return 1 if self.outp >= self.threshold else 0

    def update(self, input: [float], expected: float):
        """
        Berekenen van nieuwe weights en bias, error updaten.
        """
        error = expected - self.processInput(input)
        print(error, 'error', self.weights, self.bias, 'bias')
        self.errors.append(error)

        for i in range(len(self.weights)):
            deltaWeights = self.learningRate * error * input[i]
            self.weights[i] = self.weights[i] + deltaWeights

        deltaBias = self.learningRate * error
        self.bias = self.bias + deltaBias

        print(self.weights, 'after weights', self.bias, 'bias')

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
        return f'Uitvoer:  \nWeights: {self.weights} \nBias {self.bias} \nOutput: {self.outp}'

# x = Perceptron(threshold=1, weights=[-0.5, 0.5], bias=-1.5, learningRate=0.8)
# print(x.activatiePercep([1, 1]))
# print(x.update([1, 1], 1), 'jaaaaaaa')
# print(x.activatiePercep([1, 1]))
# y = x.error()
# print(y, 'error')
# print(x)