class Perceptron(object):
    def __init__(self, threshold: float, weights: [float], bias: float, learningRate: float) -> None:
        self.th = threshold
        self.w = weights
        self.b = bias
        self.outp = 0
        self.n = learningRate
        self.errors = []

    def activatiePercep(self, inp: [float]):
        """
        Functie verwerken van de weights en de bias optellen, akkoord geven als output boven threshold is.
        """
        for i in range(len(inp)):
            self.outp += inp[i] * self.w[i]
        self.outp += self.b

        return 1 if self.outp >= self.th else 0

    def update(self, input: [float], expected: float):
        """
        Berekenen van nieuwe weights en bias, error updaten.
        """
        error = expected - self.activatiePercep(input)
        self.errors.append(error)

        for i in range(len(self.w)):
            deltaW = self.n * error * input[i]
            self.w[i] = self.w[i] + deltaW

        deltaB = self.n * error
        self.b = self.b + deltaB

    def error(self):
        """
        Berekenen mean squared error.
        """
        return (sum(self.errors) ** 2) / len(self.errors)

    def __str__(self) -> str:
        """
        Formatten van variabelen om ze duidelijk te returnen en te printen.
        """
        return f'Uitvoer:  \nWeights: {self.w} \nBias {self.b} \nOutput: {self.outp}'

x = Perceptron(threshold=1, weights=[-0.5, 0.5], bias=-1.5, learningRate=0.8)
print(x.activatiePercep([1, 1]))
print(x.update([1, 1], 1), 'jaaaaaaa')
print(x.activatiePercep([1, 1]))
y = x.error()
print(y, 'error')
print(x)