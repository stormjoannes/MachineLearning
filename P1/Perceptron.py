class Perceptron(object):
    def __init__(self, threshold: float, weights:[float], bias: float) -> None:
        self.th = threshold
        self.wht = weights
        self.bi = bias
        self.outp = 0

    def activatiePercep(self, inp: [float]):
        """
        Functie verwerken van de weights en de bias optellen, akkoort geven als output boven threshold is.
        """
        for i in range(len(inp)):
            self.outp += inp[i] * self.wht[i]
        self.outp += self.bi

        return 1 if self.outp >= self.th else 0

    def __str__(self) -> str:
        """
        Formatten van variabelen om ze duidelijk te returnen en te printen.
        """
        return f'Invoer:  \nWeights: {self.wht} \nBias {self.bi} \nOutput: {self.outp}'