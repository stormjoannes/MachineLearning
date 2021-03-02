import P1.Perceptron as Perceptron

class PerceptronLayer(object):
    def __init__(self, perceptron=[Perceptron]) -> None:
        self.perceptrons = perceptron
        self.outp = []

    def activatieLayer(self, inp: [float]):
        """
        Functie output per perceptron in lijst plaatsen en deze returnen.
        """
        for i in self.perceptrons:
            self.outp.append(i.activatiePercep(inp))
        return self.outp

    def __str__(self) -> str:
        """
        Formatten van variabelen om ze duidelijk te returnen en te printen.
        """
        return f'perceptrons: {self.perceptrons} \nOutputs: {self.outp}'

# PerceptronLayer([Perceptron(threshold=1, weights=[0.5, 0.5], bias=0), Perceptron(threshold=1, weights=[0.5, 1], bias=0)])