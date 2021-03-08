import P3.Neuron as Neuron

class NeuronLayer(object):
    def __init__(self, neuron=[Neuron]) -> None:
        self.neurons = neuron
        self.outp = []
        self.errors = []

    def activatieLayer(self, inp: [float]):
        """
        Functie output per neuron in lijst plaatsen en deze returnen.
        """
        for i in self.neurons:
            self.outp.append(i.activatieNeuron(inp))
        return self.outp

    def setError(self):
        for neuron in self.neurons:
            self.errors.append(neuron.setError())

    def __str__(self) -> str:
        """
        Formatten van variabelen om ze duidelijk te returnen en te printen.
        """
        return f'neurons: {self.neurons} \nOutputs: {self.outp}'