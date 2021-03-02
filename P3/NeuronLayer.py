import P3.sigmoid_neuron as Neuron

class NeuronLayer(object):
    def __init__(self, neuron=[Neuron]) -> None:
        self.neurons = neuron
        self.outp = []

    def activatieLayer(self, inp: [float]):
        """
        Functie output per neuron in lijst plaatsen en deze returnen.
        """
        for i in self.neurons:
            self.outp.append(i.activatieNeuron(inp))
        return self.outp

    def __str__(self) -> str:
        """
        Formatten van variabelen om ze duidelijk te returnen en te printen.
        """
        return f'neurons: {self.neurons} \nOutputs: {self.outp}'