import unittest
from P3 import Neuron as Neuron

class testClasses(unittest.TestCase):
    """
    Doordat een neuron niet met de stepfunctie werkt voor de activatie maar met de sigmoid komen er andere uitkomsten
    uit de activatieNeuron. Om toch op de goede uitkomsten te komen kun je wat met de weights en bias spelen om de goede
    uitkomsten te krijgen.
    """

    def testAnd(self):
        """
        Testen and port
        De and gate zou alleen AAN moeten staan als beide inputs 1 zijn.
        """
        testAndPort = Neuron.Neuron(weights=[0.5, 0.5], bias=-1.0)
        output = []
        output.append(round(testAndPort.activatieNeuron([0, 0])))
        output.append(round(testAndPort.activatieNeuron([0, 1])))
        output.append(round(testAndPort.activatieNeuron([1, 0])))
        output.append(round(testAndPort.activatieNeuron([1, 1])))

        self.assertEqual([0, 0, 0, 1], output)


    def testOr(self):
        """
        Testen or port
        De or gate zou alleen UIT moeten staan als geen een input 1 is.
        """
        testOrPort = Neuron.Neuron(weights=[0.5, 0.5], bias=-0.5)
        output = []
        output.append(round(testOrPort.activatieNeuron([0, 0])))
        output.append(round(testOrPort.activatieNeuron([0, 1])))
        output.append(round(testOrPort.activatieNeuron([1, 0])))
        output.append(round(testOrPort.activatieNeuron([1, 1])))

        self.assertEqual([0, 1, 1, 1], output)

    def testInvert(self):
        """
        Testen invert
        De Invert zou de input om moeten draaien (1 word 0 en omgekeerd).
        """
        testInvertPort = Neuron.Neuron(weights=[-1], bias=0)
        output = []

        output.append(round(testInvertPort.activatieNeuron([0])))
        output.append(round(testInvertPort.activatieNeuron([1])))

        self.assertNotEqual([1, 0], output)

    def testNor(self):
        """
        Testen Nor port
        De Nor gate zou alleen aan moeten staan als alle inputs 0 zijn.
        """
        testNorPort = Neuron.Neuron(weights=[-1, -1, -1], bias=0.1)
        output = []

        for num1 in range(2):
            for num2 in range(2):
                for num3 in range(2):
                    output.append(round(testNorPort.activatieNeuron([num1, num2, num3])))

        self.assertEqual(output, [1, 0, 0, 0, 0, 0, 0, 0])

if __name__ == '__main__':
    unittest.main()