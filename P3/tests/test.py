import unittest
from P3 import sigmoid_neuron as Neuron

class testClasses(unittest.TestCase):

    def testAnd(self):
        """
        Testen and port
        """
        testAndPort = Neuron.Neuron(weights=[0.5, 0.5], bias=-1.0)
        output = []
        output.append(round(testAndPort.activatieNeuron([0, 0])))
        output.append(round(testAndPort.activatieNeuron([0, 1])))
        output.append(round(testAndPort.activatieNeuron([1, 0])))
        output.append(round(testAndPort.activatieNeuron([1, 1])))

        self.assertEqual([0, 0, 0, 1], output)
        #De nieuwe weight 1 zou -0.5 moeten worden en de nieuwe weight 2 0.5
        #De bias zou -1.5 moeten worden


    def testOr(self):
        """
        Testen or port
        """
        testOrPort = Neuron.Neuron(weights=[0.5, 0.5], bias=-1.0)
        output = []
        output.append(round(testOrPort.activatieNeuron([0, 0])))
        output.append(round(testOrPort.activatieNeuron([0, 1])))
        output.append(round(testOrPort.activatieNeuron([1, 0])))
        output.append(round(testOrPort.activatieNeuron([1, 1])))

        self.assertEqual([0, 0, 0, 1], output)
        #De nieuwe weight 1 zou -0.5 moeten worden en de nieuwe weight 2 0.5
        #De bias zou -1.5 moeten worden

    def testInvert(self):
        """
        Testen invert
        """
        testInvertPort = Neuron.Neuron(weights=[-1], bias=0)
        output = []

        output.append(round(testInvertPort.activatieNeuron([0])))
        output.append(round(testInvertPort.activatieNeuron([1])))

        self.assertNotEqual([1, 0], output)
        # De nieuwe weight 1 zou -1 moeten worden en de nieuwe weight 2 -1
        # De bias zou 0 moeten worden

    def testNor(self):
        """
        Testen Nor port
        """
        testNorPort = Neuron.Neuron(weights=[-1, -1, -1], bias=0.1)
        output = []
        for num1 in range(2):
            for num2 in range(2):
                for num3 in range(2):
                    output.append(round(testNorPort.activatieNeuron([num1, num2, num3])))

        self.assertEqual(output, [1, 0, 0, 0, 0, 0, 0, 0])
        #De nieuwe weight 1 zou -0.5 moeten worden en de nieuwe weight 2 0.5
        #De bias zou -1.5 moeten worden

if __name__ == '__main__':
    unittest.main()