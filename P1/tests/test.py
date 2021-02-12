import unittest
from P1 import Perceptron as Perceptron
from P1 import PerceptronLayer as PcpLayer
from P1 import PerceptronNetwork as pcpNetwork

class testClasses(unittest.TestCase):

    def testAnd(self):
        """
        Testen And port.
        """
        testAndPort = Perceptron.Perceptron(threshold=1, weights=[0.5, 0.5], bias=0)
        output = []

        for i in range(2):
            for x in range(2):
                output.append(testAndPort.output([i, x]))

        self.assertEqual(output, [0, 0, 0, 1])

    def testOr(self):
        """
        Testen Or port.
        """
        testOrPort = Perceptron.Perceptron(threshold=0.5, weights=[0.5, 0.5], bias=0)
        output = []

        for i in range(2):
            for x in range(2):
                output.append(testOrPort.output([i, x]))

        self.assertEqual(output, [0, 1, 1, 1])

    def testNor(self):
        """
        Testen Nor port.
        """
        testNorPort = Perceptron.Perceptron(threshold=0, weights=[-1, -1], bias=0)
        output = []

        for i in range(2):
            for x in range(2):
                output.append(testNorPort.output([i, x]))

        self.assertEqual(output, [1, 0, 0, 0])

    def testXor(self):
        """
        Testen Xor port.
        """
        testXorPort = Perceptron.Perceptron(threshold=0, weights=[-1, -1], bias=0)
        output = []

        for i in range(2):
            for x in range(2):
                output.append(testXorPort.output([i, x]))

        self.assertEqual(output, [0, 1, 1, 0])

    if __name__ == '__main__':
        unittest.main()