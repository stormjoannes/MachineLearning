import unittest
from P3 import sigmoid_neuron as Neuron

class testClasses(unittest.TestCase):

    def testAnd(self):
        """
        Testen and port
        """
        testAndPort = Neuron.Neuron(threshold=1, weights=[-0.5, 0.5], bias=-1.5)
        output = (testAndPort.w, testAndPort.b)

        self.assertEqual(output, ([-0.5, 0.5] , -1.5))
        #De nieuwe weight 1 zou -0.5 moeten worden en de nieuwe weight 2 0.5
        #De bias zou -1.5 moeten worden


    def testOr(self):
        """
        Testen or port
        """
        testOrPort = Neuron.Neuron(threshold=0.5, weights=[-1, -1], bias=0)
        output = (testOrPort.w, testOrPort.b)

        self.assertEqual(output, ([-1,-1], 0.12))
        # De nieuwe weight 1 zou -1 moeten worden en de nieuwe weight 2 -1
        # De bias zou 0.12 moeten worden

    def testInvert(self):
        """
        Testen invert
        """
        testInvertPort = Neuron.Neuron(threshold=0, weights=[-1, -1], bias=0)
        output = (testInvertPort.w, testInvertPort.b)

        self.assertEqual(output, ([-1, -1], 0))
        # De nieuwe weight 1 zou -1 moeten worden en de nieuwe weight 2 -1
        # De bias zou 0 moeten worden

if __name__ == '__main__':
    unittest.main()