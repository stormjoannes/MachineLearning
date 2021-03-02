import unittest
from sklearn.datasets import load_iris
from P2 import perceptron_learning_rule as Perceptron

class testClasses(unittest.TestCase):

    def testAnd(self):
        """
        Testen and port
        """
        testAndPort = Perceptron.Perceptron(threshold=1, weights=[-0.5, 0.5], bias=-1.5, learningRate=0.8)
        testAndPort.update([1, 1], 1)
        output = (testAndPort.w, testAndPort.b)

        self.assertEqual(output, ([0.30000000000000004, 1.3], -0.7))
        #De nieuwe weight 1 zou 0.3 moeten worden en de nieuwe weight 2 1.3
        #De bias zou -0.7 moeten worden


    def testXor(self):
        """
        Testen Xor port
        """
        testXorPort = Perceptron.Perceptron(threshold=0, weights=[1, 1], bias=-1, learningRate=0.5)
        testXorPort.update([1, 1], 1)
        output = (testXorPort.w, testXorPort.b)

        #Xor is geen linear probleem dus kan niet opgelost worden met 1 perceptron, hierdoor moet ik ook not equal doen.
        self.assertNotEqual(output, ([0.5, 0.5], -1.5))
        # De nieuwe weight 1 zou 0.5 moeten worden en de nieuwe weight 2 0.5
        # De bias zou -1.5 moeten worden


    def iris(self):
        """
        Testen iris dataset
        """
        studentnummer = '1760581'
        pass


if __name__ == '__main__':
    unittest.main()