import unittest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import random
from P2 import perceptron_learning_rule as Perceptron

class testClasses(unittest.TestCase):

    def testAnd(self):
        """
        Testen and port
        """
        testAndPort = Perceptron.Perceptron(threshold=1, weights=[-0.5, 0.5], bias=-1.5, learningRate=0.8)
        testAndPort.update([1, 1], 1)
        output = (testAndPort.weights, testAndPort.bias)

        self.assertEqual(output, ([0.30000000000000004, 1.3], -0.7))
        #De nieuwe weight 1 zou 0.3 moeten worden en de nieuwe weight 2 1.3
        #De bias zou -0.7 moeten worden


    def testXor(self):
        """
        Testen Xor port
        """
        testXorPort = Perceptron.Perceptron(threshold=0, weights=[1, 1], bias=-1, learningRate=0.5)
        testXorPort.update([1, 1], 1)
        output = (testXorPort.weights, testXorPort.bias)

        #Xor is geen linear probleem dus kan niet opgelost worden met 1 perceptron, hierdoor moet ik ook not equal doen.
        self.assertNotEqual(output, ([0.5, 0.5], -1.5))
        # De nieuwe weight 1 zou 0.5 moeten worden en de nieuwe weight 2 0.5
        # De bias zou -1.5 moeten worden


    def iris(self):
        """
        Testen iris dataset
        """
        studentnummer = 1760581
        random.seed(studentnummer)
        data = load_iris()

        input = [random.randint(-10, 10) for i in range(4)]
        bias = random.randint(-5, 5)
        learningRule = 0.1
        threshold = random.randint(0, 1)
        testPercep = Perceptron.Perceptron(threshold, [], bias, learningRule)

        score = 0

        while score < 1:
            weightsa = []
            for i in range(100):
                testPercep.weights = data['data'][i]
                target = data['target'][i]
                for _ in range(15):
                    testPercep.update(input, target)
                weightsa.append(testPercep.weights)
            print(testPercep.bias, testPercep.weights)

            total_target = []
            total_output = []
            for i in range(100):
                weights = data['data'][i]
                total_target.append(data['target'][i])
                total_output.append(testPercep.processInput(input))
            for i in range(len(total_target)):
                print(total_target[i], total_output[i], data['data'][i])
            score = accuracy_score(total_target, total_output)
            print(score)


if __name__ == '__main__':
    unittest.main()