import unittest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import random
from P2 import perceptron_learning_rule as Perceptron

class testClasses(unittest.TestCase):

    def testAnd(self):
        """
        Testen and port
        De and gate zou alleen AAN moeten staan als beide inputs 1 zijn.
        """
        testAndPort = Perceptron.Perceptron(threshold=0, weights=[0.5, 0.5], bias=-1, learningRate=0.8)

        output = []
        output.append(testAndPort.processInput([0, 0]))
        output.append(testAndPort.processInput([0, 1]))
        output.append(testAndPort.processInput([1, 0]))
        output.append(testAndPort.processInput([1, 1]))

        self.assertEqual([0, 0, 0, 1], output)
        #De nieuwe weight 1 zou -0.5 moeten worden en de nieuwe weight 2 0.5
        #De bias zou -1.5 moeten worden


    def testXor(self):
        """
        Testen Xor port
        De Xor gate zou alleen AAN moeten staan als niet alle inputs hetzelfde zijn.
        """
        testXorPort = Perceptron.Perceptron(threshold=0, weights=[1, 1], bias=-1, learningRate=0.5)

        output = []
        output.append(testXorPort.processInput([0, 0]))
        output.append(testXorPort.processInput([0, 1]))
        output.append(testXorPort.processInput([1, 0]))
        output.append(testXorPort.processInput([1, 1]))

        self.assertEqual([0, 1, 1, 0], output)


    def iris(self):
        """
        Testen iris dataset
        """
        studentnummer = 1760581
        random.seed(studentnummer)
        data = load_iris()

        #random 4 cijferige input tussen -5 en 5
        input = [random.randint(-5, 5) for i in range(4)]
        bias = random.randint(-5, 5)
        learningRate = 0.1
        #threshold random tussen 0 en 1
        threshold = random.randint(0, 1)
        testPercep = Perceptron.Perceptron(threshold, [], bias, learningRate)

        targets = []
        outputs = []

        for i in range(100):
            #op iedere weight de weights en bias updaten
            testPercep.weights = data['data'][i]
            target = data['target'][i]
            for _ in range(15):
                #max 15 keer de update runnen om de weights en bias te veranderen
                testPercep.update(input, target)

        for i in range(100):
            #lijsten vullen  met de verwachte uitkomsten en de uitkomsten om de score te berekenen.
            targets.append(data['target'][i])
            outputs.append(testPercep.processInput(input))
        score = accuracy_score(targets, outputs)

        self.assertEqual(1, score)

if __name__ == '__main__':
    unittest.main()