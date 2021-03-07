from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from P2 import perceptron_learning_rule as Perceptron
import random


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