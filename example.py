import numpy as np
from NeuralNetwork.NeuralNetwork import NeuralNetwork

def Classification():
    X = np.matrix([[0,0],[0,1],[1,0],[1,1]])
    Y = np.matrix([1,1,1,0]).T
    NN = NeuralNetwork(X, Y)
    NN.addLayer(5, layertype="sigmoid")
    NN.addOutputLayer(layertype="sigmoid")

    print(NN.cost_function())
    NN.train(iterations=10000, alpha=0.1)
    print(NN.cost_function())

    x = NN.predict([[1,1]]).item(0,0)
    print(x)

def Regression():
    X = np.matrix([0,0.1,0.2,0.3]).T
    Y = np.matrix([0,0.1,0.2,0.3]).T
    NN = NeuralNetwork(X, Y)
    NN.addLayer(2, layertype="linear")
    NN.addOutputLayer(layertype="linear")


    print(NN.cost_function())
    NN.train(iterations=10000, alpha=0.01)
    print(NN.cost_function())

    x = NN.predict([[3]]).item(0,0)
    print(x)


def main():
    Classification()
    Regression()

if __name__ == "__main__":
    main()