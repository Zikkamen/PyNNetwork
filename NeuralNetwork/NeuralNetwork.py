import numpy as np
from NeuralNetwork.NNModels.sigmoid import SigmoidLayer
from NeuralNetwork.NNModels.linear import LinearLayer

class NeuralNetwork:

  def __init__(self, Data, Target):
    self.X = np.matrix(Data)
    self.Y = np.matrix(Target)
    self.m = len(self.X)
    self.n = len(self.X.T)
    self.Theta = []
    self.layer = {'sigmoid': SigmoidLayer, 'linear': LinearLayer}
  
  def addLayer(self, n_neurons, layertype="linear"):
    if(self.Theta == []): 
      n = self.n
    else:
      n = self.Theta[-1].getNeuronsNumber()
    
    self.Theta.append(self.layer[layertype](self.m, n, n_neurons))

  def addOutputLayer(self, layertype="linear"):
    if(self.Theta == []): 
      n = self.n
    else:
      n = self.Theta[-1].getNeuronsNumber()

    n_neurons = len(self.Y.T)
    self.Theta.append(self.layer[layertype](self.m, n, n_neurons, output=True))
  
  def predict(self, X, m=1):
    for x in self.Theta:
      X = x.predict(X, m)

    return X

  def update(self, alpha):
    A = self.X
    for x in self.Theta:
      A = x.forward(A)

    Delta = [self.Y, []]
    for x in self.Theta[-1::-1]:
      Delta = x.backward(Delta[0], Delta[1], alpha)
  
  def train(self, iterations=1000, alpha=1):
    for i in range(iterations):
      self.update(alpha)

  def cost_function(self):
    A = self.predict(self.X, self.m)
    j = self.Theta[-1].costFunction(self.Y)

    return j
  
  def showT(self):
    T_out = []

    for x in self.Theta:
      T_out.append(x.getWeights())

    return T_out