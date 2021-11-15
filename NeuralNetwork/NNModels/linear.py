import numpy as np

class LinearLayer:

  def __init__(self, m_data, m_mat, n_neurons, output=False):
    self.Weig = np.matrix(np.random.rand(m_mat + 1, n_neurons))
    self.n_ne = n_neurons
    self.m_da = m_data
    self.out  = output
    self.Bias = np.ones((m_data, 1))
  
  def getWeights(self):
    return self.Weig
  
  def getNeuronsNumber(self):
    return self.n_ne
  
  def forward(self, Matrix):
    self.Mat_b = np.hstack((self.Bias, Matrix))
    self.z     = self.Mat_b @ self.Weig
    self.A     = self.z

    return self.A

  def backward(self, Matrix, Theta, alpha=1):
    Weig = self.Weig

    if(self.out):
      self.delta = self.A - Matrix
      self.Weig -= (1/self.m_da) * (self.delta.T @ self.Mat_b).T
      return [self.delta, Weig]

    self.delta = np.multiply((Theta[1:] @ Matrix.T), self.z.T).getT()
    self.Weig -= (alpha/self.m_da) * (self.delta.T @ self.Mat_b).T
    return [self.delta, Weig]
  
  def predict(self, Matrix, m_pre):
    Bias = np.ones((m_pre, 1))
    self.Mat_b = np.hstack((Bias, Matrix))
    self.z     = np.hstack((Bias, Matrix)) @ self.Weig
    self.A     = self.z

    return self.A

  def costFunction(self, Y):
    j = (1/self.m_da) * np.sum(np.power(self.A - Y , 2))
    return j