from functions import *
import numpy as np



__all__ = ["MultilayPerceptron"]


class MultilayerPerceptron():
  
  def __init__(self, layers):
    self.layers = layers
    self.L = len(self.layers)
    self.W =[[0.0]]
    self.B = [[0.0]]
    for i in range(1, self.L):
      w_temp = np.random.randn(self.layers[i], self.layers[i-1])*np.sqrt(2/self.layers[i-1])
      b_temp = np.random.randn(self.layers[i], 1)*np.sqrt(2/self.layers[i-1])

      self.W.append(w_temp)
      self.B.append(b_temp)



  def forward_pass(self, x, predict_vector = False):
    Z =[[0.0]]
    A = [x]
    for i in range(1, self.L):
      z = (self.W[i] @ A[i-1]) + self.B[i]
      a = sigmoid(z)
      Z.append(z)
      A.append(a)

    if predict_vector == True:
      return A[-1]
    else:
      return Z, A

  def MSE(self, X, Y):
    c = 0.0
    for p in zip(X, Y):
      a = self.forward_pass(p[0], predict_vector=True)
      c += mse(a, p[1])
    return c/len(X)

  def deltas_dict(self, x, y):
    Z, A = self.forward_pass(x)
    deltas = dict()
    deltas[self.L-1] = (A[-1] - y)*sigmoid_prime(Z[-1])
    for l in range(self.L-2, 0, -1):
      deltas[l] = (self.W[l+1].T @ deltas[l+1]) * sigmoid_prime(Z[l])

    return A, deltas

  def stochastic_gradient_descent(self, X, Y, alpha = 0.04, epochs = 50):

    #print(f"Initial Cost = {self.MSE(X, Y)}")
    for k in range(epochs):
        for x, y in zip(X, Y):
            A, deltas = self.deltas_dict(x, y)
            for i in range(1, self.L):
                self.W[i] = self.W[i] - alpha*deltas[i]@A[i-1].T
                self.B[i] = self.B[i] - alpha*deltas[i]
        #print(f"{k} Cost = {self.MSE(X, Y)}")