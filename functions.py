import numpy as np
import matplotlib.pyplot as plt

__all__ = ["sigmoid", "sigmoid_prime", "mse"]

def sigmoid(z):
  return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
  return sigmoid(z)*(1.0-sigmoid(z))

def mse(a, y):
  return .5*sum((a[i]-y[i])**2 for i in range(10))[0]

