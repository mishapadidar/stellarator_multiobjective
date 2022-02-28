
import numpy as np

class eval_wrapper:
  """
  Class for wrapping a function call so that we can 
  save evaluations.
  f: function handle
  dim: function input dimension
  Usage:
    f = lambda x: np.linalg.norm(x)
    dim = 2
    func = eval_wrapper(f,dim)
    # call the function
    func(np.random.randn(dim))
    # check the history
    print(func.X)
    print(func.fX)
  """

  def __init__(self,f,dim):
    self.dim = dim
    self.X = np.zeros((0,dim))
    self.fX = np.zeros(0)
    self.func = f

  def __call__(self,xx):
    ff = self.func(xx)
    self.X = np.append(self.X,[xx],axis=0)
    self.fX = np.append(self.fX,ff)
    return ff
