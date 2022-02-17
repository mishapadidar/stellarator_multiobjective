import numpy as np

def finite_difference(f,x0,h=1e-6):
  """Compute the jacobian of f with 
  central difference
  """
  h2   = h/2.0
  dim  = len(x0)
  Ep   = x0 + h2*np.eye(dim)
  Fp   = np.array([f(e) for e in Ep])
  Em   = x0 - h2*np.eye(dim)
  Fm   = np.array([f(e) for e in Em])
  jac = (Fp - Fm)/(h)
  return jac.T

if __name__=="__main__":
    dim = 3
    A = np.random.randn(dim)
    func = lambda x: A @ x
    x0 = np.random.randn(dim)
    print(A)
    print(finite_difference(func,x0))

    dim = 3
    A = np.random.randn(dim,dim)
    func = lambda x: A @ x
    x0 = np.random.randn(dim)
    print("")
    print(A)
    print(finite_difference(func,x0))

