
import numpy as np
from scipy.linalg import lu as plu_decomp


def NewtonLinesearch(F,H,x0,max_iter=1000,ftarget=0.0,gamma_dec=0.5,c_1=1e-4,alpha_min=1e-10,verbose=False):
  """
  Newtons method to solve
    F(x) = 0
  Take steps x_kp1 = x_k - alpha*H^{-1}F
  where alpha is determined via an armijo linesearch on the
  sum of squares ||F(x)||^2.

  This is (almost) Algorithm 11.4 from Nocedal & Wright.

  WARNING: In general the linesearch method will not perform well if
    the Hessian is singular. A better alternative is to use a Newton Trust
    Region method.

  F: function handle, returns array of shape (dim_x,)
  H: function handle for jacobian of F, returns array of shape (dim_x, dim_x)
  x0: (dim_x,) array, starting point
  c_1: Armijo parameters for linesearch.
           must satisfy 0 < c_1 < c_2 < 1
  alpha_min: minimum linesearch parameter.
  """
  assert (0 < c_1 and c_1< 1), "unsuitable linesearch parameters"

  # inital guess
  x_k = np.copy(x0)
  F_k = np.copy(F(x_k))
  dim = len(x_k)

  nn = 0
  stop = False
  while stop==False:

    # compute search direction
    H_k = np.copy(H(x_k))
    #P,L,U = plu_decomp(H_k)
    #p_k = - np.linalg.solve(U,np.linalg.solve(L,np.linalg.solve(P,F_k)))
    p_k = - np.linalg.solve(H_k,F_k)

    # func and grad for linesearch
    f_k = np.linalg.norm(F_k)
    g_k = np.copy(2*H_k.T @ F_k)

    if verbose:
      print(f'{nn})','resid: ',f_k)

    # stopping criteria
    if f_k <= ftarget:
      if verbose:
        print('Exiting: ftarget reached')
      stop = True

    # linsearch with Armijo condition
    alpha_k = 1.0 # always try 1 first
    armijo = False
    while armijo==False:
      # take step
      x_kp1 = np.copy(x_k + alpha_k*p_k)
      # f_kp1
      F_kp1 = np.copy(F(x_kp1))
      f_kp1 = np.linalg.norm(F_kp1)
      # compute the armijo condition
      armijo = f_kp1 <= f_k + c_1*g_k @ (x_kp1 - x_k)

      # break if alpha is too small
      if alpha_k <= alpha_min:
        if verbose:
          print('Exiting: step size too small.')
        return x_k

      # reduce our step size
      alpha_k = gamma_dec*alpha_k;

    # reset for next iteration
    x_k  = np.copy(x_kp1)
    F_k  = np.copy(F_kp1)
    f_k  = f_kp1;

    # update iteration counter
    nn += 1

    # stopping criteria
    if nn >= max_iter:
      if verbose:
        print('Exiting: max_iter reached')
      stop = True

  return x_k
