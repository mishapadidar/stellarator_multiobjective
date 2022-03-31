import numpy as np


def NewtonLinesearch(Obj,Grad,Hess,x0,max_iter=1000,gtol=1e-8,gamma_dec=0.5,c_1=1e-4,alpha_min=1e-10,verbose=False):
  """
  Newtons method for minimization of
    min Obj(x)
  Take steps x_kp1 = x_k - alpha*Hess^{-1} @ Grad
  where alpha is determined via an armijo linesearch.

  WARNING: In general the linesearch method will not perform well if
    the Hessian is nearly singular. A better alternative is to use a Newton Trust
    Region method.

  Obj: scalar function handle, for minimization
  Grad: gradient handle, returns array of shape (dim_x,)
  Hess: hessian handle, returns array of shape (dim_x, dim_x)
  x0: (dim_x,) array, starting point
  c_1: Armijo parameters for linesearch.
           must satisfy 0 < c_1 < c_2 < 1
  alpha_min: minimum linesearch parameter.
  max_iter: int, maximum iterations
  gtol: float, 2-norm gradient tolerance
  """
  assert (0 < c_1 and c_1< 1), "unsuitable linesearch parameters"

  # inital guess
  x_k = np.copy(x0)
  f_k = np.copy(Obj(x_k))
  dim = len(x_k)

  nn = 0
  stop = False
  while stop==False:

    # compute search direction
    g_k = np.copy(Grad(x_k))
    # TODO: factor H_k
    H_k = np.copy(Hess(x_k))
    p_k = - np.linalg.solve(H_k,g_k)

    stat_cond = np.linalg.norm(g_k)
    if verbose:
      print(f'{nn})','stationary condition: ',stat_cond)

    # stopping criteria
    if stat_cond <= gtol:
      if verbose:
        print('Exiting: gtol reached')
      stop = True

    # TODO: should compue wolfe conditions
    # linsearch with Armijo condition
    alpha_k = 1.0 # always try 1 first
    armijo = False
    while armijo==False:
      # take step
      x_kp1 = np.copy(x_k + alpha_k*p_k)
      # f_kp1
      f_kp1 = np.copy(Obj(x_kp1))
      # compute the armijo condition
      armijo = f_kp1 <= f_k + c_1*alpha_k*g_k @ p_k

      # break if alpha is too small
      if alpha_k <= alpha_min:
        if verbose:
          print('Exiting: step size too small.')
        return x_k

      # reduce our step size
      alpha_k = gamma_dec*alpha_k;

    # reset for next iteration
    x_k  = np.copy(x_kp1)
    f_k  = np.copy(f_kp1)

    # update iteration counter
    nn += 1

    # stopping criteria
    if nn >= max_iter:
      if verbose:
        print('Exiting: max_iter reached')
      stop = True

  return x_k
