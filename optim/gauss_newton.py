
import numpy as np

def GaussNewton(resid,jac,x0,max_iter=1000,gtol=1e-5,gamma_dec=0.5,c_1=1e-4,alpha_min=1e-16,ftarget=0.0,verbose=False):
  """
  Gauss Newton method with linesearch to solve nonlinear least squares
    min sum_i resid_i(x)**2
  where resid is the vector of residuals.

  resid: function handle for (dim_r,) array, of residuals. dim_r must be >= dim_x
  jac: function handle for (dim_r,dim_x) array jacobian of residuals.
  x0: (dim_x,) array, starting point
  c_1: Armijo parameters for linesearch.
           must satisfy 0 < c_1 < c_2 < 1
  """
  assert (0 < c_1 and c_1< 1), "unsuitable linesearch parameters"

  # define an objective handle
  obj = lambda x: np.sum(resid(x)**2)

  # inital guess
  x_k = np.copy(x0)
  dim = len(x_k)

  # stop when gradient is flat (within tolerance)
  nn = 0
  stop = False
  while stop==False:

    # compute search direction
    J_k = np.copy(jac(x_k))
    Q,R = np.linalg.qr(J_k)
    r_k = np.copy(resid(x_k))
    p_k = - np.copy(np.linalg.solve(R.T @ R,J_k.T @ r_k))

    # func and grad
    f_k = np.sum(r_k**2)
    g_k = np.copy(2*J_k.T @ r_k)

    if verbose:
      print(f'{nn})','resid: ',f_k)

    # stopping criteria
    if np.linalg.norm(g_k) <= gtol:
      if verbose:
        print('Exiting: gtol reached')
      stop = True
    # stopping criteria
    if f_k <= ftarget:
      if verbose:
        print('Exiting: ftarget reached')
      stop = True

    # compute step
    alpha_k = 1.0 # always try 1 first
    x_kp1 = np.copy(x_k + alpha_k*p_k)
    f_kp1 = obj(x_kp1);

    # linsearch with Armijo condition
    armijo = f_kp1 <= f_k + c_1*g_k @ (x_kp1 - x_k)
    while armijo==False:
      # reduce our step size
      alpha_k = gamma_dec*alpha_k;
      # take step
      x_kp1 = np.copy(x_k + alpha_k*p_k)
      # f_kp1
      f_kp1 = obj(x_kp1);
      # compute the armijo condition
      armijo = f_kp1 <= f_k + c_1*g_k @ (x_kp1 - x_k)

      # break if alpha is too small
      if alpha_k <= alpha_min:
        if verbose:
          print('Exiting: step size too small.')
        return x_k

    # reset for next iteration
    x_k  = np.copy(x_kp1)
    f_k  = f_kp1;

    # update iteration counter
    nn += 1

    # stopping criteria
    if nn >= max_iter:
      if verbose:
        print('Exiting: max_iter reached')
      stop = True

  return x_k

