
import random
import numpy as np

def BlockCoordinateGaussNewton(resid,jac,x0,block_size=1,max_iter=1000,ftarget=0.0,gamma_dec=0.5,c_1=1e-4,alpha_min=1e-16,verbose=False):
  """
  A stochastic block coordinate Gauss Newton method with linesearch to solve nonlinear least squares
    min sum_i resid_i(x)**2
  where resid is the vector of residuals.

  Block coordinate gauss newton reduces the cost of a Gauss Newton iteration by taking a step in only
  a subset of the coordinates. At each iteration a subset of the indexes is selected, randomly in our
  case, and the jacobian of the residuals is computed only with respect to these components, i.e. J[idx].
  The selected components of x are updated via the standard gauss newton updated
    x_{k+1}[idx] = alpha_k*(J[idx]^T @ J[idx])^{-1} @ J[idx]^T @ x_k[idx]
  where the lineseach is computed against the entire sum of squares, not a subset.

  Block coordinate gauss newton is useful if the full jacobian is expensive to compute, such as with
  finite differences, or if the number of residuals is so large that solving the normal equations is
  prohibitive.

  resid: function handle for (dim_r,) array, of residuals. dim_r must be >= dim_x
  jac: function handle for (dim_r,block_size) array, jacobian of residuals.
       Has two arguments, full input vector of size (dim_x,) and second argument of size (block_size,)
       which is a vector of indexes. function returns Jacobian of all residuals with respect
       to only the indexes of x from the second argument.
  x0: (dim_x,) array, starting point
  block_size: int, number of coordinates to descend on per step, must be <= dim
  max_iter: int, maximum iterations
  ftarget: float, target function value to reach.
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
    idx_k = random.sample(range(dim),block_size) # random indexes
    J_k = np.copy(jac(x_k,idx_k))
    Q,R = np.linalg.qr(J_k)
    r_k = np.copy(resid(x_k))
    p_k = - np.copy(np.linalg.solve(R.T @ R,J_k.T @ r_k))

    # func and grad
    f_k = np.sum(r_k**2)
    g_k = np.copy(2*J_k.T @ r_k)

    if verbose:
      print(f'{nn})','resid: ',f_k)

    # stopping criteria
    if f_k <= ftarget:
      if verbose:
        print('Exiting: ftarget reached')
      stop = True

    # compute step
    alpha_k = 1.0 # always try 1 first
    x_kp1 = np.copy(x_k)
    x_kp1[idx_k] = np.copy(x_k[idx_k] + alpha_k*p_k)
    f_kp1 = obj(x_kp1);

    # linsearch with Armijo condition
    armijo = f_kp1 <= f_k + c_1*g_k @ (x_kp1[idx_k] - x_k[idx_k])
    while armijo==False:
      # reduce our step size
      alpha_k = gamma_dec*alpha_k;
      # take step
      x_kp1[idx_k] = np.copy(x_k[idx_k] + alpha_k*p_k)
      # f_kp1
      f_kp1 = obj(x_kp1);
      # compute the armijo condition
      armijo = f_kp1 <= f_k + c_1*g_k @ (x_kp1[idx_k] - x_k[idx_k])

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
