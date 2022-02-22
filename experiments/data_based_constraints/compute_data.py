import numpy as np
import sys
sys.path.append("../../problem")
sys.path.append("../../utils")
import qh_prob1
import pickle

"""
Develop data-driven constraints.
Suppose our feasible region Omega is defined by the binary function h(x)
where h(x) = 1 if x is in Omega and 0 otherwise. Our goal is to learn a
constraint function c(x) that replicates h as best as possible. It is 
key that we also aim to classify infeasible points correctly as well.
Define S as the set of all points in Rn, such that c(x) == h(x). 
Then we want to find the function c to solve
  max_c |S(c)|
which is a volume maximization problem. This approach aims to correctly
classify all points in Rn, not just Omega, which plagues the approach in 
practice as |Rn| greatly outweighs |Omega|. 

A more practical method is to maximize a volume fraction with respect to a
larger set, Gamma, containing Omega. Define the volume fraction V(c) to be the 
subset of Gamma of which c can classify correctly. Then we solve
  max_c V(c)
If Gamma is known then we need to integrate the volume of Gamma for which we can 
correctly classify, i.e. 
  max_c int_{Gamma} indicator{c(x)==h(x)} dx
In high dimensional problems we would solve this via monte-carlo integration, by 
randomly sampling from a distribution over Gamma. Ultimately this leads us to a
probabilistic interpretation. Define the random variable X which has non-zero
probability over Gamma
  max_c Prob(c(X)==h(X))

Of course if our constraint is hidden then there is no way to set Gamma apriori
which renders all of these methods useless. However, only minor modification is needed
to come upon a practical method. For instance using an unbounded distribution, such as a Gaussian,
will gaurantee that X has a non-zero probability over Gamma, making the probabilistic
approach viable. Another method of solving the probalistic approach is to sample from a uniform
distribution over a hypercube, the side-lengths of which are doubled successively until
the empircal probability of randomly sampling a feasible point is sufficiently low. 
Poor scaling of the input variables will inevitably cause these methods to fail, or have
poor convergence properties. 

For a given classifier c, the probability Prob(c(X)==h(X)) is difficult to compute. 
In practice we should use a differentiable classifier that returns a number c(x) 
between [0,1]. Rather than maximize the probability explicitly we should use
a surrogate loss, such as hinge loss. This can be well motivated by convexifying
the Chance-constrained program by with the Conditional Value-at-Risk (CVaR). This 
way we can solve the subproblem fairly well, and the difficulty lies mostly in 
our sampling routine. 

To increase the bound size we can use stochastic approximation!

"""

# load the problem
vmec_input = "../../problem/input.nfp4_QH_warm_start"
prob = qh_prob1.QHProb1(vmec_input)
x0 = prob.x0
dim_x = prob.dim_x
dim_F = prob.dim_F

# parameters
max_iter = 50
# number of points per iteration
n_points_per = 100 # need more than 1
n_points = max_iter*n_points_per
# growth factor
growth_factor = 2
# initial box size
max_pert = 0.001 
ub = x0 + max_pert
lb = x0 - max_pert

# match the seeds
seed = prob.sync_seeds()

# storage
X = np.zeros((0,dim_x)) # points
FX = np.zeros((0,dim_F)) # function values
CX = np.zeros((0,1)) # constraint values

for ii in range(max_iter):
  print("\n\n\n")
  print("iteration: ",ii)
  # sample
  Y = np.random.uniform(lb,ub,(n_points_per,dim_x))
  FY = prob.evalp(Y)
  # compute constraint values
  CY = (FY[:,0] != np.inf).reshape((-1,1))
  # save data
  X = np.copy(np.vstack((X,Y)))
  FX = np.copy(np.vstack((FX,FY)))
  CX = np.copy(np.vstack((CX,CY)))
  # correct scaling
  idx = CX.flatten().astype(bool)
  lb = np.copy(np.min(X[idx],axis=0))
  ub = np.copy(np.max(X[idx],axis=0))
  # enlarge
  diff = (ub-lb)/4
  ub = np.copy(ub + growth_factor*diff)
  lb = np.copy(lb - growth_factor*diff)

# compute tightest bound constraints
idx = CX.flatten().astype(bool)
lb = np.min(X[idx],axis=0)
ub = np.max(X[idx],axis=0)
  
# dump a pickle file
outfile = f"samples_{seed}.pickle"
outdata = {}
outdata['X'] = X
outdata['FX'] = FX
outdata['CX'] = CX
outdata['ub'] = ub
outdata['lb'] = lb
outdata['n_points'] = n_points
outdata['seed'] = seed
pickle.dump(outdata,open(outfile,"wb"))
