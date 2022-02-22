## Multi-Objective Optimization of Stellarator Plasma Boundaries

### Discuss
- Switch to SurfaceRZPseudoSpectral coordinates
  > `simsopt` vesion `0.7.0` does not have this feature. Need to upgrade to get access.
- VMEC evaluation print statements take approximately 0.25sec - 0.35sec
- Discuss the 1D plots and 2d plots.
- How do VMEC resolution parameters effect sim failures?

### ToDO
- make non-axis aligned 1d slices of feasible region.
- make 2d slices of feasible region.
- look at the effect of the VMEC resolution parameters on simulation failures.
- Constraints
  - build data-based bound constraints.
  - rescale the space according to the bounds.
- choose a method of solving the unrelaxable bound constrained MOO problem.
  - Bi-objective MOO
    - With Bi-objective we can use a discrete set of
      reference points within a Lp or Tchebycheff approach.
      First need to (globally) minimize each objective to bound the pareto front.
      Use multistart (fault tolerant)GD or direct search.
  - DFO MOO solvers
    - NSGA-II, DMS, MultiMADS
- optimize and save all evaluation data.
- plot the pareto front.
- locally expand pareto front around non-dominated points.
- look at convexity vs non-convexity.
- choose a few points of interest.
- Look at sensitivity to surface labels.


### Completed
  - [x] Talk to Matt about:
    - [x] turning off VMEC output.
    - [x] vmec discretization.
    - [x] built in finite difference for QS and aspect.
    - [x] parallelism: How can we concurrently evaluate the objective?
    - [x] sim failure feasible region ansatz.
  - [x] Set up vmec resolution parameters in the QH prob. 
     > Denote as `vmec.indata.mpol` and `vmec.indata.ntor`. Keep this higher than 
       the max Fourier modes of the boundary.
  - [x] write a finite difference jacobian.
  - [x] test finite difference jacobian.
  - [x] Write a parallel jacobian computation using working groups.
  - [x] Run Matt's QS test.
     > Seemed succesful. Here is the print out:
     > `xtol` termination condition is satisfied. 
     > Function evaluations 30, initial cost 1.5177e-01, final cost 1.7038e-03, first-order optimality 6.35e+01.
     > Time to solve:  303.22242522239685
     > Final Obj:  0.0034076812956755353
     > Final Total QS:  0.0034056233511294337
  - [x] Performing Timing Test
     > Using more mpi resources while keeping the number of worker groups fixed does not speed up compuation.
     > In fact I found that it was fastest to use a single mpi rank per worker group. 
     > Thus it is best to use only 1 rank per MPIPartition, and leverage concurrent computations instead.
     >
     > At max_mode = 1 (8 dofs) the computational time is 1.5sec. 
     > At max_mode = 2 (24 dofs) the computational time is 3.3sec. 
     > At max_mode = 3 (48 dofs) the computational time is 4.1sec. 
     >
     > VMEC print statements take 0.25sec - 0.35sec.
  - [x] Upgrade concurrent evals and jacobian in qh\_prob1.py so that arbitrary number of
        points can be evaluated.
  - [x] Make 1d-plots to show finite difference fidelity, multi-modality, and simulation failures.
       > Plot shows no noise even up to 1e-9 step size. This shows pretty amazing precision in VMEC solves at the
       > current resolution (by default set to 5 = 2dofs + 3).
       > Functions look like convex quadratics around qh_prob1's x0.
       > Simulation failures may not be connected in space. Some directions fail, then stop failing, then fail again 
       > along a 1d line segment, which implies non-convexity of the feasible region.
       > Lastly, bounds for each variable differ greatly.
  - [x] Write a script to sample the feasible region.
  - [x] Consdier three approximations to the feasible region: safe approximation, outer approximation, best approximation.
       > safe approximation is useful to optimization methods as it gaurantees to avoids failures. Typically safe approximation
       > should use a non-convex feasible region approximant, so that it consumes as much of the feasible region as possible. 
       > The 2d plots show that the safe approximation would be quite small for our problem.
       >
       > The best approximation is typically a nonlinear classification methods, such as an SVM. It can be used as a penalty in 
       > objective functions to push methods away from the boundary and improve the use of the computational budget. Nonlinear,
       > nonconvex regions, such as the best approximation, are difficult to sample uniformly.
       >
       > The safe approximation can be any shape so long as it contains the feasible region. Bound constrained safe
       > approximations are useful for sampling from the feasible region, rescaling input parameters, and bounding 
       > global and multi-objective optimization methods.
  
