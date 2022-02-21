## Multi-Objective Optimization of Stellarator Plasma Boundaries

### Discuss
- Switch to SurfaceRZPseudoSpectral coordinates
  > `simsopt` vesion `0.7.0` does not have this feature. Need to upgrade to get access.
- VMEC evaluation print statements take approximately 0.25sec - 0.35sec
- Discuss the 1D plots.

### ToDO
- Exploratory Analysis
  - make 2d slices of feasible region for SIMSOPT meeting.
- Constraints
  - build data-based bound constraints.
  - rescale the space according to the bounds.
  - build three approximations to the VMEC feasible region
    - safe approximation, outer approximation, best approximation.
    - Have Matt take a look at points which evaluate but are 
      outside of the safe approximation
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
       > along a 1d line segment. This is most likely due to non-bound constraints being projected onto the 1d slices.
       > Lastly, bounds for each variable differ greatly.

