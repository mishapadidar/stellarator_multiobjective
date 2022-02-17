## Multi-Objective Optimization of Stellarator Plasma Boundaries

### ToDO
- Untested features and problems
  - Concurrent evaluation test
- Exploratory Analysis
  - Switch to SurfaceRZPseudoSpectral coordinates
  - time evaluations and finite difference.
  - determine smart use of parallel resources (concurrent vs joint evals)
  - make 1d plots to show finite difference fidelity, 
    multi-modality, and simulation failures
  - Select VMEC discretization based on fidelity
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
