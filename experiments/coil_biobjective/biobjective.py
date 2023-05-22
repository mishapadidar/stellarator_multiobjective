import os
from pathlib import Path
import numpy as np
import pickle
import uuid
from scipy.optimize import minimize
import sys
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves, \
    CurveLength, curves_to_vtk, MeanSquaredCurvature, CurveSurfaceDistance, \
    ArclengthVariation
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from simsopt.objectives import SquaredFlux, QuadraticPenalty

"""
Run with simsopt 0.12.2
  mpiexec -n 1 python3 biobjective.py ${constraint_name} ${constraint_target} ${start_type} ${ncoils}
i.e.
  mpiexec -n 1 python3 biobjective.py length 5.0 warm 4

To use the warm start option, run in 'warm_mode' mode.
In warm mode, epsilon constraint problems are solved in sequence, warm
starting one solve from the solution of the previous.

If not using the warm start option, then i suggest submitting the 
jobs to slurm using ./batch_submit_biobjective.sh
"""

warm_mode = False
if warm_mode:
    # no cmd line args
    constraint_name = 'length'
    constraint_target_list = np.linspace(12.0,30.0,36)[::-1]
    #constraint_target_list = np.linspace(12.0,16.5,8)[::-1]
    #constraint_target_list = np.linspace(16.11,25.2,19)
    start_type = "warm"
    ncoils = 4
else:
    # cmd line args
    constraint_name = sys.argv[1] # length or curvature
    constraint_target_list = [float(sys.argv[2])] # float, typically in (5,50)
    ncoils = int(sys.argv[3]) # default is 4
    # only can do cold starts in batch mode
    start_type = "cold"


# penalty options
n_penalty_solves = 3
penalty_gamma = 10
initial_penalty_weight = 0.1

# solver options
maxiter = 30000
gtol = 1e-14
options = {'gtol':gtol, 'maxiter': maxiter, 'maxcor': 300} #'iprint': 5}

# coil-surface distance params
coil_surf_dist_penalty_weight = 1e6
coil_surf_dist_rhs = 0.01

# arc length variation params
arc_length_variation_penalty_weight = 1e-4
#arc_length_variation_penalty_weight = 0.0

# coil params
order = 5 # num fourier modes per coils
current = 1e5
# surface definition
vmec_input = '../../vmec_input_files/input.LandremanPaul2021_QA'
ntheta=nphi=32


##################################
## Done with options
##################################

objective_names = ['Quadratic flux'] + [constraint_name]

# Initialize the boundary magnetic surface:
surf = SurfaceRZFourier.from_vmec_input(vmec_input, range="half period", nphi=nphi, ntheta=ntheta)
R0 = surf.get("rc(0,0)")
R1 = R0/2

print('surf minor radius',surf.minor_radius())

# plot the surface
if not os.path.exists("./output"):
    os.makedirs("./output")
if not os.path.exists("./output/surf_full"):
    quadpoints_phi = np.linspace(0,1,128)
    quadpoints_theta = np.linspace(0,1,128)
    surf_plot = SurfaceRZFourier.from_vmec_input(vmec_input, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    surf_plot.to_vtk("./output/surf_full")
    quadpoints_phi = np.linspace(0,1/surf.nfp/2,128)
    quadpoints_theta = np.linspace(0,1,128)
    surf_plot = SurfaceRZFourier.from_vmec_input(vmec_input, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    surf_plot.to_vtk("./output/surf_half_period")

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, surf.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1.0) * 1e5 for i in range(ncoils)]
base_currents[0].fix_all()
coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, True)
bs = BiotSavart(coils)
bs.set_points(surf.gamma().reshape((-1, 3)))

# write vtk files
#curves = [c.curve for c in coils]
#curves_to_vtk(curves, "coils_init")

# Define the individual terms objective function:
Jflux = SquaredFlux(surf, bs)
Jlength = [CurveLength(c) for c in base_curves]
Jlength_total = sum(Jlength)
Jmsc = [MeanSquaredCurvature(c) for c in base_curves]
Jmsc_total = sum(Jmsc)
# arc length variation
Jalv = [ArclengthVariation(c) for c in base_curves]
Jalv_total = sum(Jalv)
# coil to surface distance
Jcoil_surf_dist = CurveSurfaceDistance(base_curves,surf,coil_surf_dist_rhs)


if constraint_name == "length":
    constraint = Jlength_total
elif constraint_name == 'curvature':
    constraint = Jmsc_total


for li, constraint_target in enumerate(constraint_target_list):

    print("")
    print(f"Solve {li})")
    print("constraint bound:",constraint_target)
    print("initital Bnormal:",Jflux.J())
    print("initial constraint:",constraint.J())
    print("initial constraint violation:",max(constraint.J()-constraint_target,0.0))
    print("initial cs-dist:",Jcoil_surf_dist.shortest_distance())
    print("initial arc length variation:",Jalv_total.J())

    # initial run params
    penalty_weight = initial_penalty_weight

    # penalty solve
    for ii in range(n_penalty_solves):
    
        # penalty objective
        penalty_weight = penalty_weight * penalty_gamma
        JF = Jflux + penalty_weight * QuadraticPenalty(constraint,constraint_target, f='max')\
            + coil_surf_dist_penalty_weight*Jcoil_surf_dist\
            + arc_length_variation_penalty_weight*Jalv_total
        def fun(dofs):
            JF.x = dofs
            return JF.J(), JF.dJ()
      
        # get starting point
        if (li == 0) and (ii == 0):
            x0 = np.copy(JF.x)
            dofs = np.copy(x0)
    
        # solve
        res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
                       options=options, tol=1e-15)
        xopt = res.x

        # print some stuff
        JF.x = xopt
        Fopt = np.array([Jflux.J(),constraint.J()])
        print(f"P{ii} Bnormal:",Jflux.J())
        print(f"P{ii} constraint:",constraint.J())
        print(f"P{ii} constraint violation:",max(constraint.J()-constraint_target,0.0))
        print(f"P{ii} cs-dist:",Jcoil_surf_dist.shortest_distance())
        print(f"P{ii} arc length variation:",Jalv_total.J())
    
    print("final Bnormal:",Jflux.J())
    print("final constraint:",constraint.J())
    print("final constraint violation:",max(constraint.J()-constraint_target,0.0))
    print("final cs-dist:",Jcoil_surf_dist.shortest_distance())
    print("final arc length variation:",Jalv_total.J())

    # Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
    outputdir = f"./output/biobjective/{constraint_name}"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    iden = str(uuid.uuid4())
    filename_body = f"/biobjective_eps_con_{constraint_name}_{constraint_target}_{start_type}_ncoils_{ncoils}"\
                  + f"_{iden}"
    outfilename = outputdir + filename_body + ".pickle"
    outdata = {}
    outdata['xopt'] = xopt
    outdata['constraint_target'] = constraint_target
    outdata['Fopt'] = Fopt
    outdata['start_type'] = start_type
    outdata['ncoils'] = ncoils
    outdata['order'] = order
    outdata['ntheta'] = ntheta
    outdata['nphi'] = nphi
    outdata['objective_names'] = objective_names
    outdata['constraint_name'] = constraint_name
    outdata['coil_surf_dist_penalty_weight'] = coil_surf_dist_penalty_weight
    outdata['arc_length_variation_penalty_weight'] = arc_length_variation_penalty_weight
    outdata['coil_surf_dist_rhs'] = coil_surf_dist_rhs
    outdata['min_coil_surf_dist'] = Jcoil_surf_dist.shortest_distance()
    outdata['total_arc_length_variation'] = Jalv_total.J()
    pickle.dump(outdata, open(outfilename,"wb"))

    # write vtk files
    #curves = [c.curve for c in bs.coils]
    outfilename = outputdir + filename_body
    curves_to_vtk(base_curves, outfilename,close=True)
    
    # set starting point for next solve
    if start_type == "warm":
        JF.x = np.copy(xopt)
        dofs = np.copy(xopt)
    else:
        JF.x = np.copy(x0)
        dofs = np.copy(x0)

