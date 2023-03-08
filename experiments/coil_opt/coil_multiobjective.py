import os
from pathlib import Path
import numpy as np
import pickle
from scipy.optimize import minimize

from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves, \
    CurveLength, curves_to_vtk
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from simsopt.objectives import SquaredFlux, QuadraticPenalty

# initial problem params
length_target_list = np.linspace(5,50,100)
#length_target = 18.0
# optimization params
maxiter = 400
gtol = 1e-6
# coil params
ncoils = 4
order = 5 # num fourier modes per coils
current = 1e5
# surface definition
vmec_input = '../../vmec_input_files/input.LandremanPaul2021_QA'
ntheta=nphi=32

# Initialize the boundary magnetic surface:
surf = SurfaceRZFourier.from_vmec_input(vmec_input, range="half period", nphi=nphi, ntheta=ntheta)
R0 = surf.get("rc(0,0)")
R1 = R0/2
print(R0,R1)
# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, surf.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1.0) * 1e5 for i in range(ncoils)]
base_currents[0].fix_all()
coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, True)
bs = BiotSavart(coils)
bs.set_points(surf.gamma().reshape((-1, 3)))

# Define the individual terms objective function:
Jflux = SquaredFlux(surf, bs)
Jlengths = [CurveLength(c) for c in base_curves]
Jtotal_length = sum(Jlengths)

# define the objective
n_penalty_solves = 4
penalty_gamma = 10

for length_target in length_target_list:

    length_weight = 1.0

    # penalty solve
    for ii in range(n_penalty_solves):
    
        length_weight = length_weight * penalty_gamma
        JF = Jflux + length_weight * QuadraticPenalty(Jtotal_length, length_target, f='max')
        def fun(dofs):
            JF.x = dofs
            return JF.J(), JF.dJ()
    
        dofs = JF.x
        res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
                       options={'gtol':gtol, 'maxiter': maxiter, 'maxcor': 300, 'iprint': 5}, tol=1e-15)
        xopt = res.x
        JF.x = xopt
        Fopt = np.array([Jflux.J(),Jtotal_length.J()])
        print("Fopt",Fopt)
    
    # Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
    outputdir = "./output"
    outfilename = outputdir + f"/coils_eps_con_length_{length_target}.json"
    outdata = {}
    outdata['xopt'] = xopt
    outdata['length_target'] = length_target
    outdata['Fopt'] = Fopt
    pickle.dump(outdata, open(outfilename,"wb"))
    
