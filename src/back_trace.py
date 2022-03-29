#!/usr/bin/python
# The script performs backtracing of cosmic rays in galactic magnetic field assuming
# 5 different magnetic field models
# The file containing the observed points list is passed to the script as its first parameter
# The input file should have two columns: galactic coordinates l and b in degrees for each observation
# Cosmic ray particle spectrum and mass composition as well as magnetic field models' parameters are hardcoded
# below. Change them as needed

import sys
from crpropa import *
import numpy as np

# loading observed cosmic ray arrival directions
observed_points = np.loadtxt(sys.argv[1])[:,:2]

Z=1  # cosmic ray paricle charge
A=1  # cosmic ray particle atomic mass

PID = -nucleusId(A,Z)  # CRPropa particle type id

Emin = 25.0 # minimal cosmic ray energy in EeV
Emax = 300.0 # maximal cosmic ray energy in EeV

spec_alpha = -11  # cosmic ray spectrum power (dN/dE ~ E^{spec_alpha})  set to -10 for E=Emin
# values of spec_alpha<=-10 are treated as monochromatic spectrum with E=Emin
# values of spec_alpha>=10 are treated as monochromatic spectrum with E=Emax

# sample particle energy
def sample_E(count):
    if spec_alpha >= 10:
        return np.full(count, Emax)
    if spec_alpha <= -10:
        return np.full(count, Emin)
    r = np.random.rand(count)
    assert spec_alpha != -1, 'not implemented yet'
    E = np.power(r * np.power(Emax, 1+spec_alpha) + (1-r)*np.power(Emin, 1+spec_alpha), 1./(1.+spec_alpha))
    return E


# _____________________________________________________________________
# The main parameter that defines how "precise" is
# the grid.
# Maximum step in tracking a nucleus, parsec
max_step = 25
tolerance = 1e-4

# Model of the Galactic Magnetic Field
# ------------------------------------
# Terral, Ferriere 2017 - Constraints from Faraday rotation on the magnetic
# field structure in the galactic halo, DOI: 10.1051/0004-6361/201629572,
# arXiv:1611.10222
# GMF = "TF17"

# Pshirkov, Tinyakov, Kronberg, Newton-McGee, ApJ 2011:
# NB: check ASS, BSS, Halo below in the code!
#GMF = "PTKN11"

# Jasson, Farrar, 2012:
#GMF = "JF12"
# In case of striated and/or turbulent components:
striated = 1
turbulent = 1

# JF + solenoidal improvements: https://arxiv.org/abs/1809.07528
# Here I assume striated = turbulent = 1.  The model has two more
# parameters, I keep the default values:
# JF12FieldSolenoidal (double delta=3 *kpc, double zs=0.5 *kpc)
#GMF = "JF12sol"
# Two parameters of the JF12sol model, kpc.
# 3 and 0.5 are the defaults
delta_sol = 3
zs_sol = 0.5

# Jasson, Farrar, 2012, modified by the Planck Collab., 2016
# Striated and turbulent components are on.
#GMF = "JF12Planck"

random_seed = 2**23     # only relevant for the JF models with ST

# Radius of the Milky Way, kpc
galaxy_radius = 20

# _____________________________________________________________________

def calc_deflections(name_GMF):
    seed_text = ', random seed=' + str(random_seed)
    if name_GMF == "JF12":
        B = JF12Field()

        if striated:
            B.randomStriated(random_seed)
            name_GMF = name_GMF + 'S'
        if turbulent:
            B.randomTurbulent(random_seed)
            name_GMF = name_GMF + 'T'

    elif name_GMF == "JF12sol":
        B = JF12FieldSolenoidal(delta_sol * kpc, zs_sol * kpc)

        # Basename of an output file

        B.randomStriated(random_seed)
        B.randomTurbulent(random_seed)

    elif name_GMF == "JF12Planck":
        B = PlanckJF12bField()

        # Basename of an output file

        B.randomStriated(random_seed)
        B.randomTurbulent(random_seed)

    elif name_GMF== 'PTKN11':
        B = PT11Field()
        #B.setUseASS(True)
        B.setUseBSS(True)
        B.setUseHalo(True)
        seed_text = ''

    elif name_GMF == "TF17":
        B = TF17Field()
        seed_text = ''

    else:
        raise ValueError('Unknown GMF!')

    # _____________________________________________________________________
    # Backtracking

    # Convert energy to EeV (CRPropa)
    energies = sample_E(observed_points.shape[0])

    # Position of the observer
    position = Vector3d(-8.5, 0, 0) * kpc


    backtracking_results = np.zeros((observed_points.shape[0], 7)) # Z, E, initial and final coords + deflection angle

    # The main cycle over all points in input_file
    for i, (l, b) in enumerate(observed_points):
        energy = energies[i] * EeV
        theta_ini = np.deg2rad(90-b)  # convert to colatitude
        phi_ini = np.deg2rad(l)

        assert phi_ini <= np.pi

        # Simulation setup
        sim = ModuleList()
        sim.add(PropagationCK(B, tolerance, 0.01 * parsec, max_step * parsec))
        obs = Observer()
        # Detects particles upon exiting a sphere
        # NB: Size of the Milky Way
        # https://www.space.com/29270-milky-way-size-larger-than-thought.html
        # https://arxiv.org/abs/1503.00257
        obs.add(ObserverSurface(Sphere(Vector3d(0.), galaxy_radius * kpc)))
        sim.add(obs)

        # Assign initial direction (detected on Earth) and "shoot"
        direction = Vector3d()
        direction.setRThetaPhi(1, theta_ini, phi_ini)
        p = ParticleState(PID, energy, position, direction)
        c = Candidate(p)
        sim.run(c)

        # Obtain direction at the Milky Way "border"
        d1 = c.current.getDirection()
        theta_res = d1.getTheta()
        phi_res = d1.getPhi()
        deflection = direction.getAngleTo(d1)

        # Convert coordinates and deflections to degrees:
        theta_res_deg,phi_res_deg,deflection_deg \
        = np.rad2deg([theta_res,phi_res,deflection])

        # Here we convert colatitudes to latitudes: 90-theta_ini if needed
        backtracking_results[i,:] = [Z, energy/EeV, l, b, phi_res_deg, 90 - theta_res_deg, deflection_deg]


    output_file = sys.argv[1] + "_" + name_GMF
    output_header = 'Z\tE/EeV\tl_observed\tb_observed\tl_ini\tb_ini\tdeflection' + seed_text
    np.savetxt(output_file, backtracking_results, fmt='%11.5f',
            header=output_header, comments='#')


for mf in ["JF12Planck", "JF12sol", "JF12", "PTKN11", "TF17"]:
    calc_deflections(mf)
