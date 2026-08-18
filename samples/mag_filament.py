import espressomd
from pressomancy.helper_functions import BondWrapper, api_agnostic_feature_check
from pressomancy.magnetodynamics import required_features_for, susceptibility_from_kT
from pressomancy.simulation import Simulation, Filament
import numpy as np
import logging
if espressomd.version.major()==5:
    from espressomd.magnetostatics import DipolarDirectSum
elif espressomd.version.major()==4:
    from espressomd.magnetostatics import DipolarDirectSumCpu
else:
    raise ImportError(f"Unsupported ESPResSo version: {espressomd.version}. Please use version 4 or 5.")
from espressomd.io.writer import vtf

sigma = 1.
n_fil = 10
density=0.1
box_dim = [10,10,10]*np.ones(3)
logging.info('box_dim: ', box_dim)
sim_inst = Simulation(box_dim=box_dim)
sim_inst.set_sys()
bond_hndl=BondWrapper(espressomd.interactions.FeneBond(k=10, d_r_max=3*sigma, r_0=0))
configuration=Filament.config.specify(sigma=sigma, size=2.26,n_parts=2, espresso_handle=sim_inst.sys,bond_handle=bond_hndl)
filaments = [Filament(config=configuration) for x in range(n_fil)]

sim_inst.store_objects(filaments)
sim_inst.set_objects(filaments)
for filament in filaments:
    filament.add_anchors(type_name='real')
    filament.bond_overlapping_virtualz(crit=0.13)
    filament.add_dipole_to_embedded_virt(type_name='real',dip_magnitude=1.)

sim_inst.set_vdW(key=('real',),lj_eps=3.)

kT = 1.0
sim_inst.sys.thermostat.set_langevin(kT=kT, gamma=1.0, seed=sim_inst.seed)
sim_inst.set_H_ext()
sim_inst.set_H_ext(H=(0,0,6.66))

pats_to_magnetize=sim_inst.sys.part.select(lambda p:p.type==sim_inst.part_types['to_be_magnetized'])
sim_inst.sys.integrator.run(0)
sim_inst.avoid_explosion(F_TOL=1e-2)
if espressomd.version.major()==5:
    sim_inst.init_magnetic_inter(DipolarDirectSum(prefactor=1))
else:
    sim_inst.init_magnetic_inter(DipolarDirectSumCpu(prefactor=1))

DIPM_SAT = 1.732
MAG_SUSC_0 = susceptibility_from_kT(DIPM_SAT, kT)
if all(api_agnostic_feature_check(f) for f in required_features_for('langevin')):
    sim_inst.set_magnetization_model(pats_to_magnetize, 'langevin',
                                     dipm_sat=DIPM_SAT,
                                     mag_susc_0=MAG_SUSC_0)
    # Check that the explicit mutual magnetization actually converges here. A
    # ratio below 1 means the one iterate per timestep scheme resolves the
    # physics; see pressomancy.magnetodynamics.contraction_ratio for the
    # saturation caveat that goes with this diagnostic.
    ratios = sim_inst.probe_magnetization_convergence(pats_to_magnetize, n_iter=20)
    if len(ratios):
        logging.info(f'mutual magnetization contraction ratio: {ratios[-1]:.3g}')
        assert ratios[-1] < 1., (
            f'magnetization iteration is not contracting (ratio {ratios[-1]:.3g}); '
            f'lower mag_susc_0 or the dipolar prefactor')
sim_inst.sys.integrator.run(1)
