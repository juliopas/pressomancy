import espressomd
import espressomd.version
import numpy as np

from pressomancy.helper_functions import api_agnostic_feature_check
from pressomancy.magnetodynamics import required_features_for
from pressomancy.simulation import Simulation, PointDipolePermanent, PointDipoleMagnetizable

if espressomd.version.major() == 5:
    from espressomd.magnetostatics import DipolarDirectSum
elif espressomd.version.major() == 4:
    from espressomd.magnetostatics import DipolarDirectSumCpu
else:
    raise RuntimeError(f"Unsupported ESPResSo version {espressomd.version}")

MODEL = 'langevin'
HAS_MAGNETIZABLE_FEATURES = all(
    api_agnostic_feature_check(feature)
    for feature in required_features_for(MODEL)
)

box_l = [10, 10, 10]
H = 1

HM_dipm = 1.
SM_dipm = 1.
SM_Xi_0 = 0.1

sim_inst = Simulation(box_dim=box_l)
sim_inst.set_sys(timestep=0.1)
sim_inst.sys.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=sim_inst.seed)

pos = np.array([[0., 0., 0.],
                [0., 0., 1.]])
dip = np.array([[0., 0., 1.],
                [0., 0., 1.]])

config_pdp = PointDipolePermanent.config.specify(dipm=HM_dipm, espresso_handle=sim_inst.sys)

# Test two permanent point dipoles aligned in the same direction as the field.
pdp_list = [PointDipolePermanent(config=config_pdp) for _ in range(2)]
sim_inst.store_objects(pdp_list)
sim_inst.set_objects(pdp_list)

for i, part in enumerate(sim_inst.sys.part.select(type=sim_inst.part_types["pdp_real"])):
    part.pos = pos[i]
    part.dip = dip[i]
    part.fix = [True, True, True]

sim_inst.sys.integrator.run(0)
sim_inst.set_H_ext(H=[0, 0, H])

assert np.array_equal(sim_inst.sys.part.all().pos, pos), f"{sim_inst.sys.part.all().pos},\n{pos}"
assert np.array_equal(sim_inst.sys.part.all().dip, dip), f"{sim_inst.sys.part.all().dip},\n{dip}"
sim_inst.reinitialize_instance()
sim_inst.set_sys(timestep=0.0001)

if HAS_MAGNETIZABLE_FEATURES:
    config_pdm = PointDipoleMagnetizable.config.specify(
        magnetization_model=MODEL,
        dipm_sat=SM_dipm,
        mag_susc_0=SM_Xi_0,
        espresso_handle=sim_inst.sys,
    )

    # Test two magnetizable point dipoles aligned in the same direction as the field.
    pdm_list = [PointDipoleMagnetizable(config=config_pdm) for _ in range(2)]
    sim_inst.store_objects(pdm_list)
    sim_inst.set_objects(pdm_list)

    for i, part in enumerate(sim_inst.sys.part.select(type=sim_inst.part_types["pdm_virt"])):
        part_real = sim_inst.sys.part.by_id(part.vs_relative[0])
        part_real.pos = pos[i]
        part.pos = pos[i]
        part_real.fix = [True, True, True]

    sim_inst.sys.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=sim_inst.seed)
    if espressomd.version.major() == 5:
        sim_inst.init_magnetic_inter(DipolarDirectSum(prefactor=1))
    else:
        sim_inst.init_magnetic_inter(DipolarDirectSumCpu(prefactor=1))

    sim_inst.set_H_ext(H=[0, 0, H])
    sim_inst.sys.integrator.run(10)

    virts = sim_inst.sys.part.select(type=sim_inst.part_types["pdm_virt"])

    def langevin_moment(field_magnitude):
        '''m_sat * L(alpha), alpha = 3 * chi_0 * |H| / m_sat, as espresso defines it.'''
        alpha = 3. * SM_Xi_0 * field_magnitude / SM_dipm
        return SM_dipm * (1. / np.tanh(alpha) - 1. / alpha)

    # Each moment must be the Langevin response to the total field it actually sees.
    H_tot = np.linalg.norm(np.asarray(virts.dip_fld) + np.asarray([0, 0, H]), axis=1)
    expected_dipm = langevin_moment(H_tot)

    assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pdm_real"]).pos, pos), f"{sim_inst.sys.part.select(type=sim_inst.part_types['pdm_real']).pos},\n{pos}"
    assert np.array_equal(virts.pos, pos), f"{virts.pos},\n{pos}"
    assert np.array_equal(sim_inst.sys.part.select(type=sim_inst.part_types["pdm_real"]).dip, dip * 0.), f"{sim_inst.sys.part.select(type=sim_inst.part_types['pdm_real']).dip},\n{dip * 0.}"
    assert np.allclose(virts.dipm, expected_dipm, rtol=1e-6), f"{virts.dipm},\n{expected_dipm}"
    # the moments align with the field, which points along z
    assert np.allclose(np.asarray(virts.dip)[:, :2], 0.), f"{virts.dip}"

    poss_for_next_test = virts.pos
    dips_for_next_test = virts.dip

    sim_inst.reinitialize_instance()
