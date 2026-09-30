'''
poly-BRACO sample: quartets -> quadriplexes -> filaments, plus crowders bonded into
filaments of their own, with steric, patch and thermostat setup, an HDF5 trajectory
of the Filament group (bonds included) and a rewind of the running system to a
saved frame through ``load_from_src``.

The rewind is the "existing tree" recipe: the objects are already built and placed,
so ``load_from_src`` is called without ``place_from`` and only copies the stored
state (here every particle position of the four owned types) onto them. Placing
from the file is not possible for these objects: a filament here is made of
quadriplexes, and no stored particle sits at a monomer centre. ``bonds`` stays
``False`` because every object built its own bonds; the file's link count is only
compared against the live one.
'''
from pressomancy.simulation import Simulation, Crowder, Filament, Quartet, Quadriplex
from pressomancy.infra import BondWrapper
import espressomd
import h5py
import numpy as np
import logging
import os
import tempfile
N_avog = 6.02214076e23

sigma = 1.
rho_si = 0.6*N_avog
no_obj=30
N = int(no_obj/3)
vol = N/rho_si
box_l = pow(vol, 1/3)
_box_l = box_l/0.4e-09
box_dim = _box_l*np.ones(3)
_rho = N/pow(_box_l, 3)

sheets_per_quad = 3
part_per_filament = 2
no_crowders=10
part_per_ligand=2

sim_inst = Simulation(box_dim=box_dim)
sim_inst.set_sys()
logging.info(f'box_dim: {sim_inst.sys.box_l}')

quartet_configuration = Quartet.config.specify(espresso_handle=sim_inst.sys, type='solid')
quartets = [Quartet(config=quartet_configuration) for x in range(no_obj)]
sim_inst.store_objects(quartets)

bond_quad = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2*1.5))
grouped_quartets = [quartets[i:i+sheets_per_quad]
                    for i in range(0, len(quartets), sheets_per_quad)]
quadriplex_configuration_list = [Quadriplex.config.specify(size=np.sqrt(3)*5., espresso_handle=sim_inst.sys, bond_handle=bond_quad, associated_objects=elem) for elem in grouped_quartets]

quadriplex = [Quadriplex(config=configuration) for configuration in quadriplex_configuration_list]
sim_inst.store_objects(quadriplex)

bond_pass = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2*1.5))
grouped_quadriplexes = [quadriplex[i:i+part_per_filament:]
                        for i in range(0, len(quadriplex), part_per_filament)]
filament_configuration_list = [Filament.config.specify(size=quadriplex[0].params['size']*part_per_filament+np.sqrt(3)*bond_pass.r_0+(part_per_filament-1), n_parts=part_per_filament, espresso_handle=sim_inst.sys, bond_handle=bond_pass, associated_objects=elem, spacing=6.) for elem in grouped_quadriplexes]
all_filaments=[]
filaments = [Filament(config=configuration) for configuration in filament_configuration_list]
all_filaments.extend(filaments)
sim_inst.store_objects(filaments)
sim_inst.set_objects(filaments)


for filament in filaments:        
    filament.bond_quadriplexes()

sim_inst.sys.integrator.run(0)
# Crowder has no sigma: the crowders' WCA sigma is the one `set_steric` sets below (1 here), not the 6 of their size.
crowder_configuration=Crowder.config.specify(size=6., espresso_handle=sim_inst.sys)
crowders = [Crowder(config=crowder_configuration)
            for x in range(no_crowders)]
sim_inst.store_objects(crowders)
grouped_crowders = [crowders[i:i+part_per_ligand]
                for i in range(0, len(crowders), part_per_ligand)]

bender_pass = BondWrapper(espressomd.interactions.FeneBond(
    k=10, r_0=6, d_r_max=6*1.5))
filament_configuration_list = [Filament.config.specify(sigma=6,size=6*part_per_ligand, n_parts=part_per_ligand, espresso_handle=sim_inst.sys, bond_handle=bender_pass, associated_objects=elem) for elem in grouped_crowders]

crowder_filaments = [Filament(config=elem) for elem in filament_configuration_list]
all_filaments.extend(crowder_filaments)
sim_inst.store_objects(crowder_filaments)
sim_inst.set_objects(crowder_filaments)

for filament in crowder_filaments:
    filament.bond_center_to_center(type_name='crowder')

sim_inst.set_steric(key=('real', 'virt','crowder'), wca_eps=1.)

for el in quadriplex:
    el.add_patches_triples()
sim_inst.set_vdW(key=('patch',), lj_eps=5, lj_sigma=2.)
sim_inst.set_vdW_custom(pairs=[('patch','crowder'),], lj_eps=[5.,], lj_sigma=[1.,])
sim_inst.sys.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=sim_inst.seed)


# --- HDF5: write two frames, integrate on, rewind to the first one -----------------
# Every owned type is restored from the frame with one grouped key: one property
# list shared by the four (source, local) type pairs.
POLY_BRACO_STATE = {(('real', 'real'), ('virt', 'virt'),
                     ('patch', 'patch'), ('crowder', 'crowder')): [('pos', 'pos')]}

with tempfile.TemporaryDirectory() as tmpdirname:
    path = os.path.join(tmpdirname, "poly_BRACO.h5")
    sim_inst.io_dict['bonds'] = True
    sim_inst.inscribe_part_group_to_h5(group_type=[Filament], h5_data_path=path, mode='NEW')
    sim_inst.write_part_group_to_h5(step=0)
    checkpoint_pos = sim_inst.sys.part.all().pos.copy()

    sim_inst.sys.integrator.run(1)
    sim_inst.write_part_group_to_h5(step=1)
    sim_inst.sys.integrator.run(5)
    assert not np.allclose(sim_inst.sys.part.all().pos, checkpoint_pos), \
        'the system did not move, so the rewind below proves nothing'
    sim_inst.io_dict['h5_file'].close()
    sim_inst.io_dict['h5_file'] = None

    # Rewind the running system to the first saved frame.
    sim_inst.load_from_src(all_filaments, path, src_to_loc=POLY_BRACO_STATE, step=0)
    assert np.allclose(sim_inst.sys.part.all().pos, checkpoint_pos, rtol=1e-05, atol=1e-08), \
        'positions did not come back from the saved frame'

    with h5py.File(path, 'r') as src_file:
        stored_links = int(src_file['connectivity/Filament/bonds'].attrs['n_links'])
    live_links = sum(len(part.bonds) for part in sim_inst.sys.part.all())
    assert live_links == stored_links, \
        f'the live topology has {live_links} links, the file stored {stored_links}'
