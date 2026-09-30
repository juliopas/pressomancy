import numpy as np
import espressomd
from pressomancy.infra import BondWrapper
from pressomancy.object_classes import Filament, Quartet, Quadriplex, RaspberrySphere
from .create_system import sim_inst, BaseTestCase

class FilamentTest(BaseTestCase):

    # positions used here are literal (not lattice-placed); .pos is never folded, so the
    # topology/np.isclose asserts are box-size independent. BUT: the quadriplex layouts in
    # make_stuff() sit at z = 0, 6, 12, 18, which all fold onto the same z = 0 image in this
    # 6^3 box, and quartet_monomers' reals are exactly 6 apart (min-image bond length 0). No
    # assert here reads a folded position or an energy on that geometry -- only
    # test_bond_nearest_part integrates, on the (unaffected, ~0.92) raspberry bonds -- so this
    # box is safe today. Enlarge it first before adding an energy check on the
    # quadriplex/quartet-monomer geometry.
    box_dim = (6, 6, 6)

    part_per_fil = 4
    pos = np.array([[float(iid), 0., 0.] for iid in range(part_per_fil)])
    ori = np.tile(np.array([[1., 0., 0.]]), (part_per_fil, 1))

    def setUp(self) -> None:
        super().setUp()
        self.instance = self._plain_filament()

    def _plain_filament(self):
        """A freshly stored, freshly placed Filament with no associated objects."""
        instance = Filament(config=Filament.config.specify(
            n_parts=self.part_per_fil, espresso_handle=sim_inst.sys))
        sim_inst.store_objects([instance])
        instance.set_object(pos=self.pos, ori=self.ori)
        return instance

    def make_stuff(self):
        quartets = [Quartet(config=Quartet.config.specify(espresso_handle=sim_inst.sys)) for _ in range(3*self.part_per_fil)]
        sim_inst.store_objects(quartets)
        quadriplexes = []
        for start in range(0, len(quartets), 3):
            grouped = quartets[start:start + 3]
            quadriplex = Quadriplex(config=Quadriplex.config.specify(
                associated_objects=grouped, espresso_handle=sim_inst.sys, bonding_mode='ftf'))
            quadriplexes.append(quadriplex)
        sim_inst.store_objects(quadriplexes)
        instance = Filament(config=Filament.config.specify(
            n_parts=self.part_per_fil, espresso_handle=sim_inst.sys, associated_objects=quadriplexes))
        sim_inst.store_objects([instance])
        pos = np.array([[0., 0., 6. * iid] for iid in range(self.part_per_fil)])
        ori = np.tile(np.array([[0., 0., 1.]]), (self.part_per_fil, 1))
        instance.set_object(pos=pos, ori=ori)
        return quadriplexes, instance

    def test_add_anchors(self):
        """add_anchors places one front/back virtual per real particle, bound to it via vs_relative."""
        self.instance.add_anchors('real')
        self.assertEqual(len(self.instance.fronts_indices), self.part_per_fil)
        self.assertEqual(len(self.instance.backs_indices), self.part_per_fil)
        front_handles = list(sim_inst.sys.part.by_ids(self.instance.fronts_indices))
        back_handles = list(sim_inst.sys.part.by_ids(self.instance.backs_indices))
        self.assertEqual({handle.vs_relative[0] for handle in front_handles}, {part.id for part in self.instance.type_part_dict['real']})
        self.assertEqual({handle.vs_relative[0] for handle in back_handles}, {part.id for part in self.instance.type_part_dict['real']})

    def test_bond_anchors(self):
        """bond_anchors and bond_overlapping_virtualz(crit=0.) produce the same front->back bond topology."""
        rows = [
            ('bond_anchors', lambda: self.instance.bond_anchors()),
            ('bond_overlapping_virtualz', lambda: self.instance.bond_overlapping_virtualz(crit=0.)),
        ]
        for label, bond_call in rows:
            with self.subTest(label):
                BaseTestCase.cleanup(self.box_dim)
                self.instance = self._plain_filament()
                self.instance.add_anchors('real')
                bond_call()
                front_handles = list(sim_inst.sys.part.by_ids(self.instance.fronts_indices))
                self.assertEqual([bond[0][1] for bond in [handle.bonds for handle in front_handles[:-1]]], self.instance.backs_indices[1:])
                self.assertEqual(front_handles[-1].bonds, ())

    def test_add_dipole_to_embedded_virt(self):
        """add_dipole_to_embedded_virt places one magnetizable virtual per real particle, bound to it."""
        self.instance.add_dipole_to_embedded_virt(type_name='real', dip_magnitude=2.)
        self.assertEqual(len(self.instance.magnetizable_virts), self.part_per_fil)
        virt_handles = list(sim_inst.sys.part.by_ids(self.instance.magnetizable_virts))
        self.assertEqual({handle.type for handle in virt_handles}, {self.instance.part_types['to_be_magnetized']})
        self.assertEqual({handle.vs_relative[0] for handle in virt_handles}, {part.id for part in self.instance.type_part_dict['real']})

    def test_add_dipole_to_type(self):
        """add_dipole_to_type sets every real particle's dipole moment to the given magnitude."""
        self.instance.add_dipole_to_type('real', dip_magnitude=3.)
        for part in self.instance.type_part_dict['real']:
            self.assertTrue(np.allclose(part.dipm, 3))

    def test_bond_center_to_center(self):
        """bond_center_to_center chains real particles directly, or an associated object's real particles."""
        def plain():
            instance = self._plain_filament()
            instance.bond_center_to_center(type_name='real')
            self.assertEqual([bond[0][1] for bond in [part.bonds for part in instance.type_part_dict['real'][:-1]]], [part.id for part in instance.type_part_dict['real'][1:]])

        def quartet_monomers():
            quartets = [Quartet(config=Quartet.config.specify(espresso_handle=sim_inst.sys)) for _ in range(self.part_per_fil)]
            sim_inst.store_objects(quartets)
            instance = Filament(config=Filament.config.specify(
                n_parts=self.part_per_fil, espresso_handle=sim_inst.sys, associated_objects=quartets))
            sim_inst.store_objects([instance])
            pos = np.array([[0., 0., 6. * iid] for iid in range(self.part_per_fil)])
            ori = np.tile(np.array([[0., 0., 1.]]), (self.part_per_fil, 1))
            instance.set_object(pos=pos, ori=ori)
            instance.bond_center_to_center(type_name='real')
            self.assertEqual([quartet.type_part_dict['real'][0].bonds[0][1] for quartet in quartets[:-1]], [quartet.type_part_dict['real'][0].id for quartet in quartets[1:]])

        for label, row in [('plain', plain), ('quartet monomers', quartet_monomers)]:
            with self.subTest(label):
                BaseTestCase.cleanup(self.box_dim)
                row()

    def test_bond_nearest_part(self):
        """bond_nearest_part bonds each pair of adjacent associated objects through their nearest particles, at zero bonded energy."""
        rasp_sigm = 3
        sample_bond_r0 = 0.83
        spacing = rasp_sigm + sample_bond_r0
        raspberry_equilibrium_r0 = 0.9197313953641069
        raspberries_config = RaspberrySphere.config.specify(
            size=rasp_sigm, espresso_handle=sim_inst.sys)
        raspberries = [RaspberrySphere(config=raspberries_config) for _ in range(self.part_per_fil)]
        sim_inst.store_objects(raspberries)
        bond_hndl = BondWrapper(espressomd.interactions.FeneBond(k=10, d_r_max=3 * rasp_sigm, r_0=raspberry_equilibrium_r0))
        self.instance = Filament(config=Filament.config.specify(
            n_parts=self.part_per_fil,
            espresso_handle=sim_inst.sys,
            bond_handle=bond_hndl,
            associated_objects=raspberries,
        ))
        sim_inst.store_objects([self.instance])
        pos = np.array([[0., 0., spacing * idx] for idx in range(self.part_per_fil)])
        ori = np.tile(np.array([[0., 0., 1.]]), (self.part_per_fil, 1))
        self.instance.set_object(pos=pos, ori=ori)
        self.instance.bond_nearest_part('virt')
        sim_inst.sys.integrator.run(0)
        energy = sim_inst.sys.analysis.energy()

        no_bonds = sum(len(part.bonds) for raspberry in raspberries for part in raspberry.type_part_dict['virt'])
        self.assertEqual(no_bonds, self.part_per_fil - 1)
        self.assertAlmostEqual(energy['bonded'], 0)

    def test_bending_potential(self):
        """add_bending_potential bonds each interior real particle to its neighbours, leaving the ends unbonded."""
        angle_harmonic = espressomd.interactions.AngleHarmonic(bend=1., phi0=3.)
        self.instance.add_bending_potential(type_name='real',bond_handle=angle_harmonic)
        self.assertEqual(self.instance.type_part_dict['real'][0].bonds, ())
        self.assertEqual(self.instance.type_part_dict['real'][-1].bonds, ())
        for iid in range(1, self.part_per_fil - 1):
            self.assertEqual(self.instance.type_part_dict['real'][iid].bonds[0][0], angle_harmonic)
            self.assertEqual(
                self.instance.type_part_dict['real'][iid].bonds[0][1:],
                (self.instance.type_part_dict['real'][iid + 1].id, self.instance.type_part_dict['real'][iid - 1].id),
            )

    def test_bond_quadriplexes(self):
        """bond_quadriplexes adds one hinge bond per monomer gap, or all matching corner pairs (4x as many)."""
        rows = [
            ('hinge', 'hinge', self.part_per_fil - 1),
            ('all', 'all', 4 * (self.part_per_fil - 1)),
        ]
        for label, mode, expected_delta in rows:
            with self.subTest(label):
                BaseTestCase.cleanup(self.box_dim)
                quadriplexes, instance = self.make_stuff()
                pre_bonds = sum(len(corner.bonds) for quadriplex in quadriplexes for quartet in quadriplex.associated_objects for corner in quartet.corner_particles)
                instance.bond_quadriplexes(mode=mode)
                post_bonds = sum(len(corner.bonds) for quadriplex in quadriplexes for quartet in quadriplex.associated_objects for corner in quartet.corner_particles)
                self.assertEqual(post_bonds - pre_bonds, expected_delta)
