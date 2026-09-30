from itertools import product
import numpy as np
import espressomd
from .create_system import sim_inst, BaseTestCase, bonds_using
from pressomancy.object_classes.quadriplex_class import Quartet, Quadriplex

def bonded_energy():
    """The bonded energy after ``integrator.run(0)`` has populated the accumulator."""
    sim_inst.sys.integrator.run(0)
    return sim_inst.sys.analysis.energy()['bonded']

class QuadriplexTest(BaseTestCase):
    box_dim = (6, 6, 6)  # a quartet's virtual sites sit up to 2.955 from their real (5^3 breaks the relation)

    def setUp(self) -> None:
        super().setUp()
        self.instance_ftf=Quadriplex(config=Quadriplex.config.specify(espresso_handle=sim_inst.sys, bonding_mode='ftf'))
        self.instance_ctc=Quadriplex(config=Quadriplex.config.specify(espresso_handle=sim_inst.sys, bonding_mode='ctc'))
        sim_inst.store_objects([self.instance_ftf, self.instance_ctc])

    @staticmethod
    def patched_quadriplex(quartet_types):
        """An ftf Quadriplex of ``quartet_types`` (center, top, bottom) at the origin, every quartet with h-bond patches."""
        quartet_triplet = [Quartet(config=Quartet.config.specify(alias='quartet', type=quartet_type, espresso_handle=sim_inst.sys))
                           for quartet_type in quartet_types]
        quadriplex_config = Quadriplex.config.specify(
            associated_objects=quartet_triplet,
            espresso_handle=sim_inst.sys,
            bonding_mode='ftf',
            size=np.sqrt(3)*5,
        )
        quadriplex=Quadriplex(config=quadriplex_config)
        sim_inst.store_objects([quadriplex,])
        quadriplex.set_object(pos=np.array([0,0,0]),ori=np.array([0,0,1]))
        for quartet in quartet_triplet:
            quartet.add_h_bond_patches()
        return quadriplex

    def test_add_patches_triples(self):
        """One patch on the top and one on the bottom real, each excluding only the other."""
        self.instance_ftf.set_object(
            pos=np.array([0,0,0]),ori=np.array([0,0,1]))
        self.instance_ftf.add_patches_triples()
        patch_parts = self.instance_ftf.type_part_dict['patch']
        self.assertEqual(len(patch_parts), 2)
        top_real = self.instance_ftf.associated_objects[1].type_part_dict['real'][0]
        bottom_real = self.instance_ftf.associated_objects[2].type_part_dict['real'][0]
        owner_ids = {top_real.id, bottom_real.id}
        self.assertEqual({patch.vs_relative[0] for patch in patch_parts}, owner_ids)

        patch_by_owner = {patch.vs_relative[0]: patch for patch in patch_parts}
        self.assertEqual(set(int(part_id) for part_id in patch_by_owner[top_real.id].exclusions), {patch_by_owner[bottom_real.id].id})
        self.assertEqual(set(int(part_id) for part_id in patch_by_owner[bottom_real.id].exclusions), {patch_by_owner[top_real.id].id})

    def test_add_bending_potential(self):
        """One center-top-bottom angle per corner (ftf) or per real (ctc), each at its equilibrium."""

        def asserts():
            for central, top, bottom in zip(center_parts, top_parts, bottom_parts):
                angle_bonds = [bond for bond in central.bonds if bond[0] == int_nhdl]
                self.assertEqual(len(angle_bonds), 1)
                self.assertEqual(angle_bonds[0][1:], (top.id, bottom.id))

        self.assertAlmostEqual(bonded_energy(), 0)
        int_nhdl=espressomd.interactions.AngleHarmonic(bend=1, phi0=np.pi)
        sim_inst.sys.bonded_inter.add(int_nhdl)
        self.instance_ftf.set_object(
            pos=np.array([0,0,0]),ori=np.array([0,0,1]))
        self.instance_ftf.add_bending_potential(bending_potential_handle=int_nhdl)
        self.assertEqual(bonds_using(int_nhdl), 4)
        self.assertAlmostEqual(bonded_energy(), 0)
        center_parts = self.instance_ftf.associated_objects[0].corner_particles
        top_parts = self.instance_ftf.associated_objects[1].corner_particles
        bottom_parts = self.instance_ftf.associated_objects[2].corner_particles
        asserts()
        self.instance_ctc.set_object(
            pos=np.array([10,10,10]),ori=np.array([0,0,1]))
        self.instance_ctc.add_bending_potential(bending_potential_handle=int_nhdl)
        self.assertEqual(bonds_using(int_nhdl), 4 + 1)
        self.assertAlmostEqual(bonded_energy(), 0)
        center_parts = np.atleast_1d(self.instance_ctc.associated_objects[0].type_part_dict['real'][0])
        top_parts = np.atleast_1d(self.instance_ctc.associated_objects[1].type_part_dict['real'][0])
        bottom_parts = np.atleast_1d(self.instance_ctc.associated_objects[2].type_part_dict['real'][0])
        asserts()

    def test_add_dihedrals_and_extra_bendings(self):
        """Each fold adds its exact dihedrals and extra bendings, every one at its equilibrium."""
        folds = {'antiparallel': ['brokenB', 'brokenA', 'brokenA'],
                 'hybrid': ['brokenA', 'brokenB', 'brokenA'],
                 'parallel': ['brokenA', 'brokenA', 'brokenA']}
        for fold, quartet_types in folds.items():
            with self.subTest(fold):
                BaseTestCase.cleanup(self.box_dim)
                quadriplex = self.patched_quadriplex(quartet_types)
                self.assertAlmostEqual(bonded_energy(), 0)
                dihedral = espressomd.interactions.Dihedral(bend=10, mult=1, phase=np.pi/2.)
                sim_inst.sys.bonded_inter.add(dihedral)
                quadriplex.add_dihedrals(dihedral_potential_handle=dihedral)
                self.assertEqual(bonds_using(dihedral), 2 * 4)  # one per src corner, top-center and center-bottom
                self.assertAlmostEqual(bonded_energy(), 0)
                angle_another = espressomd.interactions.AngleHarmonic(bend=10.0, phi0=np.pi/2.)
                sim_inst.sys.bonded_inter.add(angle_another)
                quadriplex.add_extra_bendings(bending_potential_handle=angle_another)
                self.assertEqual(bonds_using(angle_another), 3 * 4 * 2)  # squareB + squareA per corner of all 3 quartets
                self.assertAlmostEqual(bonded_energy(), 0)

    def test_mark_covalent_bonds(self):
        """One corner of the top and of the bottom quartet is marked, none of the central one."""
        self.instance_ftf.set_object(pos=np.array([0,0,0]),ori=np.array([0,0,1]))
        self.instance_ftf.mark_covalent_bonds(part_type=666)
        self.assertEqual([sum(corner.type == 666 for corner in quartet.corner_particles)
                          for quartet in self.instance_ftf.associated_objects], [0, 1, 1])


class QuartetTest(BaseTestCase):
    box_dim = QuadriplexTest.box_dim

    def test_add_h_bond_patches(self):
        """Each corner gets two squareA and two squareB patches, and its squareB pair overlaps exactly one other corner's squareA pair."""
        for quartet_alias,quartet_type in product(['quartet', 'quartet_11x11'],['brokenA', 'brokenB']):
            with self.subTest(f'{quartet_alias} {quartet_type}'):
                BaseTestCase.cleanup(self.box_dim)
                quartet = Quartet(config=Quartet.config.specify(espresso_handle=sim_inst.sys, type=quartet_type, alias=quartet_alias))
                sim_inst.store_objects([quartet])
                quartet.set_object(pos=np.array([0, 0, 0]), ori=np.array([0, 0, 1]))
                quartet.add_h_bond_patches()
                parts, _ = quartet.get_owned_part()
                square_a_type = quartet.part_types['squareA']
                square_b_type = quartet.part_types['squareB']

                patch_map = {}
                for corner in quartet.corner_particles:
                    related = [part for part in parts if part.vs_relative[0] == corner.id]
                    square_a_parts = [part for part in related if part.type == square_a_type]
                    square_b_parts = [part for part in related if part.type == square_b_type]
                    self.assertEqual(len(square_a_parts), 2)
                    self.assertEqual(len(square_b_parts), 2)
                    patch_map[corner.id] = {'squareA': square_a_parts, 'squareB': square_b_parts}

                for left_corner in quartet.corner_particles:
                    partner_count = 0
                    for right_corner in quartet.corner_particles:
                        if right_corner.id == left_corner.id:
                            continue
                        overlaps = sum(
                            np.isclose(
                                np.linalg.norm(np.array(left_patch.pos) - np.array(right_patch.pos)),
                                0.0,
                                atol=1e-8,
                            )
                            for left_patch in patch_map[left_corner.id]['squareB']
                            for right_patch in patch_map[right_corner.id]['squareA']
                        )
                        if overlaps == 2:
                            partner_count += 1
                    self.assertEqual(partner_count, 1, msg=f"corner {left_corner.id}")

    def test_object_contracts(self):
        """Every particle has its declared type and exclusion count, and part_types tracks exactly the owned particles."""
        for quartet_alias, (quartet_type, pos) in product(['quartet', 'quartet_11x11'], zip(['solid', 'brokenA', 'brokenB'], [(0, 0, 0), (10, 10, 10), (20, 20, 20)])):
            with self.subTest(f'{quartet_alias} {quartet_type}'):
                BaseTestCase.cleanup(self.box_dim)
                quartet = Quartet(config=Quartet.config.specify(espresso_handle=sim_inst.sys, type=quartet_type, alias=quartet_alias))
                sim_inst.store_objects([quartet])
                quartet.set_object(pos=pos, ori=np.array([0, 0, 1]))
                parts, _ = quartet.get_owned_part()
                tracked_ids = set()

                if quartet.params['type'] == 'solid':
                    expected_no_excl = len(parts) - 1
                else:
                    recipe = quartet.recipe_dictA if quartet.params['type'] == 'brokenA' else quartet.recipe_dictB
                    expected_no_excl = len(next(iter(recipe['assoc'].values())))

                for type_name, expected_type in quartet.part_types.items():
                    self.assertIn(type_name, sim_inst.part_types)
                    self.assertEqual(sim_inst.part_types[type_name], expected_type)
                    for part in quartet.type_part_dict.get(type_name, []):
                        tracked_ids.add(part.id)
                        self.assertEqual(part.type, expected_type)
                        core_part = sim_inst.sys.part.by_id(part.id)
                        self.assertEqual(core_part.type, expected_type)
                        expected_exclusions = 0 if type_name == 'cation' else expected_no_excl
                        self.assertEqual(len(core_part.exclusions), expected_exclusions, msg=type_name)

                self.assertEqual(tracked_ids, {part.id for part in parts})
