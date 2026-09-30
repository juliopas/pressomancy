import pressomancy.object_classes
import inspect
import itertools
import warnings
from unittest import mock
import numpy as np
from espressomd.constraints import HomogeneousMagneticField
from .create_system import sim_inst, BaseTestCase
from pressomancy.infra import MissingFeature, api_agnostic_feature_check
from pressomancy.object_classes import Crowder, Elastomer, Filament, Quadriplex
from pressomancy.geometry import WCA_CONTACT_FACTOR, min_img_dist


def crowders(n, size=2, **params):
    """``n`` Crowders of ``size``."""
    return [Crowder(config=Crowder.config.specify(size=size, espresso_handle=sim_inst.sys, **params))
            for _ in range(n)]


def filament(**params):
    """A Filament of ``params``; the notice that it infers its monomer size is silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Filament(config=Filament.config.specify(espresso_handle=sim_inst.sys, **params))


# Simulate not compiled features
def missing_feature(name):
    """Patch the feature check that `pressomancy.simulation` calls to report ``name`` (None: nothing) absent."""
    return mock.patch('pressomancy.simulation.api_agnostic_feature_check',
                      lambda feature: feature != name and api_agnostic_feature_check(feature))


class SimulationTest(BaseTestCase):
    box_dim = (16, 16, 16)  # every default config fits; the default Elastomer fills it with 567 particles

    def test_every_class_stores_sets_and_deletes(self):
        """Each object class's default config registers its part_types, creates particles and deletes them all."""
        classes = [member for _, member in inspect.getmembers(pressomancy.object_classes, inspect.isclass)]
        exercised = []
        for cls in classes:
            with self.subTest(cls.__name__):
                BaseTestCase.cleanup(self.box_dim)  # a failed row must not hand its particles on
                try:
                    instance = [cls(config=cls.config.specify(espresso_handle=sim_inst.sys))]
                    sim_inst.store_objects(instance)
                    sim_inst.set_objects(instance)
                except MissingFeature as excp:
                    self.skipTest(str(excp))
                self.assertLessEqual(cls.part_types.items(), sim_inst.part_types.items())
                self.assertGreater(len(sim_inst.sys.part), 0)
                exercised.append(cls.__name__)
                instance[0].delete_owned_parts()
                self.assertEqual(len(sim_inst.sys.part), 0)
        self.assertGreater(len(exercised), 0, "every object class was skipped")

    def test_assignment_reaches_the_wrapped_simulation(self):
        """An attribute set on the singleton lands on the Simulation it wraps."""
        old_seed = sim_inst.seed
        sim_inst.seed = old_seed + 1
        self.assertEqual(sim_inst.instance.seed, old_seed + 1)

    def test_modify_system_attribute_refuses_anything_unlisted(self):
        """The permission miss used to be a silent no-op, so an object that asked
        for the wrong attribute went on as if it had been granted."""
        requester = object()
        recorded = []
        sim_inst.modify_system_attribute(requester, 'part_types', recorded.append)
        self.assertEqual(recorded, [sim_inst.part_types])
        for attribute in ('objects', 'no_such_attribute'):
            with self.subTest(attribute):
                with self.assertRaisesRegex(PermissionError, f"'{attribute}'"):
                    sim_inst.modify_system_attribute(requester, attribute, recorded.append)
        self.assertEqual(len(recorded), 1)  # a refused call runs nothing

    def test_store_objects_guards(self):
        """Duplicates, half-stored monomers and a missing feature are refused and store nothing;
        unstored monomers are stored along."""
        def twice():
            one = crowders(1)
            return [], one * 2

        def half_stored():
            monomers = crowders(2)
            return monomers[:1], [filament(size=4, n_parts=2, associated_objects=monomers)]

        def unstored_monomers():
            return [], [filament(size=4, n_parts=2, associated_objects=crowders(2))]

        feature = Crowder.required_features[0]
        filament_only = next(f for f in Filament.required_features if f not in Crowder.required_features)
        rows = [  # (label, build -> (stored first, then refused), feature reported missing, exception, fragment)
            ('the same object twice', twice, None, ValueError, 'common elements'),
            ('monomers partly stored', half_stored, None, ValueError, 'not all associated'),
            (f'{feature} missing', lambda: ([], crowders(1)), feature, MissingFeature,
             f'Missing required features: {feature}'),
            (f'{filament_only} missing after the monomers', unstored_monomers, filament_only, MissingFeature,
             f'Missing required features: {filament_only}'),
        ]
        for label, build, missing, exception, fragment in rows:
            with self.subTest(label):
                BaseTestCase.cleanup(self.box_dim)
                stored, refused = build()
                sim_inst.store_objects(stored)
                before = (list(sim_inst.objects), sim_inst.no_objects, dict(sim_inst.part_types))
                with missing_feature(missing), self.assertRaisesRegex(exception, fragment):
                    sim_inst.store_objects(refused)
                self.assertEqual((sim_inst.objects, sim_inst.no_objects, dict(sim_inst.part_types)), before)

        with self.subTest('monomers not stored'):
            monomers = crowders(2)
            chain = filament(size=4, n_parts=2, associated_objects=monomers)
            stored = len(sim_inst.objects)
            sim_inst.store_objects([chain])
            self.assertEqual(sim_inst.objects[stored:], monomers + [chain])
            self.assertEqual(sim_inst.no_objects, len(sim_inst.objects))

    def test_place_objects_refuses_mismatched_lengths(self):
        """Mismatched counts raise before anything is placed; zip used to place the first 2 of 3."""
        objects = crowders(3)
        sim_inst.store_objects(objects)
        positions = [[4., 4., 4.], [8., 8., 8.], [12., 12., 12.]]
        rows = [  # (label, positions, orientations, fragment)
            ('2 positions', positions[:2], None, '3 objects, 2 positions'),
            ('2 orientations', positions, [[0., 0., 1.]] * 2, '3 objects, 3 positions and 2 orientations'),
        ]
        for label, row_positions, orientations, fragment in rows:
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, fragment):
                    sim_inst.place_objects(objects, row_positions, orientations)
                self.assertEqual(len(sim_inst.sys.part), 0)

    def test_set_H_ext_replaces_and_get_H_ext_sums(self):
        """set_H_ext leaves one homogeneous field, the latest; get_H_ext sums them all (zeros for none)."""
        np.testing.assert_array_equal(sim_inst.get_H_ext(), np.zeros(3))
        sim_inst.set_H_ext((0., 0., 1.))
        sim_inst.set_H_ext((1., 0., 0.))
        fields = [c for c in sim_inst.sys.constraints if isinstance(c, HomogeneousMagneticField)]
        self.assertEqual([list(field.H) for field in fields], [[1., 0., 0.]])
        sim_inst.sys.constraints.add(HomogeneousMagneticField(H=[0., 2., 0.]))
        np.testing.assert_array_equal(sim_inst.get_H_ext(), [1., 2., 0.])

    def test_interaction_setters(self):
        """set_steric/set_vdW set every pair of the key, self-pairs too; the custom setters only the listed pairs."""
        sim_inst.part_types.update({'a': 1, 'b': 2})
        inter = sim_inst.sys.non_bonded_inter
        pairs = [('a', 'a'), ('a', 'b'), ('b', 'b')]

        def wca(eps, sigma):
            return {'epsilon': eps, 'sigma': sigma}

        def lj(eps, sigma, cutoff, r_min=0.):
            return {'epsilon': eps, 'sigma': sigma, 'cutoff': cutoff, 'shift': 0., 'offset': 0., 'min': r_min}

        unset = {'wca': wca(0., 0.), 'lennard_jones': lj(0., 0., -1.)}
        rows = [  # (label, call, interaction, {pair: params}; the pairs not listed stay unset)
            ('set_steric', lambda: sim_inst.set_steric(key=('a', 'b'), wca_eps=2., sigma=1.5),
             'wca', {pair: wca(2., 1.5) for pair in pairs}),
            ('set_vdW', lambda: sim_inst.set_vdW(key=('a', 'b'), lj_eps=2., lj_sigma=1.5),
             'lennard_jones', {pair: lj(2., 1.5, 2.5 * 1.5) for pair in pairs}),
            ('set_steric_custom', lambda: sim_inst.set_steric_custom(pairs=[('a', 'b')], wca_eps=[3.], sigma=[0.5]),
             'wca', {('a', 'b'): wca(3., 0.5)}),
            ('set_vdW_custom', lambda: sim_inst.set_vdW_custom(pairs=[('a', 'b')], lj_eps=[3.], lj_sigma=[0.5],
                                                               lj_cutoffs=[1.], r_min=0.25),
             'lennard_jones', {('a', 'b'): lj(3., 0.5, 1., 0.25)}),
            ('set_vdW_custom, default cutoff', lambda: sim_inst.set_vdW_custom(pairs=[('a', 'b')], lj_eps=[3.],
                                                                               lj_sigma=[0.5]),
             'lennard_jones', {('a', 'b'): lj(3., 0.5, 2.5 * 0.5)}),
        ]
        for label, call, interaction, expected in rows:
            with self.subTest(label):
                inter.reset()
                call()
                self.assertEqual(
                    {pair: getattr(inter[sim_inst.part_types[pair[0]], sim_inst.part_types[pair[1]]],
                                   interaction).get_params() for pair in pairs},
                    {pair: expected.get(pair, unset[interaction]) for pair in pairs})

        two = [('a', 'a'), ('a', 'b')]
        refused = [  # (label, call, fragment)
            ('set_steric_custom: 1 sigma for 2 pairs',
             lambda: sim_inst.set_steric_custom(pairs=two, wca_eps=[1., 1.], sigma=[1.]), 'epsilon and sigma'),
            ('set_vdW_custom: 1 epsilon for 2 pairs',
             lambda: sim_inst.set_vdW_custom(pairs=two, lj_eps=[1.], lj_sigma=[1., 1.]), 'epsilon and sigma'),
            ('set_vdW_custom: 1 cutoff for 2 pairs',
             lambda: sim_inst.set_vdW_custom(pairs=two, lj_eps=[1., 1.], lj_sigma=[1., 1.], lj_cutoffs=[2.5]),
             'cutoffs'),
        ]
        for label, call, fragment in refused:
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, fragment):
                    call()

    def test_avoid_explosion_restores_the_integrator(self):
        """Two overlapping WCA particles are pushed apart, and force_cap and time_step come back."""
        system = sim_inst.sys
        # cleanup() resets neither: a failed restore must not leak a smaller step or a force cap
        self.addCleanup(sim_inst.set_sys, min_global_cut=BaseTestCase.min_global_cut)
        self.addCleanup(setattr, system, 'force_cap', 0)
        system.non_bonded_inter[0, 0].wca.set_params(epsilon=1., sigma=1.)
        system.part.add(pos=[8., 8., 8.], type=0)
        system.part.add(pos=[8.8, 8., 8.], type=0)
        time_step = system.time_step
        system.integrator.run(0)
        initial = np.linalg.norm(system.part.all().f, axis=1).max()
        sim_inst.avoid_explosion(F_TOL=0.1)
        self.assertEqual((system.force_cap, system.time_step), (0, time_step))
        system.integrator.run(0)
        self.assertLess(np.linalg.norm(system.part.all().f, axis=1).max(), initial)

    def test_rebind_sys_moves_every_handle(self):
        """Simulation and manager (so reinitialize_instance won't revert) get the handle; the slice cache is dropped."""
        self.addCleanup(sim_inst.rebind_sys, sim_inst.sys)  # runs before BaseTestCase's reset, even on a failure
        stand_in = object()
        sim_inst._h5_writer._slice_cache['dummy'] = None
        sim_inst.rebind_sys(stand_in)
        self.assertIs(sim_inst.sys, stand_in)
        self.assertIs(sim_inst._espressomd_system, stand_in)
        self.assertEqual(sim_inst._h5_writer._slice_cache, {})

    def test_set_sys_configures_the_system(self):
        """set_sys sets periodicity, time_step, skin and min_global_cut; it refuses without VIRTUAL_SITES_RELATIVE."""
        # cleanup() resets neither time_step nor skin: restore what create_system set
        self.addCleanup(sim_inst.set_sys, min_global_cut=BaseTestCase.min_global_cut)
        system = sim_inst.sys
        system.periodicity = (False, True, True)
        system.cell_system.skin = 0.3
        sim_inst.set_sys(timestep=0.02, min_global_cut=2.)
        self.assertEqual((list(system.periodicity), system.time_step, system.cell_system.skin, system.min_global_cut),
                         ([True] * 3, 0.02, 0.5, 2.))
        with missing_feature('VIRTUAL_SITES_RELATIVE'), self.assertRaisesRegex(MissingFeature, 'VirtualSitesRelative'):
            sim_inst.set_sys()

    def test_mark_for_collision_detection(self):
        """Refused with no object of the type stored or no mark_covalent_bonds; a Quadriplex gets 2 marked corners."""
        with self.assertRaisesRegex(ValueError, 'correct type object'):
            sim_inst.mark_for_collision_detection()
        sim_inst.store_objects(crowders(1))
        with self.assertRaisesRegex(TypeError, 'mark_covalent_bonds'):
            sim_inst.mark_for_collision_detection(object_type=Crowder)
        self.assertNotIn('marked', sim_inst.part_types)  # a refused call registers nothing
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            quadriplex = Quadriplex(config=Quadriplex.config.specify(espresso_handle=sim_inst.sys))
            sim_inst.store_objects([quadriplex])
            sim_inst.set_objects([quadriplex])
        sim_inst.mark_for_collision_detection()
        self.assertEqual(sim_inst.part_types['marked'], 666)
        self.assertEqual(len(sim_inst.sys.part.select(type=666)), 2)


class SetObjectsTest(BaseTestCase):
    box_dim = (30, 30, 30)

    def test_every_earlier_call_is_avoided(self):
        """A third call used to be checked against the first call only, and put 10 of these
        crowders exactly on second-call sites. Centres come from the particles, so the check
        does not trust the obstacles it tests."""
        groups = [[filament(size=10, n_parts=5) for _ in range(12)], crowders(150), crowders(150)]
        for group in groups:
            sim_inst.store_objects(group)
            sim_inst.set_objects(group)
        centres = [np.array([np.mean([part.pos for part in obj.get_owned_part()[0]], axis=0) for obj in group])
                   for group in groups]
        sizes = [group[0].params['size'] for group in groups]
        for i, j in itertools.combinations(range(len(groups)), 2):
            dist = np.linalg.norm(min_img_dist(centres[i][:, None], centres[j][None], sim_inst.sys.box_l), axis=-1)
            with self.subTest(calls=(i, j)):
                self.assertGreaterEqual(dist.min(), 0.5 * (sizes[i] + sizes[j]) - 1e-9)

    def test_one_call_needs_one_type_and_layout(self):
        """One call places one type and one build (size and spacing both come from objects[0]); n_parts
        alone does not split a call, the default build does not read it."""
        rows = [  # (label, objects, fragment; None: accepted, particles placed), refused rows first
            ('Crowder and Filament', crowders(1) + [filament(size=4, n_parts=2)], 'same type', 0),
            ('size 1 and 2', crowders(1, size=1) + crowders(1), 'size, num_monomers', 0),
            ('spacing None and 1.5', [filament(size=10, n_parts=5, spacing=spacing) for spacing in (None, 1.5)],
             'size, num_monomers', 0),
            ('n_parts 1 and 2', crowders(1, n_parts=1) + crowders(1, n_parts=2), None, 2),
        ]
        for label, objects, fragment, placed in rows:
            with self.subTest(label):
                sim_inst.store_objects(objects)
                if fragment is None:
                    sim_inst.set_objects(objects)
                else:
                    with self.assertRaisesRegex(ValueError, fragment):
                        sim_inst.set_objects(objects)
                self.assertEqual(len(sim_inst.sys.part), placed)

    @staticmethod
    def min_distance(objects, points):
        """Smallest minimum-image distance from the objects' particle means to ``points``."""
        centres = np.array([np.mean([part.pos for part in obj.get_owned_part()[0]], axis=0) for obj in objects])
        return np.linalg.norm(min_img_dist(centres[:, None], np.asarray(points)[None], sim_inst.sys.box_l),
                              axis=-1).min()

    def test_deleted_objects_free_their_space(self):
        """32 crowders of size 10 take every FCC site of the 30^3 box. Deleting them used to leave
        their volumes blocked, so the next call raised "Cannot fit"."""
        first, extra, second = (crowders(n, size=10) for n in (32, 1, 32))
        sim_inst.store_objects(first + extra + second)
        sim_inst.set_objects(first)
        with self.assertRaisesRegex(ValueError, "Cannot fit"):
            sim_inst.set_objects(extra)  # the box is full
        for crowder in first:
            crowder.delete_owned_parts()
        sim_inst.set_objects(second)
        self.assertEqual(len(sim_inst.sys.part), 32)

    def test_hand_placed_obstacles_are_avoided(self):
        """An object placed by hand used to be invisible to set_objects, and one across x=0 is avoided by
        its unfolded mean (folded, it would sit at the box centre). Of the ~4000 sites for size 2, ~8% lie
        within reach of a size-14 obstacle, so, if completely random, the 300 random ones only miss it by chance with probability ~1e-11."""
        rows = [  # (label, build obstacle, positions, orientations, obstacle centre)
            ('size-14 crowder at the centre', lambda: crowders(1, size=14), [[15., 15., 15.]], None, [15., 15., 15.]),
            ('2-monomer filament across x=0', lambda: [filament(size=14, n_parts=2)],
             [[[-0.5, 15., 15.], [0.5, 15., 15.]]], [[[1., 0., 0.]] * 2], [0., 15., 15.]),
        ]
        for label, build, positions, orientations, centre in rows:
            with self.subTest(label):
                BaseTestCase.cleanup(self.box_dim)
                obstacle, small = build(), crowders(300)
                sim_inst.store_objects(obstacle + small)
                sim_inst.place_objects(obstacle, positions, orientations)
                sim_inst.set_objects(small)
                self.assertGreaterEqual(self.min_distance(small, [centre]), 0.5 * (14 + 2) - 1e-9)

    def test_placed_objects_are_not_placed_again(self):
        """The re-place guard also looks through associated objects: a Filament whose monomers already
        own particles is refused, and nothing is added."""
        def direct():
            placed = crowders(3)
            return placed, placed, placed

        def via_monomers():
            monomers = crowders(2)
            chain = filament(size=4, n_parts=2, associated_objects=monomers)
            return monomers + [chain], monomers, [chain]

        for label, build in (('crowders', direct), ('Filament of placed monomers', via_monomers)):
            with self.subTest(label):
                stored, placed, again = build()
                sim_inst.store_objects(stored)
                sim_inst.set_objects(placed)
                before = len(sim_inst.sys.part)
                with self.assertRaisesRegex(ValueError, "already own particles"):
                    sim_inst.set_objects(again)
                self.assertEqual(len(sim_inst.sys.part), before)

    def test_overflow_warns(self):
        """Five monomers at spacing 1 reach 2 from their centre, beyond size/2 = 1."""
        chain = filament(size=2, n_parts=5, spacing=1)
        sim_inst.store_objects([chain])
        with self.assertWarnsRegex(UserWarning, r"Filament particle centres reach 2 .* beyond size/2 = 1"):
            sim_inst.set_objects([chain])

    def test_set_objects_after_an_elastomer_raises(self):
        """An Elastomer (size=None) lays itself out in its slab, without the lattice; set_objects has
        no sphere to keep others away from it, so it refuses to place anything after it."""
        elastomer = Elastomer(config=Elastomer.config.specify(box_E=[30, 30, 8], n_parts=400,
                                                              sigma=1. / WCA_CONTACT_FACTOR,
                                                              espresso_handle=sim_inst.sys, seed=sim_inst.seed))
        small = crowders(3)
        sim_inst.store_objects([elastomer] + small)
        sim_inst.set_objects([elastomer])
        self.assertEqual(len(sim_inst.sys.part), len(elastomer.get_owned_part()[0]))
        with self.assertRaisesRegex(ValueError, "Elastomer has no placement size"):
            sim_inst.set_objects(small)
        self.assertEqual(len(sim_inst.sys.part), len(elastomer.get_owned_part()[0]))
