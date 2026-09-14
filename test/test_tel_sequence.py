import unittest

import espressomd
import numpy as np
from .create_system import sim_inst, BaseTestCase
from pressomancy.object_classes.tel_sequence import (TELSEQ_RULES, TelSeq,
                                                     _geometric_corner_local_ids,
                                                     _validate_telseq_rules,
                                                     _VALIDATED_TELSEQ_ALIASES)
from pressomancy.object_classes.quadriplex_class import Quartet, Quadriplex
from pressomancy.helper_functions import BondWrapper, api_agnostic_feature_check


class TelSeqRulesGeometryTest(unittest.TestCase):
    '''
    The TELSEQ_RULES-vs-geometry cross-check itself.

    Deliberately not feature-gated: it only reads resource files and does
    numpy geometry, so it stays exercised even on builds where TelSeq's own
    MORSE-gated tests cannot run.
    '''

    def setUp(self) -> None:
        self._saved = {alias: dict(params) for alias, params in TELSEQ_RULES.items()}
        _VALIDATED_TELSEQ_ALIASES.clear()

    def tearDown(self) -> None:
        for alias, params in self._saved.items():
            TELSEQ_RULES[alias] = params
        _VALIDATED_TELSEQ_ALIASES.clear()

    def test_shipped_rules_match_the_resource_geometry(self):
        for alias in TELSEQ_RULES:
            with self.subTest(alias=alias):
                _validate_telseq_rules(alias)

    def test_corner_ids_are_the_quartet_block_corners(self):
        # the rule table's top/bottom blocks must both reduce to the same
        # single-quartet-local corner set that the geometry gives
        for alias, params in TELSEQ_RULES.items():
            with self.subTest(alias=alias):
                quartet_size = params['block_size'] // 3
                expected = _geometric_corner_local_ids(alias)
                self.assertEqual(len(expected), 4, 'a tetrad has four corners')
                self.assertEqual({i - quartet_size for i in params['top']}, expected)
                self.assertEqual({i - 2 * quartet_size for i in params['bottom']}, expected)

    def test_desynchronised_rules_are_rejected(self):
        alias = 'quartet'
        corrupted = dict(TELSEQ_RULES[alias])
        corrupted['top'] = [idx + 1 for idx in corrupted['top']]
        TELSEQ_RULES[alias] = corrupted
        with self.assertRaises(ValueError):
            _validate_telseq_rules(alias)

    def test_validation_is_memoised_per_alias(self):
        _validate_telseq_rules('quartet')
        self.assertIn('quartet', _VALIDATED_TELSEQ_ALIASES)
        # a later corruption is not re-checked for an alias already validated,
        # which is what keeps the check off the hot path
        TELSEQ_RULES['quartet'] = {**TELSEQ_RULES['quartet'], 'top': [0, 0, 0, 0]}
        _validate_telseq_rules('quartet')


if all(api_agnostic_feature_check(feature) for feature in TelSeq.required_features):
    class TelSeqTest(BaseTestCase):

        def tearDown(self) -> None:
            self.cleanup()
            self.assertEqual(len(sim_inst.sys.part), 0)

        def _build_tel(self, fold_type, alias='quartet', pos=None):
            quad_bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=3.0))
            diag_bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=np.sqrt(2) * 4.2, d_r_max=3.0))
            across_bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=4.2, d_r_max=3.0))

            quartets = [Quartet(config=Quartet.config.specify(
                alias=alias,
                espresso_handle=sim_inst.sys,
                type='brokenA',
            )) for _ in range(6)]
            quadriplexes = []
            for idx in range(0, len(quartets), 3):
                quadriplexes.append(Quadriplex(config=Quadriplex.config.specify(
                    espresso_handle=sim_inst.sys,
                    associated_objects=quartets[idx:idx + 3],
                    bonding_mode='ftf',
                    bond_handle=quad_bond,
                )))

            tel = TelSeq(config=TelSeq.config.specify(
                n_parts=2,
                espresso_handle=sim_inst.sys,
                associated_objects=quadriplexes,
                bond_handle=quad_bond,
                diag_bond_handle=diag_bond,
                across_bond_handle=across_bond,
                type=fold_type,
            ))
            sim_inst.store_objects([tel])
            if pos is None:
                pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]])
            tel.set_object(pos=pos, ori=np.array([0.0, 0.0, 1.0]))
            return tel

        def _assert_quartet_real_particle_directors(self, tel, expected_director):
            expected_director = np.array(expected_director, dtype=float)
            for quadriplex in tel.associated_objects:
                for quartet in quadriplex.associated_objects:
                    self.assertGreater(len(quartet.type_part_dict['real']), 0)
                    for particle in quartet.type_part_dict['real']:
                        self.assertTrue(np.allclose(particle.director, expected_director))

        def test_antiparallel_z_axis_uses_x_axis_side_orientation(self):
            tel = self._build_tel('antiparallel')
            expected_orientor = np.array([1.0, 0.0, 0.0])
            self.assertTrue(np.allclose(tel.orientor, np.array([0.0, 0.0, 1.0])))
            self._assert_quartet_real_particle_directors(tel, expected_orientor)

        def test_antiparallel_x_axis_uses_y_axis_side_orientation(self):
            tel = self._build_tel(
                'antiparallel',
                pos=np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
            )
            expected_orientor = np.array([0.0, 1.0, 0.0])
            self.assertTrue(np.allclose(tel.orientor, np.array([1.0, 0.0, 0.0])))
            self._assert_quartet_real_particle_directors(tel, expected_orientor)

        def test_parallel_and_hybrid_use_chain_orientation(self):
            for fold_type in ('parallel', 'hybrid'):
                tel = self._build_tel(fold_type)
                self._assert_quartet_real_particle_directors(tel, np.array([0.0, 0.0, 1.0]))
                tel=[]
                self.cleanup()

        def test_high_resolution_wrap_into_tel(self):
            for fold_type in ('parallel', 'hybrid', 'antiparallel'):
                tel = self._build_tel(fold_type,alias='quartet_11x11')
                # who_am_i is a monotonic allocator that cleanup() never resets, so its
                # absolute value depends on how many objects earlier tests built. Assert
                # the relationship instead: this TelSeq's quadriplexes are consecutive.
                owned_ids = [quad.who_am_i for quad in tel.associated_objects]
                self.assertEqual(owned_ids, list(range(owned_ids[0], owned_ids[0] + len(owned_ids))))
                second_corner_ids = []
                second_corner_ids.extend(part.id for part in tel.associated_objects[1].associated_objects[1].corner_particles)
                second_corner_ids.extend(part.id for part in tel.associated_objects[1].associated_objects[2].corner_particles)
                # Likewise relative: the second quadriplex's corners must come after the
                # first's, rather than exceeding a hardcoded particle id.
                first_corner_ids = [part.id for part in tel.associated_objects[0].associated_objects[1].corner_particles]
                self.assertGreater(max(second_corner_ids), max(first_corner_ids))
                tel.wrap_into_Tel()
                bonded_corners = []
                for monomer in tel.associated_objects:
                    for quartet in monomer.associated_objects[1:]:
                        bonded_corners.extend(part for part in quartet.corner_particles if len(part.bonds) > 0)
                self.assertTrue(bonded_corners)
                monomer=None
                tel=None
                self.cleanup()
