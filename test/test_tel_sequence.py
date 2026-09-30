import unittest

import espressomd
import numpy as np
from .create_system import sim_inst, BaseTestCase, BoxTestCase, bonds_using
from pressomancy.object_classes.tel_sequence import (TELSEQ_RULES, TelSeq,
                                                     _validate_telseq_rules,
                                                     _VALIDATED_TELSEQ_ALIASES)
from pressomancy.object_classes.quadriplex_class import Quartet, Quadriplex
from pressomancy.infra import BondWrapper, api_agnostic_feature_check


class TelSeqRulesGeometryTest(unittest.TestCase):
    '''
    The TELSEQ_RULES-vs-geometry cross-check itself.

    Deliberately not feature-gated: it only reads resource files and does
    numpy geometry, so it stays exercised even on builds where TelSeq's own
    MORSE-gated tests cannot run.
    '''

    def setUp(self) -> None:
        saved = {alias: dict(params) for alias, params in TELSEQ_RULES.items()}
        self.addCleanup(_VALIDATED_TELSEQ_ALIASES.clear)
        self.addCleanup(TELSEQ_RULES.update, saved)
        _VALIDATED_TELSEQ_ALIASES.clear()

    def test_shipped_rules_match_the_resource_geometry(self):
        """Every shipped alias' top/bottom ids are the corners of its resource geometry."""
        for alias in TELSEQ_RULES:
            with self.subTest(alias):
                _validate_telseq_rules(alias)

    def test_desynchronised_rules_are_rejected(self):
        """Shifted top ids no longer match the geometry and are refused."""
        alias = 'quartet'
        corrupted = dict(TELSEQ_RULES[alias])
        corrupted['top'] = [idx + 1 for idx in corrupted['top']]
        TELSEQ_RULES[alias] = corrupted
        with self.assertRaisesRegex(ValueError, 'does not match the corner particles'):
            _validate_telseq_rules(alias)


@unittest.skipIf(not all(api_agnostic_feature_check(feature) for feature in TelSeq.required_features),
                 f'TelSeq needs {TelSeq.required_features}')
class TelSeqTest(BoxTestCase):
    box_dim = (12, 12, 12)  # the diag bonds are 5.657 long: under half the box, as minimum image needs

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

    def test_orientation(self):
        """Antiparallel quartets face a side axis, parallel and hybrid ones the chain axis."""
        x_axis, y_axis, z_axis = np.eye(3)
        along_z = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]])
        along_x = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        rows = [('antiparallel along z', 'antiparallel', along_z, z_axis, x_axis),
                ('antiparallel along x', 'antiparallel', along_x, x_axis, y_axis),
                ('parallel', 'parallel', along_z, z_axis, z_axis),
                ('hybrid', 'hybrid', along_z, z_axis, z_axis)]
        for label, fold_type, pos, chain_axis, director in rows:
            with self.subTest(label):
                BaseTestCase.cleanup(self.box_dim)
                tel = self._build_tel(fold_type, pos=pos)
                self.assertTrue(np.allclose(tel.orientor, chain_axis))
                for quadriplex in tel.associated_objects:
                    for quartet in quadriplex.associated_objects:
                        self.assertGreater(len(quartet.type_part_dict['real']), 0)
                        for particle in quartet.type_part_dict['real']:
                            self.assertTrue(np.allclose(particle.director, director))

    def test_high_resolution_wrap_into_tel(self):
        """Each fold adds its rule's diag and across bonds per monomer and one link between the two."""
        # _rule_maker per monomer: parallel walks n=3 diagonals; hybrid and antiparallel make
        # 1 diag + 2 across. The counts do not depend on the random start corner.
        rule_pairs = {'parallel': (3, 0), 'hybrid': (1, 2), 'antiparallel': (1, 2)}
        for fold_type, (diag_per_monomer, across_per_monomer) in rule_pairs.items():
            with self.subTest(fold_type):
                BaseTestCase.cleanup(self.box_dim)
                tel = self._build_tel(fold_type, alias='quartet_11x11')
                self.assertEqual([quad.who_am_i for quad in tel.associated_objects], [0, 1])
                second_corner_ids = []
                second_corner_ids.extend(part.id for part in tel.associated_objects[1].associated_objects[1].corner_particles)
                second_corner_ids.extend(part.id for part in tel.associated_objects[1].associated_objects[2].corner_particles)
                first_corner_ids = [part.id for part in tel.associated_objects[0].associated_objects[1].corner_particles]
                self.assertGreater(max(second_corner_ids), max(first_corner_ids))
                link_bond = tel.params['bond_handle'].get_raw_handle()
                quadriplex_bonds = bonds_using(link_bond)  # the monomers' own ftf bonds share the link's handle
                tel.wrap_into_Tel()
                n_monomers = len(tel.associated_objects)
                self.assertEqual(bonds_using(tel.params['diag_bond_handle'].get_raw_handle()), n_monomers * diag_per_monomer)
                self.assertEqual(bonds_using(tel.params['across_bond_handle'].get_raw_handle()), n_monomers * across_per_monomer)
                self.assertEqual(bonds_using(link_bond) - quadriplex_bonds, n_monomers - 1)
