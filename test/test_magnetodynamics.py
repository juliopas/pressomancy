import functools
import unittest
from unittest import mock

import numpy as np

import espressomd.interactions
from espressomd.magnetostatics import DipolarDirectSum

from pressomancy.infra import BondWrapper, MissingFeature, api_agnostic_feature_check
from pressomancy.magnetodynamics import (MAGNETIZATION_MODELS, MOMENT_CARRIER_PROPAGATION,
                                         Propagation, configure_magnetization,
                                         contraction_ratio, required_features_for,
                                         susceptibility_from_kT, validate_model)
from pressomancy.simulation import (Filament, PointDipoleMagnetizable,
                                    PointDipoleSuperparamagnetic)
from .create_system import sim_inst, BaseTestCase, BoxTestCase

AVAILABLE_MODELS = [name for name in MAGNETIZATION_MODELS
                    if all(api_agnostic_feature_check(f)
                           for f in required_features_for(name))]


def langevin_moment(field, m_sat, chi_0):
    '''Espresso's Langevin magnetisation curve, see langevin_magnetization.cpp.'''
    alpha = 3. * chi_0 * field / m_sat
    return m_sat * (1. / np.tanh(alpha) - 1. / alpha)


def froelich_kennelly_moment(field, m_sat, chi_0):
    '''Espresso's Froelich-Kennelly magnetisation curve, see froelich_kennelly.cpp.'''
    return chi_0 * m_sat / (m_sat + chi_0 * field) * field


CLOSED_FORM = {'langevin': langevin_moment,
               'froelich_kennelly': froelich_kennelly_moment}


def _make_pair(m_sat, pos=(5., 5., 5.)):
    '''A fixed real anchor plus an unbound particle at its position, moment seeded along z.'''
    anchor = sim_inst.sys.part.add(pos=list(pos), fix=[True] * 3)
    virt = sim_inst.sys.part.add(pos=anchor.pos, rotation=[True] * 3, dip=[0., 0., m_sat])
    return anchor, virt


class ValidationTest(unittest.TestCase):
    '''Refusals that need no live espresso particle.'''

    def test_unknown_model_is_refused_everywhere(self):
        '''The helpers and both object configs refuse an unknown model name.'''
        cases = {
            'validate_model': lambda: validate_model('ideal'),
            'required_features_for': lambda: required_features_for('ideal'),
            'PointDipoleMagnetizable': lambda: PointDipoleMagnetizable(
                config=PointDipoleMagnetizable.config.specify(
                    magnetization_model='ideal', espresso_handle=sim_inst.sys)),
            'PointDipoleSuperparamagnetic': lambda: PointDipoleSuperparamagnetic(
                config=PointDipoleSuperparamagnetic.config.specify(
                    magnetization_model='ideal', espresso_handle=sim_inst.sys)),
        }
        for label, call in cases.items():
            with self.subTest(label), self.assertRaisesRegex(ValueError, 'Unknown magnetization model'):
                call()

    def test_unphysical_dipm_or_kT_is_refused(self):
        '''`PointDipoleSuperparamagnetic.__init__` fails early, not at `set_object`.'''
        for dipm, kT, fragment in [(0., 1., 'dipm must'), (-1., 1., 'dipm must'),
                                   (1., 0., 'kT must'), (1., -2., 'kT must')]:
            with self.subTest(f'susceptibility_from_kT({dipm}, {kT})'), \
                    self.assertRaisesRegex(ValueError, fragment):
                susceptibility_from_kT(dipm, kT)
        for dipm, kT, fragment in [(0., 1., 'dipm must'), (1., 0., 'kT must')]:
            with self.subTest(f'PointDipoleSuperparamagnetic(dipm={dipm}, kT={kT})'), \
                    self.assertRaisesRegex(ValueError, fragment):
                PointDipoleSuperparamagnetic(config=PointDipoleSuperparamagnetic.config.specify(
                    dipm=dipm, kT=kT, espresso_handle=sim_inst.sys))


@unittest.skipIf(not AVAILABLE_MODELS, 'no magnetization model compiled in this espresso build')
class ConfigureMagnetizationTest(BoxTestCase):
    '''`configure_magnetization` on bare anchor + virtual pairs, and through objects and `Simulation`.'''

    box_dim = (20, 20, 20)
    m_sat = 1.5
    chi_0 = 0.3

    def test_configure_binds_and_switches_models(self):
        '''Every available model in turn, then back to the first; only the first step passes the anchor.'''
        anchor, virt = _make_pair(self.m_sat)
        models = AVAILABLE_MODELS + AVAILABLE_MODELS[:1]
        for step, model in enumerate(models):
            dipm_sat = 1. + 0.5 * step
            chi_0 = 0.1 * (len(models) - 1 - step)  # distinct at every step, the last is the boundary 0
            with self.subTest(f'step {step}: {model}'):
                configure_magnetization(virt, model, dipm_sat, chi_0,
                                        anchor=anchor if step == 0 else None)
                self.assertEqual(virt.dipm_sat, dipm_sat)
                self.assertEqual(virt.mag_susc_0, chi_0)
                self.assertEqual({m: getattr(virt, MAGNETIZATION_MODELS[m][1]) for m in AVAILABLE_MODELS},
                                 {m: m == model for m in AVAILABLE_MODELS})
                self.assertTrue(virt.is_virtual())
                self.assertEqual(virt.vs_relative[0], anchor.id)
                self.assertEqual(int(virt.propagation), MOMENT_CARRIER_PROPAGATION)

    def test_refusals_leave_the_particle_untouched(self):
        '''Each refusal raises before the particle is bound, propagated or flagged.'''
        def with_anchor(anchor, virt):
            return anchor

        def plain(anchor, virt):
            return None

        def never_related(anchor, virt):
            virt.propagation = Propagation.TRANS_VS_RELATIVE | Propagation.ROT_VS_RELATIVE
            return None

        def langevin_coupled(anchor, virt):
            virt.vs_auto_relate_to(anchor, couple_to_langevin=True)
            return None

        model, m_sat, chi_0 = AVAILABLE_MODELS[0], self.m_sat, self.chi_0
        # (label, model, dipm_sat, mag_susc_0, bind, feature reported missing, exception, fragment)
        rows = [
            ('unknown model', 'ideal', m_sat, chi_0, with_anchor, None,
             ValueError, 'Unknown magnetization model'),
            ('dipm_sat 0', model, 0., chi_0, with_anchor, None, ValueError, 'dipm_sat'),
            ('dipm_sat -1', model, -1., chi_0, with_anchor, None, ValueError, 'dipm_sat'),
            ('mag_susc_0 -0.1', model, m_sat, -0.1, with_anchor, None, ValueError, 'mag_susc_0'),
            ('anchor=None, plain particle', model, m_sat, chi_0, plain, None,
             ValueError, 'virtual sites relative: False'),
            ('anchor=None, never related', model, m_sat, chi_0, never_related, None,
             ValueError, 'related to particle id: -1'),
            ('anchor=None, coupled to langevin', model, m_sat, chi_0, langevin_coupled, None,
             ValueError, 'cannot be kept'),
        ] + [
            (f'{name}: {MAGNETIZATION_MODELS[name][0]} missing', name, m_sat, chi_0, with_anchor,
             MAGNETIZATION_MODELS[name][0], MissingFeature,
             f'Missing required features: {MAGNETIZATION_MODELS[name][0]}')
            for name in AVAILABLE_MODELS
        ]
        for label, row_model, dipm_sat, mag_susc_0, bind, missing, exception, fragment in rows:
            with self.subTest(label):
                sim_inst.sys.part.all().remove()
                anchor, virt = _make_pair(m_sat)
                anchor_arg = bind(anchor, virt)
                before = (int(virt.propagation), virt.is_virtual())
                # the real check, except that `missing` (None: nothing) is reported absent
                with mock.patch('pressomancy.magnetodynamics.api_agnostic_feature_check',
                                lambda f: f != missing and api_agnostic_feature_check(f)):
                    with self.assertRaisesRegex(exception, fragment):
                        configure_magnetization(virt, row_model, dipm_sat, mag_susc_0, anchor=anchor_arg)
                    self.assertEqual((int(virt.propagation), virt.is_virtual()), before)
                    for m in AVAILABLE_MODELS:
                        self.assertFalse(getattr(virt, MAGNETIZATION_MODELS[m][1]))
                    if missing is not None:
                        for other in AVAILABLE_MODELS:
                            if other != row_model:
                                configure_magnetization(virt, other, m_sat, chi_0, anchor=anchor)
                                self.assertTrue(getattr(virt, MAGNETIZATION_MODELS[other][1]))

    def test_moment_follows_closed_form(self):
        '''One isolated carrier sees only H_ext, so a single step lands on its model's curve.'''
        h_hat = np.array([1., 2., 2.]) / 3.
        carriers = []  # (label, virtual site, m_sat, m(H))
        for model in AVAILABLE_MODELS:
            anchor, virt = _make_pair(self.m_sat)
            configure_magnetization(virt, model, self.m_sat, self.chi_0, anchor=anchor)
            carriers.append((model, virt, self.m_sat,
                             functools.partial(CLOSED_FORM[model], m_sat=self.m_sat, chi_0=self.chi_0)))
        if 'langevin' in AVAILABLE_MODELS:
            dipm, kT = 2.0, 0.5

            def classical_langevin(H):
                alpha = dipm * H / kT
                return dipm * (1. / np.tanh(alpha) - 1. / alpha)

            obj = PointDipoleSuperparamagnetic(config=PointDipoleSuperparamagnetic.config.specify(
                dipm=dipm, kT=kT, espresso_handle=sim_inst.sys))
            sim_inst.store_objects([obj])
            sim_inst.place_objects([obj], [np.array([15., 15., 15.])], [np.array([0., 0., 1.])])
            carriers.append(('PointDipoleSuperparamagnetic', obj.type_part_dict['pds_virt'][0], dipm,
                             classical_langevin))

        for H in (0.25, 1., 5., 50., 1e4):
            for _, virt, m_sat, _ in carriers:
                virt.dip = [0., 0., m_sat]
                self.assertLess(np.dot(virt.dip, h_hat) / m_sat, 0.9)
            sim_inst.set_H_ext(H=H * h_hat)
            sim_inst.sys.integrator.run(1)
            for label, virt, m_sat, moment in carriers:
                with self.subTest(carrier=label, H=H):
                    np.testing.assert_allclose(np.copy(virt.dip), moment(H) * h_hat, rtol=1e-9)
                    self.assertLess(virt.dipm, m_sat)

    def test_simulation_configures_object_virtuals(self):
        '''`set_magnetization_model` on the virtuals `add_dipole_to_embedded_virt` made; a bad dipm_sat is refused.'''
        bond_hndl = BondWrapper(espressomd.interactions.FeneBond(k=10, d_r_max=3., r_0=0))
        config = Filament.config.specify(sigma=1., size=2.26, n_parts=2,
                                         espresso_handle=sim_inst.sys, bond_handle=bond_hndl)
        filaments = [Filament(config=config) for _ in range(2)]
        sim_inst.store_objects(filaments)
        sim_inst.set_objects(filaments)
        for filament in filaments:
            filament.add_dipole_to_embedded_virt(type_name='real', dip_magnitude=1.)
        targets = list(sim_inst.sys.part.select(type=sim_inst.part_types['to_be_magnetized']))
        self.assertGreater(len(targets), 0)

        model, dipm_sat = AVAILABLE_MODELS[0], 1.732
        sim_inst.set_magnetization_model(targets, model, dipm_sat=dipm_sat, mag_susc_0=dipm_sat ** 2 / 3.)
        for part in targets:
            with self.subTest(part=part.id):
                self.assertEqual(part.dipm_sat, dipm_sat)
                self.assertEqual(part.mag_susc_0, dipm_sat ** 2 / 3.)
                self.assertTrue(getattr(part, MAGNETIZATION_MODELS[model][1]))
                self.assertEqual(int(part.propagation), MOMENT_CARRIER_PROPAGATION)
        with self.assertRaisesRegex(ValueError, 'dipm_sat'):
            sim_inst.set_magnetization_model(targets, model, dipm_sat=-1., mag_susc_0=1.)


@unittest.skipIf('langevin' not in AVAILABLE_MODELS,
                 'the langevin magnetization model is not compiled in this espresso build')
class ConvergenceTest(BoxTestCase):
    '''`contraction_ratio` measures the fixed-point iteration on moment vectors without advancing time.'''

    box_dim = (20, 20, 20)
    m_sat = 1.732

    def _chain(self, chi_0):
        '''Five touching magnetizable spheres head to tail along z, H=1 along z.'''
        parts = []
        for i in range(5):
            anchor, virt = _make_pair(self.m_sat, pos=(10., 10., 5. + i))
            configure_magnetization(virt, 'langevin', self.m_sat, chi_0, anchor=anchor)
            parts.append(virt)
        sim_inst.init_magnetic_inter(DipolarDirectSum(prefactor=1.))
        sim_inst.set_H_ext(H=[0., 0., 1.])
        return parts

    def _tilted_ring(self):
        '''A flat ring of six (χ₀=0.8) whose moments start tilted out of plane: the iteration mostly rotates them.'''
        parts, tilt = [], 0.3 * np.pi
        for i in range(6):
            phi = 2. * np.pi * i / 6
            anchor, virt = _make_pair(self.m_sat, pos=(10. + np.cos(phi), 10. + np.sin(phi), 10.))
            virt.dip = self.m_sat * np.array([np.sin(tilt) * np.cos(phi),
                                              np.sin(tilt) * np.sin(phi),
                                              np.cos(tilt)])
            configure_magnetization(virt, 'langevin', self.m_sat, 0.8, anchor=anchor)
            parts.append(virt)
        sim_inst.init_magnetic_inter(DipolarDirectSum(prefactor=1.))
        sim_inst.set_H_ext(H=[0., 0., 0.1])
        return parts

    def _settled_chain(self, chi0, n_particles=6):
        '''The regime-sweep geometry: PointDipoleMagnetizable at contact, H=0.01; returns (last ratio, mean |m|).'''
        spacing = 2.0 ** (1.0 / 6.0)
        cfg = PointDipoleMagnetizable.config.specify(
            magnetization_model='langevin', dipm_sat=1., mag_susc_0=chi0,
            espresso_handle=sim_inst.sys)
        objs = [PointDipoleMagnetizable(config=cfg) for _ in range(n_particles)]
        sim_inst.store_objects(objs)
        centre = 0.5 * sim_inst.sys.box_l[2] - 0.5 * spacing * (n_particles - 1)
        mid = 0.5 * sim_inst.sys.box_l[0]
        sim_inst.place_objects(
            objs,
            [np.array([mid, mid, centre + i * spacing]) for i in range(n_particles)],
            [np.array([0., 0., 1.]) for _ in range(n_particles)])
        sim_inst.init_magnetic_inter(DipolarDirectSum(prefactor=1.0))
        sim_inst.set_H_ext(H=(0, 0, 0.01))
        virt = [p for o in objs for p in o.get_owned_part()[0]
                if int(p.type) == PointDipoleMagnetizable.part_types['pdm_virt']]
        ratios = sim_inst.probe_magnetization_convergence(virt, n_iter=40)
        moment = float(np.mean([float(np.linalg.norm(p.dip)) for p in virt]))
        return float(ratios[-1]), moment

    def test_probe_tracks_moment_vectors(self):
        '''On a rotation-dominated ring the probe equals the stacked-vector ratios, not the magnitude ones.'''
        n_iter, tol = 10, 1e-12
        parts = self._tilted_ring()
        snapshots = []
        for _ in range(n_iter):
            sim_inst.sys.integrator.run(0, recalc_forces=True)
            snapshots.append(np.array([p.dip for p in parts]))
        snapshots = np.array(snapshots)
        vector_inc = np.linalg.norm(np.diff(snapshots, axis=0).reshape(n_iter - 1, -1), axis=1)
        magnitude_inc = np.linalg.norm(np.diff(np.linalg.norm(snapshots, axis=2), axis=0), axis=1)
        # no increment reaches tol, so the probe keeps the full series and no truncation is needed here
        self.assertGreater(vector_inc.min(), tol)
        self.assertLess((magnitude_inc / vector_inc).mean(), 0.25)

        sim_inst.sys.part.all().remove()
        actual = sim_inst.probe_magnetization_convergence(self._tilted_ring(), n_iter=n_iter, tol=tol)
        np.testing.assert_allclose(actual, vector_inc[1:] / vector_inc[:-1], rtol=1e-9)
        self.assertFalse(np.allclose(actual, magnitude_inc[1:] / magnitude_inc[:-1]))

    def test_ratio_scales_linearly_with_chi0(self):
        '''Unsaturated chains contract with ratio ∝ χ₀ (3.8·χ₀ here), as the module docstring states.'''
        n_iter, slopes = 10, []
        for chi_0 in (0.02, 0.05, 0.1):
            with self.subTest(chi_0=chi_0):
                sim_inst.sys.part.all().remove()
                parts = self._chain(chi_0)
                time_before = sim_inst.sys.time
                ratios = sim_inst.probe_magnetization_convergence(parts, n_iter=n_iter)
                self.assertEqual(sim_inst.sys.time, time_before)
                self.assertEqual(len(ratios), n_iter - 2)
                self.assertTrue(np.all(ratios < 1.))
                slopes.append(ratios[-1] / chi_0)
        np.testing.assert_allclose(slopes, slopes[0], rtol=0.05)

    def test_truncation_and_argument_contract(self):
        '''A loose tol cuts the series short; a lone particle leaves no ratio; bad arguments are refused.'''
        n_iter = 10
        ratios = sim_inst.probe_magnetization_convergence(self._chain(chi_0=0.02), n_iter=n_iter, tol=1e-6)
        self.assertTrue(0 < len(ratios) < n_iter - 2)
        self.assertTrue(np.all(ratios < 1.))
        sim_inst.sys.part.all().remove()

        anchor, virt = _make_pair(self.m_sat, pos=(10., 10., 10.))
        configure_magnetization(virt, 'langevin', self.m_sat, 0.1, anchor=anchor)
        self.assertEqual(len(contraction_ratio(sim_inst.sys, [virt], n_iter=n_iter)), 0)
        for label, part_list, kwargs, fragment in [('n_iter=1', [virt], {'n_iter': 1}, 'n_iter'),
                                                   ('tol=0', [virt], {'tol': 0.}, 'tol'),
                                                   ('empty part_list', [], {}, 'empty')]:
            with self.subTest(label), self.assertRaisesRegex(ValueError, fragment):
                contraction_ratio(sim_inst.sys, part_list, **kwargs)

    def test_documented_regime_points(self):
        '''Pins two points of the regime table in magnetodynamics.py; if either moves, re-measure the table.'''
        # χ₀=1.0 is the documented false positive: it contracts, but around a saturated fixed point
        for chi_0, saturated in ((0.1, False), (1.0, True)):
            with self.subTest(chi_0=chi_0):
                BaseTestCase.cleanup(self.box_dim)
                ratio, moment = self._settled_chain(chi_0)
                self.assertLess(ratio, 1.)
                if saturated:
                    self.assertGreater(moment, 0.5)
                else:
                    self.assertLess(moment, 0.5)


if __name__ == '__main__':
    unittest.main()
