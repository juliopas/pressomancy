import unittest
import numpy as np

import espressomd
import espressomd.magnetostatics

from pressomancy.helper_functions import (BondWrapper, MissingFeature,
                                          api_agnostic_feature_check)
from pressomancy.magnetodynamics import (MAGNETIZATION_MODELS, configure_magnetization,
                                         contraction_ratio, required_features_for,
                                         susceptibility_from_kT, validate_model)
from pressomancy.simulation import (Filament, PointDipoleMagnetizable,
                                    PointDipoleSuperparamagnetic)
from .create_system import sim_inst, BaseTestCase

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


class ModelRegistryTest(unittest.TestCase):
    '''Validation that does not need a live espresso system.'''

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            validate_model('not_a_model')
        with self.assertRaises(ValueError):
            required_features_for('ideal')

    def test_required_features_are_model_specific(self):
        for name, (feature, _) in MAGNETIZATION_MODELS.items():
            self.assertIn(feature, required_features_for(name))
            for other, (other_feature, _) in MAGNETIZATION_MODELS.items():
                if other != name:
                    self.assertNotIn(other_feature, required_features_for(name))

    def test_object_config_rejects_unknown_model(self):
        with self.assertRaises(ValueError):
            PointDipoleMagnetizable(
                config=PointDipoleMagnetizable.config.specify(
                    magnetization_model='ideal', espresso_handle=sim_inst.sys))


@unittest.skipIf(not AVAILABLE_MODELS,
                 'no magnetization model compiled in this espresso build')
class ConfigureMagnetizationTest(BaseTestCase):
    '''Exercises the helper directly on a bare anchor plus virtual site pair.'''

    m_sat = 1.5
    chi_0 = 0.4

    def tearDown(self) -> None:
        self.cleanup()
        self.assertEqual(len(sim_inst.sys.part), 0)

    def _make_pair(self):
        anchor = sim_inst.sys.part.add(pos=[5., 5., 5.], fix=[True] * 3)
        virt = sim_inst.sys.part.add(pos=anchor.pos, rotation=[True] * 3,
                                     dip=[0., 0., self.m_sat])
        return anchor, virt

    def test_parameters_land_on_the_particle(self):
        for model in AVAILABLE_MODELS:
            with self.subTest(model=model):
                anchor, virt = self._make_pair()
                configure_magnetization(virt, model, self.m_sat, self.chi_0,
                                        anchor=anchor)
                self.assertEqual(virt.dipm_sat, self.m_sat)
                self.assertEqual(virt.mag_susc_0, self.chi_0)
                self.assertTrue(getattr(virt, MAGNETIZATION_MODELS[model][1]))
                self.assertTrue(virt.is_virtual())
                sim_inst.sys.part.all().remove()

    def test_models_are_mutually_exclusive(self):
        if len(AVAILABLE_MODELS) < 2:
            self.skipTest('needs both models compiled')
        anchor, virt = self._make_pair()
        first, second = AVAILABLE_MODELS[0], AVAILABLE_MODELS[1]
        configure_magnetization(virt, first, self.m_sat, self.chi_0, anchor=anchor)
        configure_magnetization(virt, second, self.m_sat, self.chi_0)
        self.assertTrue(getattr(virt, MAGNETIZATION_MODELS[second][1]))
        self.assertFalse(getattr(virt, MAGNETIZATION_MODELS[first][1]),
                         'switching models must clear the previous one')

    def test_parameter_validation_precedes_espresso(self):
        anchor, virt = self._make_pair()
        model = AVAILABLE_MODELS[0]
        with self.assertRaises(ValueError):
            configure_magnetization(virt, model, 0., self.chi_0, anchor=anchor)
        with self.assertRaises(ValueError):
            configure_magnetization(virt, model, -1., self.chi_0, anchor=anchor)
        with self.assertRaises(ValueError):
            configure_magnetization(virt, model, self.m_sat, -0.1, anchor=anchor)

    def test_unknown_model_raises_before_touching_the_particle(self):
        anchor, virt = self._make_pair()
        with self.assertRaises(ValueError):
            configure_magnetization(virt, 'ideal', self.m_sat, self.chi_0,
                                    anchor=anchor)
        self.assertFalse(virt.is_virtual(),
                         'a rejected model must not bind the virtual site')


@unittest.skipIf(not AVAILABLE_MODELS,
                 'no magnetization model compiled in this espresso build')
class MagnetizationCurveTest(BaseTestCase):
    '''A single isolated particle must follow the closed form of its model.

    With one particle there is no dipolar field, so H_tot is the external field
    alone and the fixed point is reached in a single step.
    '''

    m_sat = 2.0
    chi_0 = 0.3

    def tearDown(self) -> None:
        self.cleanup()
        self.assertEqual(len(sim_inst.sys.part), 0)

    def test_moment_follows_closed_form(self):
        for model in AVAILABLE_MODELS:
            for H in (0.25, 1.0, 5.0, 50.0):
                with self.subTest(model=model, H=H):
                    anchor = sim_inst.sys.part.add(pos=[5., 5., 5.], fix=[True] * 3)
                    virt = sim_inst.sys.part.add(pos=anchor.pos, rotation=[True] * 3,
                                                 dip=[0., 0., self.m_sat])
                    configure_magnetization(virt, model, self.m_sat, self.chi_0,
                                            anchor=anchor)
                    sim_inst.set_H_ext(H=[0., 0., H])
                    sim_inst.sys.integrator.run(1)
                    expected = CLOSED_FORM[model](H, self.m_sat, self.chi_0)
                    self.assertAlmostEqual(virt.dipm, expected, places=9)
                    # the moment aligns with the field
                    np.testing.assert_allclose(np.asarray(virt.dip)[:2], 0.,
                                               atol=1e-12)
                    sim_inst.sys.part.all().remove()

    def test_saturates_below_m_sat(self):
        model = AVAILABLE_MODELS[0]
        anchor = sim_inst.sys.part.add(pos=[5., 5., 5.], fix=[True] * 3)
        virt = sim_inst.sys.part.add(pos=anchor.pos, rotation=[True] * 3,
                                     dip=[0., 0., self.m_sat])
        configure_magnetization(virt, model, self.m_sat, self.chi_0, anchor=anchor)
        sim_inst.set_H_ext(H=[0., 0., 1e4])
        sim_inst.sys.integrator.run(1)
        self.assertLess(virt.dipm, self.m_sat)
        self.assertGreater(virt.dipm, 0.95 * self.m_sat)


@unittest.skipIf(not AVAILABLE_MODELS,
                 'no magnetization model compiled in this espresso build')
class SimulationConfiguratorTest(BaseTestCase):
    '''Simulation.set_magnetization_model on virtuals an object created for us.'''

    sigma = 1.
    m_sat = 1.732

    def tearDown(self) -> None:
        self.filaments = None
        self.cleanup()
        self.assertEqual(len(sim_inst.sys.part), 0)

    def setUp(self) -> None:
        bond_hndl = BondWrapper(espressomd.interactions.FeneBond(
            k=10, d_r_max=3 * self.sigma, r_0=0))
        config = Filament.config.specify(sigma=self.sigma, size=2.26, n_parts=2,
                                         espresso_handle=sim_inst.sys,
                                         bond_handle=bond_hndl)
        self.filaments = [Filament(config=config) for _ in range(2)]
        sim_inst.store_objects(self.filaments)
        sim_inst.set_objects(self.filaments)
        for filament in self.filaments:
            filament.add_dipole_to_embedded_virt(type_name='real', dip_magnitude=1.)

    def test_configures_filament_virtuals(self):
        model = AVAILABLE_MODELS[0]
        targets = list(sim_inst.sys.part.select(
            type=sim_inst.part_types['to_be_magnetized']))
        self.assertGreater(len(targets), 0)
        sim_inst.set_magnetization_model(targets, model,
                                         dipm_sat=self.m_sat,
                                         mag_susc_0=self.m_sat ** 2 / 3.)
        for part in targets:
            self.assertEqual(part.dipm_sat, self.m_sat)
            self.assertTrue(getattr(part, MAGNETIZATION_MODELS[model][1]))

    def test_rejects_bad_parameters(self):
        targets = list(sim_inst.sys.part.select(
            type=sim_inst.part_types['to_be_magnetized']))
        with self.assertRaises(ValueError):
            sim_inst.set_magnetization_model(targets, AVAILABLE_MODELS[0],
                                             dipm_sat=-1., mag_susc_0=1.)


class SusceptibilityFromKTTest(unittest.TestCase):
    '''The dipm/kT parameterisation of a superparamagnetic point dipole.'''

    def test_matches_the_langevin_initial_slope(self):
        for dipm, kT in [(1., 1.), (1.732, 2.5), (0.25, 0.1)]:
            with self.subTest(dipm=dipm, kT=kT):
                self.assertAlmostEqual(susceptibility_from_kT(dipm, kT),
                                       dipm ** 2 / (3. * kT))

    def test_is_the_slope_of_the_classical_curve_at_zero_field(self):
        # chi_0 must equal dm/dH of m*L(m*H/kT) as H -> 0. The field cannot be
        # taken arbitrarily small here: 1/tanh(a) - 1/a subtracts two numbers of
        # order 1/a, so the cancellation swamps the result long before the
        # O(a**2) truncation of the slope does.
        dipm, kT = 1.4, 0.8
        h = 1e-3
        alpha = dipm * h / kT
        moment = dipm * (1. / np.tanh(alpha) - 1. / alpha)
        self.assertAlmostEqual(moment / h, susceptibility_from_kT(dipm, kT),
                               places=6)

    def test_rejects_unphysical_arguments(self):
        for dipm, kT in [(0., 1.), (-1., 1.), (1., 0.), (1., -2.)]:
            with self.subTest(dipm=dipm, kT=kT):
                with self.assertRaises(ValueError):
                    susceptibility_from_kT(dipm, kT)

    def test_object_config_rejects_unphysical_arguments(self):
        for dipm, kT in [(0., 1.), (1., 0.), (-1., 1.)]:
            with self.subTest(dipm=dipm, kT=kT):
                with self.assertRaises(ValueError):
                    PointDipoleSuperparamagnetic(
                        config=PointDipoleSuperparamagnetic.config.specify(
                            dipm=dipm, kT=kT, espresso_handle=sim_inst.sys))

    def test_object_config_rejects_unknown_model(self):
        with self.assertRaises(ValueError):
            PointDipoleSuperparamagnetic(
                config=PointDipoleSuperparamagnetic.config.specify(
                    magnetization_model='ideal', espresso_handle=sim_inst.sys))


@unittest.skipIf(not AVAILABLE_MODELS,
                 'no magnetization model compiled in this espresso build')
class ContractionRatioTest(BaseTestCase):
    '''The convergence probe must measure the iteration without advancing time.'''

    m_sat = 1.732

    def tearDown(self) -> None:
        self.cleanup()
        self.assertEqual(len(sim_inst.sys.part), 0)

    def _chain(self, chi_0, n=5, spacing=1.):
        '''n touching magnetizable spheres head to tail along z.'''
        parts = []
        for i in range(n):
            anchor = sim_inst.sys.part.add(pos=[10., 10., 5. + i * spacing],
                                           fix=[True] * 3)
            virt = sim_inst.sys.part.add(pos=anchor.pos, rotation=[True] * 3,
                                         dip=[0., 0., self.m_sat])
            configure_magnetization(virt, AVAILABLE_MODELS[0], self.m_sat, chi_0,
                                    anchor=anchor)
            parts.append(virt)
        sim_inst.init_magnetic_inter(
            espressomd.magnetostatics.DipolarDirectSum(prefactor=1.))
        sim_inst.set_H_ext(H=[0., 0., 1.])
        return parts

    @staticmethod
    def _reset_cluster():
        sim_inst.sys.magnetostatics.clear()
        sim_inst.sys.part.all().remove()

    @staticmethod
    def _increment_series(parts, n_iter, scalar):
        increments = []
        previous = None
        for _ in range(n_iter):
            sim_inst.sys.integrator.run(0, recalc_forces=True)
            current = (np.array([p.dipm for p in parts]) if scalar
                       else np.array([p.dip for p in parts]).ravel())
            if previous is not None:
                increments.append(np.linalg.norm(current - previous))
            previous = current
        return np.asarray(increments)

    @staticmethod
    def _ratios(increments, tol=1e-12):
        converged = np.flatnonzero(increments <= tol)
        cut = int(converged[0]) if converged.size else len(increments)
        increments = increments[:cut]
        if len(increments) < 2:
            return np.empty(0)
        return increments[1:] / increments[:-1]

    def _tilted_ring(self, chi_0, n=6, radius=1.0, tilt=0.3*np.pi, H=(0., 0., 0.1)):
        parts = []
        for i in range(n):
            phi = 2. * np.pi * i / n
            anchor = sim_inst.sys.part.add(
                pos=[10. + radius * np.cos(phi), 10. + radius * np.sin(phi), 10.],
                fix=[True] * 3)
            moment = self.m_sat * np.array([np.sin(tilt) * np.cos(phi),
                                            np.sin(tilt) * np.sin(phi),
                                            np.cos(tilt)])
            virt = sim_inst.sys.part.add(pos=anchor.pos, rotation=[True] * 3,
                                         dip=moment)
            configure_magnetization(virt, AVAILABLE_MODELS[0], self.m_sat, chi_0,
                                    anchor=anchor)
            parts.append(virt)
        sim_inst.init_magnetic_inter(
            espressomd.magnetostatics.DipolarDirectSum(prefactor=1.))
        sim_inst.set_H_ext(H=list(H))
        return parts

    def test_rotation_dominated_series_tracks_moment_vectors(self):
        n_iter = 10

        vec_inc = self._increment_series(
            self._tilted_ring(chi_0=0.8), n_iter, scalar=False)
        self._reset_cluster()
        sca_inc = self._increment_series(
            self._tilted_ring(chi_0=0.8), n_iter, scalar=True)
        self._reset_cluster()
        # the fixture must actually be rotation-dominated, otherwise the rest is vacuous
        live = vec_inc > 1e-12
        self.assertTrue(live.any(), "no resolvable increments")
        self.assertLess((sca_inc[live] / vec_inc[live]).mean(), 0.25,
                        "fixture is not roation-dominated, otherwise this ration would be small")

        expected_vector = self._ratios(vec_inc)
        expected_scalar = self._ratios(sca_inc)
        actual = sim_inst.probe_magnetization_convergence(
            self._tilted_ring(chi_0=0.8), n_iter=n_iter)
        np.testing.assert_allclose(
            actual, expected_vector, rtol=1e-9,
            err_msg='probe does not track the stacked moment vectors')
        self.assertFalse(
            np.allclose(actual, expected_scalar),
            "probe output matches a magnitude-only metric."
            "contraction_ratio has regressed to reading p.dipm instead of p.dip")

    def test_weak_coupling_contracts(self):
        parts = self._chain(chi_0=0.1)
        ratios = sim_inst.probe_magnetization_convergence(parts, n_iter=10)
        self.assertGreater(len(ratios), 0)
        self.assertLess(ratios[-1], 1.)

    def test_stronger_coupling_contracts_more_slowly(self):
        weak = self._chain(chi_0=0.05)
        weak_ratio = sim_inst.probe_magnetization_convergence(weak, n_iter=10)[-1]
        sim_inst.sys.magnetostatics.clear()
        sim_inst.sys.part.all().remove()
        strong = self._chain(chi_0=0.4)
        strong_ratio = sim_inst.probe_magnetization_convergence(strong, n_iter=10)[-1]
        self.assertLess(weak_ratio, strong_ratio)

    def test_probe_does_not_advance_the_simulation(self):
        parts = self._chain(chi_0=0.1)
        time_before = sim_inst.sys.time
        pos_before = np.copy(sim_inst.sys.part.all().pos)
        sim_inst.probe_magnetization_convergence(parts, n_iter=10)
        self.assertEqual(sim_inst.sys.time, time_before)
        np.testing.assert_allclose(sim_inst.sys.part.all().pos, pos_before)

    def test_converged_series_is_truncated_not_reported_as_noise(self):
        parts = self._chain(chi_0=0.02)
        ratios = sim_inst.probe_magnetization_convergence(parts, n_iter=10)
        self.assertGreater(len(ratios), 0)
        self.assertLess(ratios[-1], 1.)

    def test_exact_fixed_point_reports_no_ratio(self):
        anchor = sim_inst.sys.part.add(pos=[10., 10., 10.], fix=[True] * 3)
        virt = sim_inst.sys.part.add(pos=anchor.pos, rotation=[True] * 3,
                                     dip=[0., 0., self.m_sat])
        configure_magnetization(virt, AVAILABLE_MODELS[0], self.m_sat, 0.1,
                                anchor=anchor)
        sim_inst.set_H_ext(H=[0., 0., 1.])
        ratios = contraction_ratio(sim_inst.sys, [virt], n_iter=10)
        self.assertEqual(len(ratios), 0)


class MissingFeatureTest(unittest.TestCase):
    '''Models the build does not provide must fail loudly, not silently.'''

    def test_uncompiled_model_raises_missing_feature(self):
        unavailable = [name for name in MAGNETIZATION_MODELS
                       if name not in AVAILABLE_MODELS]
        if not unavailable:
            self.skipTest("every model is compiled in this espresso build")
        anchor = sim_inst.sys.part.add(pos=[5., 5., 5.])
        virt = sim_inst.sys.part.add(pos=anchor.pos)
        try:
            with self.assertRaises(MissingFeature):
                configure_magnetization(virt, unavailable[0], 1., 1., anchor=anchor)
        finally:
            sim_inst.sys.part.all().remove()


if __name__ == '__main__':
    unittest.main()
