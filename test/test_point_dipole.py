import unittest
from pressomancy.simulation import PointDipolePermanent, PointDipoleMagnetizable, PointDipoleSuperparamagnetic
from pressomancy.infra import api_agnostic_feature_check
from pressomancy.magnetodynamics import required_features_for, susceptibility_from_kT
from .create_system import sim_inst, BaseTestCase

import numpy as np
import espressomd.propagation

Propagation = espressomd.propagation.Propagation

MODEL = 'langevin'
_model_missing = [feature for feature in required_features_for(MODEL) if not api_agnostic_feature_check(feature)]


class PointDipoleTest(BaseTestCase):
    box_dim = (6, 6, 6)  # 11 objects: 10 of default size 1, 1 of size 2; two set_objects calls

    config=PointDipolePermanent.config.specify(
        dipm=1.2, size=2., espresso_handle=sim_inst.sys)

    def setUp(self) -> None:
        super().setUp()
        self.mag_part = [PointDipolePermanent(config=PointDipolePermanent.config.specify(espresso_handle=sim_inst.sys)) for _ in range(10)]
        self.mag_part.append(PointDipolePermanent(config=self.config))
        sim_inst.store_objects(self.mag_part)
        # One set_objects call per size.
        sim_inst.set_objects(self.mag_part[:10])
        sim_inst.set_objects(self.mag_part[10:])

    def test_set_object_generic(self):
        """part_types and dipole moments match the default and per-object config."""
        for name, type_id in PointDipolePermanent.part_types.items():
            self.assertEqual(sim_inst.part_types[name], type_id)
        self.assertEqual(sim_inst.part_types["pdp_real"], 61)

        parts = sim_inst.sys.part.select(type=sim_inst.part_types['pdp_real'])
        self.assertEqual(len(parts), len(self.mag_part))
        self.assertEqual(len(self.mag_part), 11)

        moments = sorted(np.linalg.norm(p.dip) for p in parts)
        np.testing.assert_allclose(moments[:10], [1.0] * 10, rtol=1e-6)
        np.testing.assert_allclose(moments[10], 1.2, rtol=1e-6)


@unittest.skipIf(_model_missing, f"Missing required features: {', '.join(_model_missing)}")
class PointDipoleMagnetizableTest(BaseTestCase):
    box_dim = (6, 6, 6)  # 11 objects: 10 of default size 1, 1 of size 0.5; two set_objects calls

    config=PointDipoleMagnetizable.config.specify(
        magnetization_model=MODEL, dipm_sat=1.5, mag_susc_0=0.5,
        size=0.5, espresso_handle=sim_inst.sys)

    def setUp(self) -> None:
        super().setUp()
        self.mag_part = [PointDipoleMagnetizable(config=PointDipoleMagnetizable.config.specify(espresso_handle=sim_inst.sys)) for _ in range(10)]
        self.mag_part.append(PointDipoleMagnetizable(config=self.config))
        sim_inst.store_objects(self.mag_part)
        # One set_objects call per size.
        sim_inst.set_objects(self.mag_part[:10])
        sim_inst.set_objects(self.mag_part[10:])

    def test_set_object(self):
        """part_types, model config, anchor binding and the saturated seed match the default and per-object config."""
        for name, type_id in PointDipoleMagnetizable.part_types.items():
            self.assertEqual(sim_inst.part_types[name], type_id)
        self.assertEqual(sim_inst.part_types["pdm_real"], 62)
        self.assertEqual(sim_inst.part_types["pdm_virt"], 622)

        specified_virt = self.mag_part[-1].type_part_dict['pdm_virt'][0]
        self.assertEqual(specified_virt.mag_susc_0, 0.5)
        self.assertEqual(specified_virt.dipm_sat, 1.5)
        self.assertAlmostEqual(specified_virt.dipm, self.mag_part[-1].params['dipm_sat'])

        for obj in self.mag_part[:-1]:
            p_virt = obj.type_part_dict['pdm_virt'][0]
            self.assertEqual(p_virt.mag_susc_0, 0.1)
            self.assertEqual(p_virt.dipm_sat, 1.)
            self.assertAlmostEqual(p_virt.dipm, obj.params['dipm_sat'])

        self.assertIs(specified_virt.langevin_magnetization_is_enabled, True)
        self.assertIs(specified_virt.froelich_kennelly_is_enabled, False)

        p_virt = next(iter(sim_inst.sys.part.select(type=sim_inst.part_types["pdm_virt"])))
        self.assertTrue(p_virt.is_virtual())
        anchor = sim_inst.sys.part.by_id(p_virt.vs_relative[0])
        self.assertEqual(anchor.type, sim_inst.part_types["pdm_real"])
        self.assertEqual(p_virt.propagation, (Propagation.TRANS_VS_RELATIVE |
                                      Propagation.ROT_VS_INDEPENDENT))


@unittest.skipIf(_model_missing, f"Missing required features: {', '.join(_model_missing)}")
class PointDipoleSuperparamagneticTest(BaseTestCase):
    box_dim = (6, 6, 6)  # 11 objects: 10 of default size 1, 1 of size 0.5; two set_objects calls

    config=PointDipoleSuperparamagnetic.config.specify(
        magnetization_model=MODEL, dipm=2., kT=0.5,
        size=0.5, espresso_handle=sim_inst.sys)

    def setUp(self) -> None:
        super().setUp()
        self.mag_part = [PointDipoleSuperparamagnetic(config=PointDipoleSuperparamagnetic.config.specify(espresso_handle=sim_inst.sys)) for _ in range(10)]
        self.mag_part.append(PointDipoleSuperparamagnetic(config=self.config))
        sim_inst.store_objects(self.mag_part)
        # One set_objects call per size.
        sim_inst.set_objects(self.mag_part[:10])
        sim_inst.set_objects(self.mag_part[10:])

    def test_set_object(self):
        """part_types, susceptibility-from-kT config, anchor binding and the saturated seed match the default and per-object config."""
        for name, type_id in PointDipoleSuperparamagnetic.part_types.items():
            self.assertEqual(sim_inst.part_types[name], type_id)
        self.assertEqual(sim_inst.part_types["pds_real"], 63)
        self.assertEqual(sim_inst.part_types["pds_virt"], 633)

        specified_virt = self.mag_part[-1].type_part_dict['pds_virt'][0]
        expected_susc = susceptibility_from_kT(2., 0.5)
        self.assertEqual(specified_virt.mag_susc_0, expected_susc)
        self.assertEqual(specified_virt.dipm_sat, 2.)
        self.assertAlmostEqual(specified_virt.dipm, self.mag_part[-1].params['dipm'])

        default_susc = susceptibility_from_kT(1., 1.)
        for obj in self.mag_part[:-1]:
            p_virt = obj.type_part_dict['pds_virt'][0]
            self.assertEqual(p_virt.mag_susc_0, default_susc)
            self.assertEqual(p_virt.dipm_sat, 1.)
            self.assertAlmostEqual(p_virt.dipm, obj.params['dipm'])

        self.assertIs(specified_virt.langevin_magnetization_is_enabled, True)
        self.assertIs(specified_virt.froelich_kennelly_is_enabled, False)

        p_virt = next(iter(sim_inst.sys.part.select(type=sim_inst.part_types["pds_virt"])))
        self.assertTrue(p_virt.is_virtual())
        anchor = sim_inst.sys.part.by_id(p_virt.vs_relative[0])
        self.assertEqual(anchor.type, sim_inst.part_types["pds_real"])
        self.assertEqual(p_virt.propagation, (Propagation.TRANS_VS_RELATIVE |
                                      Propagation.ROT_VS_INDEPENDENT))
