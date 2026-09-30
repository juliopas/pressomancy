import unittest
from pressomancy.simulation import SWPart
from pressomancy.infra import api_agnostic_feature_check
from .create_system import sim_inst, BaseTestCase

import numpy as np
import espressomd
import espressomd.propagation

Propagation = espressomd.propagation.Propagation

_sw_missing = [feature for feature in SWPart.required_features if not api_agnostic_feature_check(feature)]

@unittest.skipIf(_sw_missing, f"Missing required features: {', '.join(_sw_missing)}")
class SWPartTest(BaseTestCase):
    box_dim = (6, 6, 6)  # 11 objects of size 1/0.5, placed via two set_objects calls

    config = SWPart.config.specify(
        anisotropy_field_inv=0.3, sat_mag=2.5, anisotropy_energy=8.,
        sw_dt_incr=2.0e-10, sw_tau0_inv=5.0e8,
        size=0.5, espresso_handle=sim_inst.sys)

    def setUp(self) -> None:
        super().setUp()
        self.mag_part = [SWPart(config=SWPart.config.specify(espresso_handle=sim_inst.sys)) for _ in range(10)]
        self.mag_part.append(SWPart(config=self.config))
        sim_inst.store_objects(self.mag_part)
        # One set_objects call per size.
        sim_inst.set_objects(self.mag_part[:10])
        sim_inst.set_objects(self.mag_part[10:])

    def test_set_object(self):
        """part_types, magnetodynamics config, anchor binding and the saturated seed match the default and per-object config."""
        for name, type_id in SWPart.part_types.items():
            self.assertEqual(sim_inst.part_types[name], type_id)
        self.assertEqual(sim_inst.part_types["sw_real"], 9)
        self.assertEqual(sim_inst.part_types["sw_virt"], 10)

        specified_virt = self.mag_part[-1].type_part_dict['sw_virt'][0]
        self.assertIs(specified_virt.magnetodynamics['is_enabled'], True)
        self.assertEqual(specified_virt.magnetodynamics['anisotropy_field_inv'], 0.3)
        self.assertEqual(specified_virt.magnetodynamics['sat_mag'], 2.5)
        self.assertEqual(specified_virt.magnetodynamics['anisotropy_energy'], 8.)
        self.assertEqual(specified_virt.magnetodynamics['sw_dt_incr'], 2.0e-10)
        self.assertEqual(specified_virt.magnetodynamics['sw_tau0_inv'], 5.0e8)

        for obj in self.mag_part[:-1]:
            p_virt = obj.type_part_dict['sw_virt'][0]
            self.assertEqual(p_virt.magnetodynamics['anisotropy_field_inv'], 0.175)
            self.assertEqual(p_virt.magnetodynamics['sat_mag'], 1.75)

        p_virt = next(iter(sim_inst.sys.part.select(type=sim_inst.part_types["sw_virt"])))
        self.assertTrue(p_virt.is_virtual())
        anchor = sim_inst.sys.part.by_id(p_virt.vs_relative[0])
        self.assertEqual(anchor.type, sim_inst.part_types["sw_real"])
        self.assertEqual(p_virt.propagation, (Propagation.TRANS_VS_RELATIVE |
                                      Propagation.ROT_VS_INDEPENDENT))

        # a zero magnitude dipole cannot be used to infer an orientation on I/O,
        # and SWPart.set_object itself asserts dipm matches sat_mag before returning
        for obj in self.mag_part:
            p_virt = obj.type_part_dict['sw_virt'][0]
            np.testing.assert_allclose(p_virt.dipm, p_virt.magnetodynamics['sat_mag'])
