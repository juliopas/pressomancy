from pressomancy.simulation import SWPart
from pressomancy.helper_functions import api_agnostic_feature_check
from .create_system import sim_inst, BaseTestCase

import numpy as np
import espressomd

if all(api_agnostic_feature_check(feature) for feature in SWPart.required_features):
    import espressomd.propagation
    Propagation = espressomd.propagation.Propagation

    class SWPartTest(BaseTestCase):
        config = SWPart.config.specify(
            anisotropy_field_inv=0.3, sat_mag=2.5, anisotropy_energy=8.,
            sw_dt_incr=2.0e-10, sw_tau0_inv=5.0e8,
            size=0.5, espresso_handle=sim_inst.sys)

        def tearDown(self) -> None:
            self.mag_part=None
            self.cleanup()
            self.assertEqual(len(sim_inst.sys.part),0)

        def setUp(self) -> None:
            self.mag_part = [SWPart(config=SWPart.config.specify(espresso_handle=sim_inst.sys)) for _ in range(10)]
            self.mag_part.append(SWPart(config=self.config))
            sim_inst.store_objects(self.mag_part)
            sim_inst.set_objects(self.mag_part)

        def test_set_object_generic(self):
            for name, type_id in SWPart.part_types.items():
                assert sim_inst.part_types[name] == type_id
            assert sim_inst.part_types["sw_real"] == 9 and sim_inst.part_types["sw_virt"] == 10

        def test_model_written_to_virtual_site(self):
            specified_virt = self.mag_part[-1].type_part_dict['sw_virt'][0]
            assert specified_virt.magnetodynamics['is_enabled'] is True
            assert specified_virt.magnetodynamics['anisotropy_field_inv'] == 0.3
            assert specified_virt.magnetodynamics['sat_mag'] == 2.5
            assert specified_virt.magnetodynamics['anisotropy_energy'] == 8.
            assert specified_virt.magnetodynamics['sw_dt_incr'] == 2.0e-10
            assert specified_virt.magnetodynamics['sw_tau0_inv'] == 5.0e8

            for obj in self.mag_part[:-1]:
                p_virt = obj.type_part_dict['sw_virt'][0]
                assert p_virt.magnetodynamics['anisotropy_field_inv'] == 0.175, 'default config object must keep the default'
                assert p_virt.magnetodynamics['sat_mag'] == 1.75

        def test_virtual_site_binding(self):
            p_virt = next(iter(sim_inst.sys.part.select(type=sim_inst.part_types["sw_virt"])))
            assert p_virt.is_virtual()
            anchor = sim_inst.sys.part.by_id(p_virt.vs_relative[0])
            assert anchor.type == sim_inst.part_types["sw_real"]
            if espressomd.version.major() == 5:
                assert p_virt.propagation == (Propagation.TRANS_VS_RELATIVE |
                                              Propagation.ROT_VS_INDEPENDENT)

        def test_moment_seeded_at_saturation(self):
            # a zero magnitude dipole cannot be used to infer an orientation on I/O,
            # and SWPart.set_object itself asserts dipm matches sat_mag before returning
            for obj in self.mag_part:
                p_virt = obj.type_part_dict['sw_virt'][0]
                np.testing.assert_allclose(p_virt.dipm, p_virt.magnetodynamics['sat_mag'])
