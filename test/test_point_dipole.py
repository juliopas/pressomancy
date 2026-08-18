from pressomancy.simulation import PointDipolePermanent, PointDipoleMagnetizable
from pressomancy.helper_functions import api_agnostic_feature_check
from pressomancy.magnetodynamics import required_features_for
from create_system import sim_inst, BaseTestCase


class PointDipoleTest(BaseTestCase):
    H_ext = [0,0,3.]
    config=PointDipolePermanent.config.specify(
        dipm=1.2, size=2., espresso_handle=sim_inst.sys)

    def tearDown(self) -> None:
        self.mag_part=None
        self.cleanup()
        self.assertEqual(len(sim_inst.sys.part),0)

    def setUp(self) -> None:
        self.mag_part = [PointDipolePermanent(config=PointDipolePermanent.config.specify(espresso_handle=sim_inst.sys)) for _ in range(10)]
        self.mag_part.append(PointDipolePermanent(config=self.config))
        sim_inst.store_objects(self.mag_part)
        sim_inst.set_objects(self.mag_part)

    def test_set_object_generic(self):

        assert sim_inst.part_types["pdp_real"] == 61

MODEL = 'langevin'

if all(api_agnostic_feature_check(feature) for feature in required_features_for(MODEL)):
    import espressomd.propagation
    Propagation = espressomd.propagation.Propagation

    class PointDipoleMagnetizableTest(BaseTestCase):
        H_ext = [0,0,3.]
        config=PointDipoleMagnetizable.config.specify(
            magnetization_model=MODEL, dipm_sat=1., mag_susc_0=0.5,
            size=0.5, espresso_handle=sim_inst.sys)

        def tearDown(self) -> None:
            self.mag_part=None
            self.cleanup()
            self.assertEqual(len(sim_inst.sys.part),0)

        def setUp(self) -> None:
            self.mag_part = [PointDipoleMagnetizable(config=PointDipoleMagnetizable.config.specify(espresso_handle=sim_inst.sys)) for _ in range(10)]
            self.mag_part.append(PointDipoleMagnetizable(config=self.config))
            sim_inst.store_objects(self.mag_part)
            sim_inst.set_objects(self.mag_part)

        def test_set_object_generic(self):
            assert sim_inst.part_types["pdm_real"] == 62 and sim_inst.part_types["pdm_virt"] == 666

        def test_model_written_to_virtual_site(self):
            p_virt = next(iter(sim_inst.sys.part.select(type=sim_inst.part_types["pdm_virt"])))
            # defaults come from the class level config
            assert p_virt.dipm_sat == 1.
            assert p_virt.mag_susc_0 == 1.
            assert p_virt.langevin_magnetization_is_enabled is True
            assert p_virt.froelich_kennelly_is_enabled is False

        def test_virtual_site_binding(self):
            p_virt = next(iter(sim_inst.sys.part.select(type=sim_inst.part_types["pdm_virt"])))
            assert p_virt.is_virtual()
            anchor = sim_inst.sys.part.by_id(p_virt.vs_relative[0])
            assert anchor.type == sim_inst.part_types["pdm_real"]
            assert p_virt.propagation == (Propagation.TRANS_VS_RELATIVE |
                                          Propagation.ROT_VS_INDEPENDENT)

        def test_moment_seeded_at_saturation(self):
            # a zero magnitude dipole cannot be used to infer an orientation on I/O
            for p_virt in sim_inst.sys.part.select(type=sim_inst.part_types["pdm_virt"]):
                assert p_virt.dipm > 0.

        def test_specified_config_overrides_defaults(self):
            assert self.config['dipm_sat'] == 1.
            assert self.config['mag_susc_0'] == 0.5
            assert self.mag_part[-1].required_features == required_features_for(MODEL)
