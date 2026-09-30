'''
``EGGPart``: a magnetic particle driven by ESPResSo's egg-model Brownian
dipole dynamics, built from a real reference particle plus a virtual ``yolk``
particle carrying the dipole moment. Gated behind the ``EGG_MODEL`` feature.
'''
from pressomancy.object_classes.part_class import GenericPart
from pressomancy.object_classes.object_class import ObjectConfigParams
from pressomancy.infra import TypeDictSafe, SimulationType
import espressomd
import espressomd.propagation
Propagation = espressomd.propagation.Propagation

class EGGPart(GenericPart):
    """
    Magnetic particle represented by ESPResSo's egg-model dynamics.

    An ``EGGPart`` is built as a real reference particle plus a virtual
    ``yolk`` particle carrying the dipole moment. The reference particle
    controls the yolk position through a relative virtual-site relation, while
    the yolk orientation is evolved independently by the egg-model Brownian
    rotational dynamics.

    The ``ori`` argument passed to :meth:`set_object` initializes both the
    reference frame and the initial yolk dipole direction. The easy axis is
    configured separately through ``axis_quat_body``, which is interpreted in
    the body-fixed frame of the real reference particle. By default,
    ``axis_quat_body`` is the identity quaternion, so the easy axis is aligned
    with ``ori``.
    """
    required_features = GenericPart.required_features + ['EGG_MODEL', 'DIPOLES', 'VIRTUAL_SITES_RELATIVE']
    simulation_type = SimulationType('egg_part', 74)
    part_types = TypeDictSafe({'yolk': 11})
    config = ObjectConfigParams(
        dipm=1,
        gamma=1.,
        anisotropy_energy=1.,
        axis_quat_body=[1, 0, 0, 0],
    )

    def __init__(self, config: ObjectConfigParams):
        """
        Initialize an egg-model particle wrapper.

        Parameters
        ----------
        config : ObjectConfigParams
            Configuration containing an ESPResSo system handle, dipole moment,
            egg friction, anisotropy energy, and body-frame easy-axis
            quaternion.
        """
        self.sys = config['espresso_handle']
        self.params = config
        self.associated_objects = config['associated_objects']
        self.type_part_dict = {key: [] for key in EGGPart.part_types}

    def set_object(self, pos, ori):
        """
        Add the real reference particle and egg-model yolk to the system.

        Parameters
        ----------
        pos : array_like
            Initial position of the particle pair.
        ori : array_like
            Initial director for the reference particle and the initial yolk
            dipole direction.

        Returns
        -------
        EGGPart
            The configured object instance.
        """
        particl_real = self.add_particle(
            type_name='real', pos=pos, rotation=(True, True, True),
            director=ori)
        particl_virt = self.add_particle(
            type_name='yolk', pos=pos, rotation=(True, True, True),
            director=ori, dipm=self.params['dipm'])
        particl_virt.vs_auto_relate_to(particl_real)

        magnetodynamics_setup = {
            "is_enabled": True,
            "gamma": self.params['gamma'],
            "anisotropy_energy": self.params['anisotropy_energy'],
            "axis_quat_body": self.params['axis_quat_body'],
        }
        particl_virt.magnetodynamics.egg = magnetodynamics_setup
        particl_virt.propagation = (Propagation.TRANS_VS_RELATIVE |
                                    Propagation.ROT_VS_INDEPENDENT)

        return self
