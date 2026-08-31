from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams
from pressomancy.helper_functions import PartDictSafe, SinglePairDict
from pressomancy.magnetodynamics import (COMMON_FEATURES, configure_magnetization,
                                         required_features_for, susceptibility_from_kT,
                                         validate_model)

class PointDipolePermanent(metaclass=Simulation_Object):
    '''
    Class that contains permanent magnetic point dipole particles relevant paramaters and methods. At construction one must pass an espresso handle because the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a PointDipolePermanent. Therefore many relevant parameters are class specific, not instance specific.
    '''

    required_features=['DIPOLES', 'ROTATION']
    numInstances = 0
    simulation_type= SinglePairDict('point_dipole_permanent', 3)
    part_types = PartDictSafe({'pdp_real': 61})
    config = ObjectConfigParams(
         dipm=1.
    )

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a PointDipolePermanent object requires the specification of particle size and a handle to the espresso system
        '''
        self.sys=config['espresso_handle']
        self.params=config
        self.associated_objects=config['associated_objects']
        self.type_part_dict=PartDictSafe({key: [] for key in PointDipolePermanent.part_types.keys()})
        assert self.associated_objects is None, "Point dipoles can not have associated objects. They are singular particles, as basic as possible."
        PointDipolePermanent.numInstances += 1

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        dipm= self.params['dipm']
        hndl = self.add_particle(type_name='pdp_real', pos=pos, rotation=[True, True, True], dip=(dipm * ori))

        return self

class PointDipoleMagnetizable(metaclass=Simulation_Object):
    '''
    Class that contains magnetizable point dipole particles relevant paramaters and methods. At construction one must pass an espresso handle because the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a PointDipoleMagnetizable. Therefore many relevant parameters are class specific, not instance specific.

    The dipole moment is carried by a virtual particle whose magnitude and
    direction are updated by espresso every timestep, following the
    magnetization model named in ``config['magnetization_model']``. See
    :mod:`pressomancy.magnetodynamics` for the available models and their
    parameters.

    Because the required features depend on which model is picked, the class
    attribute only lists what every model needs; the model specific feature is
    added to the instance in __init__ and is what Simulation.sanity_check reads.
    '''

    required_features=list(COMMON_FEATURES) + ['ROTATION']
    numInstances = 0
    simulation_type= SinglePairDict('point_dipole_magnetizable', 4)
    part_types = PartDictSafe({'pdm_real': 62, 'pdm_virt': 622})
    config = ObjectConfigParams(
        magnetization_model='langevin',
        dipm_sat=1.,
        mag_susc_0=0.1,
    )

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a PointDipoleMagnetizable object requires the specification of particle size and a handle to the espresso system
        '''
        self.sys=config['espresso_handle']
        self.params=config
        validate_model(config['magnetization_model'])
        self.required_features=required_features_for(config['magnetization_model'])
        self.associated_objects=config['associated_objects']
        self.type_part_dict=PartDictSafe({key: [] for key in PointDipoleMagnetizable.part_types.keys()})
        assert self.associated_objects is None, "Point dipoles can not have associated objects. They are singular particles, as basic as possible."
        PointDipoleMagnetizable.numInstances += 1

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. A real anchor carries the steric interaction and the rotational degrees of freedom, while a virtual site bound to it carries the magnetic moment that espresso updates. The moment is seeded at saturation along ori, since a zero magnitude dipole cannot be used to infer an orientation on I/O.

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        particl_real=self.add_particle(type_name='pdm_real', pos=pos, rotation=[True, True, True], director=ori)

        particl_virt=self.add_particle(type_name='pdm_virt', pos=pos, rotation=[True, True, True], dip=self.params['dipm_sat']*ori)
        configure_magnetization(particl_virt,
                                model=self.params['magnetization_model'],
                                dipm_sat=self.params['dipm_sat'],
                                mag_susc_0=self.params['mag_susc_0'],
                                anchor=particl_real)

        return self

class PointDipoleSuperparamagnetic(metaclass=Simulation_Object):
    '''
    Class that contains superparamagnetic point dipole particles relevant paramaters and methods. At construction one must pass an espresso handle because the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instanse there will be only one type of a PointDipoleSuperparamagnetic. Therefore many relevant parameters are class specific, not instance specific.

    Same machinery as PointDipoleMagnetizable, but parameterised the way a
    superparamagnetic particle is usually described: by the magnitude of its
    point dipole and the thermal energy it fluctuates against, rather than by a
    susceptibility. The moment then follows the classical Langevin law
    m(H) = dipm * L(dipm * H / kT), which espresso reproduces once the
    susceptibility is set to dipm**2 / (3 kT). See
    pressomancy.magnetodynamics.susceptibility_from_kT for that conversion.

    Note that kT here describes the internal magnetic degree of freedom only.
    It is independent of the thermostat temperature, which governs the
    translational and rotational motion of the carrier particle, so the two need
    not agree if that is what the model calls for.
    '''

    required_features=list(COMMON_FEATURES) + ['ROTATION']
    numInstances = 0
    simulation_type= SinglePairDict('point_dipole_superparamagnetic', 7)
    part_types = PartDictSafe({'pds_real': 63, 'pds_virt': 633})
    config = ObjectConfigParams(
        magnetization_model='langevin',
        dipm=1.,
        kT=1.,
    )

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a PointDipoleSuperparamagnetic object requires the specification of particle size and a handle to the espresso system
        '''
        self.sys=config['espresso_handle']
        self.params=config
        validate_model(config['magnetization_model'])
        # fail here rather than at set_object time if dipm or kT are unphysical
        susceptibility_from_kT(config['dipm'], config['kT'])
        self.required_features=required_features_for(config['magnetization_model'])
        self.associated_objects=config['associated_objects']
        self.type_part_dict=PartDictSafe({key: [] for key in PointDipoleSuperparamagnetic.part_types.keys()})
        assert self.associated_objects is None, "Point dipoles can not have associated objects. They are singular particles, as basic as possible."
        PointDipoleSuperparamagnetic.numInstances += 1

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos paramater passed is commesurate with n_part. A real anchor carries the steric interaction and the rotational degrees of freedom, while a virtual site bound to it carries the magnetic moment that espresso updates. The moment is seeded at dipm along ori, since a zero magnitude dipole cannot be used to infer an orientation on I/O.

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        dipm = self.params['dipm']
        particl_real=self.add_particle(type_name='pds_real', pos=pos, rotation=[True, True, True], director=ori)

        particl_virt=self.add_particle(type_name='pds_virt', pos=pos, rotation=[True, True, True], dip=dipm*ori)
        configure_magnetization(particl_virt,
                                model=self.params['magnetization_model'],
                                dipm_sat=dipm,
                                mag_susc_0=susceptibility_from_kT(dipm, self.params['kT']),
                                anchor=particl_real)

        return self
