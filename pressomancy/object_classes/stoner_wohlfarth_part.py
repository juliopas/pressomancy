'''
``SWPart``: a magnetic particle driven by ESPResSo's thermal Stoner-Wohlfarth
(tSW) kinetic Monte Carlo magnetization model. Requires ESPResSo to be built
with ``THERMAL_STONER_WOHLFARTH`` (which itself requires NLOPT).
'''
from pressomancy.object_classes.part_class import GenericPart
from pressomancy.object_classes.object_class import ObjectConfigParams
from pressomancy.helper_functions import PartDictSafe, SinglePairDict
import espressomd
import numpy as np
if espressomd.version.major() == 5:
    import espressomd.propagation
    Propagation = espressomd.propagation.Propagation
# else:
#     raise ImportError(f"Unsupported espressomd version: {espressomd.version.major()}. This code requires espressomd version 5 or higher.")

class SWPart(GenericPart):

    '''
    Class that contains SWPart relevant parameters and methods. At construction one must pass an espresso handle because the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instance there will be only one type of a SWPart. Therefore many relevant parameters are class specific, not instance specific.
    '''
    required_features=GenericPart.required_features + ['THERMAL_STONER_WOHLFARTH', 'DIPOLES', 'VIRTUAL_SITES_RELATIVE']

    numInstances = 0
    simulation_type= SinglePairDict('sw_part', 13)
    part_types = PartDictSafe({'sw_real': 9,'sw_virt': 10})
    config = ObjectConfigParams(
        anisotropy_field_inv=0.175, # inverse anisotropy field (1/H_k) in reduced units
        sat_mag=1.75, # saturation magnetisation in reduced units
        anisotropy_energy=5., # anisotropy energy K * V in reduced units
        sw_dt_incr=1.0e-10, # kinetic Monte Carlo time increment [s]
        sw_tau0_inv=1.0e9  # inverse attempt time (1/tau_0) [1/s]
    )

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a SWPart object requires the specification of particle size and a handle to the espresso system
        '''
        super().__init__(config)

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos parameter passed is commensurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of filament stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        magnetodynamics_setup={'is_enabled': True,'anisotropy_field_inv':self.params['anisotropy_field_inv'],'sat_mag':self.params['sat_mag'],'anisotropy_energy' : self.params['anisotropy_energy'], 'sw_dt_incr':self.params['sw_dt_incr'],'sw_tau0_inv':self.params['sw_tau0_inv']}

        particl_real=self.add_particle(type_name='sw_real', pos=pos, rotation=(True, True, True), director=ori)

        particl_virt=self.add_particle(type_name='sw_virt', pos=pos, rotation=(False, False, False), dip=self.params['sat_mag']*ori)
        particl_virt.magnetodynamics = magnetodynamics_setup
        particl_virt.vs_auto_relate_to(particl_real)
        if espressomd.version.major() == 5:
            particl_virt.propagation = Propagation.TRANS_VS_RELATIVE | Propagation.ROT_VS_INDEPENDENT

        written = particl_virt.magnetodynamics
        assert written['is_enabled'], \
                f"The magnetization model was not enabled on particle {particl_virt.id}. " \
                f"Espresso ignores a magnetodynamics dict it does not recognise instead of " \
                f"raising, so this usually means the parameter names have changed. " \
                f"Wrote {magnetodynamics_setup}, read back {written}."
        if not np.allclose(particl_virt.dipm, written['sat_mag']):
            raise ValueError(f"Error in setting dipole moment for virtual particle. Expected {written['sat_mag']}, got {particl_virt.dipm}. Check the parameters passed to the config object and the logic in this method.")

        return self
