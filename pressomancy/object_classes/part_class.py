from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams 
from pressomancy.helper_functions import PartDictSafe, SinglePairDict

class GenericPart(metaclass=Simulation_Object):

    '''
    Generic simulation particle object.

    This class provides a minimal wrapper around an ESPResSo particle. It
    manages the particle's ESPResSo handle, configuration parameters, and
    associated objects, and provides a method for creating a real particle
    with rotational degrees of freedom.
    '''
    required_features=['ROTATION']
    numInstances = 0
    simulation_type= SinglePairDict('generic_particle', 42)
    part_types = PartDictSafe({'real': 1,'virt': 2})
    config = ObjectConfigParams(
        espresso_part_kwargs=dict(),
        alias=None
    )

    def __init__(self, config: ObjectConfigParams):
        self.sys=config['espresso_handle']
        self.params=config
        self.associated_objects=self.params['associated_objects']
        self.type_part_dict=PartDictSafe({key: [] for key in GenericPart.part_types.keys()})
        GenericPart.numInstances += 1

    def set_object(self,  pos, ori):
        particle=self.add_particle(type_name='real', pos=pos, rotation=(True, True, True), **self.params['espresso_part_kwargs'])
        particle.director = ori
        return self
