from pressomancy.object_classes.point_dipole import (PointDipolePermanent,
                                                     PointDipoleMagnetizable,
                                                     PointDipoleSuperparamagnetic)
from pressomancy.object_classes.quadriplex_class import Quadriplex, Quartet
from pressomancy.object_classes.crowder_class import Crowder
from pressomancy.object_classes.filament_class import Filament
from pressomancy.object_classes.otp_molecule_class import OTP
from pressomancy.object_classes.stoner_wohlfarth_part import SWPart
from pressomancy.object_classes.egg_model_part import EGGPart
from pressomancy.object_classes.raspberry_sphere import RaspberrySphere
from pressomancy.object_classes.tel_sequence import TelSeq
from pressomancy.object_classes.elastomer import Elastomer
from pressomancy.object_classes.part_class import GenericPart
from pressomancy.object_classes.rigid_obj import GenericRigidObj
from pressomancy.object_classes.multicore_particle import MulticorePart

#: Maps an object class name to the class itself, for resolving names read back
#: out of an HDF5 file (the connectivity datasets store class names as strings).
#: This exists so code outside this package -- notably ``pressomancy.io`` -- can
#: resolve a name without relying on a wildcard import having populated its own
#: module globals, which is how the lookup used to work.
OBJECT_CLASS_REGISTRY = {
    cls.__name__: cls
    for cls in (
        PointDipolePermanent,
        PointDipoleMagnetizable,
        PointDipoleSuperparamagnetic,
        Quadriplex,
        Quartet,
        Crowder,
        Filament,
        OTP,
        SWPart,
        EGGPart,
        RaspberrySphere,
        TelSeq,
        Elastomer,
        GenericPart,
        GenericRigidObj,
        MulticorePart,
    )
}



