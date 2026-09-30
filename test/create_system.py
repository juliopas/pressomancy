from pressomancy.simulation import Simulation
import logging
import unittest
import gc

class BaseTestCase(unittest.TestCase):
    """A class that touches the shared system: it resets into its ``box_dim`` before its first test and after
    every test (``addCleanup``, so also when a subclass setUp fails half-way). There is no class-end restore:
    the next class resets into its own box. A class that needs no system extends ``unittest.TestCase`` instead."""

    box_dim=(20,20,20)
    min_global_cut=1

    @classmethod
    def setUpClass(cls):
        # Configure logging for tests
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)

        if logger.hasHandlers():
            logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        BaseTestCase.cleanup(cls.box_dim)

    def setUp(self):
        super().setUp()
        self.addCleanup(BaseTestCase.cleanup, self.box_dim)

    @staticmethod
    def reset_io_state():
        """Close whatever HDF5 file is open and clear every field of `io_dict`."""
        io_dict = sim_inst.io_dict
        h5_file = io_dict.get('h5_file')
        if h5_file is not None:
            h5_file.flush()
            h5_file.close()
        io_dict['h5_file'] = None
        io_dict['flat_part_view'].clear()
        io_dict['registered_observables'] = {}
        io_dict['registered_group_type'] = None
        io_dict['bonds'] = False
        io_dict['bond_links'] = {}
        sim_inst._h5_writer._slice_cache.clear()

    @staticmethod
    def cleanup(box_dim):
        """Reset the simulation instance, leaving it in ``box_dim`` (pass the class's own, ``self.box_dim``)."""
        BaseTestCase.reset_io_state()
        sim_inst.reinitialize_instance()
        # Each setter rebuilds the cell grid even for an unchanged value.
        if tuple(sim_inst.sys.box_l) != tuple(box_dim):
            sim_inst.sys.box_l=box_dim
        if sim_inst.sys.min_global_cut != BaseTestCase.min_global_cut:
            sim_inst.sys.min_global_cut=BaseTestCase.min_global_cut
        gc.collect()


sim_inst = Simulation(box_dim=BaseTestCase.box_dim)
sim_inst.set_sys(min_global_cut=BaseTestCase.min_global_cut)


def bonds_using(handle):
    """The number of ``part.bonds`` entries, over every particle, whose bond is the espresso bond ``handle``."""
    return sum(bond[0] == handle for part in sim_inst.sys.part.all() for bond in part.bonds)
