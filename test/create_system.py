from pressomancy.simulation import Simulation
import logging
import unittest
import gc

class BaseTestCase(unittest.TestCase):

    box_dim=(50,50,50)
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
    def cleanup(box_dim=None):
        """Reset the simulation instance after each test, leaving it in ``box_dim``
        (default: the shared box). A class that works in its own box resets into it
        between its tests and restores the shared one once, as a class cleanup."""
        box_dim = BaseTestCase.box_dim if box_dim is None else box_dim
        BaseTestCase.reset_io_state()
        sim_inst.reinitialize_instance()
        # Each setter rebuilds the cell grid (~0.1-0.25 s in 50^3) even for an unchanged value.
        if tuple(sim_inst.sys.box_l) != tuple(box_dim):
            sim_inst.sys.box_l=box_dim
        if sim_inst.sys.min_global_cut != BaseTestCase.min_global_cut:
            sim_inst.sys.min_global_cut=BaseTestCase.min_global_cut
        gc.collect()


class BoxTestCase(BaseTestCase):
    """A class that works in its own ``box_dim``: reset into it before its first test and after every
    test (``addCleanup``, so also when a subclass setUp fails half-way); the shared box comes back once,
    as a class cleanup (so also when a subclass setUpClass fails). A cell-grid rebuild costs ~125 ms in
    the shared 50^3 box and a few ms in a small one."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.addClassCleanup(BaseTestCase.cleanup)
        BaseTestCase.cleanup(cls.box_dim)

    def setUp(self):
        super().setUp()
        self.addCleanup(BaseTestCase.cleanup, self.box_dim)


sim_inst = Simulation(box_dim=BaseTestCase.box_dim)
sim_inst.set_sys(min_global_cut=BaseTestCase.min_global_cut)
