import types

from .create_system import BaseTestCase, sim_inst
from pressomancy.infra import (MissingFeature, SimulationExistsException, SimulationType,
                               TypeDictSafe, particle_attribute_check)
from pressomancy.object_classes import OBJECT_CLASS_REGISTRY
from pressomancy.object_classes.object_class import (ObjectConfigParams,
                                                     Simulation_Object,
                                                     SIMULATION_TYPE_OWNERS)
from pressomancy.object_classes.filament_class import Filament
from pressomancy.object_classes.crowder_class import Crowder
from pressomancy.simulation import Simulation


def make_object_class(name, simulation_type):
    """A throwaway object class built through the metaclass, minimal required attrs."""
    return Simulation_Object(name, (), {
        'required_features': [],
        'part_types': TypeDictSafe(),
        'simulation_type': simulation_type,
        'config': ObjectConfigParams(),
    })


class ObjectConfigParamsTest(BaseTestCase):
    def test_sigma_only_where_it_is_read(self):
        """`sigma` is no common key: a class that ignores it rejects it."""
        with self.assertRaisesRegex(ValueError, "Invalid keys"):
            Crowder.config.specify(sigma=1.)
        self.assertEqual(Filament.config.specify(sigma=2.)['sigma'], 2.)


class TypeDictSafeTest(BaseTestCase):
    """`part_types` is a strict `str` -> `int` bijection: a typo or a duplicate number must fail loudly instead of reaching espresso as a bogus type."""

    def test_entries_and_mapping_are_enforced(self):
        with self.assertRaises(TypeError):
            TypeDictSafe({1: 1})
        with self.assertRaises(TypeError):
            TypeDictSafe({'real': []})
        with self.assertRaises(TypeError):
            TypeDictSafe({'real': True})
        types_ = TypeDictSafe({'real': 1})
        with self.assertRaises(TypeError):
            types_['virt'] = 2.
        self.assertEqual(dict(types_), {'real': 1})

        with self.assertRaises(RuntimeError):
            types_['real'] = 2
        with self.assertRaises(RuntimeError):
            types_['virt'] = 1
        types_['virt'] = 2
        self.assertEqual(dict(types_), {'real': 1, 'virt': 2})

        types_ = TypeDictSafe([('real', 1)])
        self.assertEqual(types_['real'], 1)
        with self.assertRaises(KeyError):
            types_['nonmang']
        self.assertEqual(dict(types_), {'real': 1})


class SimulationTypeTest(BaseTestCase):
    """The metaclass owns the uniqueness of `simulation_type` across object classes: the name and the number both end up in HDF5 output, so neither may be reused."""

    throwaway = 'ThrowawayObject'

    def tearDown(self):
        SIMULATION_TYPE_OWNERS.pop(self.throwaway, None)
        super().tearDown()

    def test_pair_uniqueness_is_enforced(self):
        taken = Filament.simulation_type
        with self.assertRaises(ValueError):
            make_object_class(self.throwaway, SimulationType(taken.key, 9999))
        with self.assertRaises(ValueError):
            make_object_class(self.throwaway, SimulationType('throwaway', taken.value))
        self.assertNotIn(self.throwaway, SIMULATION_TYPE_OWNERS)

        with self.assertRaises(TypeError):
            make_object_class(self.throwaway, SimulationType('throwaway', '9999'))
        with self.assertRaises(TypeError):
            make_object_class(self.throwaway, SimulationType(9999, 9999))

        pair = SimulationType('throwaway', 9999)
        make_object_class(self.throwaway, pair)
        make_object_class(self.throwaway, pair)
        self.assertEqual(SIMULATION_TYPE_OWNERS[self.throwaway], pair)


class ReinitializeCountersTest(BaseTestCase):
    """`reinitialize_instance` must leave the object classes as a fresh interpreter would: `who_am_i` restarts at zero, and so do the metaclass bookkeeping attributes."""

    def tearDown(self):
        BaseTestCase.cleanup()
        super().tearDown()

    def test_counters_rewind_for_every_registered_class(self):
        filaments = [Filament(config=Filament.config.specify(
            n_parts=3, sigma=1., size=3., espresso_handle=sim_inst.sys))
            for _ in range(2)]
        sim_inst.store_objects(filaments)
        self.assertEqual([obj.who_am_i for obj in filaments], [0, 1])
        self.assertEqual(Filament.numInstances, 2)
        self.assertEqual(len(Filament.live_instances), 2)

        del filaments
        BaseTestCase.cleanup()

        for object_class in OBJECT_CLASS_REGISTRY.values():
            with self.subTest(object_class=object_class.__name__):
                self.assertEqual(object_class.instance_id_counter, 0)
                self.assertEqual(object_class.numInstances, 0)
                self.assertEqual(len(object_class.live_instances), 0)

        again = Filament(config=Filament.config.specify(
            n_parts=3, sigma=1., size=3., espresso_handle=sim_inst.sys))
        self.assertEqual(again.who_am_i, 0)


class SingletonTest(BaseTestCase):
    """A second `Simulation(...)` while one is live is refused; the live instance stays exactly what it was."""

    def test_second_instance_is_refused(self):
        instance = sim_inst.instance
        with self.assertRaises(SimulationExistsException):
            Simulation(box_dim=(20, 20, 20))
        self.assertIs(sim_inst.instance, instance)


class ParticleAttributeCheckTest(BaseTestCase):
    """An existing particle attribute passes silently; a missing one raises `MissingFeature`."""

    def test_missing_attribute_raises(self):
        handle = types.SimpleNamespace(dip=[0., 0., 1.])
        particle_attribute_check(handle, 'dip')
        with self.assertRaises(MissingFeature), self.assertLogs(level='WARNING'):
            particle_attribute_check(handle, 'no_such_attribute')
