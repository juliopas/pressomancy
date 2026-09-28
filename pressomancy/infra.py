'''
Core infrastructure shared across pressomancy.

Covers the ``ManagedSimulation`` singleton decorator;
the ``MissingFeature``/``SimulationExistsException`` exceptions;
the guarded container types (``SimulationType``, ``TypeDictSafe``); ``RoutineWithArgs``;
the ESPResSo capability probes (``api_agnostic_feature_check``,
``particle_attribute_check``); ``BondWrapper``;
and the git provenance helpers (``get_repo_context``, ``get_submission_creator_info``).

Importing this module fails loudly on ESPResSo 4: pressomancy is ESPResSo 5
only.
'''
import inspect
import logging
import subprocess
import sys as sysos
import weakref
from typing import NamedTuple
from pathlib import Path

import numpy as np

import espressomd
import espressomd.code_features
import espressomd.version

class MissingFeature(Exception):
    pass


#: pressomancy targets the ESPResSo 5 python API exclusively; a v4 interpreter
#: would fail later, in unrelated places, so refuse it here.
if espressomd.version.major() != 5:
    raise MissingFeature(
        f"pressomancy requires ESPResSo 5.x, but this interpreter provides "
        f"{espressomd.version.friendly()}. Run your scripts with the pypresso "
        f"of an ESPResSo 5 build.")


class ManagedSimulation:
    """
    A decorator class to manage a singleton instance of a simulation object.

    The `ManagedSimulation` class enforces that only one instance of a decorated simulation class can exist at a time. It provides methods to initialize, reinitialize, and manage the instance, while maintaining a shared `espressomd.System` object. This class is especially useful for simulations where global state must be consistent across multiple components.

    Attributes
    ----------
    aClass : type
        The class being decorated and managed as a singleton.
    instance : object, optional
        The single instance of the decorated class. Initially set to `None`.
    init_args : tuple
        Arguments used during the initialization of the decorated class.
    init_kwargs : dict
        Keyword arguments used during the initialization of the decorated class.
    _espressomd_system : espressomd.System
        The shared ESPResSo system object, initialized during the first instantiation.
    __name__ : str
        The name of the singleton instance, including the decorated class name.
    __qualname__ : str
        The qualified name of the singleton instance, including the decorated class's qualified name.

    Methods
    -------
    __call__(*args, **kwargs):
        Creates and initializes the singleton instance, or raises an exception if it already exists.
    reinitialize_instance():
        Recreates the instance while preserving the shared ESPResSo system object and resets the system state.
    __getattr__(name):
        Forwards attribute access to the instance, raising an error if the instance is uninitialized.
    """

    # Internal attributes that belong to ManagedSimulation itself
    internal_attrs = {"aClass", "instance", "init_args", "init_kwargs", "_espressomd_system", "__name__", "__qualname__"}

    def __init__(self, aClass):
        """
        Initializes the ManagedSimulation decorator.

        Parameters
        ----------
        aClass : type
            The class to be decorated and managed as a singleton.
        """
        self.aClass = aClass
        setattr(aClass, 'reinitialize_instance', self.reinitialize_instance)
        self.instance = None
        self.init_args = ()
        self.init_kwargs = {}
        self._espressomd_system = None  # Shared espressomd.System instance
        self.__name__ = f"Singleton({aClass.__name__})"
        self.__qualname__ = f"Singleton({aClass.__qualname__})"

    def __call__(self, *args, **kwargs):
        """
        Creates and initializes the singleton instance, or raises an exception if it already exists.

        If the instance does not exist, initializes the shared ESPResSo system object and the decorated class.
        If the instance already exists, raises a `SimulationExistsException`.

        Parameters
        ----------
        *args : tuple
            Positional arguments for the decorated class constructor.
        **kwargs : dict
            Keyword arguments for the decorated class constructor. Includes optional `box_dim` to specify the
            simulation box dimensions.

        Returns
        -------
        ManagedSimulation
            The ManagedSimulation instance (not the decorated class instance).

        Raises
        ------
        SimulationExistsException
            If an instance of the decorated class already exists.
        """
        if self.instance is None:
            # Initialize the ESPResSo system object
            if self._espressomd_system is None:
                box_dim = kwargs.get('box_dim', [10, 10, 10])  # Default box dimensions
                self._espressomd_system = self.aClass._sys(box_l=box_dim)

            # Instantiate the decorated class and set its system attribute
            self.instance = self.aClass(*args, **kwargs)
            self.instance.sys = self._espressomd_system
            # Back-reference so Simulation.rebind_sys can keep the cached handle
            # here in step after a checkpoint load.
            self.instance._manager = self
            self.init_args = args
            self.init_kwargs = kwargs
        else:
            # Raise exception if a second instance is attempted
            frame = inspect.currentframe().f_back
            raise SimulationExistsException(
                f"An instance of {self.aClass.__name__} already exists at {frame.f_code.co_filename}, line {frame.f_lineno}"
            )
        return self  # Return the ManagedSimulation instance

    def reinitialize_instance(self):
        """
        Recreates the singleton instance without affecting the shared ESPResSo system object.

        This method resets the decorated class instance while preserving the ESPResSo system object.
        It releases registered simulation objects and clears particles,
        interactions, and thermostat settings in the system, ensuring a clean state.
        It also rewinds the per-class instance counters of every object class
        (see `_rewind_object_class_counters`), so that the new simulation starts
        from the state a freshly started interpreter would give it.
        """
        if self.instance is not None:
            for obj in self.instance.objects:
                obj.delete_owned_parts()
            self.instance.objects = []
            self.instance.no_objects = 0
            self.instance.part_types.clear()
            self.instance.part_positions = []
            self.instance.volume_centers = []
            self.instance.volume_size = None
            self.instance.partitioned = None
            self.instance = self.aClass(*self.init_args, **self.init_kwargs)
            self.instance.sys = self._espressomd_system
            # Back-reference so Simulation.rebind_sys can keep the cached handle
            # here in step after a checkpoint load.
            self.instance._manager = self
            self.instance.sys.part.clear()
            self.instance.sys.non_bonded_inter.reset()
            self.instance.sys.bonded_inter.clear()
            self.instance.sys.constraints.clear()
            self.instance.sys.thermostat.turn_off()
            self.instance.sys.integrator.set_vv()
            self.instance.sys.lb = None
            # Both setters rebuild the cell system even when nothing changes, which
            # is the dominant cost of a reset in a large box: skip the no-ops.
            if self.instance.sys.magnetostatics.solver is not None:
                self.instance.sys.magnetostatics.clear()
            if not all(self.instance.sys.periodicity):
                self.instance.sys.periodicity = (True, True, True)
            self.instance.sys.time = 0.
            self._rewind_object_class_counters()

    @staticmethod
    def _rewind_object_class_counters():
        """
        Rewinds the per-class instance bookkeeping of every simulation object class.

        The metaclass owns three class attributes per object class:
        `instance_id_counter` (source of `who_am_i`), `numInstances` (monotonic
        creation count) and `live_instances` (a `weakref.WeakSet` that racks all
        live instances of an Object). None of them is rewound when instances die,
        so a fresh simulation in a live interpreter would otherwise keep handing
        out `who_am_i` values from wherever the previous simulation stopped --
        which the HDF5 seeding layer, matching `who_am_i` against a source file,
        cannot tolerate. Rewinding here makes `reinitialize_instance()` equivalent
        to a restarted process.

        The classes are taken from `OBJECT_CLASS_REGISTRY`, which lists every
        class built by the metaclass (imported lazily: `object_classes` imports
        this module).
        """
        from pressomancy.object_classes import OBJECT_CLASS_REGISTRY
        for object_class in OBJECT_CLASS_REGISTRY.values():
            object_class.instance_id_counter = 0
            object_class.numInstances = 0
            object_class.live_instances = weakref.WeakSet()


    def __getattr__(self, name):
        """
        Forwards attribute access to the singleton instance.

        Parameters
        ----------
        name : str
            The name of the attribute to access.

        Returns
        -------
        object
            The requested attribute from the instance.

        Raises
        ------
        AttributeError
            If the singleton instance has not been initialized.
        """
        if self.instance is None:
            raise AttributeError(f"Instance of {self.aClass.__name__} has not been initialized.")
        return getattr(self.instance, name)

    def __setattr__(self, name, value):
        """
        Forwards attribute setting to the singleton instance.
        Does not forward attributes intended for ManagedSimulation, as defined in the self.internal_attrs set.

        Parameters
        ----------
        name : str
            The name of the attribute to set.
        value :
            Value to set the attribute to.

        Raises
        ------
        AttributeError
            If the singleton instance has not been initialized.
        """
        if name in self.internal_attrs:
            object.__setattr__(self, name, value)
        elif self.instance is None:
            raise AttributeError(f"Instance of {self.aClass.__name__} has not been initialized.")
        else:
            # Forward to Simulation instance
            setattr(self.instance, name, value)

    def __dir__(self):
        """Attributes of the wrapper plus those of the wrapped instance, if created."""
        names = set(super().__dir__())
        if self.instance is not None:
            names.update(dir(self.instance))
        return sorted(names)

class SimulationExistsException(Exception):
    def __init__(self, message):
        super().__init__(message)

class SimulationType(NamedTuple):
    """The ``(name, numeric type)`` pair identifying an object class.

    Uniqueness of the pair across object classes is enforced by the
    ``Simulation_Object`` metaclass, not here.
    """

    key: str
    value: int

class TypeDictSafe(dict):
    """
    A ``str`` -> ``int`` mapping of type name to numeric espresso type, kept a bijection.

    `TypeDictSafe` enforces that:
    - Keys are `str` and values are `int` (`bool` is not an `int` here).
    - A key cannot be reassigned to a different value.
    - A value cannot be associated with more than one key.
    - A missing key raises `KeyError`, like a plain `dict`.

    Used for `Simulation.part_types` and the class-level `part_types` of every simulation object.

    Methods
    -------
    sanity_check(key, value):
        Validates the types of the key and value and the uniqueness constraints.
    __setitem__(key, value):
        Sets a key-value pair in the dictionary after passing a sanity check.
    update(*args, **kwargs):
        Updates the dictionary with key-value pairs from another dictionary or iterable, enforcing sanity checks.
    key_for(value):
        Reverse lookup: the key(s) mapping to the given value(s).
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the dictionary from an optional mapping or iterable of pairs and keyword arguments.

        Parameters
        ----------
        *args : tuple
            At most one mapping or iterable of ``(key, value)`` pairs.
        **kwargs : dict
            Additional entries.

        Raises
        ------
        TypeError
            If an entry has a non-`str` key or a non-`int` value.
        RuntimeError
            If the initial data violates the uniqueness constraints.
        """
        if len(args) > 1:
            raise TypeError(
                f"TypeDictSafe expected at most 1 positional argument, got {len(args)}"
            )
        super().__init__()
        self.update(*args, **kwargs)

    def sanity_check(self, key, value):
        """
        Ensures the entry is a `str` -> `int` pair that does not violate uniqueness.

        Parameters
        ----------
        key : object
            The key to validate.
        value : object
            The value to validate.

        Raises
        ------
        TypeError
            If the key is not a `str` or the value is not an `int` (`bool` excluded).
        RuntimeError
            If the key already exists with a different value or the value is already associated with another key.
        """
        if not isinstance(key, str):
            raise TypeError(
                f"TypeDictSafe keys must be str: entry {key!r}: {value!r} has a "
                f"'{type(key).__name__}' key."
            )
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"TypeDictSafe values must be int: entry {key!r}: {value!r} has a "
                f"'{type(value).__name__}' value."
            )
        current_value = self.get(key)
        if current_value == value:
            return

        if current_value is not None:
            raise RuntimeError(
                f"Key '{key}' already exists in `TypeDictSafe` with a different value '{current_value}'. "
                f"Attempted to reset it to '{value}', which is not allowed."
            )

        if value in self.values():
            existing_key = next(k for k, v in self.items() if v == value)
            raise RuntimeError(
                f"Value '{value}' is already associated with key '{existing_key}' in `TypeDictSafe`. "
                f"New entries must have unique values: {key}"
            )

    def __setitem__(self, key, value):
        """
        Sets a key-value pair in the dictionary after validating with a sanity check.

        Parameters
        ----------
        key : object
            The key to add or update.
        value : object
            The value to associate with the key.

        Raises
        ------
        TypeError
            If the key or value has the wrong type.
        RuntimeError
            If the key or value violates the uniqueness constraints.
        """
        self.sanity_check(key, value)
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        """
        Updates the dictionary with key-value pairs from another dictionary or iterable.

        Parameters
        ----------
        *args : tuple
            At most one mapping or iterable of ``(key, value)`` pairs.
        **kwargs : dict
            Additional entries.

        Raises
        ------
        TypeError
            If an entry has a non-`str` key or a non-`int` value.
        RuntimeError
            If any key-value pair violates the uniqueness constraints.
        """
        if args:
            iterable = args[0]
            for key, value in (iterable.items() if isinstance(iterable, dict) else iterable):
                self.sanity_check(key, value)
                super().__setitem__(key, value)

        for key, value in kwargs.items():
            self.sanity_check(key, value)
            super().__setitem__(key, value)

    def setdefault(self, key, default=None):
        """
        Insert `key` with `default` if absent, validating like `__setitem__`.
        """
        if key not in self:
            self[key] = default
        return super().__getitem__(key)

    def key_for(self, value):
        """
        Return the key(s) mapping to `value`.

        `value` may be a scalar or an array-like of values; the returned list holds
        the key found for each of them, in order.

        Raises:
            KeyError:   if no key maps to one of the requested values.
        """
        rek_keys = []
        for val in np.atleast_1d(value):
            matches = [k for k, v in self.items() if v == val]
            if not matches:
                raise KeyError(f"No key found for value {val}")
            rek_keys.extend(matches)

        return rek_keys

class RoutineWithArgs:
    """
    A wrapper class to manage callable routines with configurable arguments.

    The `RoutineWithArgs` class provides a way to encapsulate a callable function,
    allowing it to be called with predefined arguments. If no function is provided
    during initialization, a default routine (`generic_routine_per_volume`) is used.

    Attributes
    ----------
    func : callable
        The function to be called. Defaults to `generic_routine_per_volume`.
    num_monomers : int
        The number of monomers or items to process within the routine.

    Methods
    -------
    __call__(**kwargs)
        Invokes the encapsulated function with the provided keyword arguments.
    generic_routine_per_volume(**kwargs)
        A default routine to generate points within a spherical volume. Must be
        implemented by subclasses or overridden.
    """

    def __init__(self, func=None, num_monomers=1, monomer_size=1., spacing=None):
        """
        Initializes the RoutineWithArgs instance.

        Parameters
        ----------
        func : callable, optional
            The function to encapsulate. If not provided, `generic_routine_per_volume` is used.
        num_monomers : int, optional
            The number of monomers or items to process. Defaults to 1.
            partition_cuboid_volume only runs the routine when this exceeds 1; otherwise
            it just places one point at each volume centre.
        monomer_size : float, optional
            Diameter of a single monomer, used by partition_cuboid_volume as the minimum
            allowed separation when rejecting overlapping placements. Defaults to 1.0.
        spacing : float, optional
            Fixed centre-to-centre distance between consecutive monomers, passed through to
            the routine. If None, the routine spreads the monomers across the volume radius
            instead. Defaults to None.
        """
        if func is None:
            self.func = self.generic_routine_per_volume
        else:
            self.func = func
        self.num_monomers = num_monomers
        self.spacing = spacing
        self.monomer_size = monomer_size

    def __call__(self, **kwargs):
        """
        Invokes the encapsulated function with the provided keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            The arguments to pass to the encapsulated function.

        Returns
        -------
        object
            The result of the encapsulated function call.
        """
        return self.func(**kwargs)
    @staticmethod
    def generic_routine_per_volume(**kwargs):
        """
        A placeholder for a default routine to generate points within a spherical volume.

        This method must be implemented by subclasses or overridden by specific instances.

        Parameters
        ----------
        **kwargs : dict
            The arguments required for the routine.

        Raises
        ------
        NotImplementedError
            If the method is called without being overridden.
        """
        raise NotImplementedError("Implement point generation method within a sphere.")

def api_agnostic_feature_check(feature_name):
    """
    Checks whether the running ESPResSo build has a given compile-time feature
    enabled.

    :param feature_name: str | name of the ESPResSo feature (e.g. 'DIPOLES')
    :return: bool | True if the feature is compiled in, False if it is not.
        A name this build does not know at all (features of out-of-tree forks,
        such as 'EGG_MODEL', reach here from the objects that guard on them)
        also yields False, logged as a warning rather than raised.
    """
    try:
        return espressomd.code_features.has_features(feature_name)
    except RuntimeError:
        logging.warning(f'feature check for {feature_name} failed with exception {sysos.exc_info()}')
        return False

def particle_attribute_check(part_hndl, attribute_name):
    """
    Checks that a particle handle exposes a given attribute.

    :param part_hndl: ParticleHandle | particle to check
    :param attribute_name: str | name of the attribute to look up
    :return: None
    :raises MissingFeature: if the attribute is not present, e.g. because the
        ESPResSo build lacks the feature that would expose it
    """
    try:
        getattr(part_hndl,attribute_name)
    except AttributeError:
        logging.warning(f'particle attribute check for {attribute_name} failed with exception {sysos.exc_info()}')
        raise MissingFeature(f"Particle attribute {attribute_name} not found. Please ensure your ESPResSo installation supports this attribute.")

class BondWrapper:
    """Transparent proxy around an espresso bond.
    """

    #: Attributes that belong to the wrapper, not to the wrapped espresso bond.
    _wrapper_attrs = frozenset({"_bond_handle", "name"})

    def __init__(self, bond_handle):
        self._bond_handle = bond_handle
        self.name = bond_handle.__class__.__name__

    def __getattr__(self, name):
        return getattr(self._bond_handle, name)

    def __setattr__(self, name, value):
        if name in BondWrapper._wrapper_attrs:
            super().__setattr__(name, value)
        else:
            setattr(self._bond_handle, name, value)

    def __delattr__(self, name):
        delattr(self._bond_handle, name)

    def __repr__(self):
        return f"BondWrapper({self._bond_handle!r})"

    def get_raw_handle(self):
        """Return the wrapped espresso bond object."""
        return self._bond_handle

def get_repo_context(path):
    """Return the enclosing git repository root and a provenance version string.

    The version string stores the current branch and commit hash and appends ``-dirty`` when
    the repository has uncommitted changes. If the path is not inside a git
    repository, or if the git query fails, the version falls back to ``unknown``.
    """
    path = Path(path).resolve()
    search_root = path if path.is_dir() else path.parent
    repo_root = None
    for candidate in (search_root, *search_root.parents):
        if (candidate / ".git").exists():
            repo_root = candidate
            break
    if repo_root is None:
        return None, "unknown"
    try:
        branch = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return repo_root, "unknown"
    suffix = "-dirty" if dirty else ""
    return repo_root, f"{branch}@{commit}{suffix}"

def get_submission_creator_info():
    """Return H5MD creator metadata for the active submission script.

    The creator name is reported as ``repo_name/repo_relative_path`` when the
    script belongs to a git repository, and as ``unknown/<script_name>``
    otherwise. The accompanying version string follows :func:`get_repo_context`.
    """
    package_root = Path(__file__).resolve().parent
    current_file = Path(__file__).resolve()
    frame = inspect.currentframe()
    script_path = None
    while frame is not None:
        filename = frame.f_code.co_filename
        if filename:
            candidate = Path(filename).resolve()
            if candidate != current_file and package_root not in candidate.parents:
                script_path = candidate
                break
        frame = frame.f_back

    if script_path is None:
        return "unknown/unknown", "unknown"
    repo_root, version = get_repo_context(script_path)
    if repo_root is None:
        return f"unknown/{script_path.name}", "unknown"
    relpath = script_path.relative_to(repo_root).as_posix()
    return f"{repo_root.name}/{relpath}", version
