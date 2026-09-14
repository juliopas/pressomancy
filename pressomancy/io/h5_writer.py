'''
HDF5 (H5MD-style) output for pressomancy.

``H5Writer`` owns everything to do with writing particle groups and observable
streams to an HDF5 file: the ``io_dict`` state, the inscription lifecycle across
the ``NEW``/``LOAD``/``LOAD_NEW``/``INIT_SRC`` modes, per-frame appends, and the
offline ``mk_src_file`` surgery.

It used to live inside :class:`pressomancy.simulation.Simulation`, where the
inscription path was a 213-line method built from four closures. ``Simulation``
now holds one ``H5Writer`` and delegates to it; ``Simulation.io_dict`` is a
property onto this object's dict, so the public API is unchanged.
'''
import logging
import shutil
from collections import defaultdict
from numbers import Integral
from pathlib import Path

import h5py
import numpy as np

from pressomancy.analysis import H5DataSelector
from pressomancy.object_classes import OBJECT_CLASS_REGISTRY
from pressomancy.io.bonds import write_bonds, verify_bond_params, check_bond_count
from pressomancy.helper_functions import get_repo_context, get_submission_creator_info


class H5Writer:
    """Owns the HDF5 output state and the write path for one :class:`Simulation`."""

    def __init__(self, owner):
        #: The Simulation this writer belongs to. Held as a back-reference so the
        #: writer can read live system state without six parameters on every call.
        self._owner = owner
        self.author_name = "unknown"
        self.author_email = "unknown"
        self.io_dict = {
            'h5_file': None,
            'properties': [('id', 1, np.int32), ('type', 1, np.int16),
                           ('pos', 3, np.float64), ('f', 3, np.float64),
                           ('director', 3, np.float64), ('dip', 3, np.float64),
                           ('image_box', 3, np.int32)],
            'bonds': False,
            'bond_links': {},
            'flat_part_view': defaultdict(list),
            'registered_group_type': None,
            'registered_observables': {},
        }
        #: Per group: a ParticleSlice over the group's ids in *sorted* order, plus
        #: the permutation that restores flat_part_view order. See _refresh_slice.
        self._slice_cache = {}

    # -- live state borrowed from the owning Simulation ---------------------
    @property
    def sys(self):
        return self._owner.sys

    @property
    def objects(self):
        return self._owner.objects

    @property
    def part_types(self):
        return self._owner.part_types

    # -- bulk frame reads ---------------------------------------------------
    def _group_slice(self, group_name):
        """Return ``(slice, inverse_permutation)`` for reading a group in bulk.

        Reading properties off a ParticleSlice is ~6x faster than a per-particle
        getattr loop, but a slice built from a non-monotonic id list is **not
        self-consistent about row order**: espresso dispatches ``type``, ``q``,
        ``pos`` and ``pos_folded`` through an optimised path and everything else
        through a per-id loop, and before espresso commit 45376706e those two
        paths disagreed -- the optimised one returned pid-sorted rows while the
        other returned them in the requested order. Mixing both into one frame
        would interleave two particle orderings with nothing to flag it.

        So the slice is always built from **sorted** ids, where the two orders
        coincide and every espresso version agrees, and the inverse permutation
        restores ``flat_part_view`` order afterwards. That is correct for most
        (likely all) espresso builds.

        The cache is rebuilt whenever the group's particle set changes.
        """
        handles = self.io_dict['flat_part_view'][group_name]
        cached = self._slice_cache.get(group_name)
        if cached is not None and cached[0] == len(handles) and cached[3] is handles:
            return cached[1], cached[2]

        ids = [int(part.id) for part in handles]

        if len(set(ids)) != len(ids):
            raise ValueError(
                f"Group '{group_name}' has duplicate particle ids in flat_part_view; "
                f"{len(ids)} entries but only {len(set(ids))} distinct ids. Bulk "
                f"property reads cannot be ordered unambiguously."
            )

        order = np.argsort(ids, kind='stable')
        inverse = np.argsort(order, kind='stable')
        part_slice = self.sys.part.by_ids([ids[k] for k in order])
        self._slice_cache[group_name] = (len(handles), part_slice, inverse, handles)
        return part_slice, inverse

    def _read_frame(self, group_name, prop, dim, dtype):
        """One frame of ``prop`` for a group, in ``flat_part_view`` order."""
        part_slice, inverse = self._group_slice(group_name)
        values = np.asarray(getattr(part_slice, prop)).reshape(len(inverse), dim)
        return values[inverse].astype(dtype, copy=False)

    def _collect_instances_recursively(self, roots):
        """
        Traverse each root in `roots` and return a flat preorder list
        of every object reachable via `.associated_objects`.
        Raises RuntimeError on any duplicate.
        """
        seen = set()
        result = []
        def traverse(obj):
            if obj in seen:
                raise RuntimeError(f"Duplicate object detected during recursion: {obj!r}")
            seen.add(obj)
            result.append(obj)
            for child in getattr(obj, "associated_objects", []) or []:
                traverse(child)
        for root in roots:
            traverse(root)
        return result

    def set_author(self, name, email='unknown'):
        """Set default author metadata for newly created HDF5 files."""
        self.author_name = name
        self.author_email = email

    def _restore_part_types_from_metadata(self, h5_file, group_type):
        """Restore ``part_types`` from optional HDF5 metadata or infer them from the file.

        If ``/parameters/pressomancy/part_types`` is present, it is used as the
        authoritative source. Otherwise a light fallback infers the mapping from a
        single stored timestep together with the ownership connectivity datasets.
        """
        try:
            part_types_group = h5_file["parameters/pressomancy/part_types"]
        except KeyError:
            part_types_group = None

        if part_types_group is not None:
            for key, value in part_types_group.attrs.items():
                self.part_types.update({key: int(value)})
            return

        observed_numeric_types = set()
        for grp_typ in group_type:
            data_view = H5DataSelector(h5_file, particle_group=grp_typ.__name__)
            observed_numeric_types.update(
                int(val) for val in np.unique(data_view.timestep[-1].type)
            )

        object_names = set()
        connectivity_root = h5_file.get("connectivity")
        if connectivity_root is None:
            logging.warning(
                "No 'connectivity' group in the H5 file, so part types cannot be inferred "
                "from ownership. All observed numeric types will be reported as unmatched."
            )
        else:
            for particle_group in connectivity_root.values():
                for dataset_name in particle_group.keys():
                    if not dataset_name.startswith("ParticleHandle_to_"):
                        continue
                    object_names.add(dataset_name.removeprefix("ParticleHandle_to_"))

        recovered = {}
        unmatched = []
        for numeric_type in sorted(observed_numeric_types):
            matched_key = None
            for object_name in sorted(object_names):
                object_cls = OBJECT_CLASS_REGISTRY.get(object_name)
                if object_cls is None:
                    logging.warning(
                        "Object class '%s' is named in the H5 file but is not importable here; "
                        "skipping it while recovering part types.", object_name
                    )
                    continue
                for key, value in object_cls.part_types.items():
                    if value == numeric_type:
                        matched_key = key
                        self.part_types.update({key: int(value)})
                        break
                if matched_key is not None:
                    recovered[matched_key] = numeric_type
                    break
            if matched_key is None:
                unmatched.append(numeric_type)

        if recovered or unmatched:
            logging.warning(
                "Recovered part types from H5 fallback. Matched=%s Unmatched numeric types=%s",
                recovered,
                unmatched,
            )

    def _inscribe_h5_stream(self, mode, force_resize_to_size, setup, new_kernel,load_new_kernel, load_kernel, resize_kernel):
        """Run the shared HDF5 inscription mode and resize lifecycle."""
        if mode not in ('NEW', 'LOAD', 'LOAD_NEW', 'INIT_SRC'):
            raise ValueError(f"Unknown mode: {mode}")
        if force_resize_to_size is not None:
            if not (mode in ('LOAD', 'LOAD_NEW')):
                raise ValueError('force_resize_to_size can only be used in LOAD or LOAD_NEW mode')

        setup()

        if mode in ['NEW', 'INIT_SRC']:
            h5md_group = self.io_dict['h5_file'].require_group("h5md")
            author_group = h5md_group.require_group("author")
            creator_group = h5md_group.require_group("creator")
            h5md_group.attrs["version"] = np.array([1, 0], dtype=np.int32)
            author_group.attrs["name"] = self.author_name
            author_group.attrs["email"] = self.author_email
            creator_name, creator_version = get_submission_creator_info()
            creator_group.attrs["name"] = creator_name
            creator_group.attrs["version"] = creator_version
            parameters_group = self.io_dict['h5_file'].require_group("parameters")
            pressomancy_group = parameters_group.require_group("pressomancy")
            _, pressomancy_version = get_repo_context(Path(__file__).resolve())
            pressomancy_group.attrs["version"] = pressomancy_version
            part_types_group = pressomancy_group.require_group("part_types")
            for key, value in self.part_types.items():
                if isinstance(value, (int, np.integer)):
                    part_types_group.attrs[key] = int(value)
            sim_inst_group = self.io_dict['h5_file'].require_group("sim_inst")
            sys_group = self.io_dict['h5_file'].require_group("sys")
            sim_inst_group.attrs["seed"] = self.seed
            sim_inst_group.attrs["kT"] = self.kT
            sys_group.attrs["time_step"] = self.sys.time_step
            sys_group.attrs["box_l"] = self.sys.box_l
            sys_group.attrs["periodicity"] = self.sys.periodicity
            GLOBAL_COUNTER = new_kernel()
        elif mode == 'LOAD_NEW':
            GLOBAL_COUNTER = load_new_kernel()
            logging.info(f"Loaded h5 file with GLOBAL_COUNTER={GLOBAL_COUNTER} ")
        elif mode == 'LOAD':
            GLOBAL_COUNTER = load_kernel()
            logging.info(f"Loading h5 file with GLOBAL_COUNTER={GLOBAL_COUNTER} ")

        if force_resize_to_size is not None:
            if not (type(force_resize_to_size) is int):
                raise TypeError('force_resize_to_size must be an integer')
            if not (force_resize_to_size <= GLOBAL_COUNTER):
                raise ValueError('force_resize_to_size must be smaller than or equal to the current number of timesteps saved in file')
            if force_resize_to_size == GLOBAL_COUNTER:
                logging.info(f'force_resize_to_size is equal to the current number of timesteps saved in file. No resizing will be done.')
            else:
                resize_kernel(force_resize_to_size)
                self.io_dict['h5_file'].flush()
                logging.info(f'Force resized all datasets from {GLOBAL_COUNTER} to size {force_resize_to_size}')
                GLOBAL_COUNTER = force_resize_to_size

        return GLOBAL_COUNTER

    def inscribe_part_group_to_h5(self, group_type=None, h5_data_path=None,mode='NEW', force_resize_to_size=None):
        """
        Inscribe one or more groups of simulation objects into an HDF5 file.

        This method creates (or opens) an HDF5 file and, for each `group_type`:
        - Builds a flat list of particle handles and their coordinating indices
        - Creates `/particles/<GroupName>` and corresponding property datasets
        - Creates `/connectivity/<GroupName>/ParticleHandle_to_<OwnerClass>` tables
        - Creates `/connectivity/<GroupName>/<Left>_to_<Right>` object–object tables

        Parameters
        ----------
        group_type : list of type
            A list of `SimulationObject` subclasses. All instances of each
            class in `self.objects` will be registered and inscribed.
        h5_data_path : str
            Path to the HDF5 file to write or append.
        mode : {'NEW', 'LOAD', 'LOAD_NEW', 'INIT_SRC'}, optional
            - 'NEW' : create a fresh file structure (default).
            - 'LOAD': open an existing file and resume writing using the legacy path.
            - 'LOAD_NEW': resume writing from HDF5 state and optional metadata.
            - 'INIT_SRC': create a new file while populating particle data from a source file.
        force_resize_to_size : int or None, optional
            If provided in 'LOAD' or 'LOAD_NEW' mode, truncate all registered
            particle property datasets to this number of saved frames before
            subsequent writes.

        Returns
        -------
        int
            The starting global counter for writing time steps. This is 0 in
            'NEW' and 'INIT_SRC' modes; for 'LOAD' and 'LOAD_NEW' it is the
            current number of already-saved steps.

        Raises
        ------
        ValueError
            If `mode` is not one of 'NEW', 'LOAD', 'LOAD_NEW', or 'INIT_SRC'.
        ValueError
            If `group_type` is not a list.
        ValueError
            In load modes, if different groups have mismatched saved step counts.
        AssertionError
            If `force_resize_to_size` is used outside load modes, is not an
            integer, or exceeds the number of saved frames.

        Notes
        -----
        In 'NEW' and 'INIT_SRC' modes the method writes optional H5MD-style root
        metadata under ``/h5md`` together with pressomancy-specific metadata under ``/parameters/pressomancy``. In 'LOAD_NEW' mode, this metadata is used as a convenience source for restoring ``part_types`` when available, but it is not required for successful resume.
        """
        GLOBAL_COUNTER = self._inscribe_h5_stream(
            mode, force_resize_to_size,
            setup=lambda: _particles_setup(self, group_type, mode, h5_data_path),
            new_kernel=lambda: _particles_new(self, group_type),
            load_new_kernel=lambda: _particles_load_new(self, group_type),
            load_kernel=lambda: _particles_load(self, group_type),
            resize_kernel=lambda n: _particles_resize(self, group_type, n),
        )
        return GLOBAL_COUNTER

    def inscribe_observable_group_to_h5(self, observable_defs=None, h5_data_path=None, mode='NEW', force_resize_to_size=None):
        """
        Inscribe one or more observable streams into an HDF5 file.

        This method creates or reopens datasets under ``/observables/<name>/{step,time,value}``. Observable streams may be inscribed
        independently or after particle groups have already opened the target
        HDF5 file.

        Parameters
        ----------
        observable_defs : list of tuple
            Non-empty list of observable definitions. Each definition must be
            ``(name, shape, dtype, observable_value_ref)``:

            - ``name`` is the HDF5 observable group name.
            - ``shape`` is the per-frame payload shape. Scalars may use
              ``None`` or ``()``; integer shapes are converted to one-element
              tuples.
            - ``dtype`` is converted with ``numpy.dtype`` and used for the
              ``value`` dataset.
            - ``observable_value_ref`` is the live Python object or NumPy array
              written on each observable frame.
        h5_data_path : str
            Path to the HDF5 file to write or append. This is required when no
            HDF5 handle is currently open in ``self.io_dict['h5_file']``. If a
            file is already open, the observable groups are added to that file.
        mode : {'NEW', 'LOAD', 'LOAD_NEW', 'INIT_SRC'}, optional
            - 'NEW' : create a fresh observable structure when opening a file.
            - 'LOAD': reopen existing observable datasets and validate them
              against `observable_defs`.
            - 'LOAD_NEW': same observable behavior as 'LOAD'; the file is the
              source of the saved frame count, while `observable_defs` supplies
              the live value references for future writes.
            - 'INIT_SRC': create a new observable structure, matching the
              shared HDF5 inscription lifecycle.
        force_resize_to_size : int or None, optional
            If provided in 'LOAD' or 'LOAD_NEW' mode, truncate all registered
            observable ``step``, ``time``, and ``value`` datasets to this number
            of saved frames before subsequent writes.

        Returns
        -------
        int
            The starting global counter for writing time steps. This is 0 in
            'NEW' and 'INIT_SRC' modes; for 'LOAD' and 'LOAD_NEW' it is the
            current number of already-saved observable frames.

        Raises
        ------
        ValueError
            If `observable_defs` is not a non-empty list, if a definition does
            not contain exactly four entries, if `mode` is unknown, if an
            observable is missing in load modes, if its stored shape does not
            match the requested shape, or if registered observables have
            mismatched saved step counts.
        AssertionError
            If `force_resize_to_size` is used outside load modes, is not an
            integer, or exceeds the number of saved frames.

        Notes
        -----
        In 'NEW' and 'INIT_SRC' modes the shared HDF5 inscription lifecycle
        writes H5MD-style root metadata under ``/h5md`` together with
        pressomancy-specific metadata under ``/parameters/pressomancy`` before
        the observable datasets are created.
        """
        def setup():
            if not isinstance(observable_defs, list) or not observable_defs:
                raise ValueError("observable_defs must be a non-empty list.")
            nonlocal normalised_defs
            normalised_defs = []
            for obs_def in observable_defs:
                if len(obs_def) != 4:
                    raise ValueError("Each observable definition must be (name, shape, dtype, observable_value_ref).")
                name, shape, dtype, observable_value_ref = obs_def
                if shape is None:
                    shape = tuple()
                elif isinstance(shape, Integral):
                    shape = (int(shape),)
                else:
                    shape = tuple(shape)
                normalised_defs.append((str(name), shape, np.dtype(dtype), observable_value_ref))

            self.io_dict['registered_observables'] = {
                name: {'shape': shape, 'dtype': dtype, 'value': observable_value_ref}
                for name, shape, dtype, observable_value_ref in normalised_defs
            }

            if self.io_dict['h5_file'] is None:
                if h5_data_path is None:
                    raise ValueError("h5_data_path must be provided when no HDF5 file is currently open.")
                file_mode = "w" if mode in ('NEW', 'INIT_SRC') else "a"
                self.io_dict['h5_file'] = h5py.File(h5_data_path, file_mode)


        normalised_defs = []

        def new_kernel():
            observables_group = self.io_dict['h5_file'].require_group("observables")
            for name, shape, dtype, _ in normalised_defs:
                obs_group = observables_group.require_group(name)
                if any(key in obs_group for key in ('step', 'time', 'value')):
                    raise ValueError(f"Observable '{name}' already exists in HDF5 file.")
                obs_group.create_dataset("step", shape=(0,), maxshape=(None,), dtype=np.int32)
                obs_group.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float32)
                obs_group.create_dataset(
                    "value",
                    shape=(0, *shape),
                    maxshape=(None, *shape),
                    dtype=dtype,
                    chunks=(1, *shape) if shape else (1,),
                    compression="gzip",
                    compression_opts=4,
                )
            return 0

        def load_kernel():
            observables_group = self.io_dict['h5_file'].require_group("observables")
            candidate_lens = []
            for name, shape, dtype, _ in normalised_defs:
                obs_group = observables_group.get(name)
                if obs_group is None:
                    raise ValueError(f"Observable '{name}' was not found in HDF5 file during {mode}.")
                value_dataset = obs_group["value"]
                if tuple(value_dataset.shape[1:]) != shape:
                    raise ValueError(
                        f"Observable '{name}' shape mismatch: file has {value_dataset.shape[1:]}, expected {shape}."
                    )
                candidate_lens.append(value_dataset.shape[0])

            if len(set(candidate_lens)) != 1:
                raise ValueError(f"Inconsistent step counts across observables: {candidate_lens}")
            return candidate_lens[0]

        def resize_kernel(force_resize_to_size):
            observables_group = self.io_dict['h5_file'].require_group("observables")
            for name, _, _, _ in normalised_defs:
                obs_group = observables_group[name]
                step_dataset = obs_group["step"]
                time_dataset = obs_group["time"]
                value_dataset = obs_group["value"]
                step_dataset.resize((force_resize_to_size,))
                time_dataset.resize((force_resize_to_size,))
                value_dataset.resize((force_resize_to_size, *value_dataset.shape[1:]))
        GLOBAL_COUNTER = self._inscribe_h5_stream(mode, force_resize_to_size, setup, new_kernel, load_kernel, load_kernel, resize_kernel)
        return GLOBAL_COUNTER

    def write_part_group_to_h5(self, step, unique=False):
        """Append one frame using an integer step counter and current ESPResSo time.

        step : int
            Step counter for this frame, e.g. simulation integration step.
            In append mode (unique=False) it must strictly exceed the last written step (H5MD requirement).
        unique : bool, optional
            If True, allow non-increasing steps: place the frame at its sorted
            position, overwriting a frame at the same step.
            Defaults to False.

        Note:
            This overwrites at the sorted position rather than inserting.
            An exact step match is overwritten in place (idempotent re-save),
            but a genuinely new step that sorts before existing frames will
            clobber its neighbour rather than slot in — e.g. writing step=5
            into [10, 20, 30] yields [5, 20, 30], not [5, 10, 20, 30]. Use only
            when re-saving existing steps; true out-of-order insertion is not
            supported.
        """
        if not (self.io_dict['h5_file']!=None):
            raise RuntimeError('storage file has not been inscribed!')
        if not isinstance(step, Integral):
            raise TypeError("step must be provided as an integer frame counter.")
        physical_time = float(self.sys.time)
        particles_group = self.io_dict['h5_file']["particles"]

        if not unique:
            for grp_typ in self.io_dict['registered_group_type']:
                data_grp = particles_group[grp_typ]
                for prop,_,_ in self.io_dict['properties']:
                    step_dataset = data_grp[f"{prop}/step"]
                    if step_dataset.shape[0] > 0 and step <= step_dataset[-1]:
                        raise ValueError(
                            f"step must strictly increase (got {step}, last {int(step_dataset[-1])}). Use unique=True to overwrite."
                        )

        for grp_typ in self.io_dict['registered_group_type']:
            data_grp = particles_group[grp_typ]
            for prop,_dim,_dtype in self.io_dict['properties']:
                dataset_val = data_grp[f"{prop}/value"]
                step_dataset = data_grp[f"{prop}/step"]
                time_dataset = data_grp[f"{prop}/time"]

                dataset_size = dataset_val.shape[0]
                idx = int(np.searchsorted(step_dataset[:], step)) if (unique and dataset_size > 0) else dataset_size
                if idx == dataset_size:
                    step_dataset.resize((dataset_size + 1,))
                    time_dataset.resize((dataset_size + 1,))
                    dataset_val.resize((dataset_size + 1, dataset_val.shape[1], dataset_val.shape[2]))
                elif idx > dataset_size:
                    raise ValueError("Something went horribly wrong when looking for the right spot to save this data.")
                step_dataset[idx] = step
                time_dataset[idx] = physical_time
                dataset_val[idx, :, :] = self._read_frame(grp_typ, prop, _dim, _dtype)

            if self.io_dict.get('bonds', False):
                check_bond_count(
                    self.io_dict['bond_links'].get(grp_typ),
                    self.io_dict['flat_part_view'][grp_typ],
                    group_name=grp_typ,
                    policy="raise"
                )

        logging.debug(f"Successfully wrote timestep for {self.io_dict['registered_group_type']}.")
        return step

    def write_observable_group_to_h5(self, step=None, unique=False):
        """Append one frame for every registered observable.

        The frame counter is stored in each observable ``step`` dataset and the
        current ESPResSo time is stored in the corresponding ``time`` dataset.
        Values are read from the live references registered by
        :meth:`inscribe_observable_group_to_h5`.

        :param step: int | frame counter. In append mode (unique=False) it must
            strictly exceed the last written step, matching
            :meth:`write_part_group_to_h5` so the two streams cannot drift apart.
        :param unique: bool (=False) | if True, allow a non-increasing step and
            overwrite the frame at the sorted position, as on the particle side.
        """
        if not (self.io_dict['h5_file'] != None):
            raise RuntimeError('storage file has not been inscribed!')
        if not isinstance(step, Integral):
            raise TypeError("step must be provided as an integer frame counter.")

        registered_observables = self.io_dict['registered_observables']
        if not registered_observables:
            raise ValueError("No observables have been inscribed in HDF5.")

        physical_time = float(self.sys.time)
        observables_group = self.io_dict['h5_file']["observables"]

        if not unique:
            for name in registered_observables:
                step_dataset = observables_group[name]["step"]
                if step_dataset.shape[0] > 0 and step <= step_dataset[-1]:
                    raise ValueError(
                        f"step must strictly increase (got {step}, last {int(step_dataset[-1])}) "
                        f"for observable '{name}'. Use unique=True to overwrite."
                    )

        for name, obs_data in registered_observables.items():
            obs_group = observables_group[name]
            payload = obs_data['value']
            value_dataset = obs_group["value"]
            expected_shape = tuple(value_dataset.shape[1:])
            if hasattr(payload, 'shape'):
                payload_shape = tuple(payload.shape)
            else:
                payload_shape = tuple()
            if payload_shape != expected_shape:
                raise ValueError(
                    f"Observable '{name}' shape mismatch: payload has {payload_shape}, dataset expects {expected_shape}."
                )
            step_dataset = obs_group["step"]
            time_dataset = obs_group["time"]

            dataset_size = value_dataset.shape[0]
            idx = int(np.searchsorted(step_dataset[:], step)) if (unique and dataset_size > 0) else dataset_size
            if idx == dataset_size:
                step_dataset.resize((dataset_size + 1,))
                time_dataset.resize((dataset_size + 1,))
                value_dataset.resize((dataset_size + 1, *value_dataset.shape[1:]))
            elif idx > dataset_size:
                raise ValueError("Something went horribly wrong when looking for the right spot to save this observable.")
            step_dataset[idx] = step
            time_dataset[idx] = physical_time
            value_dataset[idx] = payload

        logging.debug(f"Successfully wrote timestep for {list(registered_observables)}.")

    def write_registered_to_h5(self, step=None, unique=False):
        """Append one synchronized frame for all registered HDF5 streams.

        If particle groups are registered, their particle property datasets are
        extended. If observables are registered, their observable datasets are
        extended. Both streams receive the same integer `step` and current
        ESPResSo time.

        :param step: int | frame counter, applied to both streams.
        :param unique: bool (=False) | overwrite semantics, applied to both streams.
            Passing it here rather than to one writer is what keeps the particle and
            observable step arrays in step with each other.
        """
        if not (self.io_dict['h5_file'] is not None):
            raise RuntimeError('storage file has not been inscribed!')
        if not isinstance(step, Integral):
            raise TypeError("step must be provided as an integer frame counter.")

        registered_groups = self.io_dict['registered_group_type'] or []
        registered_observables = self.io_dict['registered_observables']
        if not registered_groups and not registered_observables:
            raise ValueError("No particle groups or observables have been inscribed in HDF5.")

        if registered_groups:
            self.write_part_group_to_h5(step=step, unique=unique)
        if registered_observables:
            self.write_observable_group_to_h5(step=step, unique=unique)

    def mk_src_file(self, original_data_file_path, dest_h5_file_path, prop_dim=None, time_step=-1):
        """
        Copy an HDF5 simulation file, shrink it to a single time step, and optionally add one-frame datasets for new particle properties.

        The operation runs in two phases:

        1) **Copy & shrink to one frame**
        The file at ``original_data_file_path`` is copied to ``dest_h5_file_path``.
        For every group under ``/particles/<Group>/<Prop>``, the datasets
        ``value``, ``step``, and ``time`` are sliced at ``time_step`` and then
        **resized to length 1** (T=1), preserving the chosen frame as the only
        frame in the destination file.

        2) **Optionally create new properties (single frame)**
        If ``prop_dim`` is provided, for each group name in
        ``self.io_dict['registered_group_type']`` this function creates a new
        property group ``/particles/<Group>/<prop>`` with the standard layout:
        - ``step`` : int32, shape ``(T,)`` (created empty, then resized to 1)
        - ``time`` : float64, shape ``(T,)`` (created empty, then resized to 1). Note this differs
          from ``inscribe_part_group_to_h5``, which creates ``time`` as float32; readers must not
          assume one dtype for this dataset.
        - ``value``: the dtype given in ``prop_dim``, shape ``(T, N, D)`` (gzip, chunked as ``(1, N, D)``)

        It then appends **one** frame (T=1), reusing the preserved
        ``step[-1]`` and ``time[-1]`` values from the kept frame, and fills ``value[-1, :, :]`` from the
        in-memory list ``self.io_dict['flat_part_view'][<Group>]`` using
        ``getattr(part, prop)`` for each particle.

        Parameters
        ----------
        original_data_file_path : str or os.PathLike
            Path to the source HDF5 file to copy.
        dest_h5_file_path : str or os.PathLike
            Destination path for the copied/modified HDF5 file. Parent directories are created if missing.
        prop_dim : iterable[tuple[str, int, dtype]] or None, optional
            Iterable of ``(prop_name, dim, dtype)`` triples describing new properties to add as
            single-frame datasets, matching the layout of ``io_dict['properties']``. The dtype is
            taken from the caller, not inferred. If ``None`` (default), the function only performs
            the copy-and-shrink phase.
        time_step : int, optional
            Index of the frame to keep during the shrink phase. Must be a valid index for all existing per-property datasets.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If expected groups/datasets (e.g., ``/particles``) are missing.
        IndexError
            If ``time_step`` is out of range for any ``step``/``time``/``value`` dataset.
        ValueError / RuntimeError
            If dataset creation for new properties fails (e.g., attempting to create a dataset that already exists, or a dtype/shape mismatch).
        AssertionError
            If the resulting destination file is not single-step (``len(selector.timestep) != 1``).

        Notes
        -----
        - **Single-step invariant:** After phase (1), the destination file contains exactly one time step (T=1) for all existing properties. The function asserts this using ``H5DataSelector(...).timestep``.
        - **Particle ordering:** New property values are taken from
        ``self.io_dict['flat_part_view'][<Group>]`` in its current order and
        written as an ``(N, dim)`` slab for the single kept frame. This assumes
        that the in-memory order matches the file's particle order.
        - **Creation semantics:** New property datasets are created with
        ``create_dataset``; if a property group already exists, this code will
        raise. Switch to existence checks (e.g., ``if 'value' in prop_group``) or ``require_dataset`` if you need idempotent behavior.
        - **Compression & chunks:** New ``value`` datasets use the caller-supplied dtype with
        chunks ``(1, N, dim)`` and ``gzip`` compression level 4 for consistency.

        Examples
        --------
        Copy a file, keep frame ``time_step=0``, and add ``director``/``image_box``:

        >>> self.mk_src_file(
        ...     original_data_file_path="src.h5",
        ...     dest_h5_file_path="dst_single.h5",
        ...     prop_dim=[("director", 3, np.float64), ("image_box", 3, np.int32)],
        ...     time_step=0,
        ... )
        """

        dst_path=Path(dest_h5_file_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if self.io_dict['h5_file'] is not None:
            self.io_dict['h5_file'].flush()
        shutil.copy2(original_data_file_path, dst_path)
        with h5py.File(dst_path, "r+") as f:
            grp_particles = f["particles"]
            for group_name in grp_particles:
                g = grp_particles[group_name]
                for _, prop_grp in g.items():
                    if not isinstance(prop_grp, h5py.Group) or "value" not in prop_grp:
                        continue
                    val = prop_grp["value"]

                    slice_data = val[time_step, ...]  # shape (1, N, D...)
                    val.resize((1,) + val.shape[1:])
                    val[0, ...] = slice_data

                    ds = prop_grp["step"]
                    step_val = ds[time_step]
                    ds.resize((1,))
                    ds[0] = step_val

                    ds = prop_grp["time"]
                    time_val = ds[time_step]
                    ds.resize((1,))
                    ds[0] = time_val

        logging.info(f"✔ Shrunk to single timestep at: {dst_path}")

        if prop_dim != None:
            with h5py.File(dst_path, "a") as h5_file_handle:
                for grp_typ in self.io_dict['registered_group_type']:
                    particles_group = h5_file_handle["particles"]
                    data_grp = particles_group[grp_typ]
                    reference_prop = data_grp["pos"]  # Use 'pos' as a reference for step/time values and particle count
                    kept_step = int(reference_prop["step"][0])
                    kept_time = float(reference_prop["time"][0])
                    total_part_num=len(self.io_dict['flat_part_view'][grp_typ])
                    for prop,dim,_dtype in prop_dim:
                        prop_group = data_grp.require_group(prop)
                        step_dataset=prop_group.create_dataset("step", shape=(0,), maxshape=(None,), dtype=np.int32)
                        time_dataset=prop_group.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float64)
                        dataset_val=prop_group.create_dataset(
                            "value",
                            shape=(0, total_part_num, dim),  # Store all particles in a single dataset
                            maxshape=(None, total_part_num, dim),
                            dtype=_dtype,
                            chunks=(1, total_part_num, dim),
                            compression="gzip",
                            compression_opts=4
                        )
                        step_dataset.resize((dataset_val.shape[0] + 1,))
                        time_dataset.resize((dataset_val.shape[0] + 1,))
                        dataset_val.resize((dataset_val.shape[0] + 1, dataset_val.shape[1], dataset_val.shape[2]))
                        step_dataset[-1] = kept_step
                        time_dataset[-1] = kept_time
                        dataset_val[-1, :, :] = self._read_frame(grp_typ, prop, dim, _dtype)
                        src_data_grp = H5DataSelector(h5_file_handle, particle_group=grp_typ)
                        # retained assert: internal invariant. Post-condition of the shrink just above.
                        assert len(src_data_grp.timestep)==1,'dataset is ragged!!!'
                        logging.info(f'appended {prop} to {dst_path}')


# ---------------------------------------------------------------------------
# Particle-group inscription, one function per mode.
# These were four closures inside a 213-line method; the bond-count write sat
# ~300 lines from its readers, which is how that mismatch went unnoticed.
# ---------------------------------------------------------------------------
def _particles_setup(writer, group_type, mode, h5_data_path):
    """Open the file and register which groups this run writes."""
    if not isinstance(group_type, list):
        raise ValueError("group_type must be a list of classes.")
    writer.io_dict['registered_group_type']=[grp_typ.__name__ for grp_typ in group_type]
    file_mode = "w" if mode in ['NEW', 'INIT_SRC'] else "a"
    writer.io_dict['h5_file'] = h5py.File(h5_data_path, file_mode)

def _particles_new(writer, group_type):
    """Create the group/schema tree for a fresh file. Returns the start counter (0)."""
    par_grp = writer.io_dict['h5_file'].require_group(f"particles")
    for grp_typ in group_type:
        data_grp = par_grp.require_group(grp_typ.__name__)
        box_grp = data_grp.require_group("box")
        box_grp.attrs["dimension"] = int(len(writer.sys.box_l))
        box_grp.attrs["boundary"] = np.array(
            ["periodic" if flag else "none" for flag in writer.sys.periodicity],
            dtype=h5py.string_dtype(encoding="ascii"),
        )
        if "edges" in box_grp:
            del box_grp["edges"]
        box_grp.create_dataset(
            "edges",
            data=np.asarray(writer.sys.box_l, dtype=np.float64),
            dtype=np.float64,
        )
        connect_grp = writer.io_dict['h5_file'].require_group(f"connectivity").require_group(grp_typ.__name__)
        logging.info(f"Inscribe: Creating group {grp_typ.__name__} in HDF5 file.")
        objects_to_register=[obj for obj in writer.objects if isinstance(obj,grp_typ)]

        coordination_indices=[]
        for cr in objects_to_register:
            part,coord=cr.get_owned_part()
            writer.io_dict['flat_part_view'][grp_typ.__name__].extend(part)
            coordination_indices.extend(coord)

        total_part_num=len(writer.io_dict['flat_part_view'][grp_typ.__name__])

        # Create the connectivity for ParticleHandle to objects that own them.
        grouped = defaultdict(list)
        for part, coords in zip(writer.io_dict['flat_part_view'][grp_typ.__name__], coordination_indices):
            for cls_name, idx in coords:
                grouped[cls_name].append((part.id, idx))

        for cls_name in sorted(grouped):
            arr = np.array(grouped[cls_name], dtype=np.int32)
            connect_grp.create_dataset(
                f"ParticleHandle_to_{cls_name}",
                data=arr,
                dtype=np.int32,
                maxshape=(arr.shape)
            )
        # Create the connectivity for objects that own each other
        pair_buckets = defaultdict(list)

        for obj in writer._collect_instances_recursively(objects_to_register):
            if not obj.associated_objects:
                continue
            left_name = obj.__class__.__name__
            for sub in obj.associated_objects:
                right_name = sub.__class__.__name__
                pair_buckets[(left_name, right_name)].append((obj.who_am_i, sub.who_am_i))

        for (left_name, right_name) in sorted(pair_buckets):
            arr = np.array(pair_buckets[(left_name, right_name)], dtype=np.int32)
            ds = connect_grp.create_dataset(
                f"{left_name}_to_{right_name}",
                data=arr,
                dtype=np.int32,
                maxshape=(arr.shape)
            )
        # Bond topology: static, written once
        # into the connectivity group
        if writer.io_dict.get('bonds', False):
            writer.io_dict['bond_links'][grp_typ.__name__] = write_bonds(
                connect_grp,
                particles=writer.io_dict['flat_part_view'][grp_typ.__name__],
                sys=writer.sys,
                step=0
            )
        # Create the datasets for each property
        for prop,dim,_dtype in writer.io_dict['properties']:
            prop_group = data_grp.require_group(prop)
            prop_group.create_dataset("step", shape=(0,), maxshape=(None,), dtype=np.int32)
            prop_group.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float32)
            prop_group.create_dataset(
                "value",
                shape=(0, total_part_num, dim),  # Store all particles in a single dataset
                maxshape=(None, total_part_num, dim),
                dtype=_dtype,
                chunks=(1, total_part_num, dim),
                compression="gzip",
                compression_opts=4
            )

    return 0

def _particles_load_new(writer, group_type):
    """Resume from a file, rebuilding the particle view from stored connectivity."""
    writer._restore_part_types_from_metadata(writer.io_dict['h5_file'], group_type)
    particles_group = writer.io_dict['h5_file']["particles"]
    candidate_lens=[]
    for grp_typ in group_type:
        data_view=H5DataSelector(writer.io_dict['h5_file'], particle_group=grp_typ.__name__)
        ids=data_view.get_connectivity_values(grp_typ.__name__)
        part_ids=[]
        for iid in ids:
            temp=data_view.select_particles_by_object(object_name=grp_typ.__name__,connectivity_value=iid)
            part_ids+=temp.timestep[-1].id.flatten().tolist()
        part_ids=[int(x) for x in part_ids]
        writer.io_dict['flat_part_view'][grp_typ.__name__].extend(writer.sys.part.by_ids(part_ids))
        data_grp = particles_group[grp_typ.__name__]
        dataset_val = data_grp["pos/value"]
        if writer.io_dict.get('bonds', False):
            connect_grp = writer.io_dict['h5_file']["connectivity"][grp_typ.__name__]
            verify_bond_params(connect_grp, writer.sys)
            n_bond_links = connect_grp["bonds"].attrs.get("n_links")
            if n_bond_links is not None:
                writer.io_dict['bond_links'][grp_typ.__name__] = int(n_bond_links)
        candidate_lens.append(dataset_val.shape[0])
    if len(set(candidate_lens)) != 1:
        raise ValueError(
            f"Inconsistent step counts across groups: {candidate_lens}"
        )
    return candidate_lens[0]

def _particles_load(writer, group_type):
    """Resume from a file, taking the particle view from the live objects."""
    particles_group = writer.io_dict['h5_file']["particles"]
    candidate_lens=[]
    for grp_typ in group_type:
        objects_to_register=[obj for obj in writer.objects if isinstance(obj,grp_typ)]
        for cr in objects_to_register:
            part,_=cr.get_owned_part()
            writer.io_dict['flat_part_view'][grp_typ.__name__].extend(part)
        data_grp = particles_group[grp_typ.__name__]
        dataset_val = data_grp["pos/value"]
        if writer.io_dict.get('bonds', False):
            connect_grp = writer.io_dict['h5_file']["connectivity"][grp_typ.__name__]
            verify_bond_params(connect_grp, writer.sys)
            n_bond_links = connect_grp["bonds"].attrs.get("n_links")
            if n_bond_links is not None:
                writer.io_dict['bond_links'][grp_typ.__name__] = int(n_bond_links)
        candidate_lens.append(dataset_val.shape[0])
    if len(set(candidate_lens)) != 1:
        raise ValueError(
            f"Inconsistent step counts across groups: {candidate_lens}"
        )
    return candidate_lens[0]

def _particles_resize(writer, group_type, force_resize_to_size):
    """Truncate every registered property dataset to the given frame count."""
    particles_group = writer.io_dict['h5_file']["particles"]
    for grp_typ in group_type:
        data_grp = particles_group[grp_typ.__name__]
        for prop,_,_ in writer.io_dict['properties']:
            dataset_val = data_grp[f"{prop}/value"]
            step_dataset = data_grp[f"{prop}/step"]
            time_dataset = data_grp[f"{prop}/time"]
            step_dataset.resize((force_resize_to_size,))
            time_dataset.resize((force_resize_to_size,))
            dataset_val.resize((force_resize_to_size, dataset_val.shape[1], dataset_val.shape[2]))
