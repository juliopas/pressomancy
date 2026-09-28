'''
HDF5 (H5MD) output for pressomancy.

``H5Writer`` owns everything to do with writing particle groups and observable
streams to an HDF5 file: the ``io_dict`` state, the inscription lifecycle across
the ``NEW``/``LOAD``/``LOAD_NEW`` modes and the per-frame appends.

Layout (``layout = "h5md-1"``)::

    /h5md                                   version [2,], author/, creator/
    /particles/<Group>/box                  attrs dimension, boundary; edges/{step,time,value} [F, D]
    /particles/<Group>/<element>/value      [F, N] scalars, [F, N, 3] vectors; element = read.element_name(attr)
    /particles/<Group>/<element>/{step,time}  ONE pair per group, hard-linked into every element
    /connectivity/<Group>/{bonds,angles,dihedrals}   see io/bonds.py
    /observables/<name>/{step,time,value}
    /parameters/pressomancy                 attrs version, layout; part_types/ attrs
    /pressomancy/system                     attrs box_l, periodicity, time_step, seed, kT
    /pressomancy/<Group>/ownership/         ParticleHandle_to_<Class>, <Left>_to_<Right>
    /pressomancy/<Group>/bond_params/       see io/bonds.py

``Simulation`` holds one ``H5Writer`` and delegates to it;
``Simulation.io_dict`` is a property that points into this object's dict, so
the public API is unchanged.
'''
import logging
import os
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from pressomancy.io.bonds import write_bonds, verify_bond_params, check_bond_count
from pressomancy.io.read import (LAYOUT, _TIMELINE, _TIME_RTOL, _require_layout, attr_name,
                                 element_name, frame_of_step, stored_steps)
from pressomancy.infra import api_agnostic_feature_check, get_repo_context, get_submission_creator_info


#: Per-particle state a checkpoint carries,
#: as (espresso attribute, dim, required feature or None),
#: in RESTORE order: ``quat`` before ``omega_lab``/``torque_lab``,
#: whose setters convert lab -> body through the *current* quat.
#: ``dip`` is not here on purpose: ``quat`` + ``dipm`` is exact.
#: ``dip_fld`` is zero on the pre-loop call, so it must be saved.
CHECKPOINT_PROPERTIES = (('pos', 3, None), ('v', 3, None), ('f', 3, None),
                         ('quat', 4, 'ROTATION'), ('omega_lab', 3, 'ROTATION'),
                         ('torque_lab', 3, 'ROTATION'), ('dipm', None, 'DIPOLES'),
                         ('dip_fld', 3, 'DIPOLE_FIELD_TRACKING'))

#: Thermostat kinds whose Philox counter a checkpoint records when active.
_THERMOSTAT_PHILOX_KINDS = ('langevin', 'brownian', 'thermalized_bond', 'dpd', 'npt_iso', 'stokesian')


def checkpoint_properties(trajectory_properties):
    """The ``io_dict['properties']`` list a checkpoint is written with.

    ``id`` and ``type``, then every ``CHECKPOINT_PROPERTIES`` entry this espresso
    build has (float64), then the trajectory's remaining entries with floating
    dtypes widened to float64 (bool/int entries such as ``fix``/``image_box`` kept).
    """
    result = [('id', None, np.int32), ('type', None, np.int16)]
    result += [(attr, dim, np.float64) for attr, dim, feature in CHECKPOINT_PROPERTIES
               if feature is None or api_agnostic_feature_check(feature)]
    present = {attr for attr, _dim, _dtype in result}
    for attr, dim, dtype in trajectory_properties:
        if attr not in present:
            present.add(attr)
            widened = np.float64 if np.issubdtype(np.dtype(dtype), np.floating) else dtype
            result.append((attr, dim, widened))
    return result


class H5Writer:
    """Owns the HDF5 output state and the write path for one :class:`Simulation`."""

    def __init__(self, owner):
        #: The Simulation instance that owns this
        self._owner = owner
        self.author_name = "unknown"
        self.author_email = "unknown"
        self.io_dict = {
            'h5_file': None,
            #: (espresso attribute, dim, dtype); dim None = scalar
            'properties': [('id', None, np.int32), ('type', None, np.int16),
                           ('pos', 3, np.float32), ('image_box', 3, np.int32)] \
                            + ([('director', 3, np.float32)] if api_agnostic_feature_check("ROTATION") else []),
            'bonds': False,
            'bond_links': {},
            'flat_part_view': defaultdict(list),
            'registered_group_type': None,
            'registered_observables': {},
        }
        #: Per group: the ParticleSlice over the group's ids. File columns are in
        #: ascending particle id, which is also the order the slice returns.
        # See _group_slice.
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

    @property
    def seed(self):
        return self._owner.seed

    @property
    def kT(self):
        return self._owner.kT

    # -- bulk frame reads ---------------------------------------------------
    def _group_slice(self, group_name):
        """``(ParticleSlice, n)`` for reading a group in bulk.

        File columns are in **ascending particle id** (the inscription helpers
        sort ``flat_part_view`` that way), which is also the order a
        ParticleSlice returns its properties in, so no reordering is needed.
        A view that is not strictly ascending (unsorted, or duplicate ids)
        raises, since its columns would not mean what the file promises.

        The cache is rebuilt whenever the group's particle set changes.
        """
        handles = self.io_dict['flat_part_view'][group_name]
        cached = self._slice_cache.get(group_name)
        if cached is not None and cached[0] == len(handles) and cached[2] is handles:
            return cached[1], cached[0]

        ids = [int(part.id) for part in handles]
        if not np.all(np.diff(ids) > 0):
            raise ValueError(
                f"Group '{group_name}': flat_part_view must hold strictly ascending particle ids "
                f"(file columns are in id order); got {ids}.")
        part_slice = self.sys.part.by_ids(ids)
        self._slice_cache[group_name] = (len(handles), part_slice, handles)
        return part_slice, len(handles)

    def _capture_frame(self, group_name, attr, dim, dtype):
        """One frame of espresso attribute ``attr`` for a group, in file column (ascending id) order.

        ``dim`` None gives ``(n,)`` (a scalar element), otherwise ``(n, dim)``.
        """
        part_slice, n = self._group_slice(group_name)
        shape = (n,) if dim is None else (n, dim)
        return np.asarray(getattr(part_slice, attr)).reshape(shape).astype(dtype, copy=False)

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

    def _check_part_types_against_file(self, h5_file):
        """Cross-check the live ``part_types`` against the file's own type table.

        A name declared by both must carry the same numeric type: appending
        frames whose ``type`` column means something else than the stored
        frames would corrupt the stream, so a disagreement raises ``RuntimeError``.
        Names only the file declares are registered, so a resumed script keeps
        the writer's vocabulary. A file without
        ``/parameters/pressomancy/part_types`` is accepted as is -- the resume is
        still guarded by the numeric ``type``/``pos`` column check of
        ``_check_load_new_columns``.
        """
        table_path = "parameters/pressomancy/part_types"
        if table_path not in h5_file:
            logging.info("%s has no '%s' table; LOAD_NEW relies on the numeric type/pos "
                         "column check.", h5_file.filename, table_path)
            return
        for key, value in h5_file[table_path].attrs.items():
            name, number = str(key), int(value)
            live = self.part_types.get(name)
            if live is None:
                self.part_types.update({name: number})
            elif int(live) != number:
                raise RuntimeError(
                    f"part_types mismatch: type '{name}' is {int(live)} live but {number} in "
                    f"{h5_file.filename}; appending to that file would store particles of a "
                    "different numeric type than the frames already in it.")

    def _inscribe_h5_stream(self, mode, force_resize_to_size, rewind_to_step, setup, new_kernel,
                            load_new_kernel, load_kernel, resize_kernel, rewind_kernel):
        """Run the shared inscription lifecycle: setup, then NEW (metadata + create) or
        LOAD/LOAD_NEW (truncate, then load).

        Truncation (``force_resize_to_size`` -> ``resize_kernel``, ``rewind_to_step``
        -> ``rewind_kernel``; one at most) runs BEFORE the load kernel, so the
        kernel's checks (live particles against the file's last frame, frame
        count) see the frames that will actually be appended to.
        """
        if mode not in ('NEW', 'LOAD', 'LOAD_NEW'):
            raise ValueError(f"Unknown mode: {mode}")
        truncating = force_resize_to_size is not None or rewind_to_step is not None
        if truncating and mode not in ('LOAD', 'LOAD_NEW'):
            raise ValueError('force_resize_to_size and rewind_to_step can only be used in LOAD or LOAD_NEW mode')
        if force_resize_to_size is not None and rewind_to_step is not None:
            raise ValueError('force_resize_to_size and rewind_to_step are mutually exclusive; give one of them.')
        for name, value in (('force_resize_to_size', force_resize_to_size), ('rewind_to_step', rewind_to_step)):
            if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))):
                raise TypeError(f'{name} must be an integer, got {value!r}')

        # --- Setup ---------------
        setup()

        # --- Truncate ------------
        if force_resize_to_size is not None:
            resize_kernel(force_resize_to_size)
        elif rewind_to_step is not None:
            rewind_kernel(rewind_to_step)
        if truncating:
            self.io_dict['h5_file'].flush()
            logging.info(f"{mode}: truncated the registered streams (force_resize_to_size="
                            f"{force_resize_to_size}, rewind_to_step={rewind_to_step}).")

        # --- Apply mode ----------
        if mode == 'NEW':
            h5_file = self.io_dict['h5_file']
            h5md_group = h5_file.require_group("h5md")
            author_group = h5md_group.require_group("author")
            creator_group = h5md_group.require_group("creator")
            h5md_group.attrs["version"] = np.array([1, 1], dtype=np.int32)
            author_group.attrs["name"] = self.author_name
            author_group.attrs["email"] = self.author_email
            creator_name, creator_version = get_submission_creator_info()
            creator_group.attrs["name"] = creator_name
            creator_group.attrs["version"] = creator_version
            pressomancy_group = h5_file.require_group("parameters/pressomancy")
            _, pressomancy_version = get_repo_context(Path(__file__).resolve())
            pressomancy_group.attrs["version"] = pressomancy_version
            pressomancy_group.attrs["layout"] = LAYOUT
            part_types_group = pressomancy_group.require_group("part_types")
            for key, value in self.part_types.items():
                part_types_group.attrs[str(key)] = int(value)
            system_group = h5_file.require_group("pressomancy/system")
            system_group.attrs["box_l"] = self.sys.box_l
            system_group.attrs["periodicity"] = self.sys.periodicity
            system_group.attrs["time_step"] = self.sys.time_step
            system_group.attrs["seed"] = self.seed
            system_group.attrs["kT"] = self.kT
            GLOBAL_COUNTER = new_kernel()
            logging.info(f"Created h5 file with GLOBAL_COUNTER={GLOBAL_COUNTER} ")
            return GLOBAL_COUNTER
        elif mode == 'LOAD_NEW':
            GLOBAL_COUNTER = load_new_kernel()
            logging.info(f"Loaded h5 file with GLOBAL_COUNTER={GLOBAL_COUNTER}. Ran check. ")
        elif mode == 'LOAD':
            GLOBAL_COUNTER = load_kernel()
            logging.info(f"Loaded h5 file with GLOBAL_COUNTER={GLOBAL_COUNTER}Did not run check. ")
        return GLOBAL_COUNTER

    def inscribe_part_group_to_h5(self, group_type=None, h5_data_path=None, mode='NEW',
                                  force_resize_to_size=None, rewind_to_step=None):
        """
        Inscribe one or more groups of simulation objects into an HDF5 file.

        This method creates (or opens) an HDF5 file and, for each `group_type`:
        - Builds a flat list of particle handles and their coordinating indices
        - Creates `/particles/<GroupName>` and one `<element>/{step,time,value}` per property
        - Creates `/pressomancy/<GroupName>/ownership/ParticleHandle_to_<OwnerClass>` tables
        - Creates `/pressomancy/<GroupName>/ownership/<Left>_to_<Right>` object–object tables

        Parameters
        ----------
        group_type : list of type
            A list of `SimulationObject` subclasses. All instances of each
            class in `self.objects` will be registered and inscribed.
        h5_data_path : str
            Path to the HDF5 file to write or append.
        mode : {'NEW', 'LOAD', 'LOAD_NEW'}, optional
            - 'NEW' : create a fresh file structure (default).
            - 'LOAD': resume writing, taking the particle view from the live
            objects.
            - 'LOAD_NEW': resume writing, rebuilding the particle view from the
              file's own id columns and checking it against the last frame.
        force_resize_to_size : int or None, optional
            'LOAD'/'LOAD_NEW' only: truncate every registered group to this
            number of saved frames (its step/time pair, every element, the box
            edges) before the load checks run.
        rewind_to_step : int or None, optional
            'LOAD'/'LOAD_NEW' only: truncate every registered group to end
            at the frame stored at this step value. That frame's time must
            equal the live ``sys.time`` (relative ``1e-6``), which
            ``restart_from_checkpoint`` sets -- so the file can only be
            rewound to the state the system is actually in.

        Returns
        -------
        int
            The starting global counter for writing frames. This is 0 in
            'NEW' mode; for 'LOAD' and 'LOAD_NEW' it is the number of saved
            frames after any truncation.

        Raises
        ------
        ValueError
            If `mode` is unknown, `group_type` is not a list,
            ``io_dict['properties']`` is malformed (it must start with
            ``('id', None, int32), ('type', None, int16), ('pos', 3, <floating>)``,
            attrs unique, dims None or positive, dtypes numpy-resolvable), a
            truncation argument is given outside load modes or both are given,
            `force_resize_to_size` exceeds the saved frames, or in load modes
            different groups have mismatched saved step counts.
        TypeError
            If `force_resize_to_size` / `rewind_to_step` is not an integer.
        KeyError
            If `rewind_to_step` is not a stored step.
        RuntimeError
            In load modes, if the file is not in the ``h5md-1`` layout, if
            ``io_dict['properties']`` does not describe the file's elements
            exactly (set, dtype and per-particle shape -- edit the list to match
            the file), if the `rewind_to_step` frame's time is not the live
            ``sys.time``; in 'LOAD_NEW' mode, if the live particles do not match
            the file's last-frame columns, or a type name is declared with a
            different number in the file.

        Notes
        -----
        In 'NEW' mode the method writes the H5MD root metadata under ``/h5md``,
        pressomancy-specific metadata under ``/parameters/pressomancy`` (incl. the
        ``layout`` attr readers check) and the system attrs under ``/pressomancy/system``.
        In 'LOAD_NEW' mode the stored ``part_types`` table is *checked* against the live one
        (``RuntimeError`` if a shared name carries a different number; names only the file
        declares are registered); a file without that table resumes fine, guarded by the
        numeric ``type``/``pos`` column check.
        """
        return self._inscribe_h5_stream(
            mode, force_resize_to_size, rewind_to_step,
            setup=lambda: _particles_setup(self, group_type, mode, h5_data_path),
            new_kernel=lambda: _particles_new(self, group_type),
            load_new_kernel=lambda: _particles_load_new(self, group_type),
            load_kernel=lambda: _particles_load(self, group_type),
            resize_kernel=lambda n: _particles_resize(self, group_type, n),
            rewind_kernel=lambda step: _particles_rewind(self, group_type, step),
        )

    def inscribe_observable_group_to_h5(self, observable_defs=None, h5_data_path=None, mode='NEW',
                                        force_resize_to_size=None, rewind_to_step=None):
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
        mode : {'NEW', 'LOAD', 'LOAD_NEW'}, optional
            - 'NEW' : create a fresh observable structure when opening a file.
            - 'LOAD': reopen existing observable datasets and validate them
              against `observable_defs`.
            - 'LOAD_NEW': same as 'LOAD'.
        force_resize_to_size : int or None, optional
            'LOAD'/'LOAD_NEW' only: truncate every registered observable's
            ``step``, ``time`` and ``value`` to this number of saved frames
            before the load checks run.
        rewind_to_step : int or None, optional
            'LOAD'/'LOAD_NEW' only: truncate every registered observable
            to end at the row whose own ``step`` is this value; that row's
            time must equal the live ``sys.time`` (relative ``1e-6``), as
            for the particle groups.

        Returns
        -------
        int
            The starting global counter for writing frames. This is 0 in
            'NEW' mode; for 'LOAD' and 'LOAD_NEW' it is the number of saved
            observable frames after any truncation.

        Raises
        ------
        ValueError
            If `observable_defs` is not a non-empty list, if a definition does
            not contain exactly four entries, if `mode` is unknown, if an
            observable is missing in load modes, if its stored shape does not
            match the requested shape, if registered observables have
            mismatched saved step counts, if a truncation argument is given
            outside load modes or both are given, or if `force_resize_to_size`
            exceeds the saved frames.
        TypeError
            If `force_resize_to_size` / `rewind_to_step` is not an integer.
        KeyError
            If `rewind_to_step` is not a stored step of every observable.
        RuntimeError
            In load modes, if the file is not in the ``h5md-1`` layout, or the
            `rewind_to_step` row's time is not the live ``sys.time``.

        Notes
        -----
        In 'NEW' mode the shared HDF5 inscription lifecycle
        writes H5MD-style root metadata under ``/h5md`` together with
        pressomancy-specific metadata under ``/parameters/pressomancy`` before
        the observable datasets are created.
        """
        return self._inscribe_h5_stream(
            mode, force_resize_to_size, rewind_to_step,
            setup=lambda: _observables_setup(self, observable_defs, mode, h5_data_path),
            new_kernel=lambda: _observables_new(self),
            load_new_kernel=lambda: _observables_load(self, mode),
            load_kernel=lambda: _observables_load(self, mode),
            resize_kernel=lambda n: _observables_resize(self, n),
            rewind_kernel=lambda step: _observables_rewind(self, step),
        )

    def write_part_group_to_h5(self, step):
        """Append one frame at integer ``step`` and the current ESPResSo time.

        Every group has ONE ``step``/``time`` pair (hard-linked into each of its
        elements), so the pair is appended once per group and then every
        element's ``value`` gets its row -- the box is an element too, so
        ``box/edges/value`` gets one ``sys.box_l`` row per frame.

        step : int
            Step counter for this frame, e.g. simulation integration step. It
            must strictly exceed the last written step (H5MD requirement);
            ``RuntimeError`` otherwise, raised for every group before anything
            is written. To continue from an earlier frame, truncate the file
            first (``inscribe_part_group_to_h5(..., rewind_to_step=...)`` or
            ``force_resize_to_size=...``); frames are never overwritten in place.
        """
        if self.io_dict['h5_file'] is None:
            raise RuntimeError('storage file has not been inscribed!')
        if not isinstance(step, (int, np.integer)):
            raise TypeError("step must be provided as an integer frame counter.")
        physical_time = float(self.sys.time)
        particles_group = self.io_dict['h5_file']["particles"]
        properties = self.io_dict['properties']

        # Guards first, for every group, so a raise leaves no group half-written.
        for grp_typ in self.io_dict['registered_group_type']:
            if self.io_dict.get('bonds', False):
                check_bond_count(
                    self.io_dict['bond_links'].get(grp_typ),
                    self.io_dict['flat_part_view'][grp_typ],
                    group_name=grp_typ,
                )
            step_dataset = particles_group[grp_typ][f"{_TIMELINE}/step"]
            if step_dataset.shape[0] > 0 and step <= step_dataset[-1]:
                raise RuntimeError(
                    f"step must strictly increase (got {step}, last {int(step_dataset[-1])}) "
                    f"for group '{grp_typ}'; to continue from an earlier frame, inscribe with "
                    "rewind_to_step or force_resize_to_size first.")

        for grp_typ in self.io_dict['registered_group_type']:
            data_grp = particles_group[grp_typ]
            step_dataset = data_grp[f"{_TIMELINE}/step"]
            time_dataset = data_grp[f"{_TIMELINE}/time"]
            idx = step_dataset.shape[0]
            step_dataset.resize((idx + 1,))
            time_dataset.resize((idx + 1,))
            step_dataset[idx] = step
            time_dataset[idx] = physical_time
            # The box is an element too: one edges row per frame.
            edges_dataset = data_grp["box/edges/value"]
            edges_dataset.resize((idx + 1, *edges_dataset.shape[1:]))
            edges_dataset[idx] = np.asarray(self.sys.box_l, dtype=np.float64)
            for attr, dim, dtype in properties:
                dataset_val = data_grp[f"{element_name(attr)}/value"]
                dataset_val.resize((idx + 1, *dataset_val.shape[1:]))
                dataset_val[idx] = self._capture_frame(grp_typ, attr, dim, dtype)

        logging.debug(f"Successfully wrote timestep for {self.io_dict['registered_group_type']}.")
        return step

    def write_observable_group_to_h5(self, step=None):
        """Append one frame for every registered observable.

        The frame counter is stored in each observable ``step`` dataset and the
        current ESPResSo time is stored in the corresponding ``time`` dataset.
        Values are read from the live references registered by
        :meth:`inscribe_observable_group_to_h5`.

        :param step: int | frame counter. It must strictly exceed the last
            written step (``RuntimeError`` otherwise), matching
            :meth:`write_part_group_to_h5` so the two streams cannot drift apart.
        """
        if self.io_dict['h5_file'] is None:
            raise RuntimeError('storage file has not been inscribed!')
        if not isinstance(step, (int, np.integer)):
            raise TypeError("step must be provided as an integer frame counter.")

        registered_observables = self.io_dict['registered_observables']
        if not registered_observables:
            raise ValueError("No observables have been inscribed in HDF5.")

        physical_time = float(self.sys.time)
        observables_group = self.io_dict['h5_file']["observables"]

        for name in registered_observables:
            step_dataset = observables_group[name]["step"]
            if step_dataset.shape[0] > 0 and step <= step_dataset[-1]:
                raise RuntimeError(
                    f"step must strictly increase (got {step}, last {int(step_dataset[-1])}) "
                    f"for observable '{name}'; to continue from an earlier frame, inscribe with "
                    "rewind_to_step or force_resize_to_size first.")

        for name, obs_data in registered_observables.items():
            obs_group = observables_group[name]
            payload = obs_data['value']
            value_dataset = obs_group["value"]
            expected_shape = tuple(value_dataset.shape[1:])
            payload_shape = tuple(payload.shape) if hasattr(payload, 'shape') else tuple()
            if payload_shape != expected_shape:
                raise ValueError(
                    f"Observable '{name}' shape mismatch: payload has {payload_shape}, dataset expects {expected_shape}."
                )
            step_dataset = obs_group["step"]
            time_dataset = obs_group["time"]
            idx = value_dataset.shape[0]
            step_dataset.resize((idx + 1,))
            time_dataset.resize((idx + 1,))
            value_dataset.resize((idx + 1, *value_dataset.shape[1:]))
            step_dataset[idx] = step
            time_dataset[idx] = physical_time
            value_dataset[idx] = payload

        logging.debug(f"Successfully wrote timestep for {list(registered_observables)}.")

    def write_registered_to_h5(self, step=None):
        """Append one synchronized frame for all registered HDF5 streams.

        If particle groups are registered, their particle property datasets are
        extended. If observables are registered, their observable datasets are
        extended. Both streams receive the same integer `step` and current
        ESPResSo time.

        :param step: int | frame counter, applied to both streams.
        """
        if self.io_dict['h5_file'] is None:
            raise RuntimeError('storage file has not been inscribed!')
        if not isinstance(step, (int, np.integer)):
            raise TypeError("step must be provided as an integer frame counter.")

        registered_groups = self.io_dict['registered_group_type'] or []
        registered_observables = self.io_dict['registered_observables']
        if not registered_groups and not registered_observables:
            raise ValueError("No particle groups or observables have been inscribed in HDF5.")

        if registered_groups:
            self.write_part_group_to_h5(step=step)
        if registered_observables:
            self.write_observable_group_to_h5(step=step)

    def write_checkpoint(self, group_type, path, step):
        """Write a verified one-frame checkpoint of ``group_type`` to ``path``.

        A checkpoint is an ordinary ``h5md-1`` file with one frame of
        ``checkpoint_properties(io_dict['properties'])`` (the restorable
        per-particle state in float64, bonds on) and a ``/pressomancy/checkpoint``
        group whose attrs are ``time`` (float64), ``step`` and one
        ``<kind>_philox_counter`` per active thermostat. ``restart_from_checkpoint``
        reads it back and ``inscribe_*(..., rewind_to_step=step)`` then re-aligns
        the trajectory.

        The file goes to ``path + '.tmp'``, is reopened read-only and verified
        (layout, exactly the one step, live-vs-file ``type``/``pos`` columns, bond
        link count, time) and only then moved onto ``path`` with ``os.replace``.
        On any failure the ``.tmp`` is left in place for inspection and a previous
        ``path`` is untouched; the package never deletes.

        :param group_type: list of object classes, as for ``inscribe_part_group_to_h5``.
        :param path: the checkpoint file; ``ValueError`` if it is the open trajectory.
        :param step: int | the step value stored with the frame; returned.
        :raises NotImplementedError: if the LB thermostat is active (the fluid
            state is not checkpointed).
        """
        if not isinstance(group_type, list):
            raise ValueError("group_type must be a list of classes.")
        trajectory = self.io_dict['h5_file']
        if trajectory is not None and Path(path).resolve() == Path(trajectory.filename).resolve():
            raise ValueError(f"write_checkpoint: {path} is the open trajectory file; "
                             "a checkpoint needs its own path.")
        lb = getattr(self.sys.thermostat, 'lb', None)   # absent when not compiled in
        if lb is not None and lb.is_active:
            raise NotImplementedError("write_checkpoint: the LB thermostat is active and the "
                                      "fluid state is not checkpointed.")

        tmp = str(path) + ".tmp"
        ckpt = H5Writer(self._owner)
        ckpt.author_name, ckpt.author_email = self.author_name, self.author_email
        ckpt.io_dict['properties'] = checkpoint_properties(self.io_dict['properties'])
        ckpt.io_dict['bonds'] = True
        try:
            ckpt.inscribe_part_group_to_h5(group_type, tmp, 'NEW')
            ckpt.write_part_group_to_h5(step)
            attrs = ckpt.io_dict['h5_file'].require_group("pressomancy/checkpoint").attrs
            attrs['time'] = np.float64(self.sys.time)
            attrs['step'] = int(step)
            for kind in _THERMOSTAT_PHILOX_KINDS:
                thermostat = getattr(self.sys.thermostat, kind, None)   # absent when not compiled in
                if thermostat is not None and thermostat.is_active:
                    attrs[f"{kind}_philox_counter"] = int(thermostat.philox_counter)
        finally:
            if ckpt.io_dict['h5_file'] is not None:
                ckpt.io_dict['h5_file'].close()

        with h5py.File(tmp, 'r') as h5_file:
            _require_layout(h5_file)
            for grp_typ in group_type:
                group_name = grp_typ.__name__
                stored = stored_steps(h5_file, group_name).tolist()
                if stored != [int(step)]:
                    raise RuntimeError(f"checkpoint {tmp}: group '{group_name}' stores steps "
                                       f"{stored}, expected [{int(step)}].")
                _check_load_new_columns(ckpt, group_name, h5_file["particles"][group_name])
                n_links = int(h5_file[f"connectivity/{group_name}/bonds"].attrs['n_links'])
                if n_links != ckpt.io_dict['bond_links'][group_name]:
                    raise RuntimeError(f"checkpoint {tmp}: group '{group_name}' stores {n_links} bond "
                                       f"links, {ckpt.io_dict['bond_links'][group_name]} were written.")
            stored_time = float(h5_file["pressomancy/checkpoint"].attrs['time'])
            if stored_time != float(self.sys.time):
                raise RuntimeError(f"checkpoint {tmp}: stored time {stored_time} is not the live "
                                   f"sys.time {float(self.sys.time)}.")
        os.replace(tmp, path)
        return int(step)


# ---------------------------------------------------------------------------
# Particle-group inscription, one function per mode.
# ---------------------------------------------------------------------------
def _validate_properties(properties):
    """``ValueError`` unless ``properties`` is a list of ``(attr, dim, dtype)`` tuples that
    opens with ``('id', None, int32), ('type', None, int16), ('pos', 3, <floating>)``
    (the columns the readers, the resume checks and the timeline rely on), with
    unique attrs, ``dim`` None (scalar) or a positive int, and numpy-resolvable dtypes."""
    if not isinstance(properties, list):
        raise ValueError("io_dict['properties'] must be a list of (attr, dim, dtype) tuples, "
                         f"not {type(properties).__name__}.")
    seen = set()
    for entry in properties:
        if not (isinstance(entry, tuple) and len(entry) == 3 and isinstance(entry[0], str)):
            raise ValueError(f"io_dict['properties'] entry {entry!r} is not an (attr: str, dim, dtype) tuple.")
        attr, dim, dtype = entry
        if dim is not None and (isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or dim <= 0):
            raise ValueError(f"io_dict['properties'] entry {entry!r}: dim must be None (scalar) or a positive int.")
        try:
            np.dtype(dtype)
        except TypeError as exc:
            raise ValueError(f"io_dict['properties'] entry {entry!r}: {dtype!r} is not a numpy dtype.") from exc
        if attr in seen:
            raise ValueError(f"io_dict['properties'] lists '{attr}' twice.")
        seen.add(attr)
    head = properties[:3]
    if (len(head) < 3
            or [(attr, dim) for attr, dim, _dtype in head] != [('id', None), ('type', None), ('pos', 3)]
            or np.dtype(head[0][2]) != np.int32 or np.dtype(head[1][2]) != np.int16
            or not np.issubdtype(np.dtype(head[2][2]), np.floating)):
        raise ValueError("io_dict['properties'] must start with ('id', None, int32), ('type', None, int16), "
                         f"('pos', 3, <floating dtype>); got {head!r}.")


def _check_properties_against_file(writer, group_name, data_grp):
    """Refuse to resume unless ``io_dict['properties']`` describes the file's elements exactly.

    The stored elements (subgroups of ``particles/<Group>`` holding a ``value``,
    the box excepted) must be the list's set, each with the list's dtype and
    per-particle shape. The file is authoritative: the script's list has to be
    edited to match it. ``RuntimeError`` names the first difference.
    """
    where = f"{data_grp.file.filename}/particles/{group_name}"
    stored = {name for name, member in data_grp.items()
              if name != 'box' and isinstance(member, h5py.Group) and 'value' in member}
    wanted = {element_name(attr): (dim, np.dtype(dtype)) for attr, dim, dtype in writer.io_dict['properties']}
    if stored != set(wanted.keys()):
        names = lambda elements: sorted(attr_name(e) for e in elements)
        raise RuntimeError(
            f"io_dict['properties'] does not match {where}: the file stores {names(stored)}, the list "
            f"names {names(wanted)} (missing from the file: {names(set(wanted) - stored)}, not in the "
            f"list: {names(stored - set(wanted))}); edit the list to match the file.")
    for element, (dim, dtype) in wanted.items():
        value = data_grp[element]['value']
        per_particle = () if dim is None else (dim,)
        if value.dtype != dtype or tuple(value.shape[2:]) != per_particle:
            raise RuntimeError(
                f"io_dict['properties'] does not match {where}: '{attr_name(element)}' is stored as "
                f"{value.dtype} with per-particle shape {tuple(value.shape[2:])}, the list says {dtype} "
                f"with {per_particle}; edit the list to match the file.")


def _particles_setup(writer, group_type, mode, h5_data_path):
    """Validate the properties list, open the file and register which groups this run writes."""
    if not isinstance(group_type, list):
        raise ValueError("group_type must be a list of classes.")
    _validate_properties(writer.io_dict['properties'])
    writer.io_dict['registered_group_type']=[grp_typ.__name__ for grp_typ in group_type]
    file_mode = "w" if mode == 'NEW' else "a"
    writer.io_dict['h5_file'] = h5py.File(h5_data_path, file_mode)

def _particles_new(writer, group_type):
    """Create the group/schema tree for a fresh file. Returns the start counter (0)."""
    h5_file = writer.io_dict['h5_file']
    part_grp = h5_file.require_group("particles")
    for grp_typ in group_type:
        group_name = grp_typ.__name__
        data_grp = part_grp.require_group(group_name)
        # save box here, in case of time-fluctuations
        # Also alows different boxes for different Objects. E.g. merge two files with different systems
        # To comply with H5MD format
        box_grp = data_grp.require_group("box")
        dimension = int(len(writer.sys.box_l))
        box_grp.attrs["dimension"] = dimension
        box_grp.attrs["boundary"] = np.array(
            ["periodic" if flag else "none" for flag in writer.sys.periodicity],
            dtype=h5py.string_dtype(encoding="ascii"),
        )
        ownership_grp = h5_file.require_group(f"pressomancy/{group_name}/ownership")
        logging.info(f"Inscribe: Creating group {group_name} in HDF5 file.")
        objects_to_register=[obj for obj in writer.objects if isinstance(obj,grp_typ)]

        # File columns are in ascending particle id: sort handles and their
        # coordination indices together (ids need not be contiguous).
        owned = []
        for cr in objects_to_register:
            part,coord=cr.get_owned_part()
            owned.extend(zip(part, coord))
        owned.sort(key=lambda pc: pc[0].id)
        writer.io_dict['flat_part_view'][group_name].extend(pc[0] for pc in owned)
        coordination_indices=[pc[1] for pc in owned]

        total_part_num=len(writer.io_dict['flat_part_view'][group_name])

        # Ownership: particle id -> who_am_i of every object that owns it, per class.
        grouped = defaultdict(list)
        for part, coords in zip(writer.io_dict['flat_part_view'][group_name], coordination_indices):
            for cls_name, idx in coords:
                grouped[cls_name].append((part.id, idx))
        for cls_name in sorted(grouped):
            ownership_grp.create_dataset(f"ParticleHandle_to_{cls_name}",
                                         data=np.array(grouped[cls_name], dtype=np.int32))
        # Ownership: objects that own each other, (left who_am_i, right who_am_i).
        pair_buckets = defaultdict(list)
        for obj in writer._collect_instances_recursively(objects_to_register):
            if not obj.associated_objects:
                continue
            left_name = obj.__class__.__name__
            for sub in obj.associated_objects:
                right_name = sub.__class__.__name__
                pair_buckets[(left_name, right_name)].append((obj.who_am_i, sub.who_am_i))
        for (left_name, right_name) in sorted(pair_buckets):
            ownership_grp.create_dataset(f"{left_name}_to_{right_name}",
                                         data=np.array(pair_buckets[(left_name, right_name)], dtype=np.int32))
        # Bond topology: static, written once.
        if writer.io_dict.get('bonds', False):
            writer.io_dict['bond_links'][group_name] = write_bonds(
                h5_file, group_name,
                particles=writer.io_dict['flat_part_view'][group_name],
                sys=writer.sys,
            )
        # One <element>/{step,time,value} per property,
        # hardlinked from pos (pos must be at position 0)
        properties = sorted(writer.io_dict['properties'], key=lambda x: x[0] != "pos")
        shared = None
        for attr, dim, dtype in properties:
            element = data_grp.require_group(element_name(attr))
            if shared is None:
                assert attr == "pos", "step/time must be stored in positions dataset. Edit with caution."
                shared = (element.create_dataset("step", shape=(0,), maxshape=(None,), dtype=np.int32),
                          element.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float32))
            else:
                element["step"], element["time"] = shared
            per_particle = () if dim is None else (dim,)
            element.create_dataset(
                "value",
                shape=(0, total_part_num, *per_particle),  # Store all particles in a single dataset
                maxshape=(None, total_part_num, *per_particle),
                dtype=dtype,
                chunks=(1, total_part_num, *per_particle),
                compression="gzip",
                compression_opts=4
            )
        # The box is a time-dependent H5MD element on the same timeline
        edges_grp = box_grp.require_group("edges")
        edges_grp["step"], edges_grp["time"] = shared
        edges_grp.create_dataset(
            "value",
            shape=(0, dimension),
            maxshape=(None, dimension),
            dtype=np.float64,
            chunks=(1, dimension),
        )

    return 0

def _particles_load_new(writer, group_type):
    """Resume from a file, rebuilding the particle view from the file's id columns.

    ``particles/<Group>/id/value[-1]`` lists the particle ids in column order
    (ascending id, which is ``flat_part_view`` order), so one read and one
    ``by_ids`` rebuild the view; ``_check_load_new_columns`` then verifies the
    live particles against the last stored frame.
    """
    h5_file = writer.io_dict['h5_file']
    _require_layout(h5_file)
    writer._check_part_types_against_file(h5_file)
    particles_group = h5_file["particles"]
    candidate_lens=[]
    for grp_typ in group_type:
        group_name = grp_typ.__name__
        data_grp = particles_group[group_name]
        _check_properties_against_file(writer, group_name, data_grp)
        part_ids = [int(x) for x in np.asarray(data_grp[f"{element_name('id')}/value"][-1])]
        writer.io_dict['flat_part_view'][group_name].extend(writer.sys.part.by_ids(part_ids))
        _check_load_new_columns(writer, group_name, data_grp)
        if writer.io_dict.get('bonds', False):
            verify_bond_params(h5_file, group_name, writer.sys)
            writer.io_dict['bond_links'][group_name] = int(
                h5_file[f"connectivity/{group_name}/bonds"].attrs["n_links"])
        candidate_lens.append(data_grp[f"{_TIMELINE}/value"].shape[0])
    if len(set(candidate_lens)) != 1:
        raise ValueError(
            f"Inconsistent step counts across groups: {candidate_lens}"
        )
    return candidate_lens[0]

def _check_load_new_columns(writer, group_name, data_grp):
    """Refuse LOAD_NEW unless the live particles are the file's columns, row for row.

    Compares the rebuilt ``flat_part_view`` against the last stored frame:
    ``type`` exactly, ``pos`` with ``np.isclose``. Appending with a different row
    order or different particles would silently corrupt every later frame.
    """
    file_types = np.asarray(data_grp[f"{element_name('type')}/value"][-1])
    file_pos = np.asarray(data_grp[f"{element_name('pos')}/value"][-1])
    n_live = len(writer.io_dict['flat_part_view'][group_name])
    if n_live != len(file_types):
        raise RuntimeError(f"LOAD_NEW: local particles do not match the file columns of "
                           f"'{group_name}': {n_live} live particles vs {len(file_types)} "
                           "stored columns; refusing to append.")
    live_types = writer._capture_frame(group_name, 'type', None, file_types.dtype)
    live_pos = writer._capture_frame(group_name, 'pos', 3, file_pos.dtype)
    bad = np.flatnonzero((live_types != file_types)
                         | ~np.all(np.isclose(live_pos, file_pos), axis=1))
    if bad.size:
        row = int(bad[0])
        raise RuntimeError(
            f"LOAD_NEW: local particles do not match the file columns of '{group_name}' at the "
            f"last frame: {bad.size} of {n_live} rows differ in type or pos (first at "
            f"column {row}: live type {live_types[row]} pos {live_pos[row].tolist()} vs "
            f"stored type {file_types[row]} pos {file_pos[row].tolist()}); refusing to append.")


def _particles_load(writer, group_type):
    """Resume from a file, taking the particle view from the live objects."""
    h5_file = writer.io_dict['h5_file']
    _require_layout(h5_file)
    particles_group = h5_file["particles"]
    candidate_lens=[]
    for grp_typ in group_type:
        group_name = grp_typ.__name__
        _check_properties_against_file(writer, group_name, particles_group[group_name])
        objects_to_register=[obj for obj in writer.objects if isinstance(obj,grp_typ)]
        owned = [part for cr in objects_to_register for part in cr.get_owned_part()[0]]
        owned.sort(key=lambda part: part.id)   # file columns are in ascending id
        writer.io_dict['flat_part_view'][group_name].extend(owned)
        if writer.io_dict.get('bonds', False):
            verify_bond_params(h5_file, group_name, writer.sys)
            writer.io_dict['bond_links'][group_name] = int(
                h5_file[f"connectivity/{group_name}/bonds"].attrs["n_links"])
        candidate_lens.append(particles_group[group_name][f"{_TIMELINE}/value"].shape[0])
    if len(set(candidate_lens)) != 1:
        raise ValueError(
            f"Inconsistent step counts across groups: {candidate_lens}"
        )
    return candidate_lens[0]

def _particles_resize(writer, group_type, n_frames):
    """Truncate every registered group to its first ``n_frames`` (``ValueError`` if it holds fewer)."""
    particles_group = writer.io_dict['h5_file']["particles"]
    for grp_typ in group_type:
        data_grp = particles_group[grp_typ.__name__]
        stored = data_grp[f"{_TIMELINE}/step"].shape[0]
        if n_frames > stored:
            raise ValueError(f"force_resize_to_size={n_frames} exceeds the {stored} frame(s) stored for "
                             f"'{grp_typ.__name__}' in {data_grp.file.filename}.")
        data_grp[f"{_TIMELINE}/step"].resize((n_frames,))
        data_grp[f"{_TIMELINE}/time"].resize((n_frames,))
        values = [member["value"] for name, member in data_grp.items() if name != "box"]
        for value in values + [data_grp["box/edges/value"]]:
            value.resize((n_frames, *value.shape[1:]))

def _particles_rewind(writer, group_type, step):
    """Truncate every registered group to end at the frame stored at ``step``.

    That frame's time must be the live ``sys.time`` (``restart_from_checkpoint``
    sets it), so a file can only be rewound to the state the system is in.

    ``RuntimeError`` unless the frame being rewound to was written at the live ``sys.time``.
    """
    h5_file = writer.io_dict['h5_file']
    for grp_typ in group_type:
        group_name = grp_typ.__name__
        frame = frame_of_step(h5_file, group_name, step)   # KeyError absent, ValueError duplicated
        data_grp = h5_file["particles"][group_name]
        stored_time = float(data_grp[f"{_TIMELINE}/time"][frame])
        live_time = float(writer.sys.time)
        if not np.isclose(stored_time, live_time, rtol=_TIME_RTOL):
            raise RuntimeError(
                f"rewind_to_step={step}: frame {frame} of {h5_file.filename}/particles/{group_name} has time {stored_time}, live sys.time is "
                f"{live_time} (tolerance: relative {_TIME_RTOL}); restart_from_checkpoint sets it."
            )
        n_frames = frame + 1
        data_grp[f"{_TIMELINE}/step"].resize((n_frames,))
        data_grp[f"{_TIMELINE}/time"].resize((n_frames,))
        values = [member["value"] for name, member in data_grp.items() if name != "box"]
        for value in values + [data_grp["box/edges/value"]]:
            value.resize((n_frames, *value.shape[1:]))
    


# ---------------------------------------------------------------------------
# Observable-stream inscription, one function per mode (same shape as the
# particle helpers). The normalised definitions live in
# ``writer.io_dict['registered_observables']``, filled by the setup step, so the
# kernels share no closure state.
# ---------------------------------------------------------------------------
def _observables_setup(writer, observable_defs, mode, h5_data_path):
    """Validate and normalise the definitions, register them, open the file if none is open."""
    if not isinstance(observable_defs, list) or not observable_defs:
        raise ValueError("observable_defs must be a non-empty list.")
    registered = {}
    for obs_def in observable_defs:
        if len(obs_def) != 4:
            raise ValueError("Each observable definition must be (name, shape, dtype, observable_value_ref).")
        name, shape, dtype, observable_value_ref = obs_def
        if shape is None:
            shape = tuple()
        elif isinstance(shape, (int, np.integer)):
            shape = (int(shape),)
        else:
            shape = tuple(shape)
        registered[str(name)] = {'shape': shape, 'dtype': np.dtype(dtype), 'value': observable_value_ref}
    writer.io_dict['registered_observables'] = registered

    if writer.io_dict['h5_file'] is None:
        if h5_data_path is None:
            raise ValueError("h5_data_path must be provided when no HDF5 file is currently open.")
        file_mode = "w" if mode == 'NEW' else "a"
        writer.io_dict['h5_file'] = h5py.File(h5_data_path, file_mode)

def _observables_new(writer):
    """Create ``/observables/<name>/{step,time,value}`` for every registered observable. Returns 0."""
    observables_group = writer.io_dict['h5_file'].require_group("observables")
    for name, spec in writer.io_dict['registered_observables'].items():
        shape = spec['shape']
        obs_group = observables_group.require_group(name)
        if any(key in obs_group for key in ('step', 'time', 'value')):
            raise ValueError(f"Observable '{name}' already exists in HDF5 file.")
        obs_group.create_dataset("step", shape=(0,), maxshape=(None,), dtype=np.int32)
        obs_group.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float32)
        obs_group.create_dataset(
            "value",
            shape=(0, *shape),
            maxshape=(None, *shape),
            dtype=spec['dtype'],
            chunks=(1, *shape) if shape else (1,),
            compression="gzip",
            compression_opts=4,
        )
    return 0

def _observables_load(writer, mode):
    """Reopen the registered observables, checking presence and per-frame shape. Returns the saved frame count."""
    _require_layout(writer.io_dict['h5_file'])
    observables_group = writer.io_dict['h5_file'].require_group("observables")
    candidate_lens = []
    for name, spec in writer.io_dict['registered_observables'].items():
        obs_group = observables_group.get(name)
        if obs_group is None:
            raise ValueError(f"Observable '{name}' was not found in HDF5 file during {mode}.")
        value_dataset = obs_group["value"]
        if tuple(value_dataset.shape[1:]) != spec['shape']:
            raise ValueError(
                f"Observable '{name}' shape mismatch: file has {value_dataset.shape[1:]}, expected {spec['shape']}."
            )
        candidate_lens.append(value_dataset.shape[0])
    if len(set(candidate_lens)) != 1:
        raise ValueError(f"Inconsistent step counts across observables: {candidate_lens}")
    return candidate_lens[0]

def _observables_resize(writer, n_frames):
    """Truncate every registered observable to its first ``n_frames`` (``ValueError`` if it holds fewer)."""
    observables_group = writer.io_dict['h5_file']["observables"]
    for name in writer.io_dict['registered_observables']:
        obs_group = observables_group[name]
        stored = obs_group["value"].shape[0]
        if n_frames > stored:
            raise ValueError(f"force_resize_to_size={n_frames} exceeds the {stored} frame(s) stored for "
                             f"observable '{name}' in {obs_group.file.filename}.")
        for name in ("step", "time", "value"):
            dataset = obs_group[name]
            dataset.resize((n_frames, *dataset.shape[1:]))

def _observables_rewind(writer, step):
    """Truncate every registered observable to end at the row stored at ``step``.

    Observables keep their own ``step``/``time`` per name, so the lookup is done
    here (same KeyError/ValueError contract as ``read.frame_of_step``) and the
    row's time must be the live ``sys.time``, as for the particle groups.

    ``RuntimeError`` unless the frame being rewound to was written at the live ``sys.time``.
    """
    h5_file = writer.io_dict['h5_file']
    for name in writer.io_dict['registered_observables']:
        obs_group = h5_file["observables"][name]
        where = f"{h5_file.filename}/observables/{name}"
        steps = np.asarray(obs_group["step"][...]).reshape(-1)
        matches = np.flatnonzero(steps == int(step))
        if matches.size == 0:
            raise KeyError(f"step {step} is not stored in {where}: it holds {steps.size} frame(s).")
        if matches.size > 1:
            raise ValueError(f"step {step} is stored in {matches.size} frames of {where} "
                             f"(indices {matches.tolist()}); the selection is ambiguous.")
        frame = int(matches[0])
        stored_time = float(obs_group["time"][frame])
        live_time = float(writer.sys.time)
        if not np.isclose(stored_time, live_time, rtol=_TIME_RTOL):
            raise RuntimeError(
                f"rewind_to_step={step}: frame {frame} of {where} has time {stored_time}, live sys.time is "
                f"{live_time} (tolerance: relative {_TIME_RTOL}); restart_from_checkpoint sets it."
            )
        n_frames = frame + 1
        for name in ("step", "time", "value"):
            dataset = obs_group[name]
            dataset.resize((n_frames, *dataset.shape[1:]))
