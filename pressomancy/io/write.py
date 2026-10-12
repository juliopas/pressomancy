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

The inscription lifecycle (crash safety; ``H5Writer._inscription``). Every file is
created in the HDF5 1.10 format (``LIBVER``) and, from the end of every inscription
call on, held in SWMR-write mode, where only frame appends happen: HDF5 orders their
disk writes so that a kill at any instant leaves a readable file (at worst a ragged
last frame, which ``rewind_to_step`` trims). Each frame write flushes and fsyncs
(``_write_frame``). One inscription call:

1. checks its arguments (``_check_inscription``, the stream's own definitions, the
   ``io_dict['properties']`` list); an error raises with an inscribed stream still
   open and appendable;
2. opens the file the stream goes into. A ``NEW`` with nothing inscribed creates the
   file at its real name (symlinks resolved) and writes the header (``/h5md``,
   ``/parameters/pressomancy``, ``/pressomancy/system``). Anything else -- a resume
   (``LOAD``/``LOAD_NEW``) or a second stream into the open file, which is closed
   first -- never modifies the file at its real name: it is copied (read through
   ``read.open_output``, so a cleanly closed, crashed or still-flagged file copies
   alike) into a fresh ``<path>.new`` (a stale one is overwritten, never read). A
   superblock older than v3 (a file from before the SWMR lifecycle) is refused: the
   copy would keep its v1 B-tree chunk indices, whose appends are not kill-safe;
3. adds the stream: creates its datasets (``NEW``), or truncates (``rewind_to_step``
   / ``force_resize_to_size``) and checks it against the file and the live system;
4. writes the caller's ``attrs`` (``{group path: {attribute name: value}}``);
5. switches to SWMR: a created file by ``swmr_mode = True``; ``.new`` is closed,
   installed by ``durable_replace`` (fsync, rename, fsync of the directory) and
   reopened directly in SWMR-write mode.

Any failure from step 2 on leaves nothing inscribed (``H5Writer._discard``). A staged
inscription that fails before its install deletes ``.new``: the real file is untouched
and the exception carries a note saying so; one that fails after it (the directory
fsync, the reopen) leaves the real file as the complete copy, without a note. A failed
first ``NEW`` leaves its partial file at the real name. A kill before the install leaves the real file intact;
after it only appends happen. Anything written to a file in SWMR mode other
than frames is not covered by the ordering guarantee (HDF5 2.0 allows it, but a kill
during that flush can leave the file unrepairable): metadata goes through ``attrs``.
'''
import contextlib
import logging
import os
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from pressomancy.io.bonds import write_bonds, verify_bond_params, check_bond_count
from pressomancy.io.read import (LAYOUT, _TIMELINE, _TIME_RTOL, _require_layout, attr_name,
                                 element_name, open_output, stored_steps, superblock_version)
from pressomancy.infra import api_agnostic_feature_check, get_repo_context, get_submission_creator_info


#: The on-disk format every file is created with: HDF5 1.10 on both bounds, the oldest
#: that supports SWMR (extensible-array chunk indices, checksummed metadata), pinned so
#: the file does not depend on the library that wrote it. ``H5Writer._inscription``'s
#: SWMR-write reopen repeats it as ``h5py.h5f.LIBVER_V110``.
LIBVER = ('v110', 'v110')


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

    def _check_inscription(self, kind, mode, h5_data_path, force_resize_to_size, rewind_to_step):
        """The argument checks every inscription runs before it touches anything (so an error keeps
        an inscribed stream open and appendable); returns the real path (symlinks resolved, so a
        replace lands beside the real file).

        ``kind`` is the ``io_dict`` registration of the inscribing stream kind
        (``'registered_group_type'`` or ``'registered_observables'``): a second inscription of one
        kind into the open file would replace the first one's registration, whose stream would then
        silently stop getting frames, so it raises.
        """
        open_file = self.io_dict['h5_file']
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
        path = os.path.realpath(os.fspath(h5_data_path))
        if open_file is not None:
            if path != os.path.realpath(open_file.filename):
                raise ValueError(f"{open_file.filename} is inscribed and open; a further stream goes into "
                                 f"that file (got h5_data_path={h5_data_path!r}).")
            if self.io_dict[kind]:
                what, each = (('particle groups', 'group') if kind == 'registered_group_type'
                              else ('observables', 'observable'))
                raise RuntimeError(f"{open_file.filename} already has {what} inscribed: inscribe every {each} "
                                   "in one call (a second inscription would replace the first one's "
                                   "registration, and its stream would silently stop getting frames).")
        return path

    def _discard(self):
        """Back to nothing inscribed: close the open file (if any) and clear every registration."""
        if self.io_dict['h5_file'] is not None:
            self.io_dict['h5_file'].close()
        self.io_dict['h5_file'] = None
        self.io_dict['flat_part_view'].clear()
        self.io_dict['registered_observables'] = {}
        self.io_dict['registered_group_type'] = None
        self.io_dict['bond_links'] = {}
        self._slice_cache.clear()

    @contextlib.contextmanager
    def _inscription(self, mode, path, attrs):
        """Steps 2-5 of the lifecycle (module docstring) around the ``with`` body, which adds the
        stream to ``io_dict['h5_file']``. ``path`` is the real path ``_check_inscription`` returned."""
        staged = None           # <path>.new, once created
        try:
            if self.io_dict['h5_file'] is None and mode == 'NEW':
                # A first NEW: the file at its real name, with the header.
                h5_file = self.io_dict['h5_file'] = h5py.File(path, 'w', libver=LIBVER)
                h5md_group = h5_file.require_group("h5md")
                h5md_group.attrs["version"] = np.array([1, 1], dtype=np.int32)
                h5md_group.require_group("author").attrs.update({"name": self.author_name,
                                                                 "email": self.author_email})
                creator_name, creator_version = get_submission_creator_info()
                h5md_group.require_group("creator").attrs.update({"name": creator_name,
                                                                  "version": creator_version})
                pressomancy_group = h5_file.require_group("parameters/pressomancy")
                pressomancy_group.attrs["version"] = get_repo_context(Path(__file__).resolve())[1]
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
            else:
                # A resume or a second stream: staged on a copy. An open file is closed first, so
                # that no dataset is ever created in SWMR mode and its frames are never at risk.
                if self.io_dict['h5_file'] is not None:
                    self.io_dict['h5_file'].close()
                version = superblock_version(path)
                if version is not None and version < 3:
                    raise RuntimeError(
                        f"{mode}: {path} has HDF5 superblock version {version} (written before the SWMR "
                        "lifecycle, or by another writer): it cannot be appended to crash-safely, and a copy "
                        "keeps its v1 B-tree chunk indices. Start a fresh NEW file.")
                with open_output(path) as src:
                    # into io_dict at once, so that _discard closes it before it is deleted
                    dst = self.io_dict['h5_file'] = h5py.File(path + '.new', 'w', libver=LIBVER)
                    staged = path + '.new'
                    dst.attrs.update(src.attrs)
                    for name in src:
                        src.copy(src[name], dst, name=name)
            yield
            h5_file = self.io_dict['h5_file']
            for group_path, values in (attrs or {}).items():
                h5_file.require_group(group_path).attrs.update(values)
            if staged is None:
                h5_file.swmr_mode = True
            else:
                h5_file.close()
                durable_replace(staged, path)
                # Reopen in ONE step, directly in SWMR-write mode: h5py's 'r+' cannot open in
                # SWMR-write, and a kill between 'r+' (whose open already writes the "open for
                # write" flag) and `swmr_mode = True` leaves a file that no reader opens and only
                # `h5clear -s` repairs (measured). The fapl pins LIBVER again: without it the upper
                # bound becomes v200. A first NEW is not staged: its file holds nothing yet, and
                # when there is no checkpoint the run's start rule recreates an unopenable output.
                fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
                fapl.set_libver_bounds(h5py.h5f.LIBVER_V110, h5py.h5f.LIBVER_V110)
                self.io_dict['h5_file'] = h5py.File(h5py.h5f.open(
                    os.fsencode(path), h5py.h5f.ACC_RDWR | h5py.h5f.ACC_SWMR_WRITE, fapl=fapl))
        except BaseException as exc:
            self._discard()
            if staged is not None:
                # .new gone = installed: the real file is the complete copy, nothing to say
                try:
                    os.remove(staged)
                except FileNotFoundError:
                    pass
                else:
                    exc.add_note(f"found on the staged copy {staged} (deleted); {path} is unchanged")
            raise

    def inscribe_part_group_to_h5(self, group_type, h5_data_path, mode='NEW',
                                  force_resize_to_size=None, rewind_to_step=None, attrs=None):
        """
        Inscribe the particle groups ``group_type`` into ``h5_data_path`` (the lifecycle:
        module docstring). When the call returns, the file is in SWMR-write mode and only
        frame appends happen to it.

        For each `group_type`:
        - Builds a flat list of particle handles and their coordinating indices
        - Creates `/particles/<GroupName>` and one `<element>/{step,time,value}` per property
        - Creates `/pressomancy/<GroupName>/ownership/ParticleHandle_to_<OwnerClass>` tables
        - Creates `/pressomancy/<GroupName>/ownership/<Left>_to_<Right>` object–object tables

        Parameters
        ----------
        group_type : list of type
            Simulation object classes; every instance in ``self.objects`` is a column. All groups
            go in ONE call: a second particle inscription into the open file raises.
        h5_data_path : str or path-like
            The file; a symlink is resolved (the file it points at is written and replaced). A
            further stream (observables) into the open file passes the same path.
        mode : {'NEW', 'LOAD', 'LOAD_NEW'}, optional
            - 'NEW': create the streams (default).
            - 'LOAD': resume them, taking the particle view from the live objects.
            - 'LOAD_NEW': resume them, rebuilding the particle view from the file's id columns;
              the live particles are checked against the last frame (type exactly, pos
              ``isclose``) and the file's ``part_types`` table against the live one (a name only
              the file declares is registered).
        force_resize_to_size, rewind_to_step : int or None, optional
            Load modes only, one at most: cut every group to this many frames, or to end at the
            frame stored at this step, whose time must equal the live ``sys.time`` (relative
            ``1e-6``; ``restart_from_checkpoint`` sets it). Runs before the load checks.
        attrs : dict or None, optional
            ``{group path: {attribute name: value}}``: the caller's metadata (run parameters,
            provenance, a per-start record), written after the streams and before the SWMR
            switch; on a resume into the installed copy. A value h5py cannot store raises h5py's
            ``TypeError`` after the stream was added: a first 'NEW' leaves its partial file at the
            real name, a staged inscription deletes ``.new``.

        Returns
        -------
        int
            The stored frame count: 0 for 'NEW', after the truncation for the load modes.

        Raises
        ------
        ValueError
            Before anything is touched: an unknown mode, a truncation argument outside the load
            modes or both given, a path other than the open file's, `group_type` not a non-empty
            list, ``io_dict['properties']`` malformed (it must start with ``('id', None, int32),
            ('type', None, int16), ('pos', 3, <floating>)``; attrs unique, dims None or positive,
            dtypes numpy-resolvable). Later: `force_resize_to_size` beyond the stored frames,
            `rewind_to_step` stored twice.
        TypeError
            A truncation argument that is not an integer (before anything is touched).
        KeyError
            `rewind_to_step` is not stored.
        RuntimeError
            Before anything is touched: particle groups already inscribed in the open file. Load
            modes: a superblock older than v3 (start a fresh NEW file), not the ``h5md-1``
            layout, a ragged file (streams with different frame counts, from a killed writer:
            resume with ``rewind_to_step``), ``io_dict['properties']`` not describing the file's
            elements exactly (set, dtype, per-particle shape: edit the list to match the file),
            the `rewind_to_step` frame's time not the live ``sys.time``; 'LOAD_NEW': the live
            particles not the last frame's columns, a type name with another number in the file.
        """
        path = self._check_inscription('registered_group_type', mode, h5_data_path,
                                       force_resize_to_size, rewind_to_step)
        if not isinstance(group_type, list) or not group_type:
            raise ValueError("group_type must be a non-empty list of classes.")
        _validate_properties(self.io_dict['properties'])
        names = [grp_typ.__name__ for grp_typ in group_type]
        with self._inscription(mode, path, attrs):
            self.io_dict['registered_group_type'] = names
            if mode == 'NEW':
                n_frames = _particles_new(self, group_type)
            else:
                n_frames = _particles_load(self, group_type, mode, force_resize_to_size, rewind_to_step)
        logging.info("%s: particle groups %s in %s, %d stored frame(s).", mode, names, path, n_frames)
        return n_frames

    def inscribe_observable_group_to_h5(self, observable_defs, h5_data_path, mode='NEW',
                                        force_resize_to_size=None, rewind_to_step=None, attrs=None):
        """
        Inscribe observable streams, ``/observables/<name>/{step,time,value}``, into
        ``h5_data_path``; as ``inscribe_part_group_to_h5`` for the lifecycle, the path, the
        modes ('LOAD_NEW' checks the ``part_types`` table, 'LOAD' does not), the truncation
        (per observable, by its own ``step``/``time``), ``attrs`` and the errors.

        Parameters
        ----------
        observable_defs : list of tuple
            Non-empty; every observable in ONE call. Each definition is
            ``(name, shape, dtype, observable_value_ref)``: ``shape`` is the per-frame shape
            (``None`` or ``()`` for a scalar, an int for one dimension), ``dtype`` the ``value``
            dataset's, ``observable_value_ref`` the live object or array read at every frame.

        Returns
        -------
        int
            The stored frame count: 0 for 'NEW', after the truncation for the load modes.

        Raises
        ------
        ValueError
            Also (before anything is touched) if `observable_defs` is not a non-empty list, a
            definition is not four entries, or two definitions share a name; in load modes, if
            an observable is missing or stored with another per-frame shape. 'NEW': if an
            observable already exists in the file.
        RuntimeError
            Also (before anything is touched) if observables are already inscribed in the open file.
        """
        path = self._check_inscription('registered_observables', mode, h5_data_path,
                                       force_resize_to_size, rewind_to_step)
        if not isinstance(observable_defs, list) or not observable_defs:
            raise ValueError("observable_defs must be a non-empty list.")
        registered = {}
        for obs_def in observable_defs:
            if len(obs_def) != 4:
                raise ValueError("Each observable definition must be (name, shape, dtype, observable_value_ref).")
            name, shape, dtype, observable_value_ref = obs_def
            name = str(name)
            if name in registered:
                raise ValueError(f"observable_defs: duplicate observable name '{name}'.")
            if shape is None:
                shape = tuple()
            elif isinstance(shape, (int, np.integer)):
                shape = (int(shape),)
            else:
                shape = tuple(shape)
            registered[name] = {'shape': shape, 'dtype': np.dtype(dtype), 'value': observable_value_ref}
        with self._inscription(mode, path, attrs):
            self.io_dict['registered_observables'] = registered
            if mode == 'NEW':
                n_frames = _observables_new(self)
            else:
                n_frames = _observables_load(self, mode, force_resize_to_size, rewind_to_step)
        logging.info("%s: observables %s in %s, %d stored frame(s).", mode, list(registered), path, n_frames)
        return n_frames

    def write_part_group_to_h5(self, step):
        """Append one particle frame; returns ``(frame, step)``, frame = the row just written.

        Every group has ONE ``step``/``time`` pair (hard-linked into each of its elements), so
        the pair gets one row per group and then every element's ``value`` gets its row -- the
        box is an element too, so ``box/edges/value`` gets one ``sys.box_l`` row per frame.
        The whole frame is captured before anything is written, then flushed and fsynced.

        step : int
            Any strictly increasing integer counter (e.g. the integration step or a coarser
            unit); the ``time`` dataset holds the physical time (``sys.time``). ``RuntimeError``
            unless it exceeds the last written step (H5MD), raised before anything is written; to
            continue from an earlier frame, inscribe with ``rewind_to_step`` or
            ``force_resize_to_size`` first: frames are never overwritten in place.
        """
        return _write_frame(self.io_dict['h5_file'], self._particle_frame(step)), int(step)

    def _particle_frame(self, step):
        """One particle frame as ``[(dataset, row)]``: every guard and every value captured,
        nothing written (``_write_frame`` writes it), so a raise here leaves no ragged frame."""
        if not self.io_dict['registered_group_type']:
            raise RuntimeError('No particle groups have been inscribed in HDF5.')
        if not isinstance(step, (int, np.integer)):
            raise TypeError("step must be provided as an integer frame counter.")
        physical_time = float(self.sys.time)
        box_l = np.asarray(self.sys.box_l, dtype=np.float64)
        particles_group = self.io_dict['h5_file']["particles"]
        rows = []
        for grp_typ in self.io_dict['registered_group_type']:
            if self.io_dict.get('bonds', False):
                check_bond_count(
                    self.io_dict['bond_links'].get(grp_typ),
                    self.io_dict['flat_part_view'][grp_typ],
                    group_name=grp_typ,
                )
            data_grp = particles_group[grp_typ]
            step_dataset = data_grp[f"{_TIMELINE}/step"]
            if step_dataset.shape[0] > 0 and step <= step_dataset[-1]:
                raise RuntimeError(
                    f"step must strictly increase (got {step}, last {int(step_dataset[-1])}) "
                    f"for group '{grp_typ}'; to continue from an earlier frame, inscribe with "
                    "rewind_to_step or force_resize_to_size first.")
            # The timeline once per group (hard-linked into every element); the box is an element too.
            rows += [(step_dataset, step), (data_grp[f"{_TIMELINE}/time"], physical_time),
                     (data_grp["box/edges/value"], box_l)]
            for attr, dim, dtype in self.io_dict['properties']:
                dataset = data_grp[f"{element_name(attr)}/value"]
                value = self._capture_frame(grp_typ, attr, dim, dtype)
                if value.shape != dataset.shape[1:]:
                    raise ValueError(f"Group '{grp_typ}': '{attr}' captured with shape {value.shape}, "
                                     f"its dataset holds rows of {dataset.shape[1:]}.")
                rows.append((dataset, value))
        return rows

    def write_observable_group_to_h5(self, step):
        """Append one frame to every registered observable; returns ``(frame, step)``, frame = the
        row just written.

        Values are read from the live references registered by
        :meth:`inscribe_observable_group_to_h5`; ``step`` and the physical time (``sys.time``)
        go into each observable's own ``step``/``time``. ``step`` is any strictly increasing
        integer counter, as for :meth:`write_part_group_to_h5`. Flushed and fsynced.
        """
        return _write_frame(self.io_dict['h5_file'], self._observable_frame(step)), int(step)

    def _observable_frame(self, step):
        """One observable frame as ``[(dataset, row)]``: every guard and every payload
        captured (copied, as the dataset's dtype), nothing written; see ``_particle_frame``."""
        registered_observables = self.io_dict['registered_observables']
        if not registered_observables:
            raise RuntimeError("No observables have been inscribed in HDF5.")
        if not isinstance(step, (int, np.integer)):
            raise TypeError("step must be provided as an integer frame counter.")

        physical_time = float(self.sys.time)
        observables_group = self.io_dict['h5_file']["observables"]

        for name in registered_observables:
            step_dataset = observables_group[name]["step"]
            if step_dataset.shape[0] > 0 and step <= step_dataset[-1]:
                raise RuntimeError(
                    f"step must strictly increase (got {step}, last {int(step_dataset[-1])}) "
                    f"for observable '{name}'; to continue from an earlier frame, inscribe with "
                    "rewind_to_step or force_resize_to_size first.")

        rows = []
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
            rows += [(obs_group["step"], step), (obs_group["time"], physical_time),
                     (value_dataset, np.array(payload, dtype=value_dataset.dtype))]
        return rows

    def write_registered_to_h5(self, step):
        """Append one frame with the same ``step`` and time to every registered stream (particle
        groups, observables or both); returns ``(frame, step)``, frame = the particle row when
        groups are registered, else the observables' row (equal when both were inscribed together).

        Both frames are captured before either is written (a failing capture writes nothing),
        then flushed and fsynced together, once. ``step`` as for :meth:`write_part_group_to_h5`.
        """
        groups, observables = self.io_dict['registered_group_type'], self.io_dict['registered_observables']
        if not groups and not observables:
            raise RuntimeError("No particle groups or observables have been inscribed in HDF5.")
        rows = (self._particle_frame(step) if groups else []) + (self._observable_frame(step) if observables else [])
        return _write_frame(self.io_dict['h5_file'], rows), int(step)

    def write_checkpoint(self, group_type, path, step):
        """Write a verified one-frame checkpoint of ``group_type`` to ``path``.

        A checkpoint is an ordinary ``h5md-1`` file with one frame of
        ``checkpoint_properties(io_dict['properties'])`` (the restorable
        per-particle state in float64, bonds on) and a ``/pressomancy/checkpoint``
        group whose attrs are ``time`` (float64), ``step`` and one
        ``<kind>_philox_counter`` per active thermostat. ``restart_from_checkpoint``
        reads it back and ``inscribe_*(..., rewind_to_step=step)`` then re-aligns
        the trajectory.

        Its bond-parameter tables follow its float64 ``pos`` (a float32 trajectory's are
        float32), so a ``bonds=True`` restart restores every bond parameter exactly.

        The file goes to ``path + '.tmp'``, is closed, reopened read-only and verified
        (layout, exactly the one step, live-vs-file ``type``/``pos`` columns, bond
        link count, time) and only then installed onto ``path`` by ``durable_replace``
        (fsync, ``os.replace``, fsync of the directory), so a node failure leaves the
        old or the new checkpoint, whole. On any failure the ``.tmp`` is left in place
        for inspection and a previous ``path`` is untouched; the package never deletes.

        :param group_type: list of object classes, as for ``inscribe_part_group_to_h5``.
        :param path: the checkpoint file; ``ValueError`` if it is the open trajectory.
        :param step: int | the step value stored with the frame (any strictly increasing
            integer counter, as for ``write_part_group_to_h5``); returned.
        :raises NotImplementedError: if the LB thermostat is active (the fluid
            state is not checkpointed).
        """
        path = os.path.realpath(path)    # a symlinked checkpoint: .tmp and the replace go beside the real file
        trajectory = self.io_dict['h5_file']
        if trajectory is not None and path == os.path.realpath(trajectory.filename):
            raise ValueError(f"write_checkpoint: {path} is the open trajectory file; "
                             "a checkpoint needs its own path.")
        lb = getattr(self.sys.thermostat, 'lb', None)   # absent when not compiled in
        if lb is not None and lb.is_active:
            raise NotImplementedError("write_checkpoint: the LB thermostat is active and the "
                                      "fluid state is not checkpointed.")

        tmp = str(path) + ".tmp"
        # The checkpoint group goes in through attrs=, i.e. before the SWMR switch, like
        # every caller's metadata; nothing structural is written after it.
        checkpoint_attrs = {'time': np.float64(self.sys.time), 'step': int(step)}
        for kind in _THERMOSTAT_PHILOX_KINDS:
            thermostat = getattr(self.sys.thermostat, kind, None)   # absent when not compiled in
            if thermostat is not None and thermostat.is_active:
                checkpoint_attrs[f"{kind}_philox_counter"] = int(thermostat.philox_counter)
        ckpt = H5Writer(self._owner)
        ckpt.author_name, ckpt.author_email = self.author_name, self.author_email
        ckpt.io_dict['properties'] = checkpoint_properties(self.io_dict['properties'])
        ckpt.io_dict['bonds'] = True
        try:
            ckpt.inscribe_part_group_to_h5(group_type, tmp, 'NEW', attrs={'pressomancy/checkpoint': checkpoint_attrs})
            ckpt.write_part_group_to_h5(step)
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
        durable_replace(tmp, path)
        return int(step)


def durable_replace(tmp, path):
    """Install the finished file ``tmp`` at ``path`` so that a node failure leaves either file, whole.

    The caller closes ``tmp`` first: an HDF5 close rewrites the superblock, so the
    fsync must come after it. Then ``tmp`` is fsynced, ``os.replace``d onto ``path``,
    and ``path``'s directory is fsynced so the rename itself reaches the disk. Both
    paths are resolved first: a symlinked ``path`` keeps its link, and the real file
    it points at is the one replaced (in its own directory).
    """
    tmp, path = os.path.realpath(tmp), os.path.realpath(path)
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    dir_fd = os.open(os.path.dirname(path), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _write_frame(h5_file, rows):
    """Write one captured frame and push it to disk; returns the frame index of its first row.

    Each ``(dataset, row)`` grows by one row and gets it; then ``flush()`` (SWMR orders the
    disk writes, so the frame survives a kill of the process) and ``fsync`` of the file. The
    flush alone does not survive a node crash: the kernel writes dirty pages back after ~30 s,
    in any order. The fsync shrinks that exposure to the ~20 ms of the frame write itself, and
    keeps the frame at a checkpoint's step (the checkpoint itself is fsynced) durable in the
    trajectory, so ``rewind_to_step`` finds it. Measured: +0.5 ms per frame (24 000 particles,
    0.57 MB per frame).
    """
    frame = rows[0][0].shape[0]
    for dataset, row in rows:
        n = dataset.shape[0]
        dataset.resize((n + 1, *dataset.shape[1:]))
        dataset[n] = row
    h5_file.flush()
    os.fsync(h5_file.id.get_vfd_handle())
    return frame


# ---------------------------------------------------------------------------
# Stream inscription: one NEW and one load function per kind, called inside
# ``H5Writer._inscription`` on the file it opened.
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


def _stream_datasets(data_grp):
    """``{name: dataset}`` of everything a particle group appends per frame: the timeline,
    the box edges and every element's ``value``."""
    datasets = {'step': data_grp[f"{_TIMELINE}/step"], 'time': data_grp[f"{_TIMELINE}/time"],
                'box/edges': data_grp["box/edges/value"]}
    datasets.update({name: member["value"] for name, member in data_grp.items() if name != "box"})
    return datasets


def _frames_stored(datasets, where):
    """The one frame count all ``datasets`` share; ``RuntimeError`` naming the ragged ones
    otherwise (a writer killed mid-append leaves some streams one row longer)."""
    lengths = {name: dataset.shape[0] for name, dataset in datasets.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"{where} is ragged: its streams hold different frame counts {lengths} "
                           "(the writer was killed mid-append); resume with rewind_to_step at the last "
                           "complete frame.")
    return next(iter(lengths.values()))


def _truncate(datasets, where, force_resize_to_size, rewind_to_step, live_time):
    """Cut the per-frame ``datasets`` of one group or observable (``step`` and ``time`` among
    them) to ``force_resize_to_size`` frames, or to end at the frame stored at ``rewind_to_step``;
    nothing when both are None.

    The ``rewind_to_step`` frame's time must be ``live_time`` (relative ``_TIME_RTOL``):
    ``restart_from_checkpoint`` sets ``sys.time``, so a file can only be rewound to the state the
    system is in (``RuntimeError`` otherwise). ``KeyError`` if the step is not stored,
    ``ValueError`` if it is stored twice, or if a dataset holds fewer frames than asked for
    (resizing it up would fill the frames with zeros).
    """
    if rewind_to_step is not None:
        steps = np.asarray(datasets['step'][...]).reshape(-1)
        matches = np.flatnonzero(steps == int(rewind_to_step))
        if matches.size == 0:
            raise KeyError(f"step {rewind_to_step} is not stored in {where}: it holds {steps.size} frame(s)"
                           + (f" with steps {int(steps[0])}..{int(steps[-1])}." if steps.size else "."))
        if matches.size > 1:
            raise ValueError(f"step {rewind_to_step} is stored in {matches.size} frames of {where} "
                             f"(indices {matches.tolist()}); the selection is ambiguous.")
        frame = int(matches[0])
        stored_time = float(datasets['time'][frame])
        if not np.isclose(stored_time, live_time, rtol=_TIME_RTOL):
            raise RuntimeError(
                f"rewind_to_step={rewind_to_step}: frame {frame} of {where} has time {stored_time}, live "
                f"sys.time is {live_time} (tolerance: relative {_TIME_RTOL}); restart_from_checkpoint sets it.")
        n_frames, what = frame + 1, f"rewind_to_step={rewind_to_step} (frame {frame})"
    elif force_resize_to_size is not None:
        n_frames, what = force_resize_to_size, f"force_resize_to_size={force_resize_to_size}"
    else:
        return
    short = {name: dataset.shape[0] for name, dataset in datasets.items() if dataset.shape[0] < n_frames}
    if short:
        raise ValueError(f"{what} exceeds the frame(s) stored for {where}: {short} hold fewer than {n_frames}.")
    for dataset in datasets.values():
        dataset.resize((n_frames, *dataset.shape[1:]))
    logging.info("%s: %s cut to %d frame(s).", what, where, n_frames)

def _particles_new(writer, group_type):
    """Create the group/schema tree of every group, bond-parameter floats in the ``pos`` dtype
    (float32 trajectories, float64 checkpoints). Returns the stored frame count (0)."""
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
                float_dtype=writer.io_dict['properties'][2][2],    # pos (_validate_properties)
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


def _particles_load(writer, group_type, mode, force_resize_to_size, rewind_to_step):
    """Resume the particle groups; returns the stored frame count after the truncation.

    Per group: the properties list against the file, then the truncation. Then one ragged
    check over every per-frame dataset of every group (a killed writer leaves some one row
    longer; groups with different counts are caught the same way). Then the particle view:
    ``LOAD`` takes it from the live objects; ``LOAD_NEW`` rebuilds it from the file's last
    ``id`` row (ascending id, the column order: one read, one ``by_ids``) and checks the live
    particles against the last frame (``_check_load_new_columns``), after the file's
    ``part_types`` table against the live one. Bond tables, when on, are verified.
    """
    h5_file = writer.io_dict['h5_file']
    _require_layout(h5_file)
    if mode == 'LOAD_NEW':
        writer._check_part_types_against_file(h5_file)
    streams = {}
    for grp_typ in group_type:
        group_name = grp_typ.__name__
        data_grp = h5_file["particles"][group_name]
        _check_properties_against_file(writer, group_name, data_grp)
        datasets = _stream_datasets(data_grp)
        _truncate(datasets, f"{h5_file.filename}/particles/{group_name}", force_resize_to_size, rewind_to_step,
                  float(writer.sys.time))
        streams.update({f"{group_name}/{name}": dataset for name, dataset in datasets.items()})
    n_frames = _frames_stored(streams, f"{h5_file.filename}/particles")
    for grp_typ in group_type:
        group_name = grp_typ.__name__
        data_grp = h5_file["particles"][group_name]
        view = writer.io_dict['flat_part_view'][group_name]
        if mode == 'LOAD_NEW':
            part_ids = [int(x) for x in np.asarray(data_grp[f"{element_name('id')}/value"][-1])]
            view.extend(writer.sys.part.by_ids(part_ids))
            _check_load_new_columns(writer, group_name, data_grp)
        else:
            owned = [part for obj in writer.objects if isinstance(obj, grp_typ) for part in obj.get_owned_part()[0]]
            view.extend(sorted(owned, key=lambda part: part.id))   # file columns are in ascending id
        if writer.io_dict.get('bonds', False):
            verify_bond_params(h5_file, group_name, writer.sys)
            writer.io_dict['bond_links'][group_name] = int(
                h5_file[f"connectivity/{group_name}/bonds"].attrs["n_links"])
    return n_frames

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


def _observables_load(writer, mode, force_resize_to_size, rewind_to_step):
    """Resume the registered observables; returns the stored frame count after the truncation.

    Per observable: presence and per-frame shape, then the truncation by its own
    ``step``/``time``; then one ragged check over all of them. ``LOAD_NEW`` first checks the
    file's ``part_types`` table, as for the particle groups, so a script that resumes
    observables alone still gets the writer's type names.
    """
    h5_file = writer.io_dict['h5_file']
    _require_layout(h5_file)
    if mode == 'LOAD_NEW':
        writer._check_part_types_against_file(h5_file)
    streams = {}
    for name, spec in writer.io_dict['registered_observables'].items():
        obs_group = h5_file.get(f"observables/{name}")
        if obs_group is None:
            raise ValueError(f"Observable '{name}' was not found in HDF5 file during {mode}.")
        if tuple(obs_group["value"].shape[1:]) != spec['shape']:
            raise ValueError(f"Observable '{name}' shape mismatch: file has {obs_group['value'].shape[1:]}, "
                             f"expected {spec['shape']}.")
        datasets = {key: obs_group[key] for key in ("step", "time", "value")}
        _truncate(datasets, f"{h5_file.filename}/observables/{name}", force_resize_to_size, rewind_to_step,
                  float(writer.sys.time))
        streams.update({f"{name}/{key}": dataset for key, dataset in datasets.items()})
    return _frames_stored(streams, f"{h5_file.filename}/observables")
