'''
Seeding a simulation from an existing HDF5 file.

``H5Init`` holds the source-file parameters declared by ``set_init_src`` and the
readers built on them: ``get_pos_ori_from_src`` supplies positions and
orientations for ``set_objects_from_src``, ``set_prop_from_src`` copies particle
properties onto already-built objects, ``set_bonds_from_src`` re-creates the
stored bond topology on them, and ``load_from_src`` chains all of them so a
script seeds a system from a file in one call (``Simulation.load_from_src`` is
the only public seeding entry; the steps stay reachable on
``Simulation.h5_init``). ``get_prop_from_src`` reads a stored property back
per object without touching the system, for everything that is not a 1:1 copy.

**Placement is optional (``place_from``).** Give ``place_from`` (a list of
*source* types, names or ints) when the file should place your objects, which
requires one stored particle of those types per monomer; leave it out when you
built the tree yourself (compound objects, running systems). With ``place_from``
the objects are placed first (``get_pos_ori_from_src`` + ``place_objects``),
then ``src_to_loc`` is copied, then the bonds; with ``place_from=None``
(the default) placement is skipped and
``get_pos_ori_from_src`` / ``set_objects_from_src`` raise ``RuntimeError``.
Nothing else changes: the same who_am_i set check, the same ne-to-one pairing
per type pair, the same frame selectors.

**Frame selection.** Every reader takes ``frame`` (a frame *index* into the
source file, python-style negatives allowed), ``step`` (a stored *step value*)
and ``time`` (a stored simulation time). All ``None`` selects the last frame.
Each given selector is resolved to a frame index; when more than one is given
they must resolve to the same frame or ``ValueError`` is raised. ``time`` matches
the nearest stored time within a relative tolerance of _POS_ATOL=1e-6
(:func:`pressomancy.io.read.frame_of_time`) and raises ``KeyError`` otherwise. A
frame index and a step value are different numbers: frames are written every
``save_h5_each`` steps and a chained run keeps counting.

**Type names.** Source type names (those of ``place_from`` and the ``src_type``
half of every ``src_to_loc`` type pair) are resolved through the *source file's*
``parameters/pressomancy/part_types``; local names through
``Simulation.part_types``. The two systems need not share a type table.

**Property names.** Both halves of a property pair are espresso attribute names
(``'pos'``, ``'type'``, ``'f'``, ``'image_box'``, ``'director'``, ``'dip'``,
``'id'``): the file uses H5MD element names, pressomancy uses espresso names, and
the translation is ``pressomancy.io.read.H5MD_NAMES``, applied only when a
dataset path is built.

**Files without the type table: use numeric source types.** Every *source*
type may be given as an ``int`` -- the numeric espresso type as stored in the
file -- instead of a name, and is then used as is, so a file that carries no
``parameters/pressomancy/part_types`` table (written by something else) can
still be read::

    sim.load_from_src(objs, path, src_to_loc={(61, 'real'): [('pos', 'pos')]},
                      place_from=[61])

Local types (the right half of every type pair) stay names. A source *name* that
the file's table cannot resolve raises ``KeyError`` saying whether the table is
absent or only lacks that name, which numeric types the group stores, and that
an int may be given instead.

**The src_to_loc mapping.** One dict says both *which* particles pair up and
*what* is copied onto them. Every tuple in it is ordered ``(source, local)``,
without exception.

- A **key** is either one type pair, ``('pdp_real', 'real')`` (or ``(61, 'real')``),
  or a tuple of type pairs, ``(('pdp_real', 'real'), ('pdp_virt', 'virt'))`` -- the
  two are told apart by whether ``key[0]`` is a source type (a name or an int) or
  a pair.
- A **value** is the list of property pairs copied for the key's type pair(s),
  e.g. ``[('pos', 'pos'), ('dip', 'dip')]``. ``[]`` is allowed and means "pair
  these particles, copy nothing": the pair still takes part in the one-to-one
  count check and in bond restoration.
- The same type pair may appear under several keys; its property lists are
  **concatenated** in first-appearance order. Listing the same property pair
  twice for one type pair raises ``ValueError``, as does a malformed key or
  value (the raise names the offending entry).
- The normalised form kept on ``H5Init.src_to_loc`` is
  ``dict[(src_type, loc_type)] -> [(src_prop, loc_prop), ...]``;
  ``list(self.src_to_loc.keys())`` is the type pairing used by the count checks and by
  ``set_bonds_from_src``.

Three recipes cover what scripts do with it::

    # checkpoint restart of simple objects: the file places them, identity over every owned
    # type, the saved state, bonds back
    # E.g.
    STATE = [('pos','pos'), ('director','director'), ('dip','dip')]
    sim.load_from_src(objs, path, src_to_loc={(('real','real'), ('virt','virt')): STATE,
                                              ('substrate','substrate'): []},
                      bonds=True, place_from=['real'], step=k)
    # re-typed start: new object classes at the saved places; only what both share; the new objects build their own bonds
    # E.g.
    sim.load_from_src(new_objs, path, src_to_loc={('pdp_real','real'): [('pos','pos'), ('director','director')]},
                      place_from=['pdp_real'])
    # compound objects / an existing tree: build and place it as usual, then only copy the state
    # (no place_from; bonds=True only for objects that do not build their own bonds)
    # E.g.
    sim.load_from_src(objs, path, src_to_loc={(('real','real'), ('virt','virt')): [('pos','pos')]})
    # anything that is not a 1:1 copy: fetch, then let the object apply it
    dips = sim.get_prop_from_src(objs, path, src_type='pdp_real', prop='dip', step=k)   # one (N_i, 3) array per object
    _ = [obj_i.set_dips(dips_i) for (obj_i, dips_i) in zip(objs, dips)]

**The zip convention.** All readers pair source and local particles the same
way: for each registered object and each ``(src_type, loc_type)`` pair, the
source particles of ``src_type`` connected to that object
(``select_particles_by_object`` order, i.e. file column order) are zipped with
the object's owned handles of ``loc_type``. Both sides are in **ascending
particle id**: the writer stores columns in id order and the reader sorts the
local handles the same way, so the rule is one sentence and can be checked on
the file alone (the ``id`` column is strictly increasing). Ids need not be
contiguous; the counts must match exactly, otherwise ``ValueError``.

**The who_am_i contract.** Source particles are attributed to a local object
through the object's ``who_am_i``, which the ``Simulation_Object`` metaclass
assigns from a per-class construction counter. The local object tree must
therefore be built in the same order (same classes, same count, same nesting)
as the one that wrote the source, so that local and stored ``who_am_i`` values
coincide. Two checks enforce it loudly: ``_resolve_frame`` requires the set of
``who_am_i`` values stored in ``pressomancy/<Group>/ownership/ParticleHandle_to_<Group>``
to equal the set of the objects being seeded, and ``set_bonds_from_src``
cross-checks the source ``pos`` of every zipped pair against the local
particle's ``pos`` before any bond is attached.

**Layout.** Every reader opens the source through ``pressomancy.io.read``, which
raises ``RuntimeError`` for a file without ``parameters/pressomancy/layout ==
"h5md-1"``.

This is the read-to-initialise half of the HDF5 layer.
``Simulation`` holds one ``H5Init`` and delegates ``load_from_src`` to it.
'''
import logging
import os

import h5py
import numpy as np

from pressomancy.io.read import (H5DataSelector, frame_of_step, frame_of_time, stored_steps,
                                 element_name, attr_name)
from pressomancy.io.write import checkpoint_properties
from pressomancy.geometry import require_min_global_cut
from pressomancy.io.bonds import read_bond_params, read_bonds

#: Absolute tolerance of the source-vs-local position cross-check in set_bonds_from_src.
_POS_ATOL = 1e-6


_TYPE_TABLE = "parameters/pressomancy/part_types"


def _is_type_number(value):
    """True for an ``int`` usable as a numeric espresso type (``bool`` is not one)."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _stored_numeric_types(src_file, group_name):
    """The numeric types stored for ``group_name`` at the last frame, sorted."""
    dataset = f"particles/{group_name}/{element_name('type')}/value"
    if group_name is None or dataset not in src_file:
        return None
    return sorted(int(x) for x in np.unique(np.asarray(src_file[dataset][-1])))


def _src_part_types(src_file, names=(), group_name=None):
    """``{source type: numeric id}`` for every entry of ``names``.

    A source type is either a *name*, resolved through the source file's
    ``parameters/pressomancy/part_types`` attrs, or an ``int``, which is the
    numeric espresso type stored in the file and passes through unchanged. A
    name that the file's table cannot resolve raises ``KeyError`` saying whether
    the table is absent or only lacks the name, which numeric types the group
    does store, and that an int may be given instead.
    """
    table = ({str(name): int(value) for name, value in src_file[_TYPE_TABLE].attrs.items()}
             if _TYPE_TABLE in src_file else None)
    resolved, missing = {}, []
    for src_type in names:
        if _is_type_number(src_type):
            resolved[int(src_type)] = int(src_type)
        elif table is not None and src_type in table:
            resolved[src_type] = table[src_type]
        else:
            missing.append(src_type)
    if missing:
        where = f"{src_file.filename}/{_TYPE_TABLE}"
        reason = (f"{where} is absent: the source file carries no type table"
                  if table is None else f"{where} declares only {sorted(table)}")
        stored = _stored_numeric_types(src_file, group_name)
        stored_msg = ("" if stored is None else
                      f" The numeric types stored for {group_name} (last frame) are {stored}.")
        raise KeyError(f"source type(s) {missing} cannot be resolved: {reason}.{stored_msg} A "
                       "source type may be a name or an int, so pass the numeric type in place "
                       "of the name (e.g. src_to_loc={(61, 'real'): [('pos', 'pos')]}, "
                       "place_from=[61]).")
    return resolved


def _str_pair(value, what):
    """``value`` as a ``(str, str)`` tuple; ``ValueError`` naming it otherwise."""
    pair = tuple(value) if isinstance(value, (tuple, list)) else ()
    if len(pair) != 2 or not all(isinstance(name, str) for name in pair):
        raise ValueError(f"{what} must be a (str, str) pair; got {value!r}.")
    return pair


def _type_pair(value, what):
    """``value`` as a ``(src_type, loc_type)`` pair; source name or int, local name."""
    pair = tuple(value) if isinstance(value, (tuple, list)) else ()
    if len(pair) != 2 or not (isinstance(pair[0], str) or _is_type_number(pair[0])) \
            or not isinstance(pair[1], str):
        raise ValueError(f"{what} must be a (source type, local type) pair, where the source type "
                         f"is a name or an int (a numeric type in the file) and the local type is "
                         f"a name; got {value!r}.")
    return (pair[0] if isinstance(pair[0], str) else int(pair[0]), pair[1])


def _normalised_src_to_loc(src_to_loc):
    """The ``src_to_loc`` grammar (module docstring) as ``{(src_type, loc_type): [(src_prop, loc_prop),]}``.

    Keys are one type pair or a tuple of type pairs; the property lists of a type
    pair appearing under several keys are concatenated in first-appearance order.
    Every malformed entry and every property pair repeated for one type pair
    raises ``ValueError`` naming the entry, so a typo can never be read as "copy
    nothing".
    """
    if src_to_loc is None:
        return {}
    if not hasattr(src_to_loc, 'items'):
        raise ValueError(f"src_to_loc must be a dict of {{type pair(s): [(src_prop, loc_prop), ...]}}; "
                         f"got {src_to_loc!r}.")
    normalised = {}
    for key, props in src_to_loc.items():
        where = f"src_to_loc key {key!r}"
        if not isinstance(key, tuple) or not key:
            raise ValueError(f"{where}: a key is one (src_type, loc_type) pair or a non-empty "
                             "tuple of such pairs.")
        type_pairs = ([_type_pair(key, where)]
                if isinstance(key[0], (str, (int, np.integer)))
                else [_type_pair(entry, where) for entry in key]
            )
        if isinstance(props, str) or not isinstance(props, (list, tuple)):
            raise ValueError(f"src_to_loc[{key!r}] must be a list of (src_prop, loc_prop) pairs "
                             f"(possibly empty); got {props!r}.")
        prop_pairs = [_str_pair(entry, f"src_to_loc[{key!r}] property") for entry in props]
        for type_pair in type_pairs:
            copied = normalised.setdefault(type_pair, [])
            for prop_pair in prop_pairs:
                if prop_pair in copied:
                    raise ValueError(f"src_to_loc: property pair {prop_pair} is listed twice for "
                                     f"type pair {type_pair}; each property is copied once.")
                copied.append(prop_pair)
    return normalised


class H5Init:
    """Source-file parameters and the readers that consume them."""

    def __init__(self, owner):
        #: the Simulation instance that owns this
        self._owner = owner
        self.src_path_h5 = None
        self.src_to_loc = {}
        self.place_from = None

    @property
    def sys(self):
        return self._owner.sys

    @property
    def part_types(self):
        return self._owner.part_types

    # -- declaring the source ---------------------------------------------------
    def set_init_src(self, path, src_to_loc=None, place_from=None):
        """Declare the source file, the ``src_to_loc`` mapping and the placement source.

        Parameters
        ----------
        path : str
            HDF5 file written by :class:`pressomancy.io.write.H5Writer`.
        src_to_loc : dict, optional
            ``{type pair(s): [(src_prop, loc_prop), ...]}``; the format is in the
            module docstring. ``None`` (the default) means ``{}``: no pairing and
            no property copy.
        place_from : iterable of str or int, optional
            *Source* types (names, or the numeric types stored in the file) whose
            particles supply positions/orientations in ``get_pos_ori_from_src`` /
            ``set_objects_from_src``. ``None`` (the default) means the objects are
            built and placed by the script itself, and those two readers refuse to
            run.

        Raises
        ------
        FileNotFoundError
            If ``path`` is not a file.
        ValueError
            If ``src_to_loc`` has a malformed key or value, repeats a property
            pair for one type pair, or ``place_from`` holds something that is
            neither a name nor an int.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"source file {path!r} does not exist.")
        # normalize before anything - in case it raises
        normalised = _normalised_src_to_loc(src_to_loc)
        places = None
        if place_from is not None:
            places = list(place_from)
            bad = [src for src in places if not (isinstance(src, str) or _is_type_number(src))]
            if bad:
                raise ValueError(f"place_from entries are source types, a name or an int (a numeric "
                                 f"type in the file); got {bad!r}.")
            places = [src if isinstance(src, str) else int(src) for src in places]
        self.src_path_h5 = path
        self.src_to_loc = normalised
        self.place_from = places

    def _require_src(self, what):
        if self.src_path_h5 is None:
            raise RuntimeError(f"{what}: no source declared; call set_init_src first.")
        return self.src_path_h5

    def _require_place_from(self, what):
        if self.place_from is None:
            raise RuntimeError(f"{what}: no place_from declared; placing objects from a file needs "
                               "the source types (a name or an int) that supply one position per "
                               "monomer. Pass "
                               "place_from=[...] (objects the file should place), or build and "
                               "place the objects yourself and only copy their properties.")

    def _require_src_to_loc(self, what):
        if not self.src_to_loc:
            raise ValueError(f"{what}: src_to_loc is empty; declare the (src_type, loc_type) "
                             "pairs that pair source and local particles.")

    # -- frame selection and the who_am_i contract ---------------------------------
    def _resolve_frame(self, registered_objs, frame, step, time, what):
        """``(objs, group_name, frame_index)`` for one reader call.

        ``objs`` is ``list(registered_objs)``: non-empty and all of one class
        (``ValueError`` otherwise); ``group_name`` is that class name, the particle
        group of the source. The frame index is resolved from the selectors as
        described in the module docstring (all ``None`` = last frame) and is
        always non-negative. While the file is open the who_am_i contract is
        checked: the set of ``who_am_i`` values stored for the group must equal
        ``{o.who_am_i for o in objs}``.
        """
        objs = list(registered_objs)
        if not objs:
            raise ValueError(f"{what}: registered_objs is empty.")
        if not all(type(o) is type(objs[0]) for o in objs):
            raise ValueError(f"{what}: registered_objs must be of one class; got "
                             f"{sorted({type(o).__name__ for o in objs})}.")
        group_name = type(objs[0]).__name__
        path = self._require_src(what)
        with h5py.File(path, "r") as src_file:
            n_frames = len(stored_steps(src_file, group_name))
            resolved = {}
            if frame is not None:
                if not -n_frames <= int(frame) < n_frames:
                    raise IndexError(f"{what}: frame {frame} is out of range; {group_name} holds "
                                     f"{n_frames} frame(s).")
                resolved['frame'] = int(frame) % n_frames
            if step is not None:
                resolved['step'] = frame_of_step(src_file, group_name, step)
            if time is not None:
                resolved['time'] = frame_of_time(src_file, group_name, time)
            if len(set(resolved.values())) > 1:
                raise ValueError(f"{what}: frame={frame}, step={step}, time={time} select different "
                                 f"frames of {group_name} ({resolved}); pass one selector, or "
                                 "selectors that agree.")
            frame_index = next(iter(resolved.values())) if resolved else n_frames - 1

            table = f"pressomancy/{group_name}/ownership/ParticleHandle_to_{group_name}"
            if table not in src_file:
                raise KeyError(f"{what}: {path} has no '{table}' table.")
            src_ids = {int(x) for x in np.unique(src_file[table][:, 1])}
            loc_ids = {int(o.who_am_i) for o in objs}
            if src_ids != loc_ids:
                raise ValueError(
                    f"{what}: the source's {group_name} who_am_i values {sorted(src_ids)} differ "
                    f"from the local ones {sorted(loc_ids)}. who_am_i is a construction counter, "
                    "so the local object tree must be built in the same order (same classes, "
                    "same count) as the one that wrote the source.")
        return objs, group_name, frame_index

    # -- the zip convention --------------------------------------------------------
    def _select_src_particles(self, src_data_grp, loc_obj, type_id, frame):
        """``(selector, n)``: source particles of numeric ``type_id`` owned by ``loc_obj`` at ``frame``.

        Row order is file column order, which is the source half of the zip convention.
        """
        selection = src_data_grp.timestep[frame].select_particles_by_object(
            object_name=loc_obj.__class__.__name__,
            connectivity_value=loc_obj.who_am_i,
            predicate=lambda subset: subset.type == type_id,
        )
        return selection, len(selection.particles)

    def _local_handles(self, loc_obj, loc_typ):
        """Owned handles of ``loc_typ`` in ascending id: the local half of the zip convention."""
        type_id = self.part_types[loc_typ]
        return sorted((x for x in loc_obj.get_owned_part()[0] if x.type == type_id), key=lambda x: x.id)

    def _zipped(self, src_data_grp, src_types, loc_obj, src_typ, loc_typ, frame, what):
        """``(selection, handles)`` of one ``(src_typ, loc_typ)`` pair for ``loc_obj``; counts must match."""
        selection, n_src = self._select_src_particles(src_data_grp, loc_obj, src_types[src_typ], frame)
        handles = self._local_handles(loc_obj, loc_typ)
        if n_src != len(handles):
            raise ValueError(
                f"{what}: {type(loc_obj).__name__}[{loc_obj.who_am_i}] {src_typ}->{loc_typ}: "
                f"{n_src} source particles vs {len(handles)} local particles; the zip convention "
                "pairs them one-to-one.")
        return selection, handles

    # -- readers ---------------------------------------------------------------------
    def get_pos_ori_from_src(self, registered_objs, frame=None, step=None, time=None):
        """Positions and orientations of ``registered_objs`` as stored in the source,
        without placing them.

        For each object, the source particles of the types named in
        ``place_from`` (resolved through the file's own type table) that are
        connected to the object are read in file column order. Positions come from
        ``pos``; orientations from ``director`` when the group stores it,
        otherwise from the normalised ``dip`` (a zero dipole raises ``ValueError``).

        Parameters
        ----------
        registered_objs : iterable
            Local objects of one class whose ``who_am_i`` are the connectivity
            values used in the source.
        frame, step, time : optional
            Frame selectors; see the module docstring. All ``None`` = last frame.
            If multiple not None, they must resolve to the same frame.

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            ``(positions_per_obj, orientations_per_obj)``, each ``(N_i, 3)``.

        Raises
        ------
        RuntimeError
            If no source has been declared with ``set_init_src``, or no
            ``place_from`` was given (the objects are placed by the script).
        KeyError
            If a requested source type name is not declared in the file's type
            table (an int source type needs no table).
        ValueError
            If the objects are empty or of several classes, their ``who_am_i``
            set differs from the source's, or orientations must come from
            ``dip`` and a dipole is zero.
        """
        self._require_src('get_pos_ori_from_src')
        self._require_place_from('get_pos_ori_from_src')
        objs, group_name, frame = self._resolve_frame(registered_objs, frame, step, time,
                                                      'get_pos_ori_from_src')
        with h5py.File(self.src_path_h5, "r") as src_file:
            src_types = _src_part_types(src_file, self.place_from, group_name)
            src_data_grp = H5DataSelector(src_file, particle_group=group_name)
            allowed_types = [src_types[name] for name in self.place_from]

            positions_per_obj, ori_per_obj = [], []
            for loc_obj in objs:
                logging.debug("get_pos_ori_from_src: %s %s from source types %s",
                              group_name, loc_obj.who_am_i, self.place_from)
                part_slice = src_data_grp.timestep[frame].select_particles_by_object(
                    object_name=group_name,
                    connectivity_value=loc_obj.who_am_i,
                    predicate=lambda subset: np.isin(subset.type, allowed_types),
                )
                positions_per_obj.append(part_slice.pos)
                try:
                    ori_per_obj.append(part_slice.director)
                except AttributeError:
                    vecs = np.asarray(part_slice.dip, dtype=float)
                    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                    if np.any(norms == 0.0):
                        raise ValueError("dip moment magnitude is 0 and cannot be used to infer "
                                         "particle orientation!") from None
                    ori_per_obj.append(vecs / norms)
        return positions_per_obj, ori_per_obj

    def set_objects_from_src(self, objects, frame=None, step=None, time=None):
        """Place ``objects`` at the positions/orientations stored in the source file.

        The source-file counterpart of ``Simulation.set_objects``; the only way to
        place objects from a file.

        Parameters
        ----------
        objects : list
            Objects of one class, already stored with ``store_objects``. Their
            ``who_am_i`` must be the connectivity values used in the source.
        frame, step, time : optional
            Frame selectors; see the module docstring. All ``None`` = last frame.

        Returns
        -------
        tuple[list, list]
            The ``(positions, orientations)`` that were placed, as returned by
            ``get_pos_ori_from_src``.

        Raises
        ------
        RuntimeError
            If no source has been declared with ``set_init_src``, or no
            ``place_from`` was given.
        """
        self._require_src('set_objects_from_src')
        self._require_place_from('set_objects_from_src')
        positions, orientations = self.get_pos_ori_from_src(objects, frame=frame, step=step, time=time)
        self._owner.place_objects(list(objects), positions, orientations)
        return positions, orientations

    def set_prop_from_src(self, registered_objs, frame=None, step=None, time=None):
        """Copy particle properties from the source onto already-built objects.

        For each object and each ``(src_type, loc_type)`` of ``src_to_loc`` the
        source particles and the local handles are zipped (module docstring);
        then every ``(src_prop, loc_prop)`` listed *for that type pair* is copied
        column by column (both espresso attribute names), so two type pairs can
        restore different properties. A
        pair with an empty property list only takes the one-to-one count check.
        For ``(dip -> director)`` the vectors are normalised; a zero dipole raises
        ``ValueError`` before any value of that pair is assigned.

        Parameters
        ----------
        registered_objs : iterable
            Local objects of one class whose owned particles are updated.
        frame, step, time : optional
            Frame selectors; see the module docstring. All ``None`` = last frame.
            If multiple not None, they must resolve to the same frame.

        Raises
        ------
        RuntimeError
            If no source has been declared with ``set_init_src``.
        KeyError
            If a local type is not registered, or a source type is not declared in
            the file's type table.
        ValueError
            If the objects are empty or of several classes, their ``who_am_i``
            set differs from the source's, a type pair has different source and
            local particle counts for an object, or a dip used for a director is
            zero.
        """
        objs, group_name, frame = self._resolve_frame(registered_objs, frame, step, time,
                                                      'set_prop_from_src')
        with h5py.File(self.src_path_h5, "r") as src_file:
            src_types = _src_part_types(src_file, [src for src, _ in self.src_to_loc], group_name)
            src_data_grp = H5DataSelector(src_file, particle_group=group_name)
            for loc_obj in objs:
                for (src_typ, loc_typ), prop_pairs in self.src_to_loc.items():
                    selection, handles = self._zipped(src_data_grp, src_types, loc_obj, src_typ,
                                                      loc_typ, frame, 'set_prop_from_src')
                    for prop_src, prop_loc in prop_pairs:
                        logging.debug("set_prop_from_src: %s %s type %s->%s prop %s->%s",
                                      group_name, loc_obj.who_am_i, src_typ, loc_typ, prop_src, prop_loc)
                        values = np.asarray(getattr(selection, prop_src))
                        if prop_src == "dip" and prop_loc == "director":
                            vecs = values.astype(float)
                            norms = np.linalg.norm(vecs, axis=1)
                            zero_rows = np.flatnonzero(norms == 0.0)
                            if zero_rows.size:
                                zero_ids = np.asarray(selection.id).reshape(-1)[zero_rows]
                                raise ValueError(f"dip moment magnitude is 0 for particle id="
                                                 f"{zero_ids.tolist()} and cannot be used to infer "
                                                 "particle orientation!")
                            values = vecs / norms[:, None]
                        for local, value in zip(handles, values):
                            setattr(local, prop_loc, value)

    def get_prop_from_src(self, registered_objs, src_type, prop, frame=None, step=None, time=None):
        """One stored property of one source type, per object, without touching the system.

        The escape hatch for everything that is not a one-to-one column copy: it
        hands back the raw values and lets the caller (or the object) decide what
        to do with them -- rescale a dipole, average, feed a model. ``src_to_loc``
        plays no part here; only the declared source and the who_am_i contract do.
        ``Simulation.get_prop_from_src`` declares the source and calls this.

        Parameters
        ----------
        registered_objs : iterable
            Local objects of one class; their ``who_am_i`` must be the
            connectivity values used in the source.
        src_type : str or int
            *Source* type: a name, resolved through the file's own type table, or
            the numeric type stored in the file.
        prop : str
            Espresso attribute name (``'pos'``, ``'dip'``, ...); the dataset read is
            ``particles/<Group>/<element_name(prop)>/value``.
        frame, step, time : optional
            Frame selectors; see the module docstring. All ``None`` = last frame.
            If multiple not None, they must resolve to the same frame.

        Returns
        -------
        list[np.ndarray]
            One ``(N_i, dim)`` array per object, rows in stored (file column)
            order, i.e. ascending particle id -- the same order
            ``set_prop_from_src`` zips.

        Raises
        ------
        RuntimeError
            If no source has been declared with ``set_init_src``.
        KeyError
            If ``src_type`` is a name and is not declared in the file's type table, the
            group stores no property ``prop`` (the message lists what it stores), or
            ``prop`` is an H5MD element name rather than an espresso one.
        ValueError
            If the objects are empty or of several classes, or their ``who_am_i``
            set differs from the source's.
        """
        objs, group_name, frame = self._resolve_frame(registered_objs, frame, step, time,
                                                      'get_prop_from_src')
        with h5py.File(self.src_path_h5, "r") as src_file:
            src_types = _src_part_types(src_file, [src_type], group_name)
            dataset = f"particles/{group_name}/{element_name(prop)}/value"
            if dataset not in src_file:
                stored = sorted(attr_name(element) for element, member in
                                src_file[f"particles/{group_name}"].items()
                                if isinstance(member, h5py.Group) and "value" in member)
                raise KeyError(f"get_prop_from_src: {self.src_path_h5} has no property {prop!r} for "
                               f"{group_name} (no '{dataset}'; the group stores {stored}).")
            src_data_grp = H5DataSelector(src_file, particle_group=group_name)
            values = []
            for loc_obj in objs:
                selection, _ = self._select_src_particles(src_data_grp, loc_obj,
                                                          src_types[src_type], frame)
                values.append(np.asarray(selection.get_property(prop)))
        return values

    def set_bonds_from_src(self, registered_objs, r_cut_override=0.0, frame=None, step=None, time=None):
        """Re-create the bond topology stored in the source file on already-placed objects.

        Reads the link tables under ``/connectivity/<Group>`` and the parameter
        tables under ``/pressomancy/<Group>/bond_params`` (written with
        ``io_dict['bonds'] = True``; see :mod:`pressomancy.io.bonds`), where
        ``<Group>`` is the class name of the registered objects, maps every stored
        particle id onto a local particle id by the zip convention over the type
        pairs of ``src_to_loc`` (their property lists play no part here), and adds
        each stored link to the mapped owner with the mapped partners, so a later
        inscription reproduces the same per-particle link layout and ``n_links``.

        Before anything is attached the mapping is cross-checked: the source
        ``pos`` of every zipped pair (at the selected frame) must equal the
        local particle's ``pos`` within ``_POS_ATOL``=1e-6, which catches an object
        tree built in a different order than the source's (see the module docstring).

        One live bond handle is registered per distinct ``(bond class, parameters)``
        among the restored links, after ``r_cut_override`` is applied; identical
        bonds (a filament's ``FeneBond``s, say) share a handle. Live bond ids are
        unrelated to the file's ``bond_id`` column; ``verify_bond_params`` compares
        by content, so that costs nothing.

        Parameters
        ----------
        registered_objs : list
            Local objects of one class, already placed (e.g. by
            ``set_objects_from_src``). Their ``who_am_i`` must be the connectivity
            values used in the source file.
        r_cut_override : float or None, optional
            Replaces the stored ``r_cut`` of every bond type that has one. The default
            ``0.0`` means the bonds never break and do not enlarge the cell grid.
            ``None`` keeps the stored values verbatim.
        frame, step, time : optional
            Frame selectors used to classify source particles by type and for the
            position cross-check; see the module docstring.
            All ``None`` = last frame.
            If multiple not None, they must resolve to the same frame.

        Returns
        -------
        int
            Number of links added.

        Raises
        ------
        RuntimeError
            If no source has been declared, or ``min_global_cut`` is too small for the
            longest restored pair bond on a multi-rank system
            (:func:`pressomancy.geometry.require_min_global_cut`).
        KeyError
            If a local type is not registered, a source type is not declared in the
            file, or the source has no bond topology for the group.
        ValueError
            If ``src_to_loc`` is empty, the objects are empty or of several classes,
            their ``who_am_i`` set differs from the source's, the type pairs do not
            pair source and local particles one-to-one, a zipped pair's positions
            differ, or a stored bond has a partner outside the mapped particles
            (dangling partner, in either direction).

        Notes
        -----
        All links are collected and validated before the system is touched, so a
        failure leaves no partial topology behind. Stored parameters are float32, so
        ``k``/``r_0`` round-trip to float32 precision only.
        """
        objs, group_name, frame = self._resolve_frame(registered_objs, frame, step, time,
                                                      'set_bonds_from_src')
        self._require_src_to_loc('set_bonds_from_src')
        type_pairs = list(self.src_to_loc.keys())
        one_to_one_error_msg = (
                    f"set_bonds_from_src: the type pairs {type_pairs} do not pair source and "
                    f"local particles one-to-one over {group_name} objects "
                    f"{[o.who_am_i for o in objs]}; a particle is mapped twice."
                    )

        with h5py.File(self.src_path_h5, "r") as src_file:
            bonds_path = f"connectivity/{group_name}/bonds"
            if bonds_path not in src_file:
                raise KeyError(f"source file {self.src_path_h5} has no bond topology at "
                               f"'{bonds_path}' (it was inscribed with io_dict['bonds'] disabled).")
            src_types = _src_part_types(src_file, [src for src, _ in type_pairs], group_name)
            src_data_grp = H5DataSelector(src_file, particle_group=group_name)

            # --- source particle id -> local particle id, plus the zipped positions ------
            id_map = {}
            zipped = []  # (object, src_ids, local_ids, src_pos) per (object, type pair)
            for loc_obj in objs:
                for src_typ, loc_typ in type_pairs:
                    selection, handles = self._zipped(src_data_grp, src_types, loc_obj, src_typ,
                                                      loc_typ, frame, 'set_bonds_from_src')
                    if not handles:
                        continue
                    src_ids = np.asarray(selection.id).reshape(-1).tolist()
                    loc_ids = [int(handle.id) for handle in handles]
                    for src_id, loc_id in zip(src_ids, loc_ids):
                        if src_id in id_map:
                            raise ValueError(one_to_one_error_msg)
                        id_map[src_id] = loc_id
                    zipped.append((loc_obj, src_ids, loc_ids, np.asarray(selection.pos, dtype=float)))
            if len(set(id_map.values())) != len(id_map):
                raise ValueError(one_to_one_error_msg)

            # --- position cross-check: one bulk read of the local positions --------------
            sorted_ids = sorted(id_map.values())
            local_pos = dict(zip(sorted_ids, np.asarray(self.sys.part.by_ids(sorted_ids).pos, dtype=float)))
            for loc_obj, src_ids, loc_ids, src_pos in zipped:
                for src_id, loc_id, pos in zip(src_ids, loc_ids, src_pos):
                    if not np.allclose(pos, local_pos[loc_id], atol=_POS_ATOL, rtol=0.0):
                        raise ValueError(
                            f"set_bonds_from_src: {group_name}[{loc_obj.who_am_i}]: source particle "
                            f"{src_id} at {pos.tolist()} is zipped with local particle {loc_id} at "
                            f"{local_pos[loc_id].tolist()} (frame {frame}); the positions differ by "
                            f"more than {_POS_ATOL}. The local object tree must be built in the same "
                            "order as the source's, and placed from the same frame.")

            # --- collect and validate the stored links (nothing touched yet) -------------
            bond_params = read_bond_params(src_file, group_name)
            pending = []  # (local owner, stored bond_id, local partners)
            for src_owner, src_partners, bond_id in read_bonds(src_file, group_name):
                dangling = [p for p in src_partners if p not in id_map]
                if src_owner not in id_map:
                    if len(dangling) < len(src_partners):
                        raise ValueError(
                            f"dangling bond: source particle {src_owner} (not mapped) owns a bond "
                            f"(bond_id {bond_id}) to mapped partner(s) "
                            f"{[p for p in src_partners if p in id_map]}; it would be silently "
                            "dropped. Include its object/type in the mapping.")
                    continue  # a bond of objects that are not being restored
                if dangling:
                    raise ValueError(
                        f"dangling bond: source particle {src_owner} has a bond (bond_id {bond_id}) "
                        f"to partner(s) {dangling} that are not among the mapped source particles "
                        f"of {group_name} objects {[o.who_am_i for o in objs]} with "
                        f"type pairs {type_pairs}.")
                pending.append((id_map[src_owner], bond_id, tuple(id_map[p] for p in src_partners)))

        # --- register one handle per distinct (bond class, parameters) --------------------
        pair_r_0 = [bond_params[bond_id][1]["r_0"]
                    for _, bond_id, partners in pending
                    if len(partners) == 1 and "r_0" in bond_params[bond_id][1]]
        if pair_r_0:
            require_min_global_cut(self.sys, max(pair_r_0))
        handle_for_params = {}
        bond_for_id = {}
        for _, bond_id, _ in pending:
            if bond_id in bond_for_id:
                continue
            bond_cls, kw = bond_params[bond_id]
            if r_cut_override is not None and "r_cut" in kw:
                kw["r_cut"] = float(r_cut_override)
            key = (bond_cls, tuple(sorted(kw.items())))
            if key not in handle_for_params:
                handle = bond_cls(**kw)
                self.sys.bonded_inter.add(handle)
                handle_for_params[key] = handle
            bond_for_id[bond_id] = handle_for_params[key]

        # --- attach ------------------------------------------------------------------------
        for local_owner, bond_id, local_partners in pending:
            self.sys.part.by_id(local_owner).add_bond((bond_for_id[bond_id], *local_partners))
        logging.info("set_bonds_from_src: %d links on %s restored with %d registered bond "
                     "parameter set(s).", len(pending), group_name, len(handle_for_params))
        return len(pending)

    def load_from_src(self, objects, path, src_to_loc, bonds=False, place_from=None,
                      r_cut_override=0.0, frame=None, step=None, time=None):
        """Seed ``objects`` from the file at ``path`` in one call.

        The one call a script needs to build a system from a file: declares the
        source (``set_init_src``), places the objects from ``place_from`` when one
        is given (``set_objects_from_src``), copies properties when ``src_to_loc``
        is non-empty (``set_prop_from_src``) and restores the stored bonds when
        ``bonds`` is true (``set_bonds_from_src``), all at the same frame.

        Give ``place_from`` when the file should place your objects, which requires
        one stored particle of those types per monomer; leave it out when you built
        the tree yourself (compound objects, running systems).

        Parameters
        ----------
        objects : list
            Stored objects of one class whose ``who_am_i`` match the source. They
            are already built and placed unless ``place_from`` is given.
        path : str
            The source HDF5 file.
        src_to_loc : dict or None
            ``{type pair(s): [(src_prop, loc_prop), ...]}``, the mapping of the
            module docstring (both halves espresso attribute names, ``('pos', 'pos')``).
            ``None``/``{}`` copies nothing (only meaningful
            together with ``place_from``); it must be non-empty when ``bonds`` is
            true.
        bonds : bool, optional
            Restore the stored bond topology. Default ``False``.
        place_from : iterable of str or int, optional
            Source types (names or ints) supplying positions/orientations. ``None`` (the
            default) skips placement: the objects must already be placed.
        r_cut_override : float or None, optional
            Passed to ``set_bonds_from_src``; overwrides the stored r_cut for
            bonds with that parameter.
        frame, step, time : optional
            Frame selectors shared by every step; see the module docstring.
            All ``None`` = last frame.
            If multiple not None, they must resolve to the same frame.

        Returns
        -------
        int
            Bond links added; ``0`` when ``bonds`` is false.

        Raises
        ------
        ValueError
            If ``src_to_loc`` is malformed, or ``bonds`` is true and it is empty
            (both before anything is placed), plus everything the chained steps
            raise. Unplaced objects without ``place_from`` fail loudly in the
            one-to-one count check of ``set_prop_from_src`` (0 local particles).
        """
        self.set_init_src(path, src_to_loc=src_to_loc, place_from=place_from)
        if self.place_from is None and not self.src_to_loc and not bonds:
            raise ValueError("load_from_src: nothing to do — give place_from to place the objects, "
                             "a non-empty src_to_loc to copy state, and/or bonds=True.")
        if bonds:
            self._require_src_to_loc('load_from_src')
        selectors = dict(frame=frame, step=step, time=time)
        if self.place_from is not None:
            self.set_objects_from_src(objects, **selectors)
        if self.src_to_loc:
            self.set_prop_from_src(objects, **selectors)
        if not bonds:
            return 0
        return self.set_bonds_from_src(objects, r_cut_override=r_cut_override, **selectors)

    def restart_from_checkpoint(self, objects, path, src_to_loc=None, bonds=False, place_from=None,
                                r_cut_override=0.0):
        """Restore ``objects`` from a checkpoint written by ``H5Writer.write_checkpoint``.

        ``load_from_src`` with the checkpoint's own recipe: every type stored in
        the file is paired with the local type of the same name and receives the
        compiled-in entries of ``CHECKPOINT_PROPERTIES`` in that order (``quat``
        before ``omega_lab``/``torque_lab``, whose setters convert lab -> body
        through the *current* quat); then ``sys.time`` is set to the stored time
        and every recorded thermostat Philox counter is written back.

        Exact continuation: the checkpoint carries ``f``/``torque_lab``, so the
        first call after a restart must be ``integrator.run(n, reuse_forces=True)``
        (a force recalculation would draw fresh noise from the restored counter).
        The continuation then matches the uninterrupted run to roundoff, not
        bitwise. The LB fluid state is not covered (``write_checkpoint`` refuses
        an active LB thermostat).

        Parameters
        ----------
        objects : list
            Stored objects of one class, built in the same order as the ones that
            wrote the checkpoint (who_am_i contract); unplaced when ``place_from``
            is given, placed by the script otherwise.
        path : str
            The checkpoint file.
        src_to_loc : dict, optional
            Extra property pairs on top of the fixed list, in the grammar of the
            module docstring, e.g. ``{('real', 'real'): [('fix', 'fix')]}``.
        bonds : bool, optional
            Restore the stored bond topology (a checkpoint always stores it).
        place_from : iterable of str, optional
            Source types the file places the objects from (one stored particle
            per monomer, as for ``load_from_src``); ``None`` (the default) means
            the script placed them (compound objects, virtual sites added by the
            object after placement).
        r_cut_override : float or None, optional
            Passed to ``set_bonds_from_src``.

        Returns
        -------
        int
            The stored step, what ``inscribe_*(..., rewind_to_step=step)`` needs.

        Raises
        ------
        ValueError
            If ``path`` has no ``/pressomancy/checkpoint`` group (a trajectory:
            use ``load_from_src``) or stores more than one frame, plus everything
            ``load_from_src`` raises.
        RuntimeError
            If the file lacks a fixed element this build restores (written by a
            build with fewer features), stores a numeric type its type table does
            not name, records the counter of a thermostat this build lacks, or of
            one that is not active: set the thermostat with its seed *before*
            calling (the counter is overridden on the active thermostat).
        """
        what = 'restart_from_checkpoint'
        self.set_init_src(path)
        objs, group_name, _ = self._resolve_frame(objects, None, None, None, what)
        fixed = [attr for attr, _dim, _dtype in checkpoint_properties([]) if attr not in ('id', 'type')]
        with h5py.File(path, "r") as src_file:
            if "pressomancy/checkpoint" not in src_file:
                raise ValueError(f"{what}: {path} is not a checkpoint file (no /pressomancy/checkpoint); "
                                 "use load_from_src.")
            attrs = dict(src_file["pressomancy/checkpoint"].attrs)
            n_frames = len(stored_steps(src_file, group_name))
            if n_frames != 1:
                raise ValueError(f"{what}: {path} stores {n_frames} frames of {group_name}; a checkpoint "
                                 "holds exactly one.")
            particles = src_file[f"particles/{group_name}"]
            missing = [attr for attr in fixed if element_name(attr) not in particles]
            if missing:
                raise RuntimeError(f"{what}: {path} stores no {missing} for {group_name}; it was "
                                   "written by an espresso build with fewer features than this one.")
            stored = {int(x) for x in np.unique(particles[element_name('type')]['value'][0])}
            table = {str(name): int(value) for name, value in src_file[_TYPE_TABLE].attrs.items()}
            unnamed = stored - set(table.values())
            if unnamed:
                raise RuntimeError(f"{what}: {path} stores numeric types {sorted(unnamed)} for "
                                   f"{group_name} that {_TYPE_TABLE} does not name.")
            names = [name for name, value in sorted(table.items(), key=lambda kv: kv[1]) if value in stored]

        mapping = _normalised_src_to_loc(src_to_loc)
        for name in names:
            pairs = [(attr, attr) for attr in fixed] + mapping.get((name, name), [])
            seen = set()
            mapping[(name, name)] = [pair for pair in pairs if not (pair in seen or seen.add(pair))]

        suffix = '_philox_counter'
        counters = []
        for name, counter in attrs.items():
            if not name.endswith(suffix):
                continue
            kind = name[:-len(suffix)]
            thermostat = getattr(self.sys.thermostat, kind, None)   # absent when not compiled in
            if thermostat is None:
                raise RuntimeError(f"{what}: {path} records the {kind} thermostat counter, but this "
                                   f"espresso build is compiled without the {kind} thermostat.")
            if not thermostat.is_active:
                raise RuntimeError(f"{what}: set the {kind} thermostat with its seed before "
                                   f"restart_from_checkpoint; {path} records its counter.")
            counters.append((thermostat, int(counter)))
        self.load_from_src(objs, path, mapping, bonds=bonds, place_from=place_from,
                           r_cut_override=r_cut_override)
        self.sys.time = float(attrs['time'])
        for thermostat, counter in counters:
            thermostat.call_method("override_philox_counter", counter=counter)
        return int(attrs['step'])
