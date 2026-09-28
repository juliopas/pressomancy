'''
Read-side API for pressomancy's HDF5 (H5MD-style) simulation output.

``H5DataSelector`` provides chainable, lazy timestep/particle slicing over a
stored simulation (``.timestep``/``.particles``/``.bonds``/``.get_property``/
``.select_particles_by_object``/``.select_particles_by_type``/
``.get_connectivity_values``); ``H5ObservableSelector`` does the same for
arbitrary recorded observables. ``TimestepAccessor``/``ParticleAccessor`` back
the chained slicing. ``read_h5_selection`` is the low-level fancy-indexing
engine every dataset read goes through, and ``stored_steps``/``frame_of_step``/
``frame_of_time``/``has_step`` map between frame indices and stored step/time
values.

Every entry that opens a pressomancy file for its particle data goes through
``_particle_group``, which checks ``parameters/pressomancy/layout`` == ``LAYOUT``
and raises ``RuntimeError`` otherwise: there is one layout in code.

Names: the file uses H5MD element names (``position``, ``species``, ``force``,
``image``); pressomancy uses espresso attribute names (``pos``, ``type``, ``f``,
``image_box``) everywhere else -- API arguments, ``H5DataSelector`` attributes,
``src_to_loc``, error messages. The translation is ``H5MD_NAMES``, applied only
when a dataset path is built (``element_name``); its inverse ``attr_name`` is
applied only where the file's own element names are enumerated.
'''
import math
from collections import namedtuple

import h5py
import numpy as np

_DEFAULT_MAX_OVERREAD_FACTOR = 4.0

#: Relative tolerance for matching a requested ``time`` to a stored frame time.
_TIME_RTOL = 1e-6

#: Value of the ``parameters/pressomancy/layout`` attr this code reads and writes.
LAYOUT = "h5md-1"

#: espresso attribute -> H5MD element name, for the elements H5MD names. Every
#: other attribute (``id``, ``director``, ``dip``, ...) is stored under its own name.
H5MD_NAMES = {'pos': 'position', 'v': 'velocity', 'type': 'species', 'f': 'force', 'image_box': 'image'}
_ATTR_NAMES = {element: attr for attr, element in H5MD_NAMES.items()}


def element_name(attr):
    """The ``particles/<Group>/<element>`` name of espresso attribute ``attr``.

    The one place the espresso -> H5MD translation happens: every particle
    dataset path in ``pressomancy.io`` is ``particles/<Group>/<element_name(attr)>/...``.
    Names outside ``H5MD_NAMES`` pass through unchanged. Passing an H5MD name
    itself (``'position'``) is a ``KeyError``: the API speaks espresso names only,
    so ``data.position`` or ``('position', 'pos')`` fail loudly instead of working
    by accident.
    """
    if attr not in H5MD_NAMES and attr in _ATTR_NAMES:
        raise KeyError(f"{attr!r} is an H5MD element name; pressomancy uses espresso attribute "
                       f"names (here {_ATTR_NAMES[attr]!r}). H5MD names exist only as dataset "
                       "paths in the file.")
    return H5MD_NAMES.get(attr, attr)


def attr_name(element):
    """The espresso attribute name of a stored element: the inverse of ``element_name``.

    Used only where the file's own element names are enumerated (error messages
    listing what a group stores); an unknown custom element passes through unchanged.
    """
    return _ATTR_NAMES.get(element, element)


#: Element whose ``step``/``time`` the readers address. Every element of a group
#: hard-links the same pair, so this is a choice of path, not of data.
_TIMELINE = element_name('pos')

#: Partner count -> (``/connectivity/<Group>/<table>``, ``/pressomancy/<Group>/bond_params/<ids>``).
#: A link with k partners is a row (owner, p0..pk-1) of the k-th table; the id
#: array is row-aligned with it. 1..3 covers every espresso bond.
LINK_TABLES = {1: ("bonds", "bond_id"), 2: ("angles", "angle_id"), 3: ("dihedrals", "dihedral_id")}


def _require_layout(h5_file):
    """Raise ``RuntimeError`` unless ``h5_file`` (any group of it) carries the current layout attr."""
    root = h5_file.file
    found = root["parameters/pressomancy"].attrs.get("layout") if "parameters/pressomancy" in root else None
    if found is not None and not isinstance(found, str):
        found = found.decode() if isinstance(found, bytes) else str(found)
    if found != LAYOUT:
        raise RuntimeError(
            f"{root.filename}: old pressomancy HDF5 layout (parameters/pressomancy/layout is "
            f"{'absent' if found is None else repr(found)}, this code reads {LAYOUT!r})")


# --------------------------------------------------------------------------- #
# index planner: NumPy outer-product semantics on h5py datasets
# --------------------------------------------------------------------------- #

def _normalize_axis_index(index, length):
    """
    Normalize one axis index into ``(kind, value)``.

    ``kind`` is one of:
        - ``'int'``   : single integer index; the axis is dropped, as in NumPy
        - ``'slice'`` : lazy slice, passed straight through to h5py
        - ``'list'``  : 1-D array of non-negative positions

    Integer indices are resolved to non-negative positions, slices are kept lazy
    unless h5py cannot express them (negative step), and every
    list/tuple/range/array/boolean-mask index becomes a 1-D array of
    non-negative positions.

    ``Ellipsis`` and ``None`` (``np.newaxis``) are rejected explicitly: this
    helper maps one index onto one leading axis, so there is no well-defined
    axis for either of them.

    Args:
        index: The index for a single axis.
        length (int): The length of that axis in the dataset.

    Returns:
        tuple: ``(kind, value)`` as described above.

    Raises:
        TypeError: If the index type or dtype is unsupported.
        ValueError: If an integer index array has more than one dimension.
        IndexError: If a position is out of bounds, or a boolean mask has the
            wrong shape.
    """
    if index is Ellipsis:
        raise TypeError(
            "Ellipsis (...) is not supported for an HDF5 selection; pass one "
            "index per leading axis explicitly. Trailing axes are read in full."
        )
    if index is None:
        raise TypeError(
            "None (np.newaxis) is not supported for an HDF5 selection; insert "
            "new axes into the returned array instead."
        )
    if isinstance(index, (bool, np.bool)):
        raise TypeError("Boolean scalars are not valid indices for an HDF5 selection.")

    if isinstance(index, (int, np.integer)):
        idx = int(index)
        if idx < 0:
            idx += length
        if not 0 <= idx < length:
            raise IndexError(f"Index {index} is out of bounds for axis of length {length}.")
        return "int", idx

    if isinstance(index, slice):
        if index.step is not None and index.step < 0:
            # h5py only supports positive steps, so materialize the positions.
            return "list", np.arange(*index.indices(length), dtype=np.intp)
        return "slice", index

    if isinstance(index, (list, tuple, range, np.ndarray)):
        positions = np.asarray(index)
        if positions.dtype == bool:
            if positions.shape != (length,):
                raise IndexError(
                    f"Boolean mask of shape {positions.shape} does not match "
                    f"axis of length {length}."
                )
            positions = np.flatnonzero(positions)
        elif positions.ndim > 1:
            raise ValueError(
                f"Multi-dimensional integer index arrays are not supported for an "
                f"HDF5 selection (got shape {positions.shape}); pass a 1-D list/array "
                f"of positions."
            )
        elif positions.size == 0:
            positions = positions.astype(np.intp)

        if not np.issubdtype(positions.dtype, np.integer):
            raise TypeError(
                f"Unsupported index dtype for an HDF5 selection: {positions.dtype}."
            )

        positions = positions.astype(np.intp, copy=False).ravel()
        if positions.size:
            positions = np.where(positions < 0, positions + length, positions)

            if positions.min() < 0 or positions.max() >= length:
                raise IndexError(f"Index list is out of bounds for axis of length {length}.")

        return "list", positions

    raise TypeError(f"Unsupported index type for an HDF5 selection: {type(index)}.")


def _axis_result_length(kind, value, length):
    """Return the result length for one normalized axis, or ``None`` for a dropped int axis."""
    if kind == "int":
        return None
    if kind == "slice":
        return len(range(*value.indices(length)))
    return int(value.size)


def _result_axis_index(normalized, axis):
    """Return the position of ``axis`` in the result, ignoring dropped int axes."""
    return sum(kind != "int" for kind, _ in normalized[:axis])


def _direct_axis_plan(kind, value):
    """
    Return ``(h5_index, post_index)`` for reading one axis directly through h5py.

    h5py requires fancy indices to be strictly increasing, so duplicate or
    unsorted positions are read through their unique sorted positions and
    restored in memory afterwards.
    """
    if kind != "list":
        return value, None

    unique_positions, inverse = np.unique(value, return_inverse=True)
    inverse = np.asarray(inverse).ravel()

    if unique_positions.size == value.size and np.array_equal(unique_positions, value):
        return unique_positions.tolist(), None

    return unique_positions.tolist(), inverse


def _bounding_axis_plan(value):
    """
    Replace a list of positions with its contiguous bounding span.

    Returns ``(h5_index, post_index)`` so the requested positions can be
    extracted from the over-read span afterwards.
    """
    low = int(value.min())
    high = int(value.max()) + 1
    return slice(low, high), np.asarray(value) - low


def _apply_post_indices(data, normalized, post_indices):
    """
    Apply the in-memory part of a read plan, skipping axes dropped by int indices.

    When more than one axis needs reordering, the gathers are combined into a
    single fancy index so the data is copied once instead of once per axis.
    """
    kept = [
        post_index
        for (kind, _), post_index in zip(normalized, post_indices)
        if kind != "int"
    ]
    # Axes beyond ``indices`` were read in full and never need reordering.
    kept.extend([None] * (data.ndim - len(kept)))

    active_axes = [axis for axis, post_index in enumerate(kept) if post_index is not None]
    if not active_axes:
        return data
    elif len(active_axes) == 1:
        axis = active_axes[0]
        return np.take(data, kept[axis], axis=axis)
    # np.ix_ builds the correct index tuples to use in data[grid]
    # np.arange(data.shape[axis]) keep the whole axes
    grids = np.ix_(
        *[
            kept[axis] if kept[axis] is not None else np.arange(data.shape[axis])
            for axis in range(data.ndim)
        ]
    )
    return data[grids]


def _axis_chunk_size(dataset, axis):
    """Return the chunk length along ``axis``, or 1 for contiguous datasets."""
    chunks = getattr(dataset, "chunks", None)
    if not chunks:
        return 1
    return max(int(chunks[axis]), 1)


def _span_read_cost(positions, chunk):
    """Elements touched when reading the contiguous bounding span of ``positions``."""
    low = int(positions.min())
    high = int(positions.max()) + 1
    if chunk <= 1:
        return high - low
    return math.ceil(high / chunk) * chunk - (low // chunk) * chunk


def _fancy_read_cost(positions, chunk):
    """Elements touched when reading ``positions`` as a fancy selection."""
    unique_positions = np.unique(positions)
    if chunk <= 1:
        return int(unique_positions.size)
    return int(np.unique(unique_positions // chunk).size) * chunk


def _estimate_bounding_costs(arr_a, arr_b, chunk_a, chunk_b):
    """
    Estimate the cost of the two possible 'fancy + bounding span' strategies.

    Costs are measured in elements actually pulled off disk, rounded out to
    chunk boundaries where the dataset is chunked, because a chunk is the
    smallest unit HDF5 can read. The non-fancy axes contribute the same
    constant factor to every strategy, so they are omitted.

    Returns:
        tuple: ``(ideal_cost, cost_fancy_a, cost_fancy_b)``, where ``ideal_cost``
        is what a perfectly targeted read would touch.
    """
    span_a = _span_read_cost(arr_a, chunk_a)
    span_b = _span_read_cost(arr_b, chunk_b)

    fancy_a = _fancy_read_cost(arr_a, chunk_a)
    fancy_b = _fancy_read_cost(arr_b, chunk_b)

    ideal_cost = fancy_a * fancy_b
    # Fancy selection on A, contiguous span on B.
    cost_fancy_a = fancy_a * span_b
    # Contiguous span on A, fancy selection on B.
    cost_fancy_b = span_a * fancy_b

    return ideal_cost, cost_fancy_a, cost_fancy_b


def _read_two_fancy_axes_with_bounding_span(dataset,
    normalized, axis_a, axis_b, arr_a, arr_b,
    keep_fancy_on_a):
    """
    Read two fancy axes using one HDF5 read plus in-memory extraction.

    One axis remains a fancy selection; the other becomes its contiguous
    bounding span. The caller decides which, so the cost model is evaluated
    exactly once.
    """
    plans = [_direct_axis_plan(kind, value) for kind, value in normalized]
    if keep_fancy_on_a:
        plans[axis_b] = _bounding_axis_plan(arr_b)
    else:
        plans[axis_a] = _bounding_axis_plan(arr_a)
    data = dataset[tuple(plan[0] for plan in plans)]
    return _apply_post_indices(data, normalized, [plan[1] for plan in plans])


def _read_two_fancy_axes_in_chunks(dataset, normalized, axis_loop, axis_keep):
    """
    Read two fancy axes through one HDF5 read per position on ``axis_loop``.

    ``axis_loop`` is pinned to a single integer position per read, leaving one
    fancy axis for h5py to handle; the results are stacked back into the axis
    position the loop axis occupies in the result. This works for any pair of
    axes and any number of indices, so it is always available as a fallback and
    no selection can degrade into reading the whole dataset.
    """
    plans = [_direct_axis_plan(kind, value) for kind, value in normalized]

    h5_indices = [plan[0] for plan in plans]
    post_indices = [plan[1] for plan in plans]
    # The looped axis is read one position at a time and needs no reordering.
    post_indices[axis_loop] = None
    # Mark the looped axis as dropped so post-indexing skips it, exactly as it
    # would for a real integer index.
    chunk_normalized = list(normalized)
    chunk_normalized[axis_loop] = ("int", 0)

    stack_axis = _result_axis_index(normalized, axis_loop)

    chunks = []
    for position in normalized[axis_loop][1]:
        h5_indices[axis_loop] = int(position)
        data = dataset[tuple(h5_indices)]
        chunks.append(_apply_post_indices(data, chunk_normalized, post_indices))

    del axis_keep  # Named for readability at the call site.

    return np.stack(chunks, axis=stack_axis)


def positions_for_axis(index, length):
    """Materialize one axis index into explicit non-negative positions.

    ``read_h5_selection`` keeps slices lazy because h5py can consume them
    directly. Callers that must index something *derived* from an axis — CSR row
    ranges, say — need the positions themselves.

    Returns:
        tuple: ``(positions, drops_axis)``. ``drops_axis`` is True for a scalar
            integer index, matching NumPy's axis-dropping behaviour.
    """
    kind, value = _normalize_axis_index(index, length)
    if kind == "int":
        return np.array([value], dtype=np.intp), True
    if kind == "slice":
        return np.arange(*value.indices(length), dtype=np.intp), False
    return value, False


def read_h5_selection(dataset, *indices, max_overread_factor=None):
    """
    Read ``dataset[indices]`` with NumPy outer-product semantics, reading only
    what is selected.

    ``h5py`` accepts at most one fancy (list) index per read and requires it to
    be strictly increasing, while NumPy broadcasts two lists *pairwise*
    (returning the diagonal, or raising for unequal lengths). This helper
    bridges both: every index is composed as an independent axis selection, and
    the data is fetched lazily from the file instead of materializing the whole
    dataset first.

    With two fancy axes the read strategy is chosen by cost. Either the smaller
    of the two 'fancy + bounding span' plans is used as a single read, or, when
    that would over-read by more than ``max_overread_factor``, one read is
    issued per position along the shorter fancy axis. Costs are chunk-aware, so
    the factor is a bound on wasted I/O rather than on element count.

    Args:
        dataset (h5py.Dataset): The dataset to read from.
        *indices: One index per leading axis; each may be an int, a slice, a
            list/tuple/range/array of positions, or a boolean mask. Axes beyond
            ``indices`` are read in full. ``Ellipsis`` and ``None`` are not
            accepted.
        max_overread_factor (float, optional): How much redundant data a single
            read may pull in, as a multiple of an ideal read. Defaults to
            ``_DEFAULT_MAX_OVERREAD_FACTOR``. Raise it to favour one large read
            over many small ones (useful on high-latency storage); lower it to
            favour many targeted reads.

    Returns:
        ndarray: The selected data, with integer indices dropping their axis, as
        in NumPy.

    Raises:
        IndexError: If more indices than dimensions are given.
        NotImplementedError: If more than two axes use list-like indices.
        ValueError: If ``max_overread_factor`` is not positive.
    """
    if max_overread_factor is None:
        max_overread_factor = _DEFAULT_MAX_OVERREAD_FACTOR
    if not max_overread_factor > 0:
        raise ValueError(f"max_overread_factor must be positive, got {max_overread_factor}.")

    shape = dataset.shape
    if len(indices) > len(shape):
        raise IndexError(f"Got {len(indices)} indices for a dataset with {len(shape)} dimensions.")

    normalized = [
        _normalize_axis_index(index, shape[axis])
        for axis, index in enumerate(indices)
    ]

    if any(kind == "list" and value.size == 0 for kind, value in normalized):
        result_shape = [
            result_length
            for axis, (kind, value) in enumerate(normalized)
            if (result_length := _axis_result_length(kind, value, shape[axis]))
            is not None
        ]
        result_shape.extend(shape[len(indices):])
        return np.empty(tuple(result_shape), dtype=dataset.dtype)

    fancy_axes = [axis for axis, (kind, _) in enumerate(normalized) if kind == "list"]
    # Zero or one fancy axis can be handled directly by h5py.
    if len(fancy_axes) <= 1:
        plans = [_direct_axis_plan(kind, value) for kind, value in normalized]
        data = dataset[tuple(plan[0] for plan in plans)]
        return _apply_post_indices(data, normalized, [plan[1] for plan in plans])
    if len(fancy_axes) > 2:
        raise NotImplementedError("Selections with more than two list-like axis indices are not supported.")
    else:
        axis_a, axis_b = fancy_axes
        arr_a = normalized[axis_a][1]
        arr_b = normalized[axis_b][1]
        ideal_cost, cost_fancy_a, cost_fancy_b = _estimate_bounding_costs(
            arr_a,
            arr_b,
            _axis_chunk_size(dataset, axis_a),
            _axis_chunk_size(dataset, axis_b),
        )
    # Prefer a single HDF5 read whenever the over-reading is acceptable.
    if min(cost_fancy_a, cost_fancy_b) <= max_overread_factor * ideal_cost:
        return _read_two_fancy_axes_with_bounding_span(
            dataset,
            normalized,
            axis_a,
            axis_b,
            arr_a,
            arr_b,
            keep_fancy_on_a=cost_fancy_a <= cost_fancy_b,
        )
    # Read by chunks, looping over the smallest axis array
    if arr_b.size <= arr_a.size:
        return _read_two_fancy_axes_in_chunks(dataset, normalized, axis_b, axis_a)
    return _read_two_fancy_axes_in_chunks(dataset, normalized, axis_a, axis_b)


# --------------------------------------------------------------------------- #
# bonds of a selection
# --------------------------------------------------------------------------- #

BondLink = namedtuple("BondLink", "bond_id bond_type params partners")


class BondSelection:
    """The bonds owned by the particles of an :class:`H5DataSelector` view.

    Topology is static, so this ignores the timestep slice: ``data.bonds`` and
    ``data.timestep[5].bonds`` are the same object. The particle slice is
    honoured: a link is kept when its owner (column 0 of the
    ``connectivity/<Group>/{bonds,angles,dihedrals}`` tables, a column index that
    ``bonds.read_link_tables`` resolves to a particle id) is one of the selected
    particle ids (``id/value`` at frame 0; ids are static).

    A bond is stored once, on its owner. If the particle slice excludes an
    owner but includes its partner, that bond is absent. Links are listed
    table by table (pairs, then angles, then dihedrals), rows in stored order.

    Attributes:
        particle_ids (ndarray): ``(P,)`` selected particle ids, in view order.
        owner (ndarray): ``(L,)`` particle id owning each link.
        bond_id (ndarray): ``(L,)`` registered bond id of each link.
        n_partners (ndarray): ``(L,)`` valid partner count per link.
        partners (ndarray): ``(L, max_partners)`` partner ids, ``-1`` padded.
    """

    def __init__(self, params_grp, particle_ids, owner, bond_id, n_partners, partners):
        self._grp = params_grp
        self.particle_ids = particle_ids
        self.owner = owner
        self.bond_id = bond_id
        self.n_partners = n_partners
        self.partners = partners
        self._params = None

    def __len__(self):
        return int(self.owner.size)

    @property
    def params(self):
        """``{bond_id: (type_name, {param: value})}``, read once and cached.

        Only the registered bonds are described here — a handful of rows — so
        this reads every class table of ``pressomancy/<Group>/bond_params``
        (the compound datasets; the ``*_id`` arrays are skipped) regardless of
        the slice.
        """
        if self._params is None:
            table = {}
            for type_name, dset in self._grp.items():
                if type_name in {id_name for _, id_name in LINK_TABLES.values()}:
                    continue
                rows = dset[...]
                names = [n for n in rows.dtype.names if n != "bond_id"]
                for row in rows:
                    kw = {n: (row[n].tolist() if getattr(row[n], "shape", ())
                              else row[n].item()) for n in names}
                    table[int(row["bond_id"])] = (type_name, kw)
            self._params = table
        return self._params

    def by_particle(self):
        """``{owner_id: [BondLink, ...]}`` for the particles of the view.

        Keys are the selected particle ids **in view order**, and every selected
        particle gets one — a particle owning no bond maps to an empty list, so
        the dict lines up column-for-column with ``get_property`` output. Values
        carry the resolved bond type and parameters, and only the valid partners
        (padding dropped), which is what a caller wanting per-particle rows needs
        instead of the raw CSR arrays.
        """
        table = self.params
        out = {int(pid): [] for pid in self.particle_ids}
        for i in range(len(self)):
            bond_id = int(self.bond_id[i])
            if bond_id not in table:
                raise KeyError(
                    f"link {i} carries bond id {bond_id}, which has no row in any class "
                    f"table of {self._grp.name}.")
            bond_type, params = table[bond_id]
            partners = tuple(int(p) for p in self.partners[i, :self.n_partners[i]])
            out[int(self.owner[i])].append(
                BondLink(bond_id, bond_type, params, partners))
        return out


# --------------------------------------------------------------------------- #
# selectors
# --------------------------------------------------------------------------- #

def _particle_group(h5_file, group):
    """``particles/<group>`` of an open file; ``KeyError`` naming the groups present otherwise.

    The one place the layout attr is checked (``RuntimeError`` for an old file):
    ``H5DataSelector``, the frame lookups and ``H5Init`` all come through here.
    """
    if not isinstance(h5_file, h5py.Group):
        raise TypeError(
            "h5_file must be an open h5py.File (or a Group holding 'particles'), "
            f"not {type(h5_file).__name__}; open the file yourself and pass it in.")
    _require_layout(h5_file)
    if "particles" not in h5_file:
        raise KeyError(f"{h5_file.file.filename}: '{h5_file.name}' holds no "
                       "'particles' group.")
    names = list(h5_file["particles"])
    if group not in names:
        raise KeyError(f"'{group}' is not a particle group of "
                       f"{h5_file.file.filename} (it holds {names}).")
    return h5_file["particles"][group]


def _particle_dims(h5_file, particle_group):
    """``(n_frames, n_particles)`` shared by every ``<element>/value`` dataset of ``particles/<group>``.

    Scalar elements are ``[F, N]``, vectors ``[F, N, 3]``; only the two leading
    dims are compared.

    Raises:
        ValueError: If the group is absent (naming the groups present), holds no
            ``value`` dataset, or two elements disagree on the leading dims.
    """
    try:
        group = _particle_group(h5_file, particle_group)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    dims = None
    for element, member in group.items():
        if not (isinstance(member, h5py.Group) and "value" in member):
            continue
        shape = member["value"].shape
        if len(shape) < 2:
            continue
        current = tuple(int(x) for x in shape[:2])
        if dims is None:
            dims = current
        elif dims != current:
            raise ValueError(
                f"Inconsistent shape for property '{attr_name(element)}' of particles/{particle_group}: "
                f"expected (timesteps, particles)={dims}, got {current}")
    if dims is None:
        raise ValueError(f"particles/{particle_group} holds no '<prop>/value' dataset.")
    return dims


def _observable_dims(h5_file, observable_name):
    """``(n_frames,)`` shared by the ``step``/``time``/``value`` streams of ``observables/<name>``."""
    if "observables" not in h5_file or observable_name not in h5_file["observables"]:
        present = list(h5_file["observables"]) if "observables" in h5_file else []
        raise ValueError(f"Observable '{observable_name}' not found in "
                         f"{h5_file.file.filename} (it holds {present}).")
    group = h5_file["observables"][observable_name]
    missing = [name for name in ("step", "time", "value") if name not in group]
    if missing:
        raise ValueError(f"Observable '{observable_name}' must contain step, time, and "
                         f"value datasets; missing {missing}.")
    lengths = tuple(int(group[name].shape[0]) for name in ("step", "time", "value"))
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Observable '{observable_name}' has inconsistent step/time/value lengths: {lengths}")
    return (lengths[2],)


class H5DataSelector:
    """
    A simplified interface to access simulation data stored in an HDF5 file. The H5DataSelector maintains internal slice information for timesteps (axis 0) and particles (axis 1) and supports chaining via its accessor properties:

      - .timestep: Provides an interface for slicing/iterating over timesteps.
      - .particles: Provides an interface for slicing/iterating over particles.

    **Usage Examples:**

      # Slice timesteps 10 to 20 (inclusive of the start, exclusive of the stop)
      time_subset = data.timestep[10:20]

      # Slice particles 0 to 100
      particle_subset = data.particles[:100]

      # Chain slicing:
      sub = data.timestep[:5].particles[0:10]

      # Retrieve particle data by its espresso attribute name, e.g. positions:
      pos = data.pos  # This calls get_property("pos"), which reads position/value

      # Retrieve saved frame counters and physical times:
      step = data.step
      time = data.time

      # Ownership-based particle selection:
      filament_particles = data.select_particles_by_object("Filament", connectivity_value=0)
      monomer_subset = filament_particles.particles[10:20].timestep[5]

      # Predicate-based particle selection:
      final_type = data.select_particles_by_object(
          "Filament",
          connectivity_value=0,
          predicate=lambda subset: subset.type == 1,
      )

    **Notes:**
      - Direct indexing on a top-level H5DataSelector is disallowed to ensure explicit axis selection. Use the accessor properties (.timestep or .particles) for slicing.
      - `.timestep[...]` slices by frame index on axis 0. It does not query the stored HDF5 `step` or `time` datasets; use `frame_of_step`/`frame_of_time` to translate a stored value into a frame index.
      - `.step` and `.time` read the group's one step/time pair (hard-linked into every element; addressed as `position/step`, `position/time`).
      - Iteration and len() are only defined on the accessor objects.
      - Properties are named by espresso attribute (`pos`, `type`, `f`, `image_box`, `director`,
        `dip`, `id`); the file's H5MD element names (`position`, `species`, `force`, `image`) are
        dataset paths only (`element_name`), and `data.position` is an AttributeError.
      - Property data keeps the stored trailing axes: vectors are `(..., 3)`, scalars (`id`,
        `type`) have none, and the timestep/particle selections compose as an outer
        product, never pairwise.
      - Predicates are evaluated on the current H5DataSelector view. They select particles, not timesteps: the returned selector preserves the current timestep slice and only narrows the particle slice.
      - `common_dims` is `(n_frames, n_particles)`, taken from the group's `<element>/value` datasets at construction.

    **Raises:**
      - RuntimeError if the file is not in the `h5md-1` layout.
      - ValueError if the particle group is absent or the stored element datasets disagree on their leading dimensions.
    """
    def __init__(self, h5_file, particle_group, ts_slice=None, pt_slice=None):
        self.h5_file = h5_file
        self.particle_group = particle_group  # e.g., "Filament"
        self.common_dims = _particle_dims(h5_file, particle_group)

        # Set default slices if not provided (select all timesteps/particles)
        self.ts_slice = ts_slice if ts_slice is not None else slice(None)
        self.pt_slice = pt_slice if pt_slice is not None else slice(None)

    def __getitem__(self, key):
        raise TypeError(
            "Direct indexing on a H5DataSelector is not allowed. "
            "Use the 'timestep' or 'particles' accessor for slicing instead."
        )

    def __iter__(self):
        raise TypeError("H5DataSelector objects are not iterable. Use the '.timestep' or '.particles' accessor for iteration.")

    def __len__(self):
        raise TypeError("len() is ambiguous on H5DataSelector objects. Use '.timestep' or '.particles' accessor to get the length of the relevant axis.")

    @property
    def timestep(self):
        """
        Accessor for slicing/iterating over the timestep axis.

        Returns:
            TimestepAccessor: An accessor object for timestep operations.

        Usage:
            # Slicing timesteps 10 to 20:
            data.timestep[10:20]

            # Iterating over each timestep in a slice:
            for t in data.timestep[5:10]:
                process(t)
        """
        return TimestepAccessor(self)

    def _with_timestep(self, ts_slice):
        return H5DataSelector(
            self.h5_file,
            self.particle_group,
            ts_slice=ts_slice,
            pt_slice=self.pt_slice,
        )

    @property
    def particles(self):
        """
        Accessor for slicing/iterating over the particle axis.

        Returns:
            ParticleAccessor: An accessor object for particle operations.

        Usage:
            # Slicing particles 0 to 100:
            data.particles[0:100]

            # Iterating over each particle in a slice:
            for p in data.particles[20:30]:
                process(p)
        """
        return ParticleAccessor(self)

    def get_property(self, prop):
        """
        Retrieve data for a given property from the HDF5 dataset applying the current slices.

        Args:
            prop (str): The espresso attribute name (e.g., 'pos', 'f', 'type'); the
                dataset read is ``particles/<group>/<element_name(prop)>/value``.

        Returns:
            ndarray: The dataset with the applied timestep and particle slices.
        """
        ds = self.h5_file[f"particles/{self.particle_group}/{element_name(prop)}/value"]
        return read_h5_selection(ds, self.ts_slice, self.pt_slice)

    @property
    def bonds(self):
        """Bonds owned by the currently selected particles.

        Returns:
            BondSelection: See that class for the slicing semantics.

        Raises:
            AttributeError: If the file was written without bond storage.
        """
        conn_grp = self.h5_file.get(f"connectivity/{self.particle_group}")
        if conn_grp is None or "bonds" not in conn_grp:
            raise AttributeError(
                f"No bond topology stored for group '{self.particle_group}'. "
                "The run was inscribed with io_dict['bonds'] disabled."
            )
        # bonds.py imports this module, hence the local import
        from pressomancy.io.bonds import read_link_tables
        params_grp = self.h5_file[f"pressomancy/{self.particle_group}/bond_params"]
        rows, _ = positions_for_axis(self.pt_slice, self.common_dims[1])
        particle_ids = read_h5_selection(
            self.h5_file[f"particles/{self.particle_group}/{element_name('id')}/value"], 0, rows)
        kept = []   # (owner, bond_id, partners) per table
        for owners, partners, bond_ids in read_link_tables(self.h5_file, self.particle_group):
            mask = np.isin(owners, particle_ids)
            kept.append((owners[mask], bond_ids[mask], partners[mask]))
        max_partners = max((p.shape[1] for _, _, p in kept), default=1)
        partners = np.full((sum(len(o) for o, _, _ in kept), max_partners), -1, dtype=np.int32)
        n_partners = np.empty(partners.shape[0], dtype=np.int32)
        start = 0
        for _, _, p in kept:
            partners[start:start + len(p), :p.shape[1]] = p
            n_partners[start:start + len(p)] = p.shape[1]
            start += len(p)
        return BondSelection(
            params_grp,
            particle_ids=particle_ids,
            owner=np.concatenate([o for o, _, _ in kept]) if kept else np.empty(0, dtype=np.int32),
            bond_id=np.concatenate([b for _, b, _ in kept]) if kept else np.empty(0, dtype=np.int32),
            n_partners=n_partners,
            partners=partners,
        )

    @property
    def step(self):
        """Return saved frame counters from the particle group's one step dataset (`position/step`)."""
        ds = self.h5_file[f"particles/{self.particle_group}/{_TIMELINE}/step"]
        return read_h5_selection(ds, self.ts_slice)

    @property
    def time(self):
        """Return saved physical times from the particle group's one time dataset (`position/time`)."""
        ds = self.h5_file[f"particles/{self.particle_group}/{_TIMELINE}/time"]
        return read_h5_selection(ds, self.ts_slice)

    def get_box(self):
        """Return box metadata for the current particle group.

        ``edges`` is a time-dependent H5MD element on the group's timeline, so
        it is returned as ``[F, D]`` over the selected frames (like
        :attr:`step`/:attr:`time`), one row per frame.
        """
        box_path = f"particles/{self.particle_group}/box"
        box_group = self.h5_file[box_path]
        dimension = int(box_group.attrs["dimension"])
        boundary_raw = np.atleast_1d(box_group.attrs["boundary"]).tolist()
        boundary = tuple(
            item.decode("ascii") if isinstance(item, bytes) else str(item)
            for item in boundary_raw
        )
        edges = np.asarray(read_h5_selection(box_group["edges/value"], self.ts_slice),
                           dtype=np.float64)
        return {
            "dimension": dimension,
            "boundary": boundary,
            "edges": edges,
        }

    def get_connectivity_values(self, object_name, predicate=None):
        """
        Return object IDs present in ``pressomancy/<group>/ownership/ParticleHandle_to_<object_name>``.

        When a predicate is provided, each candidate object ID is first mapped
        to the particle indices of the current particle group. This keeps the
        returned per-object subset aligned with ``particles/<particle_group>``
        even when ``ParticleHandle_to_<object_name>`` is not in that particle
        group's storage order.

        Connectivity IDs are static. The predicate is therefore an object-ID
        filter: it may inspect particle properties on the per-object subset, but
        it must return one scalar truth value for the candidate object ID. If it
        inspects time-dependent arrays, reduce them explicitly with ``any``,
        ``all``, or an explicit timestep selection.

        Parameters
        ----------
        object_name : str
            Name of the connected object type (e.g. "Filament").
        predicate : callable, optional
            Function that accepts a per-object H5DataSelector subset and returns
            True when that object ID should be kept. The subset is provided only
            for inspection and preserves the current selector's timestep slice.

        Returns
        -------
        ndarray of shape (N,)
            Array of object IDs.
        """
        ownership = f"pressomancy/{self.particle_group}/ownership"
        ds_path = f"{ownership}/ParticleHandle_to_{object_name}"
        ds_father_path = f"{ownership}/ParticleHandle_to_{self.particle_group}"
        obj_connectivity = self.h5_file[ds_path][:]
        obj_ids = obj_connectivity[:,-1]
        father_particle_handles = self.h5_file[ds_father_path][:,0]
        ids=np.unique(obj_ids)
        if predicate is None:
            return ids

        ret_ids=[]
        for i in ids:
            object_particle_handles = obj_connectivity[obj_ids == i, 0]
            filter_mask = np.isin(father_particle_handles, object_particle_handles)
            subset = H5DataSelector(self.h5_file, self.particle_group, ts_slice=self.ts_slice,
                                    pt_slice=np.flatnonzero(filter_mask).tolist())
            if predicate(subset):
                ret_ids.append(i)
        return np.array(ret_ids)

    def select_particles_by_object(self, object_name, connectivity_value=None,predicate=None):
        """
        Select a subset of particles based on an ownership table.

        ``pressomancy/<group>/ownership/ParticleHandle_to_<object_name>`` stores
        particle ids, so this method maps those ids back to indices in
        ``particles/<particle_group>`` before composing a new particle slice.
        The current timestep slice is preserved.

        A predicate can further narrow the selected particles. It is evaluated on
        the current H5DataSelector subset, not on individual particle objects.
        Predicate masks select particles only:

        - a 1D mask must have shape ``(n_particles,)``;
        - a 2D mask must have shape ``(n_timesteps, n_particles)`` and is
          reduced with ``all`` over the current timestep context.

        This means ``predicate=lambda subset: subset.type == value`` keeps
        particles matching the value at every timestep in the current view,
        while ``predicate=lambda subset: subset.timestep[-1].type == value``
        classifies particles by the last timestep of the current view and then
        returns those particles across the original timestep slice.

        Args:
            object_name (str):
                Name of the connectivity object (e.g., "Filament").
            connectivity_value (int or float or array-like or None):
                Object ID value or values to match in the connectivity map.
                Defaults to None (all values).
            predicate (callable):
                Function taking an H5DataSelector and returning a particle mask as described above.

        Returns:
            H5DataSelector: A new selector with the same timestep slice and the
            particle slice set to the selected particle indices.
        """
        ownership = f"pressomancy/{self.particle_group}/ownership"
        # Get particles' ids from the ownership table of object_name
        ds_name = f"{ownership}/ParticleHandle_to_{object_name}"
        # Get the correct indices from the ownership table of the 'father' object. This is where the particle slices are applied to
        ds_father_name = f"{ownership}/ParticleHandle_to_{self.particle_group}"

        if connectivity_value is None:
             # Get all ids from object
            object_particle_indices = self.h5_file[ds_name][:,0]
        else:
            # Get only ids from object with connectivity value
            connectivity_map = self.h5_file[ds_name][:,1]
            connectivity_value=np.atleast_1d(connectivity_value)
            filter_mask = np.isin(connectivity_map, connectivity_value)
            object_particle_indices = np.ravel(self.h5_file[ds_name][:,0][filter_mask])
        # Get the correct indices for the selected particles from the parent's particle list
        father_particles_indices = self.h5_file[ds_father_name][:,0]
        filter_mask = np.isin(father_particles_indices, object_particle_indices)
        particle_indices = np.flatnonzero(filter_mask)
        subset=H5DataSelector(self.h5_file, self.particle_group, ts_slice=self.ts_slice, pt_slice=particle_indices.tolist())

        if predicate is not None:
            mask = np.asarray(predicate(subset))
            while mask.ndim > 1 and mask.shape[-1] == 1:
                mask = np.squeeze(mask, axis=-1)
            if mask.ndim == 2:
                mask = np.all(mask, axis=0)
            elif mask.ndim != 1:
                raise ValueError("Predicate must resolve to a particle mask.")
            particle_indices=particle_indices[mask]
            subset=H5DataSelector(self.h5_file, self.particle_group, ts_slice=self.ts_slice, pt_slice=particle_indices.tolist())
        return subset

    def select_particles_by_type(self, type_ids, frame=0):
        """The particles of the current selection whose stored ``type`` is in ``type_ids``.

        Particle types are static over a run, so the column is read at a single
        frame (``frame``, a frame *index* into the file) rather than once per
        stored frame. The timestep slice is preserved; only the particle slice
        narrows.

        This composes on the current particle slice by indexing the type column
        directly. ``select_particles_by_object`` cannot express "the current
        selection": it rebuilds the particle indices from the ownership table
        of the named object, so calling it with ``self.particle_group`` returns
        the whole group and drops whatever slice the selector already carried.

        Args:
            type_ids (int or array-like): The type value(s) to keep.
            frame (int): Frame index the type column is read at. Defaults to 0.

        Returns:
            H5DataSelector: A new selector over the matching particles, in the
            order they occupy in the current view.

        Raises:
            ValueError: If no particle of the current selection has one of those
                types; the message names the types that are present.
        """
        rows, _ = positions_for_axis(self.pt_slice, self.common_dims[1])
        ds = self.h5_file[f"particles/{self.particle_group}/{element_name('type')}/value"]
        types = np.asarray(read_h5_selection(ds, int(frame), rows)).reshape(-1)
        wanted = np.atleast_1d(type_ids)
        mask = np.isin(types, wanted)
        if not mask.any():
            raise ValueError(
                f"No particle of this selection has a type in {wanted.tolist()} at "
                f"frame {int(frame)} of group '{self.particle_group}'; the types "
                f"present are {np.unique(types).tolist()}.")
        return H5DataSelector(self.h5_file, self.particle_group,
                              ts_slice=self.ts_slice, pt_slice=rows[mask].tolist())

    def get_connectivity_map(self, parent_key, child_key):
        """
        The ``[parent_id, child_id]`` table linking two object classes.

        Parameters
        ----------
        parent_key : str
            Name of the parent object type (e.g. "Filament").
        child_key : str
            Name of the child object type (e.g. "Quadriplex").

        Returns
        -------
        ndarray of shape (N, 2)
            ``[parent_id, child_id]`` pairs.

        Raises
        ------
        KeyError
            If ``pressomancy/<group>/ownership/<parent>_to_<child>`` is not in the
            file; the message names the dataset path.
        """
        ds_path = f"pressomancy/{self.particle_group}/ownership/{parent_key}_to_{child_key}"
        if ds_path not in self.h5_file:
            raise KeyError(f"Connectivity map '{ds_path}' not found in {self.h5_file.file.filename}.")
        return self.h5_file[ds_path][:]

    def get_child_ids(self, parent_key, child_key, parent_id):
        """
        Sorted child object IDs connected to ``parent_id``.

        Parameters
        ----------
        parent_key : str
            Name of the parent object type.
        child_key : str
            Name of the child object type.
        parent_id : int
            The identifier for the parent object.

        Returns
        -------
        List[int]
            Sorted child who_am_i IDs. ``KeyError`` if the map is absent
            (see ``get_connectivity_map``).
        """
        conn = self.get_connectivity_map(parent_key, child_key)
        return sorted(int(cid) for cid in conn[conn[:, 0] == parent_id, 1])

    def get_parent_ids(self, parent_key, child_key, child_id):
        """
        Sorted parent object IDs connected to ``child_id``.

        Parameters
        ----------
        parent_key : str
            Name of the parent object type.
        child_key : str
            Name of the child object type.
        child_id : int
            The who_am_i identifier of the child object.

        Returns
        -------
        List[int]
            Sorted parent who_am_i IDs. ``KeyError`` if the map is absent
            (see ``get_connectivity_map``).
        """
        conn = self.get_connectivity_map(parent_key, child_key)
        return sorted(int(pid) for pid in conn[conn[:, 1] == child_id, 0])

    def __getattr__(self, attr):
        """
        Delegate attribute access to property retrieval.

        This allows a shorthand for accessing stored properties such that:
            data.pos
        is equivalent to:
            data.get_property('pos')

        Args:
            attr (str): The attribute name.

        Returns:
            The property data if available.

        Raises:
            AttributeError: If the property does not exist.
        """
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError(attr)
        if 'particle_group' not in self.__dict__ or 'h5_file' not in self.__dict__:
            raise AttributeError(attr)
        try:
            return self.get_property(attr)
        except KeyError as exc:
            detail = exc.args[0] if exc.args else exc
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {attr!r}: {detail}") from exc

    def __repr__(self):
        return (f"<H5DataSelector(particle_group={self.particle_group}, "
                f"ts_slice={self.ts_slice}, pt_slice={self.pt_slice})>")


class H5ObservableSelector:
    """
    A simplified interface to access observable data stored in an HDF5 file.

    The selector maintains internal slice information for the timestep axis only
    and exposes direct access to the stored ``step``, ``time``, and ``value``
    datasets under ``/observables/<name>``. ``common_dims`` is ``(n_frames,)``.
    """
    def __init__(self, h5_file, observable_name, ts_slice=None):
        self.h5_file = h5_file
        self.observable_name = observable_name
        self.common_dims = _observable_dims(h5_file, observable_name)
        self.ts_slice = ts_slice if ts_slice is not None else slice(None)

    def __getitem__(self, key):
        raise TypeError(
            "Direct indexing on a H5ObservableSelector is not allowed. "
            "Use the 'timestep' accessor for slicing instead."
        )

    def __iter__(self):
        raise TypeError("H5ObservableSelector objects are not iterable. Use the '.timestep' accessor for iteration.")

    def __len__(self):
        raise TypeError("len() is ambiguous on H5ObservableSelector objects. Use '.timestep' to get the number of selected frames.")

    @property
    def timestep(self):
        return TimestepAccessor(self)

    def _with_timestep(self, ts_slice):
        return H5ObservableSelector(
            self.h5_file,
            self.observable_name,
            ts_slice=ts_slice,
        )

    @property
    def value(self):
        ds = self.h5_file[f"observables/{self.observable_name}/value"]
        return read_h5_selection(ds, self.ts_slice)

    @property
    def step(self):
        ds = self.h5_file[f"observables/{self.observable_name}/step"]
        return read_h5_selection(ds, self.ts_slice)

    @property
    def time(self):
        ds = self.h5_file[f"observables/{self.observable_name}/time"]
        return read_h5_selection(ds, self.ts_slice)

    def __repr__(self):
        return (f"<H5ObservableSelector(observable_name={self.observable_name}, "
                f"ts_slice={self.ts_slice})>")


class TimestepAccessor:
    """
    An accessor for slicing and iterating over timesteps of an HDF5 selector.

    It composes new timestep slices with any existing slice and delegates selector
    reconstruction to the parent selector.
    """
    def __init__(self, sim_data):
        """
        Initialize the TimestepAccessor.

        Args:
            sim_data (H5DataSelector or H5ObservableSelector): The parent selector.
        """
        self.sim_data = sim_data

    def __getitem__(self, key):
        """
        Compose the new timestep index with the current slice.

        Args:
            key (int, slice, or list/tuple): The new timestep index or slice.

        Returns:
            H5DataSelector or H5ObservableSelector: A new selector with the
            updated timestep slice.

        Raises:
            IndexError: If the new index is out of bounds relative to the effective timestep indices.
        """
        total_timesteps = self.sim_data.common_dims[0]
        composed = _compose_index(self.sim_data.ts_slice, key, total_timesteps)
        return self.sim_data._with_timestep(composed)

    def __iter__(self):
        """
        Iterate over the effective timestep indices.

        Yields:
            H5DataSelector or H5ObservableSelector: Each selector has ts_slice set
            to a single timestep.
        """
        total_timesteps = self.sim_data.common_dims[0]
        ts_slice = self.sim_data.ts_slice
        if isinstance(ts_slice, slice):
            indices = list(range(*ts_slice.indices(total_timesteps)))
        elif isinstance(ts_slice, (list, tuple)):
            indices = ts_slice
        else:
            indices = [ts_slice]
        for idx in indices:
            yield self.sim_data._with_timestep(idx)

    def __len__(self):
        """
        Return the number of timesteps in the current slice.

        Returns:
            int: The count of timesteps.
        """
        total_timesteps = self.sim_data.common_dims[0]
        ts_slice = self.sim_data.ts_slice
        if isinstance(ts_slice, slice):
            start, stop, step = ts_slice.indices(total_timesteps)
            return len(range(start, stop, step))
        elif isinstance(ts_slice, (list, tuple)):
            return len(ts_slice)
        elif isinstance(ts_slice, int):
            return 1
        else:
            return total_timesteps

    def __repr__(self):
        return f"<TimestepAccessor(ts_slice={self.sim_data.ts_slice})>"


class ParticleAccessor:
    """
    An accessor for slicing and iterating over particles of a H5DataSelector.

    It composes new particle indices with any existing slice and enables iteration
    where each iteration yields a H5DataSelector corresponding to a single particle index.
    """
    def __init__(self, sim_data):
        """
        Initialize the ParticleAccessor.

        Args:
            sim_data (H5DataSelector): The parent data selector instance.
        """
        self.sim_data = sim_data

    def __getitem__(self, key):
        """
        Compose the new particle index with the current slice.

        Args:
            key (int, slice, or list/tuple): The new particle index or slice.

        Returns:
            H5DataSelector: A new selector with the updated particle slice.

        Raises:
            IndexError: If the new index is out of range for the effective particle indices.
        """
        total_particles = self.sim_data.common_dims[1]
        composed = _compose_index(self.sim_data.pt_slice, key, total_particles)
        return H5DataSelector(self.sim_data.h5_file, self.sim_data.particle_group, ts_slice=self.sim_data.ts_slice, pt_slice=composed)

    def __iter__(self):
        """
        Iterate over the effective particle indices.

        Yields:
            H5DataSelector: Each selector has pt_slice set to a single particle index.
        """
        total_particles = self.sim_data.common_dims[1]
        pt = self.sim_data.pt_slice
        if isinstance(pt, slice):
            indices = list(range(*pt.indices(total_particles)))
        elif isinstance(pt, (list, tuple)):
            indices = pt
        else:
            indices = [pt]
        for idx in indices:
            yield H5DataSelector(self.sim_data.h5_file, self.sim_data.particle_group, ts_slice=self.sim_data.ts_slice, pt_slice=idx)

    def __len__(self):
        """
        Return the number of particles in the current slice.

        Returns:
            int: The count of particles.
        """
        total_particles = self.sim_data.common_dims[1]
        pt_slice = self.sim_data.pt_slice
        if isinstance(pt_slice, slice):
            start, stop, step = pt_slice.indices(total_particles)
            return len(range(start, stop, step))
        elif isinstance(pt_slice, (list, tuple)):
            return len(pt_slice)
        elif isinstance(pt_slice, int):
            return 1
        else:
            return total_particles

    def __repr__(self):
        return f"<ParticleAccessor(pt_slice={self.sim_data.pt_slice})>"


def _compose_index(existing, new, total_length):
    """
    Compose two layers of indexing on a given axis by converting the existing index into an explicit list,
    then applying the new index. This ensures that chained indexing works similarly to Python's native list slicing.

    Args:
        existing (int, slice, or list/tuple): The current index (or composed indices).
        new (int, slice, or list/tuple): The new index to be applied.
        total_length (int): The full length of the axis in the underlying dataset.

    Returns:
        int, list, or tuple: The composed index representing the effective selection on the axis.

    Raises:
        IndexError: If the new index is out of bounds for the effective indices.
        TypeError: If unsupported types are provided for indexing.
    """
    # Convert the existing index to an explicit list.
    if isinstance(existing, slice):
        base = list(range(*existing.indices(total_length)))
    elif isinstance(existing, (list, tuple)):
        base = list(existing)
    elif isinstance(existing, int):
        base = [existing]
    else:
        raise TypeError("Unsupported type for the existing index.")

    # Apply the new indexing on the explicit list.
    if isinstance(new, int):
        try:
            result = base[new]  # Raises IndexError if out-of-bounds.
        except IndexError as e:
            raise IndexError(
                f"Index {new} is out of range for composed indices of length {len(base)}"
            ) from e
    elif isinstance(new, slice):
        result = base[new]
    elif isinstance(new, (list, tuple)):
        result = [base[i] for i in new]
    else:
        raise TypeError("Unsupported type for the new index.")
    return result


# --------------------------------------------------------------------------- #
# stored steps / times: the frame index <-> step value / time value mapping
# --------------------------------------------------------------------------- #

def stored_steps(h5_file, group):
    """The step values stored for ``group``, as an int array, frame by frame.

    A group has one ``step`` dataset, hard-linked into every element; it is
    read as ``particles/<group>/position/step``.

    Args:
        h5_file (h5py.File or h5py.Group): An **open** file (or a group holding
            ``particles``). Paths are not accepted.
        group (str): The particle group name, mandatory.

    Returns:
        ndarray: ``(n_frames,)`` of ``int``.

    Raises:
        RuntimeError: If the file is not in the ``h5md-1`` layout.
    """
    return np.asarray(_particle_group(h5_file, group)[_TIMELINE]["step"][...]).reshape(-1)

def frame_of_step(h5_file, group, step):
    """The frame *index* of ``group`` whose stored step value is ``step``.

    The inverse of ``selector.step[frame]``.

    Raises:
        KeyError: If no frame carries that step value (the message lists the
            stored steps, abbreviated).
        ValueError: If more than one does -- two runs appended the same
            counter without truncating first -- so the selection is ambiguous.
    """
    steps = stored_steps(h5_file, group)
    matches = np.flatnonzero(steps == int(step))
    where = f"{h5_file.file.filename}/particles/{group}"
    if matches.size == 0:
        stored = steps.tolist()
        shown = stored if len(stored) <= 12 else stored[:6] + ['...'] + stored[-6:]
        raise KeyError(f"step {step} is not stored in {where}: it holds "
                       f"{len(stored)} frame(s) with steps {shown}.")
    if matches.size > 1:
        raise ValueError(f"step {step} is stored in {matches.size} frames of {where} "
                         f"(indices {matches.tolist()}); the selection is ambiguous.")
    return int(matches[0])


def frame_of_time(h5_file, group, time):
    """The frame *index* of ``group`` whose stored time is nearest to ``time``.

    Read from the group's one ``time`` dataset (``particles/<group>/position/time``,
    float32). The nearest stored time must match ``time`` within a relative
    ``_TIME_RTOL``=1e-6.

    Raises:
        KeyError: If the group holds no frames, or the nearest stored time is
            outside the tolerance (the message names it and its frame).
    """
    where = f"{h5_file.file.filename}/particles/{group}"
    times = np.asarray(_particle_group(h5_file, group)[_TIMELINE]["time"][...], dtype=float).reshape(-1)
    if times.size == 0:
        raise KeyError(f"{where} holds no frames.")
    index = int(np.argmin(np.abs(times - float(time))))
    if not math.isclose(times[index], float(time), rel_tol=_TIME_RTOL):
        raise KeyError(f"time {time} is not stored in {where}: the nearest stored time is "
                       f"{times[index]} (frame {index}); tolerance is a relative {_TIME_RTOL}.")
    return index


def has_step(h5_file, group, step):
    """Whether ``group`` holds a frame stored at ``step``. Same inputs as above."""
    return bool(np.any(stored_steps(h5_file, group) == int(step)))
