"""Lazy, NumPy-like selections on :mod:`h5py` datasets."""

import math
import numpy as np
from collections import defaultdict

_DEFAULT_MAX_OVERREAD_FACTOR = 4.0


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
    if isinstance(index, (bool, np.bool_)):
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
    # np.arange(data.shape[axis]) keep the whole axies
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


class BondSelection:
    """The bonds owned by the particles of an :class:`H5DataSelector` view.

    Topology is static, so this ignores the timestep slice: ``data.bonds`` and
    ``data.timestep[5].bonds`` are the same object. The particle slice is
    honoured, and the ordering invariant is that CSR row ``i`` of
    ``connectivity/<Group>/bonds`` describes column ``i`` of every
    ``particles/<Group>/<prop>/value`` — the writer builds both from the same
    flat particle view.

    A bond is stored once, on its owner. If the particle slice excludes an
    owner but includes its partner, that bond is absent; see ``external`` for
    the mirror case of partners outside the selection.

    Attributes:
        particle_ids (ndarray): ``(P,)`` selected particle ids, in view order.
        owner (ndarray): ``(L,)`` particle id owning each link.
        bond_id (ndarray): ``(L,)`` registered bond id of each link.
        n_partners (ndarray): ``(L,)`` valid partner count per link.
        partners (ndarray): ``(L, max_partners)`` partner ids, ``-1`` padded.
    """

    def __init__(self, grp, particle_ids, owner, bond_id, n_partners, partners):
        self._grp = grp
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
        this reads the whole ``params`` group regardless of the slice.
        """
        if self._params is None:
            table = {}
            for type_name, dset in self._grp["params"].items():
                rows = dset[...]
                names = [n for n in rows.dtype.names if n != "bond_id"]
                for row in rows:
                    kw = {n: (row[n].tolist() if getattr(row[n], "shape", ())
                              else row[n].item()) for n in names}
                    table[int(row["bond_id"])] = (type_name, kw)
            self._params = table
        return self._params

    @property
    def external(self):
        """``(L,)`` bool: True where any partner falls outside the selection."""
        inside = np.isin(self.partners, self.particle_ids) | (self.partners < 0)
        return ~inside.all(axis=1)

    def pairs(self, drop_external=False):
        """``(E, 2)`` unique undirected edges, each row sorted ascending.

        Only two-body links contribute; angle and dihedral bonds have no
        well-defined edge and are skipped. Set ``drop_external`` to exclude
        edges reaching a particle outside the selection.
        """
        mask = self.n_partners == 1
        if drop_external:
            mask &= ~self.external
        edges = np.stack([self.owner[mask], self.partners[mask, 0]], axis=1)
        return np.unique(np.sort(edges, axis=1), axis=0) if edges.size else edges

    def neighbours(self, drop_external=False):
        """``{particle_id: [(partner_id, bond_id), ...]}``, both directions."""
        out = defaultdict(list)
        keep = ~self.external if drop_external else np.ones(len(self), bool)
        for i in np.flatnonzero(keep):
            pid, bid = int(self.owner[i]), int(self.bond_id[i])
            for partner in self.partners[i, :self.n_partners[i]]:
                out[pid].append((int(partner), bid))
                out[int(partner)].append((pid, bid))
        return out

    def by_bond_type(self):
        """``{type_name: mask}`` over the links, for per-type analysis."""
        by_id = {bid: name for bid, (name, _) in self.params.items()}
        names = np.array([by_id.get(int(b), "?") for b in self.bond_id])
        return {name: names == name for name in np.unique(names)}

    def __repr__(self):
        types = ", ".join(sorted({n for n, _ in self.params.values()}))
        return (f"<BondSelection: {len(self)} links over "
                f"{self.particle_ids.size} particles; types: {types or 'none'}>")