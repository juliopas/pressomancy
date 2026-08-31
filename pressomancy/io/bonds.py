"""Normalized, static HDF5 persistence for espresso bonds.

Layout, under ``/connectivity/<Group>/bonds``::

    params/<BondTypeName>  compound, one row per *registered* bond
                           (``bond_id`` + exactly that type's parameters)
    particle_ids           int32  (N,)    particle id of each CSR row
    offsets                int64  (N+1,)  CSR row pointers into ``links``
    links                  int32  (M, 2 + max_partners)
                           columns: bond_id, n_partners, p0 .. pk  (-1 padded)

Parameters live once in ``params`` and are referenced by ``bond_id``: nothing is
repeated per occurrence, and nothing is repeated per frame because the topology
has no time axis at all. CSR rather than a vlen-of-compound: it chunks and
compresses properly, reads contiguously, and slices per particle trivially.
"""

from collections import defaultdict

import logging

import h5py
import numpy as np

import espressomd.interactions

_SKIP_PARAMS = frozenset({"bond_id", "_bond_id"})


def _bond_id_of(handle):
    """Registration id of ``handle``.

    Keyed on ``_bond_id``, never on ``id(handle)``: espresso re-creates the
    Python wrapper on every ``part.bonds`` access, and once a temporary is
    collected CPython reuses its ``id()`` for an unrelated object, which would
    silently resolve to the wrong bond.
    """
    bond_id = getattr(handle, "_bond_id", None)
    if bond_id is None or int(bond_id) < 0:
        raise RuntimeError(
            f"Cannot determine the bond id of {handle!r}; it appears not to be "
            "registered in sys.bonded_inter."
        )
    return int(bond_id)


def _registered_bonds(sys):
    """``[(bond_id, handle), ...]`` for every bond in ``sys.bonded_inter``."""
    out = []
    for entry in sys.bonded_inter:
        if isinstance(entry, tuple):
            bond_id, handle = int(entry[0]), entry[1]
        else:
            bond_id, handle = _bond_id_of(entry), entry
        out.append((bond_id, handle))
    return out


# --------------------------------------------------------------------------- #
# Safeguard for new bonds
# --------------------------------------------------------------------------- #

def count_bond_links(particles):
    """Total bond occurrences owned by ``particles``. One pass, no allocation."""
    return sum(len(part.bonds) for part in particles)


def check_bond_count(stored, particles, group_name, policy='raise'):
    """Guard against topology changes after inscription.

    Compares occurrence counts only, so a rewire that
    leaves the count unchanged will go unnoticed.
    """
    if policy == 'ignore' or stored is None:
        return None
    live = count_bond_links(particles)
    if live == stored:
        return True
    msg = (
        f"Bond topology of group '{group_name}' changed after inscription "
        f"({stored} -> {live} bond occurrences). Topology is written once and "
        "has no time axis, so this change is NOT in the file. Create all bonds "
        "before inscribe_part_group_to_h5(), or call rewrite_bonds() to "
        "re-capture, or set io_dict['bond_topology_policy']='warn'."
    )
    if policy == 'warn':
        logging.warning(msg)
    else:
        raise RuntimeError(msg)
    return False

    
# --------------------------------------------------------------------------- #
# parameter schema (derived, not hardcoded)
# --------------------------------------------------------------------------- #

def _param_items(handle):
    """``[(name, value)]`` of the parameters needed to rebuild ``handle``."""
    params = handle.get_params()
    names = sorted(params)
    return [(n, params[n]) for n in names if n not in _SKIP_PARAMS]


def h5_dtype_for(handle):
    """Compound dtype describing ``handle``'s parameters, plus ``bond_id``."""
    fields = [("bond_id", np.int32)]
    for name, value in _param_items(handle):
        arr = np.asarray(value)
        if arr.ndim != 0:
            raise NotImplementedError(
                f"Parameter '{name}' of {type(handle).__name__} is not a scalar."
                " This is currently not supported."
            )
        if arr.dtype.kind == "f":
            base = np.float32
        elif arr.dtype.kind in "iu":
            base = np.int32
        elif arr.dtype.kind == "b":
            base = np.bool
        else:
            raise NotImplementedError(
                f"Parameter '{name}' of {type(handle).__name__} has unsupported "
                f"kind '{arr.dtype.kind}'."
            )
        fields.append((name, base))
    return np.dtype(fields)


def write_bond_params(bonds_grp, sys):
    """Write ``bonds/params/<BondTypeName>``, one compound table per bond type."""
    by_type = defaultdict(list)
    for bond_id, handle in _registered_bonds(sys):
        by_type[type(handle)].append((bond_id, handle))

    params_grp = bonds_grp.require_group("params")
    for bond_cls, entries in by_type.items():
        dtype = h5_dtype_for(entries[0][1])
        names = [n for n in dtype.names if n != "bond_id"]

        table = np.empty(len(entries), dtype=dtype)
        for row, (bond_id, handle) in enumerate(entries):
            params = handle.get_params()
            table[row]["bond_id"] = bond_id
            for name in names:
                table[row][name] = params[name]

        params_grp.create_dataset(
            bond_cls.__name__, data=table,
            compression="gzip", compression_opts=4,
        )


# --------------------------------------------------------------------------- #
# connectivity
# --------------------------------------------------------------------------- #

def collect_bond_links(particles):
    """Build the CSR connectivity for ``particles``.

    ``particles`` is the flat view for one registered group, in the *same order*
    as the columns of that group's ``pos/value``. Partners may live outside the
    group; they are stored as raw particle ids and left for the reader to
    resolve, so a dangling partner is visible rather than silently dropped.

    Returns:
        tuple: ``(particle_ids, offsets, links, max_partners)``.
    """
    particle_ids = np.fromiter((int(p.id) for p in particles),
                               dtype=np.int32, count=len(particles))

    rows, max_partners = [], 1
    for part in particles:
        row = []
        for entry in part.bonds:
            handle, partners = entry[0], entry[1:]
            max_partners = max(max_partners, len(partners))
            row.append((_bond_id_of(handle), [int(x) for x in partners]))
        rows.append(row)

    counts = np.fromiter((len(r) for r in rows), dtype=np.int64, count=len(rows))
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    links = np.full((int(offsets[-1]), 2 + max_partners), -1, dtype=np.int32)
    k = 0
    for row in rows:
        for bond_id, partners in row:
            links[k, 0] = bond_id
            links[k, 1] = len(partners)
            links[k, 2:2 + len(partners)] = partners
            k += 1
    return particle_ids, offsets, links, max_partners


def write_bonds(connect_grp, particles, sys, step=0):
    """Write ``<connect_grp>/bonds``. Call once, at inscription.

    Idempotent: an existing group is deleted first, so re-inscribing a group
    does not collide.
    """
    if "bonds" in connect_grp:
        del connect_grp["bonds"]
    bonds_grp = connect_grp.require_group("bonds")
    write_bond_params(bonds_grp, sys)

    particle_ids, offsets, links, max_partners = collect_bond_links(particles)
    gz = dict(compression="gzip", compression_opts=4)
    bonds_grp.create_dataset("particle_ids", data=particle_ids, **gz)
    bonds_grp.create_dataset("offsets", data=offsets, **gz)
    # A zero-row dataset cannot be chunked, and gzip requires chunking.
    bonds_grp.create_dataset("links", data=links, **(gz if links.shape[0] else {}))

    n_links = int(offsets[-1])
    bonds_grp.attrs["static_topology"] = True
    bonds_grp.attrs["max_partners"] = int(max_partners)
    bonds_grp.attrs["n_links"] = n_links
    bonds_grp.attrs["link_columns"] = np.array(
        ["bond_id", "n_partners"] + [f"partner_{i}" for i in range(max_partners)],
        dtype=h5py.string_dtype(encoding="ascii"),
    )
    bonds_grp.attrs["captured_at_step"] = int(step)
    bonds_grp.attrs["captured_at_time"] = float(sys.time)
    return n_links


def verify_bond_params(connect_grp, sys):
    """Warn if the live ``sys.bonded_inter`` has drifted from the stored tables.

    Advisory only: on resume the file is authoritative and is not rewritten.
    """
    bonds_grp = connect_grp.get("bonds")
    if bonds_grp is None:
        logging.warning("Bonds requested but no bonds group found in %s.",
                        connect_grp.name)
        return
    stored = read_bond_params(bonds_grp)
    for bond_id, handle in _registered_bonds(sys):
        entry = stored.get(bond_id)
        if entry is None:
            logging.warning("Bond id %d (%s) is live but absent from the file.",
                            bond_id, type(handle).__name__)
            continue
        cls, kw = entry
        if cls is not type(handle):
            logging.warning("Bond id %d: file says %s, live handle is %s.",
                            bond_id, cls.__name__, type(handle).__name__)
            continue
        live = handle.get_params()
        for name, value in kw.items():
            if not np.allclose(np.asarray(live[name], dtype=float),
                               np.asarray(value, dtype=float)):
                logging.warning("Bond id %d (%s): parameter '%s' differs "
                                "(file %r, live %r).",
                                bond_id, cls.__name__, name, value, live[name])


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #

def read_bond_params(bonds_grp):
    """Return ``{bond_id: (bond_class, {param: value})}`` from ``bonds/params``."""
    table = {}
    for type_name, dset in bonds_grp["params"].items():
        bond_cls = getattr(espressomd.interactions, type_name, None)
        if bond_cls is None:
            raise NotImplementedError(
                f"File contains bond type '{type_name}', which is not exposed by "
                "espressomd.interactions in this build."
            )
        rows = dset[...]
        names = [n for n in rows.dtype.names if n != "bond_id"]
        for row in rows:
            kw = {}
            for n in names:
                v = row[n]
                kw[n] = v.tolist() if getattr(v, "shape", ()) else v.item()
            table[int(row["bond_id"])] = (bond_cls, kw)
    return table


def read_bonds(bonds_grp, instantiate=False):
    """Yield ``(particle_id, partners, bond)`` for every stored bond.

    ``partners`` is a tuple of particle ids, so angle and dihedral bonds come
    back intact. ``bond`` is the integer ``bond_id`` by default; pass
    ``instantiate=True`` to get live espresso bond objects instead, which
    requires an initialised espresso System in the process.
    """
    params = read_bond_params(bonds_grp)
    bonds = ({bid: cls(**kw) for bid, (cls, kw) in params.items()}
             if instantiate else {bid: bid for bid in params})

    particle_ids = bonds_grp["particle_ids"][...]
    offsets = bonds_grp["offsets"][...]
    links = bonds_grp["links"][...]
    for i, pid in enumerate(particle_ids):
        for link in links[offsets[i]:offsets[i + 1]]:
            n = int(link[1])
            yield int(pid), tuple(int(x) for x in link[2:2 + n]), bonds[int(link[0])]