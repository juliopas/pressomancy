"""Static HDF5 persistence for espresso bonds, in the H5MD connectivity form.

Layout, for one particle group ``<Group>``::

    /connectivity/<Group>/bonds        int32 (N1, 2)  (owner, partner)             -- always written
    /connectivity/<Group>/angles       int32 (N2, 3)  (owner, p0, p1)              -- only if such links exist
    /connectivity/<Group>/dihedrals    int32 (N3, 4)  (owner, p0, p1, p2)          -- only if such links exist
        every entry is a COLUMN INDEX (0..N-1) into the group's ``<element>/value``
        arrays, as the H5MD connectivity spec requires -- not a particle id
        every table: attr particles_group = "/particles/<Group>"
        ``bonds`` only: attrs n_links (links of ALL tables), captured_at_time
    /pressomancy/<Group>/bond_params/bond_id      int32 (N1,)  row-aligned with ``bonds``
    /pressomancy/<Group>/bond_params/angle_id     int32 (N2,)  row-aligned with ``angles``
    /pressomancy/<Group>/bond_params/dihedral_id  int32 (N3,)  row-aligned with ``dihedrals``
    /pressomancy/<Group>/bond_params/<BondClass>  compound, one row per *registered* bond
                                                  (``bond_id`` + exactly that class's parameters)

A link is stored once, on its owner (the particle whose ``part.bonds`` lists it).
Inside pressomancy a particle is an id; the file's column indices exist only in
this module: ``write_bonds`` maps ids to indices (``np.searchsorted`` over the
group's ascending id column) and ``read_link_tables`` maps them back through
``particles/<Group>/id/value[0]`` (the id column is constant across frames), so
``read_bonds``, ``H5DataSelector.bonds`` and ``set_bonds_from_src`` all see ids.
A partner that is not a particle of the group has no column and is a
``NotImplementedError`` at write time, naming the link, rather than a foreign number in
the table. The table a link lands in is decided by its partner count
(``LINK_TABLES`` in ``io/read.py``: 1 -> bonds, 2 -> angles, 3 -> dihedrals,
which covers every espresso bond). Parameters live once in the
class tables and are referenced by ``bond_id``: nothing is repeated per
occurrence, and nothing is repeated per frame because the topology has no time
axis at all. Cannot track changes to bond topology in time.
"""

import logging
from collections import defaultdict

import h5py
import numpy as np

import espressomd.interactions

from pressomancy.io.read import LINK_TABLES, element_name

_SKIP_PARAMS = frozenset({"bond_id", "_bond_id"})

#: How many offending links a single aggregated warning lists by name.
_MAX_EXAMPLES = 3

_GZ = dict(compression="gzip", compression_opts=4)


def _bond_id_of(handle):
    """Registration id of ``handle`` keyed on ``handle._bond_id``.

    Never on ``id(handle)``: espresso re-creates the Python wrapper on every
    ``part.bonds`` access, and once a temporary is collected CPython reuses
    its ``id()`` for an unrelated object, which would silently resolve to
    the wrong bond.
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
    return [(_bond_id_of(handle), handle) for handle in sys.bonded_inter]


# --------------------------------------------------------------------------- #
# Safeguard for new bonds
# --------------------------------------------------------------------------- #


def check_bond_count(stored, particles, group_name):
    """Guard against topology changes after inscription.

    Compares occurrence counts only, so a rewire that leaves the count
    unchanged will go unnoticed. ``stored`` is the ``n_links`` captured at
    inscription (``None`` when the group was inscribed without bonds).

    Raises:
        RuntimeError: If the live count differs from ``stored``.
    """
    if stored is None:
        return
    live = sum(len(part.bonds) for part in particles)
    if live != stored:
        raise RuntimeError(
            f"Bond topology of group '{group_name}' changed after inscription "
            f"({stored} -> {live} bond occurrences). Topology is written once and "
            "has no time axis, so this change is NOT in the file. Create all bonds "
            "before calling inscribe_part_group_to_h5()."
        )


# --------------------------------------------------------------------------- #
# parameter schema (derived, not hardcoded)
# --------------------------------------------------------------------------- #

def _param_items(handle):
    """``[(name, value)]`` of the parameters needed to rebuild ``handle``."""
    params = handle.get_params()
    names = sorted(params.keys())
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


def _param_signature(bond_cls, params):
    """Hashable identity of ``(bond_cls, params)``, comparable across the file.

    Floats are rounded to ``float32`` because that is what ``h5_dtype_for``
    stores, so a live ``float64`` parameter and its own stored copy produce the
    *same* signature and a lossless round trip compares equal. The class is
    keyed by name rather than by the class object, so a file read in another
    process compares against the live registry all the same.
    """
    items = []
    for name, value in sorted(params.items()):
        if name in _SKIP_PARAMS:
            continue
        arr = np.asarray(value)
        if arr.ndim:
            flat = arr.ravel()
            if flat.dtype.kind == "f":
                items.append((name, tuple(float(x) for x in np.float32(flat))))
            else:
                items.append((name, tuple(flat.tolist())))
        elif arr.dtype.kind == "f":
            items.append((name, float(np.float32(arr.item()))))
        elif arr.dtype.kind == "b":
            items.append((name, bool(arr.item())))
        elif arr.dtype.kind in "iu":
            items.append((name, int(arr.item())))
        else:
            items.append((name, arr.item()))
    return (bond_cls.__name__ if isinstance(bond_cls, type) else str(bond_cls),
            tuple(items))


def _live_links_of(part):
    """``{partners_tuple: [signature, ...]}`` for one live particle, in bond order."""
    by_partners = defaultdict(list)
    for entry in part.bonds:
        handle, partners = entry[0], tuple(int(x) for x in entry[1:])
        by_partners[partners].append(_param_signature(type(handle), handle.get_params()))
    return by_partners


def _stored_links_by_particle(h5_file, group_name):
    """``(params_table, {particle_id: [(partners, bond_id, cls, kw), ...]})``."""
    stored = read_bond_params(h5_file, group_name)
    by_particle = defaultdict(list)
    for pid, partners, bond_id in read_bonds(h5_file, group_name):
        cls, kw = stored[int(bond_id)]
        by_particle[int(pid)].append((partners, int(bond_id), cls, kw))
    return stored, by_particle


def write_bond_params(params_grp, sys):
    """Write ``<params_grp>/<BondClass>``, one compound table per registered bond class."""
    by_type = defaultdict(list)
    for bond_id, handle in _registered_bonds(sys):
        by_type[type(handle)].append((bond_id, handle))

    for bond_cls, entries in by_type.items():
        dtype = h5_dtype_for(entries[0][1])
        names = [n for n in dtype.names if n != "bond_id"]

        table = np.empty(len(entries), dtype=dtype)
        for row, (bond_id, handle) in enumerate(entries):
            params = handle.get_params()
            table[row]["bond_id"] = bond_id
            for name in names:
                table[row][name] = params[name]

        params_grp.create_dataset(bond_cls.__name__, data=table, **_GZ)


# --------------------------------------------------------------------------- #
# connectivity
# --------------------------------------------------------------------------- #

def _columns_of(ids, id_rows, group_name, bond_ids):
    """Column index of every id in ``id_rows`` (L, 1 + k) within the ascending ``ids``.

    The id -> column translation of the write side. A partner id without a column
    (a particle of another group, or of no object) raises ``NotImplementedError``
    naming owner, partner and bond id: H5MD connectivity rows are indices into one
    particles group, and bonds across groups are not supported by this writer.
    """
    columns = np.searchsorted(ids, id_rows)
    in_range = columns < ids.size
    found = np.zeros_like(in_range)
    found[in_range] = ids[columns[in_range]] == id_rows[in_range]
    if not found.all():
        row, col = np.argwhere(~found)[0]
        raise NotImplementedError(
            f"Group '{group_name}': particle {int(id_rows[row, 0])} owns a bond (bond_id "
            f"{int(bond_ids[row])}) to partner {int(id_rows[row, col])}, which is not a particle "
            f"of the group. Bonds between different particle groups are not supported: H5MD "
            f"connectivity rows are column indices into particles/{group_name}.")
    return columns


def write_bonds(h5_file, group_name, particles, sys):
    """Write the link tables and parameter tables of ``group_name``. Call once, at inscription.

    ``particles`` is the flat view for one registered group, in the *same order*
    as the columns of that group's ``<element>/value`` (ascending id); link rows
    are stored as column indices into it. Returns ``n_links``, the number of
    links over all tables (what ``check_bond_count`` compares).

    Raises:
        ValueError: If ``particles`` is not in strictly ascending id order.
        NotImplementedError: If a link has a partner outside the group (bonds
            between particle groups are not supported, see ``_columns_of``).
    """
    conn_grp = h5_file.require_group(f"connectivity/{group_name}")
    params_grp = h5_file.require_group(f"pressomancy/{group_name}/bond_params")
    write_bond_params(params_grp, sys)

    ids = np.array([int(part.id) for part in particles], dtype=np.int64)
    if not np.all(np.diff(ids) > 0):
        raise ValueError(f"Group '{group_name}': particles must be in strictly ascending id order "
                         f"(file column order) to store links as column indices; got {ids.tolist()}.")
    # One row per link: (bond_id, owner id, partner ids...), bucketed by partner count.
    rows_by_arity = defaultdict(list)
    for part in particles:
        for entry in part.bonds:
            handle, partners = entry[0], [int(x) for x in entry[1:]]
            rows_by_arity[len(partners)].append((_bond_id_of(handle), int(part.id), *partners))
    n_links = sum(len(rows) for rows in rows_by_arity.values())
    rows_by_arity.setdefault(1, [])   # the H5MD 'bonds' table is always present
    unsupported = sorted(set(rows_by_arity) - set(LINK_TABLES))
    if unsupported:
        raise NotImplementedError(
            f"Group '{group_name}' has links with {unsupported} partner(s); only "
            f"{sorted(LINK_TABLES)} partner(s) have a table ({LINK_TABLES}).")
    for arity, rows in sorted(rows_by_arity.items()):
        table_name, id_name = LINK_TABLES[arity]
        arr = np.array(rows, dtype=np.int64).reshape(len(rows), 2 + arity)
        columns = _columns_of(ids, arr[:, 1:], group_name, arr[:, 0])
        # A zero-row dataset cannot be chunked, and gzip requires chunking.
        gz = _GZ if rows else {}
        table = conn_grp.create_dataset(table_name, data=columns.astype(np.int32), **gz)
        table.attrs["particles_group"] = f"/particles/{group_name}"
        params_grp.create_dataset(id_name, data=arr[:, 0].astype(np.int32), **gz)

    conn_grp["bonds"].attrs["n_links"] = n_links
    conn_grp["bonds"].attrs["captured_at_time"] = float(sys.time)
    return n_links


# For a nice error message in verify_bond_params
def _drift_description(stored_sig, live_sig):
    """Human-readable difference between two ``_param_signature`` values."""
    stored_cls, stored_params = stored_sig
    live_cls, live_params = live_sig
    if stored_cls != live_cls:
        return f"file says {stored_cls}, live bond is {live_cls}"
    live_map = dict(live_params)
    diffs = [f"'{name}' file {value!r} live {live_map.get(name, '<absent>')!r}"
             for name, value in stored_params if live_map.get(name, object()) != value]
    extra = [name for name, _ in live_params if name not in dict(stored_params)]
    if extra:
        diffs.append(f"live-only parameter(s) {extra}")
    return f"{stored_cls}: " + ", ".join(diffs or ["parameters differ"])


def verify_bond_params(h5_file, group_name, sys):
    """Warn if the live bonds have drifted from the stored tables, by *content*.

    Two independent, id-agnostic checks:

    1. **Per stored link**: the parameters of the bond actually attached to that
       particle (matched on the link's partner ids) are compared with the
       parameters the file stores for it.
    2. **Registry**: every distinct ``(class, parameters)`` set in the file's
       class tables must still exist somewhere in ``sys.bonded_inter``.

    Neither keys on the espresso registration id: a restart registers bonds in
    whatever order it rebuilds them, so file id *N* and live id *N* are
    unrelated bonds. Deduplicated registration (one handle shared by several
    stored ids) is likewise silent, because the content is what is compared.

    Advisory only: on resume the file is authoritative and is not rewritten.
    Reports are aggregated -- at most one line per kind of drift, with a count
    and up to ``_MAX_EXAMPLES``=3 examples -- so a systematic mismatch cannot bury
    the log.

    Raises:
        KeyError: If the file holds no ``connectivity/<group_name>/bonds`` table
            (the group was inscribed with ``io_dict['bonds']`` disabled).
    """
    where = f"connectivity/{group_name}"
    if f"{where}/bonds" not in h5_file:
        raise KeyError(f"{h5_file.filename} has no bond topology at '{where}/bonds' (the group "
                       "was inscribed with io_dict['bonds'] disabled); cannot resume with bonds.")
    # 1. Read the stored links (by particle) and the stored parameter tables.
    stored_params, stored_by_particle = _stored_links_by_particle(h5_file, group_name)

    live_ids = set()
    if len(sys.part):
        live_ids = {int(x) for x in np.asarray(sys.part.all().id).reshape(-1)}

    # 2. Per-link content comparison: match each stored link against the live
    # bonds on the same partners, by parameter signature rather than by id.
    missing_particles, drifted, unmatched = [], [], []
    n_links = 0
    for pid, links in stored_by_particle.items():
        n_links += len(links)
        if pid not in live_ids:
            missing_particles.append(pid)
            continue
        live_by_partners = _live_links_of(sys.part.by_id(pid))
        for partners, bond_id, cls, kw in links:
            stored_sig = _param_signature(cls, kw)
            pool = live_by_partners.get(partners)
            if pool and stored_sig in pool:
                pool.remove(stored_sig)
            elif pool:
                drifted.append((pid, partners, bond_id, stored_sig, pool.pop(0)))
            else:
                unmatched.append((pid, partners, bond_id))

    if missing_particles:
        logging.warning(
            "Bond verification: %d of %d particles that own stored bonds in %s do not "
            "exist in the live system (e.g. ids %s); their links were not checked.",
            len(missing_particles), len(stored_by_particle), where,
            missing_particles[:_MAX_EXAMPLES])
    if unmatched:
        logging.warning(
            "Bond verification: %d of %d stored links in %s have no live bond on the "
            "same partners, so the live topology differs from the file (e.g. %s).",
            len(unmatched), n_links, where,
            [f"particle {pid} -> {partners} (file bond_id {bid})"
             for pid, partners, bid in unmatched[:_MAX_EXAMPLES]])
    if drifted:
        logging.warning(
            "Bond verification: %d of %d stored links in %s are attached to a live bond "
            "with different parameters (e.g. %s). The file is authoritative and is not "
            "rewritten.",
            len(drifted), n_links, where,
            [f"particle {pid} -> {partners} (file bond_id {bid}): "
             f"{_drift_description(stored_sig, live_sig)}"
             for pid, partners, bid, stored_sig, live_sig in drifted[:_MAX_EXAMPLES]])

    # 3. Registry check: every distinct stored parameter set must still exist
    # somewhere in sys.bonded_inter, regardless of which particle owns it.
    file_sigs = {}
    for bond_id, (cls, kw) in stored_params.items():
        file_sigs.setdefault(_param_signature(cls, kw), []).append(bond_id)
    live_sigs = {_param_signature(type(handle), handle.get_params())
                 for _, handle in _registered_bonds(sys)}
    absent = [sig for sig in file_sigs if sig not in live_sigs]
    if absent:
        logging.warning(
            "Bond verification: %d of %d distinct parameter sets stored in %s are not "
            "registered in sys.bonded_inter (e.g. %s).",
            len(absent), len(file_sigs), where,
            [f"{sig[0]}{dict(sig[1])} (file bond_id(s) {file_sigs[sig][:_MAX_EXAMPLES]})"
             for sig in absent[:_MAX_EXAMPLES]])
    live_only = live_sigs - set(file_sigs)
    if live_only:
        logging.info(
            "Bond verification: %d live parameter set(s) are not in %s; they were "
            "registered after the topology was captured.",
            len(live_only), where)
    if not (missing_particles or unmatched or drifted or absent):
        logging.debug("Bond verification: %d stored links and %d distinct parameter "
                      "sets in %s all match the live system.",
                      n_links, len(file_sigs), where)


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #

def read_bond_params(h5_file, group_name):
    """``{bond_id: (bond_class, {param: value})}`` from the class tables of ``pressomancy/<group>/bond_params``.

    The class tables are the compound datasets there; the ``*_id`` arrays
    (plain int32) are skipped.
    """
    table = {}
    for type_name, dset in h5_file[f"pressomancy/{group_name}/bond_params"].items():
        if dset.dtype.names is None:
            continue
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


def read_link_tables(h5_file, group_name):
    """``[(owner_ids, partner_ids, bond_ids), ...]``, one entry per stored link table, as particle ids.

    Tables come in arity order (bonds, angles, dihedrals; absent ones skipped),
    rows in stored order: ``owner_ids`` (L,), ``partner_ids`` (L, k), ``bond_ids``
    (L,). The column -> id translation of the read side: the file's indices are
    resolved through ``particles/<group>/id/value[0]`` (the id column is constant
    across frames) before anything downstream sees them.
    """
    conn_grp = h5_file[f"connectivity/{group_name}"]
    params_grp = h5_file[f"pressomancy/{group_name}/bond_params"]
    ids = np.asarray(h5_file[f"particles/{group_name}/{element_name('id')}/value"][0]).reshape(-1)
    tables = []
    for _arity, (table_name, id_name) in sorted(LINK_TABLES.items()):
        if table_name not in conn_grp:
            continue
        links = ids[np.asarray(conn_grp[table_name][...], dtype=np.intp)]
        tables.append((links[:, 0], links[:, 1:], np.asarray(params_grp[id_name][...])))
    return tables


def read_bonds(h5_file, group_name):
    """Yield ``(owner_id, partners, bond_id)`` for every stored link of ``group_name``.

    Tables are visited in arity order (bonds, angles, dihedrals), rows in stored
    order. ``partners`` is a tuple of particle ids, so angle and dihedral bonds
    come back intact; ``bond_id`` indexes ``read_bond_params(h5_file, group_name)``.
    """
    for owners, partners, bond_ids in read_link_tables(h5_file, group_name):
        for owner, row, bond_id in zip(owners, partners, bond_ids):
            yield int(owner), tuple(int(x) for x in row), int(bond_id)
