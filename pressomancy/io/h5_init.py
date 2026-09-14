'''
Seeding a simulation from an existing HDF5 file.

``H5Init`` holds the source-file parameters declared by ``set_init_src`` and the
two readers built on them: ``set_prop_from_src`` copies particle properties out
of a source file onto already-built objects, and ``get_pos_ori_from_src``
supplies positions and orientations for ``Simulation.set_objects(mode='INIT_SRC')``.

This is the read-to-initialise half of the HDF5 layer, distinct from the write
half in :mod:`pressomancy.io.h5_writer`. ``Simulation`` holds one ``H5Init`` and
delegates to it.
'''
import logging
import sys as sysos

import h5py
import numpy as np

from pressomancy.analysis import H5DataSelector


class H5Init:
    """Source-file parameters and the readers that consume them."""

    def __init__(self, owner):
        self._owner = owner
        self.src_params_set = False
        self.src_path_h5 = None
        self.pos_ori_src_type = ['real', ]
        self.type_to_type_map = []
        self.prop_to_prop_map = []

    @property
    def sys(self):
        return self._owner.sys

    @property
    def part_types(self):
        return self._owner.part_types

    def set_init_src(self, path, pos_ori_src_type=['real',], type_to_type_map=[], prop_to_prop_map=[], declare_types=[]):
        self.src_path_h5=path
        self.pos_ori_src_type=pos_ori_src_type
        self.type_to_type_map=type_to_type_map
        self.prop_to_prop_map=prop_to_prop_map
        self.src_params_set=True
        for typ_decl in declare_types:
            for x,y in typ_decl.items():
                self.part_types[x]=y

    def set_prop_from_src(
    self,
    registered_objs=None,
    time_step: int = -1,
):
        """
        Update local particle properties from an HDF5 source file for a given group type. This loads a particle group from an HDF5 file, validates that requested type mappings exist both locally and in the source, then iterates over each group instance (connectivity ID) to copy properties from source particles into the corresponding local particles.

        Parameters
        ----------
        registered_objs : iterable
            Collection of local group instances (e.g., Filament objects) whose
            owned particles will be updated.

        time_step : int, optional
            Index of the source time step to read. ``-1`` selects the last frame.

        Notes
        -----
        * For the `(dip -> director)` mapping, vectors are normalized via
        `np.linalg.norm`. Zero-norm dipoles would raise a warning or yield NaNs
        if present—consider guarding if your data can contain zeros.

        Raises
        ------
        AssertionError
            If a mapped type is not present locally or in the source.
        KeyError
            If lookups via `self.part_types` fail (depends on your implementation).
        """

        # Open the source HDF5 and select the data group matching the requested type.
        if not (self.src_params_set==True):
            raise RuntimeError('src_params_set must be set before calling this method')
        with h5py.File(self.src_path_h5, "r") as src_file:
            src_data_grp = H5DataSelector(src_file, particle_group=registered_objs[0].__class__.__name__)

            # Discover the set of numeric type IDs present in the source for this group.
            all_src_types_numeric = np.unique(src_data_grp.type)

            # Validate that each requested (src_type -> local_type) exists both locally and in the source file.
            for src_typ, loc_typ in self.type_to_type_map:
                if not (loc_typ in self.part_types and src_typ in self.part_types):
                    raise KeyError(f"local type {loc_typ} or source type {src_typ} not found in "
                    f"simulation part types {self.part_types}")
                if not (self.part_types[src_typ] in all_src_types_numeric):
                    raise KeyError(f"source type {src_typ} with numeric id {self.part_types[src_typ]} "
                    f"not found in source data part types {all_src_types_numeric}")
            logging.info(f"simulation contains types: {self.part_types}")
            logging.info(
                f"src datafile contains types: {self.part_types.key_for(all_src_types_numeric)}"
            )

            # The two maps are consumed pairwise below, so a length mismatch would
            # silently drop the tail of the longer one.
            if not (len(self.type_to_type_map) == len(self.prop_to_prop_map)):
                raise ValueError(f"type_to_type_map ({len(self.type_to_type_map)} entries) and prop_to_prop_map "
                f"({len(self.prop_to_prop_map)} entries) must be the same length; they are "
                f"applied as aligned pairs.")

            # Iterate over each connectivity group (i.e., each distinct instance of the group).
            for loc_obj in registered_objs:

                # Apply each aligned (type mapping, property mapping) pair.
                for (src_typ, loc_typ), (prop_src, prop_loc) in zip(
                    self.type_to_type_map, self.prop_to_prop_map
                ):
                    logging.info(
                        f"Working on {loc_obj.__class__.__name__}: {loc_obj.who_am_i} type {src_typ}->{loc_typ} prop {prop_src}->{prop_loc}"
                    )

                    # Select source particles at the requested time step that belong to this group instance (connectivity == grp_id), and match the numeric type ID mapped from src_typ.
                    part_slice = src_data_grp.timestep[time_step].select_particles_by_object(
                        object_name=loc_obj.__class__.__name__,
                        connectivity_value=loc_obj.who_am_i,
                        predicate=lambda subset: subset.type == self.part_types[src_typ],
                    )

                    # Filter local particle handles to those of the destination type.
                    part_hndls = [
                        x for x in loc_obj.get_owned_part()[0]
                        if x.type == self.part_types[loc_typ]
                    ]
                    # Copy properties from source to local, element-wise.
                    for local, src in zip(part_hndls, part_slice.particles):
                        if prop_src == "dip" and prop_loc == "director":
                            # Normalize dipole to unit vector for director. Copy first:
                            # the selector's array must not be divided in place.
                            val = np.array(getattr(src, prop_src), dtype=float)
                            norm = np.linalg.norm(val)
                            if norm == 0.0:
                                raise ValueError(
                                    f"dip moment magnitude is 0 for particle id={getattr(src, 'id', src)} "
                                    f"and cannot be used to infer particle orientation!"
                                )
                            val /= norm
                            setattr(local, prop_loc, val)
                        else:
                            setattr(local, prop_loc, getattr(src, prop_src))

    def get_pos_ori_from_src(self, registered_objs, time_step: int = -1):

        """
        Load particle positions and orientations for a set of registered local objects
        from an external HDF5 source.

        This function opens the HDF5 file specified by ``self.src_path_h5``, selects the
        particle group matching the class of the first object in ``registered_objs``,
        verifies that all requested source type names (``self.pos_ori_src_type``) exist
        both in the local type map (``self.part_types``) and in the source data
        (by numeric ID), and then, for each local object, selects the subset of source
        particles connected to that object and filtered by the requested types.

        For each object, positions are read directly. Orientations are read from the
        ``director`` property when available; if it is absent, orientations are derived
        by normalizing the ``dip`` vectors. Zero-magnitude dip vectors are rejected.

        Parameters
        ----------
        registered_objs : iterable
            Local group instances (e.g., Filament objects) whose owned particles
            should be updated. All objects are assumed to belong to the same particle
            group (same class).
        time_step : int, optional
            Index of the source time step to read. Use ``-1`` to select the last
            available frame (default). The underlying selector must support negative
            indexing if ``-1`` is used.

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            Two lists, each with one entry per object in ``registered_objs``:
            ``(positions_per_obj, orientations_per_obj)``.

            - ``positions_per_obj[i]`` has shape ``(Ni, 3)`` with particle positions
            for the *i*-th object.
            - ``orientations_per_obj[i]`` has shape ``(Ni, 3)`` with unit orientation
            vectors (either the stored ``director`` or normalized ``dip``).

        Raises
        ------
        AssertionError
            If any requested type name in ``self.pos_ori_src_type`` is missing from
            ``self.part_types``, or if its corresponding numeric ID is not present in
            the source data for the selected group.
        ValueError
            If orientation must be inferred from ``dip`` and one or more dip vectors
            have zero (or nonpositive) magnitude.
        AttributeError
            Raised by ``H5DataSelector`` when ``director`` is not a stored property
            for the selected group; caught internally to fall back to normalized
            ``dip``, and logged via ``sysos.exc_info()`` in that fallback path.
        Exception
            Other exceptions may propagate from ``H5DataSelector`` or the predicate.

        Notes
        -----
        - Filtering is performed with a predicate equivalent to
        ``np.isin(subset.type, allowed_type_ids)`` where
        ``allowed_type_ids = [self.part_types[name] for name in self.pos_ori_src_type]``.
        - If ``director`` is not present, orientations are computed as
        ``dip / ||dip||`` with an explicit check against zero norms.
        - This method assumes ``self.src_params_set`` is ``True`` and that
        ``self.src_path_h5`` points to a readable HDF5 file.
        - Logging includes a summary of local and source type mappings and per-object
        load operations.

        See Also
        --------
        H5DataSelector.select_particles_by_object : Used to gather per-object subsets.
        """

        # Open the source HDF5 and select the data group matching the requested type.
        if not (self.src_params_set==True):
            raise RuntimeError('src_params_set must be set before calling this method')
        with  h5py.File(self.src_path_h5, "r") as src_file:
            src_data_grp = H5DataSelector(src_file, particle_group=registered_objs[0].__class__.__name__)

            # Discover the set of numeric type IDs present in the source for this group.
            all_src_types_numeric = np.unique(src_data_grp.type)
            requested_names = set(self.pos_ori_src_type)
            available_names = set(self.part_types.keys())

            # 1) every requested name must exist
            missing_names = requested_names - available_names
            if not (not missing_names):
                raise KeyError(f"source type(s) {sorted(missing_names)} not found in "
                f"simulation part types {sorted(available_names)}")
            # 2) the numeric ids for those names must exist in the source data
            requested_ids = {self.part_types[name] for name in requested_names}
            available_ids = set(all_src_types_numeric)

            missing_ids = requested_ids - available_ids
            if not (not missing_ids):
                raise KeyError("source data is missing type id(s): "
                f"{sorted(missing_ids)} "
                f"({[self.part_types.key_for(i) for i in sorted(missing_ids)]} by name) "
                f"not found in source data part types {sorted(available_ids)}")
            logging.info(f"simulation contains types: {dict(self.part_types)}")

            positions_per_obj,ori_per_obj=[],[]
            for loc_obj in registered_objs:
                logging.info(
                    f"Loading data for {loc_obj.__class__.__name__}: {loc_obj.who_am_i} from SRC part type {self.pos_ori_src_type}."
                )
                # Select source particles at the requested time step that belong to this group instance (connectivity == loc_obj.who_am_i), with the correct pos_ori_src_type.
                allowed_types=[self.part_types[x] for x in self.pos_ori_src_type]
                part_slice = src_data_grp.timestep[time_step].select_particles_by_object(
                    object_name=loc_obj.__class__.__name__,
                    connectivity_value=loc_obj.who_am_i,
                    predicate=lambda subset: np.isin(subset.type, allowed_types),
                )
                positions_per_obj.append(part_slice.pos)
                try:
                    ori_per_obj.append(part_slice.director)
                except AttributeError:
                    exc_type, value, traceback = sysos.exc_info()
                    logging.debug("Failed with exception [%s,%s ,%s]" %
                        (exc_type, value, traceback))
                    val = part_slice.dip
                    norm = np.linalg.norm(val,axis=1,keepdims=True)
                    if np.any(norm==0.0):
                        raise ValueError(f"dip moment magnitude is 0 and cannot be used to infer particle orientation!")
                    val /= norm
                    ori_per_obj.append(val)
                    continue
        return positions_per_obj, ori_per_obj
