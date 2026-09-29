'''
The core of pressomancy: the ``Simulation`` class that wraps an ESPResSo
``System`` handle and manages everything built on top of it.

``Simulation`` is instantiated as a process-wide singleton via the
``ManagedSimulation`` decorator (see :mod:`pressomancy.infra`).
It owns particle-type bookkeeping, storing/placing/deleting
``Simulation_Object`` instances (see :mod:`pressomancy.object_classes`),
WCA/Lennard-Jones interactions, box-wall constraints, LB fluid setup,
external magnetic fields, HDF5 (H5MD-style) I/O for particle groups and
arbitrary observables, resuming into an existing file (``LOAD``/``LOAD_NEW``)
and seeding a system from a source file (``load_from_src``).
'''
import espressomd
from espressomd import shapes
import sys as sysos
import numpy as np
import os
import warnings
from itertools import combinations_with_replacement
from pressomancy.object_classes import *
from pressomancy.infra import (ManagedSimulation, MissingFeature, TypeDictSafe,
                               api_agnostic_feature_check)
from pressomancy.geometry import (add_box_constraints_func, remove_box_constraints_func,
                                  generate_random_unit_vectors, normalize_vectors,
                                  partition_cuboid_volume,
                                  get_cross_lattice_nonintersecting_volumes)
from pressomancy.magnetodynamics import configure_magnetization, contraction_ratio
from pressomancy.io.write import H5Writer
from pressomancy.io.init import H5Init
import logging
from collections import Counter
import inspect

@ManagedSimulation
class Simulation():
    """
    A singleton class that manages an arrangement of objects inside the ESPResSo molecular dynamics framework.

    `Simulation` wraps the ESPResSo `System` handle and owns everything built on top of it: particle-type
    bookkeeping, the lifetime of `Simulation_Object` instances, non-bonded interactions, box-wall
    constraints, LB flow, external magnetic fields, and HDF5 (H5MD-style) output.

    Instantiation is managed by the `ManagedSimulation` decorator, so `Simulation(box_dim=...)` returns
    the decorator, not a bare `Simulation`; attribute access is forwarded. A second instantiation raises
    `SimulationExistsException`. Call `reinitialize_instance()` to reset state while keeping the same
    ESPResSo system.

    Attributes:
        no_objects (int): The number of objects currently stored in the simulation.
        objects (list): A list of objects stored in the simulation.
        part_types (TypeDictSafe): Maps a type name (`str`) to its integer espresso type id, as a
            strict bijection: reading an unknown name raises `KeyError`, and reassigning a name or
            reusing a number raises `RuntimeError`.
        seed (int): A random seed for reproducibility, generated at initialization.
        io_dict (dict): HDF5 output state. Notable keys:
            `properties` -- the (name, dim, dtype) tuples written for every particle each frame;
            `bonds` -- set to True *before* inscribing to write bond topology into
                `/connectivity/<Group>/bonds`.
                Topology is captured once, at inscription, and has no time
                axis, so all bonds must exist before `inscribe_part_group_to_h5` is called; a later change
                is detected and raises.
        author_name, author_email (str): Author metadata written to newly created HDF5 files.

    Methods:
        Setup and system configuration
            set_sys(timestep, min_global_cut): configure cell system, time step and
                the virtual-site scheme. NOT called automatically -- callers must invoke it.
            set_author(name, email): author metadata for new HDF5 files.
            rebind_sys(new_sys): rebind to a new espresso handle after a checkpoint load.
            modify_system_attribute(requester, attribute_name, action): permissioned mutation hook
                used by objects.

        Object management
            store_objects(iterable_list, report): register objects and their particle types.
            set_objects(objects): partition the box and place objects without overlap.
            place_objects(objects, positions, orientations): place at given coordinates, must have the positions for all the particles each object needs to place, no overlap check. **Prefer to use set_objects(objects)**.
            sanity_check(object): verify the build has the object's required features.
            mark_for_collision_detection(object_type, part_type): mark objects for covalent bonding.

        Interactions and constraints
            set_steric(key, wca_eps, sigma) / set_steric_custom(pairs, wca_eps, sigma)
            set_vdW(key, lj_eps, lj_sigma) / set_vdW_custom(pairs, lj_eps, lj_sigma, lj_cutoffs, r_min)
            add_box_constraints(...) / remove_box_constraints(...): flat walls on the box faces.
            avoid_explosion(F_TOL, MAX_STEPS, F_incr, I_incr): force-capped warmup.
            thermostat_is_off(): True when no thermostat mode is active.

        Magnetism
            init_magnetic_inter(actor_handle): attach a dipolar solver.
            set_magnetization_model(part_list, model, dipm_sat, mag_susc_0): declarative, called once.
            probe_magnetization_convergence(part_list, n_iter, tol): contraction diagnostics.
            set_H_ext(H) / get_H_ext(): external homogeneous field.

        Lattice Boltzmann
            init_lb(kT, agrid, dens, visc, gamma, timestep), create_flow_channel(slip_vel).

        HDF5 input/output
            inscribe_part_group_to_h5(group_type, h5_data_path, mode, force_resize_to_size, rewind_to_step) / 
            inscribe_observable_group_to_h5(observable_defs, h5_data_path, mode, force_resize_to_size,
                rewind_to_step): register the streams; `io_dict['properties']` (id, type, pos)
                is validated, and on LOAD/LOAD_NEW compared with the file (mismatch raises). The
                two truncation forms (frame count / stored step, exclusive) run before the load checks.
            write_part_group_to_h5(step) / write_observable_group_to_h5(step) / 
                write_registered_to_h5(step): append one frame; `step` must strictly increase.
            write_checkpoint(group_type, path, step): a verified one-frame restart file (full
                float64 state, bonds, time and thermostat counters), written via `path + '.tmp'`
                and `os.replace` (corruption safe); `restart_from_checkpoint` reads it back and
                `inscribe_*(..., rewind_to_step=step)` re-aligns the trajectory.
            restart_from_checkpoint(objects, path, src_to_loc, bonds, place_from, r_cut_override):
                mirrors `load_from_src` over every stored type with the fixed state list
                (`pressomancy.io.CHECKPOINT_PROPERTIES`), then `sys.time` and the thermostat Philox
                counters (set the thermostat first); returns the stored step. **First run afterwards**:
                `integrator.run(n, reuse_forces=True)`.
            load_from_src(objects, path, src_to_loc, bonds, place_from, r_cut_override,
                frame, step, time): the one call that seeds a system from a file (declare -> place
                -> copy properties -> restore bonds); returns the bond links added. The individual
                steps live on `pressomancy.io.init.H5Init` (`self._h5_init`).
            Placement is optional: give `place_from` (a list of SOURCE types) when the file
                should place your objects, which requires one stored particle of those types per
                monomer; leave it out when you built the tree yourself (compound objects, running
                systems) and only its state comes from the file.
            `src_to_loc` is one dict, `{type pair(s): [(src_prop, loc_prop), ...]}`, every tuple
                ordered (source, local): a key is one `(src_type, loc_type)` pair or a tuple of
                them, its value the property pairs copied for those type pairs (`[]` = pair the
                particles, copy nothing). It is also the type pairing `set_bonds_from_src` uses,
                so it must be non-empty when `bonds` is true. Grammar and recipes:
                `pressomancy.io.init`.
            get_prop_from_src(objects, path, src_type, prop, frame, step, time): read one stored
                property back per object (one `(N_i, dim)` array each, stored order) without
                touching the system -- for everything that is not a 1:1 column copy and/or compatible
                with `load_from_src`.
            Frame selectors: `frame` (INDEX, -1 = last), `step` (stored step VALUE), `time` (stored
                time VALUE, nearest within relative 1e-6); all None = last frame, multiple given
                must resole to same frame. Source type names resolve through the file's own
                `parameters/pressomancy/part_types`, local names through `part_types`. A SOURCE type
                may also be given as an `int`, the numeric type stored in the file, which needs no
                table (old files); local types stay names. The local object tree must be built in the
                same order as the source's (who_am_i contract; see pressomancy.io.init).

    Notes:
        - **ESPResSo 5.x is the only supported version.** Required ESPResSo build features are
          listed in the README.
        - The ESPResSo system handle is created and owned by the decorator; `self.sys` is bound at
          instantiation.
        - Objects must be built through the `Simulation_Object` metaclass to be safely usable here.
        - `objects`, `no_objects` and `part_types` are plain attributes with no write guard.
    """

    object_permissions=['part_types']
    _sys=espressomd.System
    def __init__(self, box_dim, use_espresso_checkpoint_system=None):
        # Object bookkeeping
        self.no_objects = 0
        self.objects = []
        self.part_types = TypeDictSafe()

        # espresso system is accessed by .sys, e.g. self.sys.part.all()
        # I/O. The writer owns io_dict and the author metadata; Simulation exposes
        # them as properties so the public API is unchanged.
        self._h5_writer = H5Writer(self)
        self._h5_init = H5Init(self)

        # System numbers stuff
        self.seed = int.from_bytes(os.urandom(2), sysos.byteorder)
        self.kT = 1.
        # self.sys=espressomd.System(box_l=box_dim) is added and managed by the singleton decrator!

    # ------------------------------------------------------------------
    # Seeding from an HDF5 source file. Implementation in io/init.py.
    # ------------------------------------------------------------------
    def load_from_src(self, objects, path, src_to_loc, bonds=False, place_from=None,
                      r_cut_override=0.0, frame=None, step=None, time=None):
        """Seed `objects` from a file in one call; returns the bond links added.
        See H5Init.load_from_src.

        Give `place_from` (a list of SOURCE types: names, or the ints stored in the file) when the
        file should place your objects, which requires one stored particle of those types per
        monomer; leave it out when you built the tree yourself (compound objects, running systems).
        """
        return self._h5_init.load_from_src(objects, path, src_to_loc, bonds=bonds,
                                           place_from=place_from, r_cut_override=r_cut_override,
                                           frame=frame, step=step, time=time)

    def restart_from_checkpoint(self, objects, path, src_to_loc=None, bonds=False, place_from=None,
                                r_cut_override=0.0):
        """Restore `objects`, `sys.time` and the thermostat counters from a `write_checkpoint` file;
        returns its step.
        See H5Init.restart_from_checkpoint.

        Every stored type is paired with the local type of the same name and gets the fixed
        state list (`pressomancy.io.CHECKPOINT_PROPERTIES`, in that order); `src_to_loc` adds
        extras (e.g. `fix`). Set the thermostat with its seed BEFORE calling, and make the first
        call afterwards `integrator.run(n, reuse_forces=True)` for an exact continuation.
        """
        return self._h5_init.restart_from_checkpoint(objects, path, src_to_loc=src_to_loc, bonds=bonds,
                                                    place_from=place_from, r_cut_override=r_cut_override)

    def get_prop_from_src(self, objects, path, src_type, prop, frame=None, step=None, time=None):
        """One stored property of one source type, per object, changing nothing.
        See H5Init.get_prop_from_src.

        Declares `path` as the source (as `load_from_src` does), then reads
        `particles/<Group>/<prop>/value` for the `src_type` particles owned by each
        object, in stored order: one `(N_i, dim)` array per object.
        """
        self._h5_init.set_init_src(path)
        return self._h5_init.get_prop_from_src(objects, src_type, prop,
                                               frame=frame, step=step, time=time)

    def set_sys(self, timestep=0.01, min_global_cut=3.0):
        '''
        Set espresso cellsystem params, and import virtual particle scheme.

        Note: this is NOT run automatically on initialisation -- callers must invoke it
        explicitly.

        :param timestep: float (=0.01) | integration time step. Note the name: espresso's own
            attribute is `time_step`, and passing `time_step=` here is silently ignored.
        :param min_global_cut: float (=3.0) | minimum global interaction range. Together with the
            skin (fixed at 0.5) this is not guaranteed optimal and should be tuned per simulation.
        :return: None
        '''
        np.random.seed(seed=self.seed)
        logging.info(f'core.seed: {self.seed}')
        self.sys.periodicity = (True, True, True)
        self.sys.time_step = timestep
        self.sys.cell_system.skin = 0.5
        self.sys.min_global_cut = min_global_cut
        if not (api_agnostic_feature_check('VIRTUAL_SITES_RELATIVE')):
            raise MissingFeature('VirtualSitesRelative must be set. If not, anything involving virtual particles will not work correctly, but it might be very hard to figure out why. I have wasted days debugging issues only to remember i commented out this line!!!')
        logging.info(f'System params have been autoset. The values of min_global_cut and skin are not guaranteed to be optimal for your simulation and should be tuned by hand!!!')

    def modify_system_attribute(self, requester, attribute_name, action):
        """
        Validates and modifies a Simulation attribute if allowed by the permissions.

        :param requester: The object requesting the modification.
        :param attribute_name: str | The name of the attribute to modify.
        :param action: callable | A function that takes the current attribute value as input and modifies it.
        :return: None
        :raises PermissionError: if `attribute_name` is not in `object_permissions` (or not an attribute).
        """
        if not (attribute_name in self.object_permissions and hasattr(self, attribute_name)):
            raise PermissionError(
                f"{requester!r} may not modify Simulation.{attribute_name!r}; "
                f"modifiable attributes are {self.object_permissions}.")
        action(getattr(self, attribute_name))

    def sanity_check(self, object):
        '''
        Method that checks if the object has the required features to be stored in the simulation. If the object has the required features it is stored in the self.objects list.
        '''

        missing_features = [feature for feature in object.required_features if not api_agnostic_feature_check(feature)]
        if missing_features:
            raise MissingFeature(f"{object.__class__.__name__} requires features: {object.required_features}.\nMissing required features: {', '.join(missing_features)}.")

    def store_objects(self, iterable_list, report=True):
        '''
        Method stores objects in the self.objects dict, if the object has a n_part and part_types attributes,
        and the list of objects passed to the method is commensurate with the system level attribute n_tot_parts.
        Populates the self.part_types attribute with types found in the objects that are stored.
        All objects that are stored should have the same types stored, but this is not checked explicitly

        :raises ValueError: if an object is already stored, or only part of an object's associated objects are
        :raises MissingFeature: if the build lacks a feature an object requires (see sanity_check)

        A refused call stores nothing: objects, no_objects and part_types are rolled back, including the
        associated objects stored along the way.
        '''
        objects_before, no_objects_before, part_types_before = list(self.objects), self.no_objects, dict(self.part_types)
        temp_dict={}
        try:
            for element in iterable_list:
                if element.params['associated_objects'] != None:
                    check_any=any(associated in self.objects for associated in element.params['associated_objects'])
                    if check_any:
                        check_all=all(associated in self.objects for associated in element.params['associated_objects'])
                        if not check_all:
                            raise ValueError(f"Some associated objects {element.params['associated_objects']} but not all associated objects are stored in the simulation. This is a sign that smth major is fucked...Suffer in silence.")
                    else:
                        self.store_objects(element.params['associated_objects'],report=False)
                if not (element not in self.objects):
                    raise ValueError("Lists have common elements!")
                self.sanity_check(element)
                element.modify_system_attribute = self.modify_system_attribute
                self.objects.append(element)
                for key, val in element.part_types.items():
                    temp_dict[key]=val
                self.no_objects += 1
        except Exception:
            self.objects[:] = objects_before
            self.no_objects = no_objects_before
            self.part_types.clear()
            self.part_types.update(part_types_before)
            raise
        self.part_types.update(temp_dict)
        if report:
            names = [element.__class__.__name__ for element in self.objects]
            counts = Counter(names)
            formatted = ", ".join(f"{count} {name}" for name, count in counts.items())
            logging.info(f"{formatted} stored")

    def set_objects(self, objects):
        """Set objects' positions and orientations in a box. Defaults to the Simulation box.
        This method places objects in the simulation box using a partitioning scheme: each object
        gets a spherical volume of diameter ``params['size']`` (its meaning: `ObjectConfigParams`),
        and the volumes are chosen so that they intersect neither each other nor any obstacle.
        
        The obstacles are derived at each call from every stored object that owns particles,
        however it was placed (`set_objects`, `place_objects`, `load_from_src`,
        `restart_from_checkpoint`): one sphere of its ``size`` at its particles' mean.
        Particles not owned by a stored object are ignored.
        Objects with ``size=None`` (`Elastomer`) lay themselves out: they skip the lattice, each
        one's ``build_function`` is called directly and its result placed, and while one of them
        owns particles nothing else can be placed. They are never checked against obstacles
        (other stored objects' placement spheres), in either direction: this method does not test
        their own placement against existing obstacles, nor can a later call place anything next
        to one that already owns particles (it raises instead, see below). The only guard against
        overlap is `Elastomer`'s own empty-``box_E`` check in `set_object`, and it sees particle
        centres only, not obstacle spheres.
        To place objects at the coordinates stored in an HDF5 file use `load_from_src` and 
        specify `place_from`.
        Parameters
        ----------
        objects : list
            A list of simulation objects to place. All objects must be instances of the same type
            and share ``params['size']`` and the ``build_function`` fields ``num_monomers``,
            ``spacing`` and ``monomer_size``.

        Raises
        ------
        ValueError
            If not all objects are of the same type, or if they differ in ``params['size']`` or in
            ``build_function.num_monomers``/``spacing``/``monomer_size`` (the partition uses
            objects[0]'s size and build function for all), or if some of them already own
            particles (`set_objects` only places new objects), or if a stored object with
            ``size=None`` owns particles.

        Warns
        -----
        UserWarning
            If a just-placed object's particle centres reach beyond ``size / 2`` from their mean
            (not checked for ``size=None`` or for associated objects).

        Notes
        -----
        The method uses partition_cuboid_volume to generate positions and orientations, and
        get_cross_lattice_nonintersecting_volumes to discard sites that intersect other objects.
        If too few sites are free, it retries adjusting the search space (increasing the factor).
        The check is volume-level: the spheres of diameter ``size`` must not intersect;
        the particles inside them are not compared.
        """

        # Ensure all objects are of the same type.
        if not (all(isinstance(item, type(objects[0])) for item in objects)):
            raise ValueError("Not all items have the same type!")
        # Placement uses objects[0]'s size and build_function for every item.
        layouts = {(obj.params['size'], obj.build_function.num_monomers, obj.build_function.spacing, obj.build_function.monomer_size) for obj in objects}
        if len(layouts) > 1:
            raise ValueError("Items differ in (size, num_monomers, spacing, monomer_size): "
                             f"{sorted(layouts, key=str)}; place them in separate set_objects calls.")
        if any(obj.get_owned_part()[0] for obj in objects):
            raise ValueError("Some items already own particles; set_objects only places new objects.")
        size = objects[0].params['size']
        # This call's objects own nothing yet; sub-objects are counted as well as their parent
        # (redundant but safe). p.pos is unfolded, so the mean stays contiguous across the box edge.
        placed = {}
        for obj in self.objects:
            pos = [p.pos for p in obj.get_owned_part()[0]]
            if pos:
                if obj.params['size'] is None:
                    raise ValueError(f"{type(obj).__name__} has no placement size (it lays itself out); "
                                     "set_objects cannot place anything next to it.")
                placed.setdefault(obj.params['size'], []).append(np.mean(pos, axis=0))
        if size is None:
            # Use their own partitioning function directly in build_function.
            # May not check against overlaps. E.g. Elastomer is intended to only have on instance set
            for obj in objects:
                orientations, positions = obj.build_function(
                    center=None, num_monomers=obj.build_function.num_monomers, sphere_radius=None,
                    spacing=obj.build_function.spacing, box_dim=self.sys.box_l)
                self.place_objects([obj], [positions], [orientations])
            return
        factor = 1
        while True:
            centers, positions, orientations = partition_cuboid_volume(
                box_dim=self.sys.box_l,
                num_spheres=len(objects) * factor,
                sphere_diameter=size,
                routine_per_volume=objects[0].build_function
            )
            free = np.ones(len(centers), dtype=bool)
            for placed_size, placed_centers in placed.items():
                res = get_cross_lattice_nonintersecting_volumes(
                    current_lattice_centers=centers,
                    current_lattice_diam=size,
                    other_lattice_centers=np.asarray(placed_centers),
                    other_lattice_diam=placed_size,
                    box_dim=self.sys.box_l
                    )
                free &= [all(res[i]) for i in range(len(centers))]
            if free.sum() >= len(objects):
                break
            factor += 1
            logging.info('Failed to find enough space; (found, needed): (%d, %d). Will retry by requesting %d times the number of parts', free.sum(), len(objects),factor)
        keep = np.flatnonzero(free)[:len(objects)]
        self.place_objects(objects, positions[keep], orientations[keep])
        # Warn for objects with particles outise their size
        reach = max(np.linalg.norm(pos - pos.mean(axis=0), axis=1).max()
                    for pos in (np.array([p.pos for p in obj.get_owned_part()[0]]) for obj in objects))
        if reach > size / 2:
            warnings.warn(f"{type(objects[0]).__name__} particle centres reach {reach:.3g} from the "
                          f"object's centre, beyond size/2 = {size / 2:.3g}: placed objects may overlap. "
                          "Set `size` to bound the build (see `size` in ObjectConfigParams).",
                          stacklevel=2)

    def place_objects(self, objects, positions, orientations=None):
        """Place objects' positions and orientations in a box.

        If orientations are not provided, random unit vectors are generated.
        This method does not guarantee non-overlapping of objects, in any way.

        Parameters
        ----------
        objects : list or array-like
            List of simulation objects to place. Can be a single object or multiple.
        positions : array-like of shape (N, 3)
            A list or array of 3D coordinates where each object will be placed.
        orientations : array-like of shape (N, 3), optional
            Orientation vectors for each object. If not provided, random unit vectors are generated.

        Raises
        ------
        ValueError
            If the number of objects, positions, and orientations do not match (checked before
            anything is placed).
        """
        objects= np.atleast_1d(objects)
        if any(obj.get_owned_part()[0] for obj in objects):
            raise ValueError("Some items already own particles; you can only places new objects.")
        if orientations is None:
            orientations = generate_random_unit_vectors(len(positions))
        else:
            orientations = normalize_vectors(orientations)
        if not len(objects) == len(positions) == len(orientations):
            raise ValueError(f"place_objects got {len(objects)} objects, {len(positions)} positions "
                             f"and {len(orientations)} orientations")
        for obj, pos, ori in zip(objects, positions, orientations):
            obj.set_object(pos, ori)
        names = [element.__class__.__name__ for element in objects]
        counts = Counter(names)
        formatted = ", ".join(f"{count} {name}" for name, count in counts.items())
        logging.info(f"{formatted} set!!!")

    def mark_for_collision_detection(self, object_type=Quadriplex, part_type=666):
        if not (any(isinstance(ele, object_type) for ele in self.objects)):
            raise ValueError("method assumes simulation holds correct type object")

        objects_iter = [ele for ele in self.objects if isinstance(ele, object_type)]
        if not (all((hasattr(ob, 'mark_covalent_bonds') and callable(getattr(ob, 'mark_covalent_bonds')))
                   for ob in objects_iter)):
            raise TypeError("method requires that stored objects have mark_covalent_bonds() method")
        self.part_types['marked'] = part_type
        for obj_el in objects_iter:
            obj_el.mark_covalent_bonds(part_type=part_type)

    def init_magnetic_inter(self, solver_handle):
        '''
        Attach a dipolar solver actor.
        '''
        self.sys.magnetostatics.clear()
        self.sys.magnetostatics.solver = solver_handle

        logging.info(f'{solver_handle} magnetic interactions actor initiated')

    def set_steric(self, key=('nonmagn',), wca_eps=1., sigma=1.):
        '''
        Set WCA interactions between particles of types given in the key parameter.

        :param key: tuple of keys from self.part_types | Default only nonmagn WCA
        :param wca_epsilon: float | strength of the steric repulsion.

        :return: None

        Interaction length is allways determined from sigma.
        '''
        logging.info(f'part types available {self.part_types.keys()} ')
        logging.info(f'WCA interactions initiated for keys: {key}')
        for key_el, key_el2 in combinations_with_replacement(key, 2):
            self.sys.non_bonded_inter[
                self.part_types[key_el], self.part_types[key_el2]
                                      ].wca.set_params(epsilon=wca_eps, sigma=sigma)

    def set_steric_custom(self, pairs=[(None, None),], wca_eps=[1.,], sigma=[1.,]):
        """
        Configures custom Weeks-Chandler-Andersen (WCA) interactions for specified particle type pairs.

        This method explicitly sets the WCA interaction parameters (epsilon and sigma) for each pair of particle types provided.
        It ensures that each interaction pair has corresponding epsilon and sigma values.

        :param pairs: list of tuples | List of particle type pairs (keys from `self.part_types`) for which interactions are defined. Defaults to [(None, None)].
        :param wca_eps: list of float | Strength of the WCA repulsion (epsilon) for each pair. Defaults to [1.0].
        :param sigma: list of float | Interaction range (sigma) for each pair. Defaults to [1.0].
        :return: None
        :raises ValueError: If the lengths of `pairs`, `wca_eps`, and `sigma` do not match.
        """
        if not (len(pairs) == len(wca_eps) and len(pairs) == len(
            sigma)):
            raise ValueError('epsilon and sigma must be specified explicitly for each type pair')
        logging.info('WCA interactions initiated')
        for (key_el, key_el2), eps, sgm in zip(pairs, wca_eps, sigma):
            self.sys.non_bonded_inter[self.part_types[key_el], self.part_types[key_el2]
                                      ].wca.set_params(epsilon=eps, sigma=sgm)

    def set_vdW(self, key=('nonmagn',), lj_eps=1., lj_sigma=1.):
        """
        Configures Lennard-Jones (LJ) interactions for specified particle types.

        This method sets the LJ interaction parameters (epsilon and sigma) for particle types listed in the `key` parameter.
        The interaction cutoff is automatically set to 2.5 times the LJ size (sigma).

        :param key: tuple of str | Particle type keys from `self.part_types` for which interactions are defined. Defaults to ('nonmagn',).
        :param lj_eps: float | Strength of the LJ attraction (epsilon). Defaults to 1.0.
        :param lj_sigma: float | Interaction range (sigma). Defaults to 1.0.
        :return: None
        """

        lj_cut = 2.5*lj_sigma
        for key_el, key_el2 in combinations_with_replacement(key, 2):
            self.sys.non_bonded_inter[self.part_types[key_el], self.part_types[key_el2]].lennard_jones.set_params(
                epsilon=lj_eps, sigma=lj_sigma, cutoff=lj_cut, shift=0)
        logging.info(f'vdW interactions initiated initiated for keys: {key}')

    def set_vdW_custom(self, pairs=[(None, None),], lj_eps=[1.,], lj_sigma=[1.,], lj_cutoffs=None, r_min=0):
        """
        Custom setter for Lennard-Jones (LJ) interactions between specified particle type pairs.

        This method allows for the explicit definition of LJ interaction parameters (epsilon and sigma) for each pair of particle types in the simulation.

        :param pairs: list of tuples | Each tuple specifies a pair of keys from `self.part_types` for which interactions are defined. Defaults to [(None, None)].
        :param lj_eps: list of float | Strength of the LJ interaction for each pair. Defaults to [1.0].
        :param lj_sigma: list of float | Interaction range (sigma) for each pair. Defaults to [1.0].
        :return: None

        :raises ValueError: If the lengths of `pairs`, `lj_eps`, and `lj_sigma` are not equal.
        """

        if not (len(pairs) == len(lj_eps) and len(pairs) == len(
            lj_sigma)):
            raise ValueError('epsilon and sigma must be specified explicitly for each type pair')
        if lj_cutoffs is None:
            for (key_el, key_el2), eps, sgm in zip(pairs, lj_eps, lj_sigma):
                lj_cut = 2.5*sgm
                self.sys.non_bonded_inter[self.part_types[key_el], self.part_types[key_el2]].lennard_jones.set_params(
                    epsilon=eps, sigma=sgm, cutoff=lj_cut, shift=0, min=r_min)
        else:
            if not (len(pairs) == len(lj_cutoffs)):
                raise ValueError('cutoffs must be specified explicitly for each type pair')
            for (key_el, key_el2), eps, sgm, cut in zip(pairs, lj_eps, lj_sigma, lj_cutoffs):
                self.sys.non_bonded_inter[self.part_types[key_el], self.part_types[key_el2]].lennard_jones.set_params(
                    epsilon=eps, sigma=sgm, cutoff=cut, shift=0, min=r_min)
        logging.info('vdW interactions initiated!')

    def add_box_constraints(self, wall_type=0, sides=['all'], inter=None, types_=None, object_types=None,
                        bottom=None, top=None, left=None, right=None, back=None, front=None):
        """
        Adds flat wall constraints to the simulation box along the specified sides.

        Thin wrapper over :func:`pressomancy.geometry.add_box_constraints_func`, which
        carries the full parameter documentation -- including the `sides` grammar ('all', 'sides',
        individual faces, and 'no-<side>' exclusions), the per-face position overrides, and how
        `inter` sets up the wall interaction.

        By default:
            bottom - z=0; top - z=self.sys.box_l[2];
            left - y=0  ; right - y=self.sys.box_l[1];
            back - x=0  ; front - x=self.sys.box_l[0];

        :param wall_type: int (=0) | particle type used for the walls. Must not collide with a type
            already present in the system, and must be passed again to `remove_box_constraints`.
        :param sides: list of str (=['all']) | which faces to build. ('bottom', 'top', 'sides', 'left', 'right', 'back', 'front', 'no-*')
        :param inter: str or list of str (=None) | interaction to enable between wall and particles.
            Currently only 'wca'.
        :param types_: list of int (=None) | particle types that interact with the walls. Defaults to
            every non-wall type in the system.
        :param object_types: list of type (=None) | object classes whose 'real' particle type should
            interact with the walls. Used instead of `types_` when `types_` is None.
        :param bottom, top, left, right, back, front: float (=None) | per-face positions, each
            defaulting to the corresponding box boundary. Passing one implicitly selects that face.

        :return: list of espressomd.constraints.ShapeBasedConstraint | the walls added, ordered
            bottom -> top -> left -> right -> back -> front. Keep it to remove a specific subset later.
        """
        wall_constraints = add_box_constraints_func(self.sys, wall_type=wall_type, sides=sides, inter=inter, types_=types_, object_types=object_types, bottom=bottom, top=top, left=left, right=right, back=back, front=front)

        return wall_constraints

    def remove_box_constraints(self, wall_constraints=None, part_types=None, object_types=None, wall_type=0):
        """ Removes wall_constraints from system. Default: removes all espressomd.shapes.Wall constraints
            whose particle type is `wall_type`.
            If part_types is not None, remove only interactions with those particle types.

            Calls geometry.remove_box_constraints_func.

        :param wall_constraints: list of espressomd.constraints.ShapeBasedConstraint | walls to remove.
            If None, walls are discovered from the system by `wall_type`.
        :param part_types: list of int | particle types to stop interacting with the box.
        :param object_types: list of type | object classes whose particle types stop interacting.
        :param wall_type: int or 'all' (=0) | particle type of the walls to remove. Must match the
            `wall_type` given to add_box_constraints, otherwise nothing is found and nothing is removed.
        """
        remove_box_constraints_func(self.sys, wall_type=wall_type, wall_constraints=wall_constraints, part_types=part_types, object_types=object_types)


    def init_lb(self, kT, agrid, dens, visc, gamma, timestep=0.01):
        """
        Initializes the lattice Boltzmann (LB) fluid for the simulation.

        This method configures an LB fluid using either CPU or GPU resources, depending on availability. It disables the thermostat, initializes particle velocities to zero, and sets the LB fluid parameters. If another active LB actor exists, it removes it before adding the new LB fluid.

        :param kT: float | Thermal energy (temperature) of the LB fluid.
        :param agrid: int | Grid resolution for the LB method.
        :param dens: float | Density of the LB fluid.
        :param visc: float | Viscosity (kinematic) of the LB fluid.
        :param gamma: float | Coupling constant for the thermostat.
        :param timestep: float | Integration time step for the LB simulation. Default is 0.01.
        :return: LBFluid | The configured lattice Boltzmann fluid object.
        """
        if not api_agnostic_feature_check('WALBERLA'):
            name = f"{type(self).__name__}.{inspect.currentframe().f_code.co_name}"
            raise MissingFeature(f"{name} requires WALBERLA. Please enable it in your ESPResSo installation.")
        self.sys.thermostat.turn_off()
        if len(self.sys.part):
            self.sys.part.all().v = (0, 0, 0)

        param_dict={'kT':kT, 'seed':self.seed, 'agrid':agrid, 'density':dens, 'kinematic_viscosity':visc, 'tau':timestep}

        if api_agnostic_feature_check('CUDA'):
            param_dict['gpu']=True
            logging.info('GPU LB method is beeing initiated')
        else:
            logging.info('CPU LB method is beeing initiated')
        lbf = espressomd.lb.LBFluid(**param_dict)

        self.sys.lb = lbf

        gamma_MD = gamma
        logging.info(f'gamma_MD: {gamma_MD}')
        self.sys.thermostat.set_lb(
            LB_fluid=lbf, gamma=gamma_MD, seed=self.seed)
        logging.info(f'LBM is set with the params {lbf.get_params()}.')
        return lbf

    def create_flow_channel(self, slip_vel=(0, 0, 0)):
        """
        Sets up LB boundaries for a flow channel.

        :param slip_vel: tuple | Velocity of the slip boundary in the format (vx, vy, vz). Default is (0, 0, 0).
        :return: None
        """
        logging.info("Setup LB boundaries.")
        top_wall = shapes.Wall(normal=[1, 0, 0], dist=1) # type: ignore
        bottom_wall = shapes.Wall( # type: ignore
            normal=[-1, 0, 0], dist=-(self.sys.box_l[0] - 1))

        self.sys.lb.add_boundary_from_shape(shape=top_wall, velocity=slip_vel)
        self.sys.lb.add_boundary_from_shape(shape=bottom_wall)

    def thermostat_is_off(self):
        """
        True when no thermostat mode is active.
        """
        thermostat = self.sys.thermostat
        if thermostat.kT is None:
            return True
        return not any(getattr(thermostat, name).is_active
                        for name in ("langevin", "brownian", "lb"))

    def avoid_explosion(self, F_TOL, MAX_STEPS=5, F_incr=100, I_incr=100):
        """
        Iteratively caps forces to prevent simulation instabilities.
        
        :param F_TOL: float | Force change tolerance between iterations to determine convergence.
        :param MAX_STEPS: int | Maximum number of steps for force iteration. Default is 5.
        :param F_incr: int | Force cap of the first iteration, doubled every iteration. Default is 100.
        :param I_incr: int | Integration steps of the first iteration, doubled every iteration. Default is 100.
        :return: None

        The method raises the timestep linearly to its original value over MAX_STEPS iterations, doubling the force cap and the number of integration steps each iteration, while monitoring the relative change of the maximum force between iterations. It stops when that change falls below F_TOL, when no force is left to relax (max |f| is 0), or after MAX_STEPS iterations; the force cap and the timestep are then restored.
        """
        timestep_og=self.sys.time_step
        timestep_icr=timestep_og/MAX_STEPS
        logging.info('iterating with a force cap.')
        self.sys.integrator.run(0)
        STEP=1
        while True:
            self.sys.time_step=timestep_icr*STEP
            old_force = np.max(np.linalg.norm(
                self.sys.part.all().f, axis=1))
            if old_force == 0.:
                logging.info('EXPLOSION AVOIDED: no force left to relax.')
                break
            self.sys.force_cap = F_incr
            self.sys.integrator.run(I_incr)
            force = np.max(np.linalg.norm(self.sys.part.all().f, axis=1))
            rel_force = np.abs((force - old_force) / old_force)
            logging.info(f'rel. force change: {rel_force:.2e}')
            if (rel_force < F_TOL) or (STEP >= MAX_STEPS):
                break
            STEP += 1
            I_incr += I_incr
            F_incr += F_incr

        self.sys.force_cap = 0
        self.sys.time_step=timestep_og
        logging.info('EXPLOSION AVOIDED sucessfully!')

    def set_magnetization_model(self, part_list, model, dipm_sat, mag_susc_0):
        '''
        Makes every particle in part_list magnetizable under one of espresso's magnetization models. The particles are expected to be virtual sites already bound to a real anchor, which is the case for virtuals created by object methods such as Filament.add_dipole_to_embedded_virt. Objects that own their magnetizable particles, such as PointDipoleMagnetizable, configure them at construction instead and do not need this method.

        The model is evaluated natively by espresso every timestep, from the total field H_tot=H_ext+dip_fld, so this method is called once at setup and not inside the integration loop.

        :param part_list: iterable(ParticleHandle) | ParticleSlice could work but prefer to wrap with the list() constructor.
        :param model: str | magnetization model name, see pressomancy.magnetodynamics.MAGNETIZATION_MODELS
        :param dipm_sat: float | saturation moment, must be > 0
        :param mag_susc_0: float | initial susceptibility, must be >= 0

        :return: None

        '''
        count = 0
        for part in part_list:
            configure_magnetization(part, model=model, dipm_sat=dipm_sat,
                                    mag_susc_0=mag_susc_0)
            count += 1
        logging.info(f"magnetization model '{model}' set on {count} particles "
                     f"with dipm_sat={dipm_sat}, mag_susc_0={mag_susc_0}")

    def probe_magnetization_convergence(self, part_list, n_iter=50, tol=1e-12):
        '''
        Measures how fast the mutual magnetization of part_list contracts, without advancing the simulation. Thin wrapper over pressomancy.magnetodynamics.contraction_ratio, see that function for what the numbers mean and for the saturation caveat.

        :param part_list: iterable(ParticleHandle) | the magnetizable particles to watch
        :param n_iter: int (=50) | number of fixed point iterates, must be at least 2
        :param tol: float (=1e-12) | increments at or below this count as converged

        :return: np.ndarray | successive contraction ratios

        '''
        ratios = contraction_ratio(self.sys, part_list, n_iter=n_iter, tol=tol)
        if len(ratios):
            logging.info(f'magnetization contraction ratio settled at {ratios[-1]:.4g}')
        return ratios

    def set_H_ext(self, H=(0, 0, 1.)):
        """
        Sets an espressomd.constraints.HomogeneousMagneticField in the simulation. Will delete any other HomogeneousMagneticField constraint if present. Safe to use for rotating or AC magnetic fileds.

        :param H: tuple | The external magnetic field vector. Default is (0, 0, 1).
        :return: None
        """
        stale = [x for x in self.sys.constraints
                 if isinstance(x, espressomd.constraints.HomogeneousMagneticField)]
        for x in stale:
            self.sys.constraints.remove(x)
            logging.info(f'Removed old H: {x}')
        ExtH = espressomd.constraints.HomogeneousMagneticField(H=list(H))
        self.sys.constraints.add(ExtH)
        logging.info(f'External field set: {ExtH.H}')

    def get_H_ext(self):
        """
        Retrieves the current external magnetic field.

        Sums over all applied homogeneus magnetic fields.

        :return: np.ndarray | The external magnetic field vector. Zero vector if no homogeneous magnetic field is applied.
        """
        fields = [ele.H for ele in list(self.sys.constraints)
                  if isinstance(ele, espressomd.constraints.HomogeneousMagneticField)]
        if not fields:
            return np.zeros(3)
        return np.asarray(fields).sum(axis=0)

    # ------------------------------------------------------------------
    # HDF5 output. The implementation lives in pressomancy/io/write.py;
    # these are thin delegators, the same pattern used for box constraints.
    # ------------------------------------------------------------------
    @property
    def io_dict(self):
        """HDF5 output state. Owned by the H5Writer; see pressomancy.io.write."""
        return self._h5_writer.io_dict

    @property
    def author_name(self):
        return self._h5_writer.author_name

    @property
    def author_email(self):
        return self._h5_writer.author_email

    def set_author(self, name, email='unknown'):
        """Set default author metadata for newly created HDF5 files."""
        return self._h5_writer.set_author(name, email)

    def _collect_instances_recursively(self, roots):
        """Flat preorder list of every object reachable via ``.associated_objects``."""
        return self._h5_writer._collect_instances_recursively(roots)

    def inscribe_part_group_to_h5(self, group_type=None, h5_data_path=None, mode='NEW',
                                  force_resize_to_size=None, rewind_to_step=None):
        """Inscribe particle groups into an HDF5 file. See H5Writer.inscribe_part_group_to_h5."""
        return self._h5_writer.inscribe_part_group_to_h5(
            group_type=group_type, h5_data_path=h5_data_path, mode=mode,
            force_resize_to_size=force_resize_to_size, rewind_to_step=rewind_to_step)

    def inscribe_observable_group_to_h5(self, observable_defs=None, h5_data_path=None, mode='NEW',
                                        force_resize_to_size=None, rewind_to_step=None):
        """Inscribe observable streams into an HDF5 file. See H5Writer.inscribe_observable_group_to_h5."""
        return self._h5_writer.inscribe_observable_group_to_h5(
            observable_defs=observable_defs, h5_data_path=h5_data_path, mode=mode,
            force_resize_to_size=force_resize_to_size, rewind_to_step=rewind_to_step)

    def write_part_group_to_h5(self, step):
        """Append one particle frame. See H5Writer.write_part_group_to_h5."""
        return self._h5_writer.write_part_group_to_h5(step)

    def write_observable_group_to_h5(self, step=None):
        """Append one observable frame. See H5Writer.write_observable_group_to_h5."""
        return self._h5_writer.write_observable_group_to_h5(step=step)

    def write_registered_to_h5(self, step=None):
        """Append one synchronized frame to every registered stream. See H5Writer.write_registered_to_h5."""
        return self._h5_writer.write_registered_to_h5(step=step)

    def write_checkpoint(self, group_type, path, step):
        """Write a verified one-frame checkpoint of `group_type` to `path` (via `path + '.tmp'` and
        `os.replace`); returns `step`. See H5Writer.write_checkpoint."""
        return self._h5_writer.write_checkpoint(group_type, path, step)

    def rebind_sys(self, new_sys):
        ''' Rebind the simulation to a new espresso system handle. This must be called after loading a checkpoint, otherwise the gloabal scope and internal reference to espressomd System will not match

        The ManagedSimulation singleton is rebound too. It caches the system handle
        and re-attaches it on reinitialize_instance(), so leaving it stale would
        silently revert to the pre-checkpoint system on the next reset.

        The writer's ParticleSlice cache is dropped as well: those slices are bound
        to the old system, and the cache revalidates on particle count and list
        identity only, neither of which changes when the handle underneath does.

        :param new_sys: espressomd.System | Global scope system handle to bind to.
        :return: None
        '''

        logging.debug('identity of local system: %s', id(self.sys))
        logging.debug('identity of loaded espresso system: %s', id(new_sys))
        object.__setattr__(self, "sys", new_sys)
        manager = getattr(self, "_manager", None)
        if manager is not None:
            object.__setattr__(manager, "_espressomd_system", new_sys)
        else:
            logging.warning('no ManagedSimulation back-reference found; the singleton still '
                            'holds the old system handle and reinitialize_instance() will '
                            'revert to it.')
        self._h5_writer._slice_cache.clear()
        logging.debug('identity of espresso system from rebind_sys: %s', id(self.sys))
        logging.info('successfully rebound to new espresso handle after checkpoint load!')
