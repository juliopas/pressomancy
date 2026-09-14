'''
The core of pressomancy: the ``Simulation`` class that wraps an ESPResSo
``System`` handle and manages everything built on top of it.

``Simulation`` is instantiated as a process-wide singleton via the
``ManagedSimulation`` decorator (see :mod:`pressomancy.helper_functions`).
It owns particle-type bookkeeping, storing/placing/deleting
``Simulation_Object`` instances (see :mod:`pressomancy.object_classes`),
WCA/Lennard-Jones interactions, box-wall constraints, LB fluid setup,
external magnetic fields, HDF5 (H5MD-style) I/O for particle groups and
arbitrary observables, and source-file-driven initialization
(``INIT_SRC``/``LOAD``/``LOAD_NEW`` modes).
'''
import espressomd
from espressomd import shapes
import espressomd.version
if espressomd.version.major() == 4:
    from espressomd.virtual_sites import VirtualSitesRelative
import sys as sysos
import numpy as np
import os
from itertools import combinations_with_replacement
from pressomancy.object_classes import *
from pressomancy.helper_functions import *
from pressomancy.magnetodynamics import configure_magnetization, contraction_ratio
from pressomancy.io.h5_writer import H5Writer
from pressomancy.io.h5_init import H5Init
import logging
from collections import Counter
import inspect

@ManagedSimulation
class Simulation():
    """
    A singleton class that manages a suspension of objects inside the ESPResSo molecular dynamics framework.

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
        part_types (PartDictSafe): Maps a type name to its integer espresso type id. Constructed with
            `default_factory=None`, so reading an unknown name raises `KeyError` rather than silently
            creating an entry.
        seed (int): A random seed for reproducibility, generated at initialization.
        part_positions (list): Per-partition particle positions produced by `set_objects`.
        volume_size (float): The size of the volume assigned to each object.
        volume_centers (list): A list of centers of the partitioned volumes.
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
            set_sys(timestep, min_global_cut, have_quaternion): configure cell system, time step and
                the virtual-site scheme. NOT called automatically -- callers must invoke it.
            set_author(name, email): author metadata for new HDF5 files.
            set_init_src(path, ...): declare an HDF5 source file for `INIT_SRC` initialization.
            rebind_sys(new_sys): rebind to a new espresso handle after a checkpoint load.
            modify_system_attribute(requester, attribute_name, action): permissioned mutation hook
                used by objects.

        Object management
            store_objects(iterable_list, report): register objects and their particle types.
            set_objects(objects, mode): partition the box and place objects without overlap.
            place_objects(objects, positions, orientations): place at given coordinates, no overlap check.
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
            inscribe_part_group_to_h5(group_type, h5_data_path, mode, force_resize_to_size)
            inscribe_observable_group_to_h5(observable_defs, h5_data_path, mode, force_resize_to_size)
            write_part_group_to_h5(step, unique), write_observable_group_to_h5(time_step, unique),
            write_registered_to_h5(time_step, unique): append one frame.
            mk_src_file(original_data_file_path, dest_h5_file_path, prop_dim, time_step): copy a file,
                shrink it to one frame, optionally append new properties.
            set_prop_from_src(registered_objs, time_step): copy properties from a source file.

    Notes:
        - **ESPResSo 5.x is the supported version.** Version-4 branches survive in a few places but
          are legacy and untested: `magnetodynamics.py` and `object_classes/multicore_particle.py`
          both need `espressomd.propagation`, which does not exist in v4, so the magnetics cannot
          run there at all. Required ESPResSo build features are listed in the README.
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
        self.part_types = PartDictSafe({}, default_factory=None)

        # Partitioning stuff
        self.partitioned=None
        self.part_positions=[]
        self.volume_size=None
        self.volume_centers=[]

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
    # Seeding from an HDF5 source file. Implementation in io/h5_init.py.
    # ------------------------------------------------------------------
    @property
    def src_params_set(self):
        return self._h5_init.src_params_set

    @property
    def src_path_h5(self):
        return self._h5_init.src_path_h5

    @property
    def pos_ori_src_type(self):
        return self._h5_init.pos_ori_src_type

    @property
    def type_to_type_map(self):
        return self._h5_init.type_to_type_map

    @property
    def prop_to_prop_map(self):
        return self._h5_init.prop_to_prop_map

    def set_init_src(self, path, pos_ori_src_type=['real',], type_to_type_map=[], prop_to_prop_map=[], declare_types=[]):
        """Declare an HDF5 source file to seed particle state from. See H5Init.set_init_src."""
        return self._h5_init.set_init_src(path, pos_ori_src_type=pos_ori_src_type,
                                          type_to_type_map=type_to_type_map,
                                          prop_to_prop_map=prop_to_prop_map,
                                          declare_types=declare_types)

    def set_prop_from_src(self, registered_objs=None, time_step: int = -1):
        """Copy particle properties from the declared source file. See H5Init.set_prop_from_src."""
        return self._h5_init.set_prop_from_src(registered_objs=registered_objs, time_step=time_step)

    def _get_pos_ori_from_src(self, registered_objs, time_step: int = -1):
        """Positions and orientations from the declared source file."""
        return self._h5_init.get_pos_ori_from_src(registered_objs, time_step=time_step)

    def set_sys(self, timestep=0.01, min_global_cut=3.0, have_quaternion=False):
        '''
        Set espresso cellsystem params, and import virtual particle scheme.

        Note: this is NOT run automatically on initialisation -- callers must invoke it
        explicitly.

        :param timestep: float (=0.01) | integration time step. Note the name: espresso's own
            attribute is `time_step`, and passing `time_step=` here is silently ignored.
        :param min_global_cut: float (=3.0) | minimum global interaction range. Together with the
            skin (fixed at 0.5) this is not guaranteed optimal and should be tuned per simulation.
        :param have_quaternion: bool (=False) | espresso 4 only. Whether relative virtual sites
            carry their own quaternion, so a virtual site can be oriented independently of its
            anchor. Ignored on espresso 5, where the scheme is always available.
        :return: None
        '''
        np.random.seed(seed=self.seed)
        logging.info(f'core.seed: {self.seed}')
        self.sys.periodicity = (True, True, True)
        self.sys.time_step = timestep
        self.sys.cell_system.skin = 0.5
        self.sys.min_global_cut = min_global_cut
        if espressomd.version.major()==4:
            self.sys.virtual_sites = VirtualSitesRelative(have_quaternion=have_quaternion)
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
        """
        if hasattr(self, attribute_name) and attribute_name in self.object_permissions:
            action(getattr(self,attribute_name))

        else:
            logging.info("Requester does not have permission to modify attributes.")

    def sanity_check(self,object):
        '''
        Method that checks if the object has the required features to be stored in the simulation. If the object has the required features it is stored in the self.objects list.
        '''

        missing_features = set(object.required_features) - set(espressomd.features())
        if missing_features:
            raise MissingFeature(f"{object.__class__.__name__} requires features: {object.required_features}.\nMissing required features: {', '.join(missing_features)}.")

    def store_objects(self, iterable_list, report=True):
        '''
        Method stores objects in the self.objects dict, if the object has a n_part and part_types attributes,
        and the list of objects passed to the method is commensurate with the system level attribute n_tot_parts.
        Populates the self.part_types attribute with types found in the objects that are stored.
        All objects that are stored should have the same types stored, but this is not checked explicitly
        '''
        temp_dict={}
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
        self.part_types.update(temp_dict)
        if report:
            names = [element.__class__.__name__ for element in self.objects]
            counts = Counter(names)
            formatted = ", ".join(f"{count} {name}" for name, count in counts.items())
            logging.info(f"{formatted} stored")

    def set_objects(self, objects, mode='NEW'):
        """Set objects' positions and orientations in a box. Defaults to the Simulation box.
        This method places objects in the simulation box using a partitioning scheme. For the first placement, it generates exactly the required number of positions. For subsequent placements, it searches for non-overlapping positions with existing objects. This guarantees non-overlapping of the objects.
        Parameters
        ----------
        objects : list
            A list of simulation objects to place. All objects must be instances of the same type.
        mode : {'NEW', 'INIT_SRC'}, optional
            'NEW' (default) partitions `self.sys.box_l` and generates fresh positions and
            orientations. 'INIT_SRC' instead reads them from the HDF5 source declared by
            `set_init_src`, via `_get_pos_ori_from_src`.

        Raises
        ------
        AssertionError
            If not all objects are of the same type.
        NotImplementedError
            If trying to place objects when more than one previous partition exists.
        Notes
        -----
        The current implementation supports placing objects either in an empty system or in a system with exactly one previous partition. The method uses partition_cuboid_volume to generate positions and orientations, and for subsequent placements, ensures no overlaps with existing objects through get_cross_lattice_nonintersecting_volumes. The method automatically adjusts the search space (by increasing the factor) if it cannot find enough non-overlapping positions in subsequent placements.
        """

        # Ensure all objects are of the same type.
        if not (all(isinstance(item, type(objects[0])) for item in objects)):
            raise ValueError("Not all items have the same type!")
        if mode=="INIT_SRC":
            positions, orientations=self._get_pos_ori_from_src(objects)
        else:
            # centeres, polymer_positions = partition_cuboid_volume_oriented_rectangles(big_box_dim=self.sys.box_l, num_spheres=len(filaments), small_box_dim=np.array([filaments[0].sigma, filaments[0].sigma, filaments[0].size]), num_monomers=filaments[0].n_parts)
            if len(self.part_positions)== 0:
                # First placement: generate exactly len(objects) positions.
                centeres, positions, orientations = partition_cuboid_volume(
                    box_lengths=self.sys.box_l,
                    num_spheres=len(objects),
                    sphere_diameter=objects[0].params['size'],
                    routine_per_volume=objects[0].build_function
                )
                self.volume_centers.append(centeres)
                self.part_positions.append(positions)
                self.volume_size = objects[0].params['size']
            elif len(self.part_positions) == 1:
                # Subsequent placements: search for positions without overlaps.
                factor = 1
                while True:
                    centeres, positions, orientations = partition_cuboid_volume(
                        box_lengths=self.sys.box_l,
                        num_spheres=len(objects) * factor,
                        sphere_diameter=objects[0].params['size'],
                        routine_per_volume=objects[0].build_function
                    )
                    res=get_cross_lattice_nonintersecting_volumes(
                        current_lattice_centers=centeres,
                        current_lattice_grouped_part_pos=positions,
                        current_lattice_diam=objects[0].params['size'],
                        other_lattice_centers=self.volume_centers[0],
                        other_lattice_grouped_part_pos=self.part_positions[0],
                        other_lattice_diam=self.volume_size,
                        box_lengths=self.sys.box_l
                        )
                    mask=[key for key,val in res.items() if all(val)]
                    positions=positions[mask]
                    orientations=orientations[mask]
                    if len(positions) >= len(objects):
                        break
                    else :
                        factor += 1
                        logging.info('Failed to find enough space; (found, needed): (%d, %d). Will retry by requesting %d times the number of parts', len(positions), len(objects),factor)
            else:
                raise NotImplementedError('The repartitioning scheme can currently handle only the case where one previos partition exists. More than than is still not supported')

        self.place_objects(objects, positions, orientations)

    def place_objects(self, objects, positions, orientations=None):
        """Set objects' positions and orientations in a box.
        This method places objects at given coordinates within the simulation box and sets their orientations.
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
        AssertionError
            If the number of objects, positions, and orientations do not match.
        """
        objects= np.atleast_1d(objects)
        if orientations is None:
            orientations = generate_random_unit_vectors(len(positions))
        else:
            orientations = normalize_vectors(orientations)
        for obj, pos, ori in zip(objects, positions, orientations):
            obj.set_object(pos, ori)
        names = [element.__class__.__name__ for element in objects]
        counts = Counter(names)
        formatted = ", ".join(f"{count} {name}" for name, count in counts.items())
        logging.info(f"{formatted} set!!!")

    def mark_for_collision_detection(self, object_type=Quadriplex, part_type=666):
        if not (any(isinstance(ele, object_type) for ele in self.objects)):
            raise ValueError("method assumes simulation holds correct type object")

        self.part_types['marked'] = part_type
        objects_iter = [ele for ele in self.objects if isinstance(ele, object_type)]
        if not (all((hasattr(ob, 'mark_covalent_bonds') and callable(getattr(ob, 'mark_covalent_bonds')))
                   for ob in objects_iter)):
            raise TypeError("method requires that stored objects have mark_covalent_bonds() method")
        for obj_el in objects_iter:
            obj_el.mark_covalent_bonds(part_type=part_type)

    def init_magnetic_inter(self, actor_handle):
        '''
        Attach a dipolar solver actor.

        ESPResSo 5 is the supported version. The v4 branch below is legacy and
        untested -- see the Notes on this class.
        '''
        if espressomd.version.major()==4:
            self.sys.actors.clear()
            self.sys.actors.add(actor_handle)
        elif espressomd.version.major()==5:
            self.sys.magnetostatics.clear()
            self.sys.magnetostatics.solver = actor_handle
        else:
            raise NotImplementedError(
                f'ESPResSo 5 is the supported version; found major version '
                f'{espressomd.version.major()}. Version 4 has a legacy, untested code path.')

        logging.info(f'{actor_handle} magnetic interactions actor initiated')

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
        :raises AssertionError: If the lengths of `pairs`, `wca_eps`, and `sigma` do not match.
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

        :raises AssertionError: If the lengths of `pairs`, `lj_eps`, and `lj_sigma` are not equal.
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

        Thin wrapper over :func:`pressomancy.helper_functions.add_box_constraints_func`, which
        carries the full parameter documentation -- including the `sides` grammar ('all', 'sides',
        individual faces, and 'no-<side>' exclusions), the per-face position overrides, and how
        `inter` sets up the wall interaction. Read it there rather than here, so the two cannot drift.

        By default:
            bottom - z=0; top - z=self.sys.box_l[2];
            left - y=0  ; right - y=self.sys.box_l[1];
            back - x=0  ; front - x=self.sys.box_l[0];

        :param wall_type: int (=0) | particle type used for the walls. Must not collide with a type
            already present in the system, and must be passed again to `remove_box_constraints`.
        :param sides: list of str (=['all']) | which faces to build.
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

            Calls helper_functions.remove_box_constraints_func.

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

        if espressomd.version.major() == 4:
            param_dict={'kT':kT, 'seed':self.seed, 'agrid':agrid, 'dens':dens, 'visc':visc, 'tau':timestep}
        elif espressomd.version.major() == 5:
            param_dict={'kT':kT, 'seed':self.seed, 'agrid':agrid, 'density':dens, 'kinematic_viscosity':visc, 'tau':timestep}

        if api_agnostic_feature_check('CUDA'):
            param_dict['gpu']=True
            logging.info('GPU LB method is beeing initiated')
            if espressomd.version.major() == 4:
                lbf = espressomd.lb.LBFluidWalberlaGPU(**param_dict)
            elif espressomd.version.major() == 5:
                lbf = espressomd.lb.LBFluid(**param_dict)
        else:
            logging.info('CPU LB method is beeing initiated')
            if espressomd.version.major() == 4:
                lbf = espressomd.lb.LBFluidWalberla(**param_dict)
            elif espressomd.version.major() == 5:
                lbf = espressomd.lb.LBFluid(**param_dict)

        if espressomd.version.major() == 4:
            if len(self.sys.actors.active_actors) == 2:
                self.sys.actors.remove(self.sys.actors.active_actors[-1])
            self.sys.actors.add(lbf)
        else:
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
        if espressomd.version.major() == 4:
            return thermostat.call_method("is_off")
        if thermostat.kT is None:
            return True
        return not any(getattr(thermostat, name).is_active
                        for name in ("langevin", "brownian", "lb"))

    def avoid_explosion(self, F_TOL, MAX_STEPS=5, F_incr=100, I_incr=100):
        """
        Iteratively caps forces to prevent simulation instabilities.
        :param F_TOL: float | Force change tolerance between iterations to determine convergence.
        :param MAX_STEPS: int | Maximum number of steps for force iteration. Default is 5.
        :param F_incr: int | Amount to increase force cap by each iteration. Default is 100.
        :param I_incr: int | Amount to increase integration steps by each iteration. Default is 100.
        :return: None

        The method gradually increases both the force cap and integration timestep while monitoring the relative force change between iterations. If the relative change falls below F_TOL or MAX_STEPS is reached, the iteration stops.
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
        logging.info('explosions avoided sucessfully!')

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
    # HDF5 output. The implementation lives in pressomancy/io/h5_writer.py;
    # these are thin delegators, the same pattern used for box constraints.
    # ------------------------------------------------------------------
    @property
    def io_dict(self):
        """HDF5 output state. Owned by the H5Writer; see pressomancy.io.h5_writer."""
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

    def inscribe_part_group_to_h5(self, group_type=None, h5_data_path=None, mode='NEW', force_resize_to_size=None):
        """Inscribe particle groups into an HDF5 file. See H5Writer.inscribe_part_group_to_h5."""
        return self._h5_writer.inscribe_part_group_to_h5(
            group_type=group_type, h5_data_path=h5_data_path, mode=mode,
            force_resize_to_size=force_resize_to_size)

    def inscribe_observable_group_to_h5(self, observable_defs=None, h5_data_path=None, mode='NEW', force_resize_to_size=None):
        """Inscribe observable streams into an HDF5 file. See H5Writer.inscribe_observable_group_to_h5."""
        return self._h5_writer.inscribe_observable_group_to_h5(
            observable_defs=observable_defs, h5_data_path=h5_data_path, mode=mode,
            force_resize_to_size=force_resize_to_size)

    def write_part_group_to_h5(self, step, unique=False):
        """Append one particle frame. See H5Writer.write_part_group_to_h5."""
        return self._h5_writer.write_part_group_to_h5(step, unique=unique)

    def write_observable_group_to_h5(self, time_step=None, unique=False):
        """Append one observable frame. See H5Writer.write_observable_group_to_h5."""
        return self._h5_writer.write_observable_group_to_h5(time_step=time_step, unique=unique)

    def write_registered_to_h5(self, time_step=None, unique=False):
        """Append one synchronized frame to every registered stream."""
        return self._h5_writer.write_registered_to_h5(time_step=time_step, unique=unique)

    def mk_src_file(self, original_data_file_path, dest_h5_file_path, prop_dim=None, time_step=-1):
        """Copy an HDF5 file, shrink it to one frame, optionally append properties."""
        return self._h5_writer.mk_src_file(original_data_file_path, dest_h5_file_path,
                                           prop_dim=prop_dim, time_step=time_step)

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
