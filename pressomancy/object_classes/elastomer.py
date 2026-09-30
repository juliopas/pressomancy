'''
``Elastomer``: a dense random-sphere-packing elastomer network. Built via
FCC-lattice packing, an optional particle- or wall-based substrate, a
thermostat-preserving "mixing" pre-equilibration (``mix_elastomer_stuff``),
a steepest-descent substrate-overlap relaxation (``relax_overlaps``), and
random/nearest-neighbor harmonic-bond curing (``cure_elastomer``,
``random_harmonic_bonds``, ``bond_to_neighbors``).
'''
import espressomd
import logging
import numpy as np
from collections import defaultdict
from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams
from pressomancy.infra import RoutineWithArgs, TypeDictSafe, SimulationType
from pressomancy.geometry import WCA_CONTACT_FACTOR, add_box_constraints_func, remove_box_constraints_func, check_free_cuboid, fcc_lattice, generate_random_unit_vectors, get_neighbours, get_neighbours_cross_lattice, require_min_global_cut
import warnings
import os
import sys as sysos


BOND_R_CUT_NEVER_BREAK = 0.


def _check_bond_r_cut(r_cut, r_catch):
    """Validate a HarmonicBond break distance for the elastomer network (see BOND_R_CUT_NEVER_BREAK)."""
    if r_cut < 0:
        raise ValueError(f"r_cut={r_cut} is negative. A negative HarmonicBond r_cut makes espresso skip the "
                         "bond loop entirely when no registered bond has a non-negative cutoff, so the "
                         f"network exerts no force. Use {BOND_R_CUT_NEVER_BREAK} (never breaks).")
    if not (r_cut == BOND_R_CUT_NEVER_BREAK or r_cut > r_catch):
        raise ValueError(f"r_cut must be {BOND_R_CUT_NEVER_BREAK} (never breaks, the default) or larger than "
                         f"any bond length (r_catch={r_catch}); got {r_cut}.")


class Elastomer(metaclass=Simulation_Object):
    '''
    Class that contains elastomer relevant parameters and methods. At construction one must pass an espresso handle because the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instance there will be only one type of a Elastomer. Therefore many relevant parameters are class specific, not instance specific.

    `config['sigma']` is the beads' LJ sigma; a bead's contact diameter is
    ``WCA_CONTACT_FACTOR * sigma``. `size` must stay None: the Elastomer lays itself out in
    `box_E` and has no placement sphere (see `size` in `ObjectConfigParams`).
    '''
    required_features=['DIPOLES', 'EXTERNAL_FORCES', 'ROTATION']
    simulation_type=SimulationType('elastomer', 98)
    part_types = TypeDictSafe({'real':1, 'substrate': 98})
    config = ObjectConfigParams(
        box_E= None,
        layer_height= None,
        n_parts= None,
        bond_type= "HarmonicBond",
        bond_K_lims= (0.01,0.1),
        bond_cutoff= 5.,
        max_bonds= 6,
        seed= None,
        size= None,
        sigma= 1
        )

    _substrate_size = 1.

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of an elastomer object requires a handle to the espresso system; sigma
        defaults to 1 and n_parts is inferred from box_E when left None, so neither is required.
        Raises ValueError if size is given.
        '''
        self.sys=config['espresso_handle']
        if config['size'] is not None:
            equivalent_sigma = config['size'] / WCA_CONTACT_FACTOR
            raise ValueError(f"Elastomer takes no size (got size={config['size']}): it lays itself out in "
                             "box_E. Set the beads through sigma (contact diameter WCA_CONTACT_FACTOR * sigma); "
                             f"use sigma={equivalent_sigma:.3f} for the same beads.")
        self._bead_size = WCA_CONTACT_FACTOR * config['sigma']
        if config['seed'] is None:
            config['seed'] = int.from_bytes(os.urandom(8), sysos.byteorder)
        if config['box_E'] is None:
            config['box_E'] = np.array(self.sys.box_l, copy=True)
            if config['layer_height'] is None:
                config['box_E'][2] = self.sys.box_l[2] / 4
                config['layer_height'] = config['box_E'][2] - self._substrate_size
            else:
                config['box_E'][2] = self._substrate_size + config['layer_height']
                if config['box_E'][2] > self.sys.box_l[2]:
                    raise ValueError("Elastomer layer is too thick - does not fit in simulations box.")
        else:
            if config['layer_height'] is None:
                config['layer_height'] = config['box_E'][2] - self._substrate_size
            elif config['box_E'][2] != config['layer_height'] + self._substrate_size:
                raise ValueError("box_E and layer_height are not compatible. Ensure that box_E[2] == layer_height.\nAlternatively, use only one of the parameters and have the other be automatically chosen.")
        assert config['layer_height'] == config['box_E'][2] - self._substrate_size
        if config['n_parts'] is None:
            config['n_parts']= int( 0.3 * (config['box_E'][0] * config['box_E'][1] * config['layer_height']) /  ( 4.1887902 * (self._bead_size/2)**3 ) ) # 0.3 * V_box / V_sphere
            warnings.warn('Inferred number of particles from volume of Elastomer (to get 0.3 volume fraction)')
        self.params=config
        self.associated_objects=self.params['associated_objects']
        if self.associated_objects is not None:
            self.part_types.update({nam: typ for obj in self.associated_objects for nam, typ in obj.part_types.items()})
            if len(self.associated_objects) != self.params["n_parts"]:
                raise ValueError("Number of elastomer particles must coincide with the number of associated objects.")
        self.substrate=None
        self.build_function=RoutineWithArgs(
            func=self.build_Elastomer,
            num_monomers=self.params['n_parts'],
            monomer_size=self._bead_size
            )
        self.type_part_dict={key: [] for key in Elastomer.part_types}

    def network_beads(self):
        """
        Handles of every bead that makes up the elastomer network.

        :return: list of espressomd ParticleHandle
        """
        owners = [self]
        if self.associated_objects:
            owners.extend(self.associated_objects)
        part_handles = [hndl
                   for obj in owners
                   for key, handles in obj.type_part_dict.items()
                   if isinstance(key, str) and "real" in key
                   for hndl in handles]
        # ParticleHandle is not hashable (it defines equality only), so deduplicate by id.
        ids = [hndl.id for hndl in part_handles]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Elastomer.network_beads: a bead is owned twice "
                               f"({len(ids) - len(set(ids))} duplicate ids)")
        return part_handles

    def set_object(self, pos, ori):
        '''
        Sets the particles in espresso according to self.build_function. Particles created here are treated according to their class set_object functions. Indices of added particles stored in self.realz_indices.append attribute.

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        pos=np.atleast_2d(pos)
        if not (check_free_cuboid(self.sys, self.params['box_E'])):
            raise RuntimeError("Elastomer must be build on empty space. Adjust box_E or remove non-elastomer particles to make space.")
        assert len(pos) == self.params['n_parts'], \
            "there is a mismatch between the pos length and Elastomer n_parts"
        if self.associated_objects is None:
            dipm= 1.
            logic = (self.add_particle(type_name='real',pos=pp, dip=(dipm * oo), rotation=(True, True, True)) for pp, oo in zip(pos, ori))
        else:
            if self.params['n_parts'] != len(self.associated_objects):
                raise ValueError(" there doesn't seem to be enough particles stored!!! ")
            if not all(hasattr(obj, 'set_object') and callable(getattr(obj, 'set_object')) for obj in self.associated_objects):
                raise TypeError("One or more objects do not implement a callable 'set_object'")
            logic = (obj_el.set_object(pos_el, ori_el)
                        for obj_el, pos_el, ori_el in zip(self.associated_objects, pos, ori))
        for part in logic:
            pass

        self.create_substrate()

        return self

    def build_Elastomer(self, center=None, sphere_radius=None, num_monomers=1, spacing=None, box_dim=None, flag='rand'):
        """
        Generates monomer positions and orientations for the elastomer packing.

        Builds an FCC lattice (see :func:`pressomancy.geometry.fcc_lattice`,
        mode ``'crystal'``) sized to ``box_E`` minus substrate clearance,
        shrinking the packing scale factor from 1.0 down to a floor of 0.85
        if needed to fit ``num_monomers`` lattice sites, then centers the
        packing in x/y and shifts it up in z to clear the substrate. This is
        invoked via ``self.build_function`` (see :class:`RoutineWithArgs`)
        by ``Simulation.set_objects``, not called directly; ``center``,
        ``sphere_radius`` and ``spacing`` are accepted for signature
        compatibility but unused: the lattice sphere is one bead
        (``WCA_CONTACT_FACTOR * sigma``).

        :param center: unused | kept for build_function signature compatibility
        :param sphere_radius: unused | kept for build_function signature compatibility
        :param num_monomers: int (=1) | number of monomer positions to return
        :param spacing: unused | kept for build_function signature compatibility
        :param box_dim: unused | kept for build_function signature compatibility
        :param flag: str (='rand') | 'rand' shuffles the lattice sites before
            truncating to ``num_monomers``; any other value takes them in
            lattice order
        :return: tuple(np.ndarray, np.ndarray) | (orientations, points), each
            shape (num_monomers, 3)
        :raises ValueError: if box_E[2] leaves no room above the substrate, or
            if num_monomers can't be fit even at the scaling floor
        """
        # function signature is determined by the build_function attribute, and should not be changed.
        sys_box_l = np.asarray(self.sys.box_l)
        box_E = np.asarray(self.params['box_E'])
        assert (box_E <= sys_box_l).all()

        z_offset = self._substrate_size + self._bead_size / 2
        box_E_eff = box_E.copy()
        box_E_eff[2] = self.params['layer_height'] - self._bead_size / 2 # layer height minus the sphere radius, to take into account for pbc volume in fcc function
        if box_E_eff[2] <= 0:
            raise ValueError("box_E[2] is too small to fit elastomer above substrate clearance.")

        scaling = 1.0
        scaling_floor = 0.85
        # Adjust scaling until we have enough sphere centers
        while True:
            sphere_centers = fcc_lattice(radius=self._bead_size / 2, box_dim=box_E_eff, scaling_factor=scaling, mode="crystal")
            volumes_to_fill=len(sphere_centers)
            if  volumes_to_fill>= num_monomers:
                break
            if scaling <= scaling_floor:
                raise ValueError(
                    f"Cannot fit {num_monomers} monomers of size {self._bead_size} into box_E "
                    f"{self.params['box_E']} (layer_height={self.params['layer_height']}) even after "
                    f"reducing the fcc scaling factor down to {scaling_floor}. Increase box_E/layer_height, "
                    f"reduce n_parts, or reduce sigma."
                )
            scaling -= 0.025

        # Randomly shuffle the available centers and select the required number of centers
        take_index = np.arange(len(sphere_centers))
        if flag=='rand':
            np.random.shuffle(take_index)
        take_index = take_index[:num_monomers]
        sphere_centers=sphere_centers[take_index]

        # Center point distribution in box_E (x/y) and enforce bottom z clearance.
        min_centers = np.min(sphere_centers, axis=0)
        max_centers = np.max(sphere_centers, axis=0)
        sphere_centers += box_E / 2 - (min_centers + max_centers) / 2
        min_centers = np.min(sphere_centers, axis=0)
        max_centers = np.max(sphere_centers, axis=0)
        z_shift = z_offset - min_centers[2]
        sphere_centers[:, 2] += z_shift

        pos_z = sphere_centers[:, 2]
        z_lo = self._substrate_size + self._bead_size/2
        z_hi = self.params['box_E'][2] - self._bead_size / 2
        assert np.all( (pos_z >= z_lo) & (pos_z <= z_hi) ), f"particle positions were placed outside of elastomer space.\n\t min({pos_z.min()}) max({pos_z.max()}). should be min({z_lo}) max({z_hi}). Contact customer services."

        points=sphere_centers
        orientations=generate_random_unit_vectors(len(sphere_centers))

        return orientations, points
    
    def _thermostat_is_off(self):
            """
            True when no thermostat mode is active.
            """
            thermostat = self.sys.thermostat
            if thermostat.kT is None:
                return True
            return not any(getattr(thermostat, name).is_active
                            for name in ("langevin", "brownian", "lb"))
    
    def _snapshot_thermostat_state(self):
        thermostat = self.sys.thermostat
        snapshot = {"is_off": self._thermostat_is_off(), "modes": []}
        if snapshot["is_off"]:
            logging.debug("Elastomer.mix_elastomer_stuff: no active thermostat to preserve")
            return snapshot

        kT = thermostat.kT
        if thermostat.langevin.is_active:
            mode = {
                "name": "langevin",
                "kT": kT,
                "gamma": np.copy(thermostat.langevin.gamma),
                "seed": thermostat.langevin.seed,
            }
            gamma_rotation = thermostat.langevin.gamma_rotation
            if gamma_rotation is not None:
                mode["gamma_rotation"] = np.copy(gamma_rotation)
            snapshot["modes"].append(mode)
        if thermostat.brownian.is_active:
            mode = {
                "name": "brownian",
                "kT": kT,
                "gamma": np.copy(thermostat.brownian.gamma),
                "seed": thermostat.brownian.seed,
            }
            gamma_rotation = thermostat.brownian.gamma_rotation
            if gamma_rotation is not None:
                mode["gamma_rotation"] = np.copy(gamma_rotation)
            snapshot["modes"].append(mode)
        if thermostat.lb.is_active and self.sys.lb is not None:
            snapshot["modes"].append({
                "name": "lb",
                "kT": kT,
                "gamma": np.copy(thermostat.lb.gamma),
                "seed": thermostat.lb.seed,
            })

        logging.info(
            "Elastomer.mix_elastomer_stuff: preserving thermostat state %s",
            [mode["name"] for mode in snapshot["modes"]],
        )
        return snapshot

    def _restore_thermostat_state(self, snapshot):
        thermostat = self.sys.thermostat
        thermostat.turn_off()
        if snapshot["is_off"]:
            logging.info("Elastomer.mix_elastomer_stuff: restored thermostat state to off")
            return

        restored = []
        for mode in snapshot["modes"]:
            if mode["name"] == "langevin":
                kwargs = {
                    "kT": mode["kT"],
                    "gamma": mode["gamma"],
                    "seed": mode["seed"],
                }
                if "gamma_rotation" in mode:
                    kwargs["gamma_rotation"] = mode["gamma_rotation"]
                thermostat.set_langevin(**kwargs)
                restored.append("langevin")
            elif mode["name"] == "brownian":
                kwargs = {
                    "kT": mode["kT"],
                    "gamma": mode["gamma"],
                    "seed": mode["seed"],
                }
                if "gamma_rotation" in mode:
                    kwargs["gamma_rotation"] = mode["gamma_rotation"]
                thermostat.set_brownian(**kwargs)
                restored.append("brownian")
            elif mode["name"] == "lb" and self.sys.lb is not None:
                thermostat.set_lb(LB_fluid=self.sys.lb, kT=mode["kT"], gamma=mode["gamma"], seed=mode["seed"])
                restored.append("lb")

        logging.info(
            "Elastomer.mix_elastomer_stuff: restored thermostat state %s",
            restored,
        )

    def mix_elastomer_stuff(self, n_iter=100, time_step=0.001, kT=1e-3, gamma=10, wall_epsilon=10):
        """
        Pre-equilibrates the packed monomers before curing, to shake out an
        even random distribution.

        Temporarily adds top/bottom WCA box walls (epsilon=``wall_epsilon``),
        switches to a short, strongly-damped Langevin run (``kT``, ``gamma``)
        at the given ``time_step``, runs ``n_iter`` integrator steps, then
        always restores the box (removing the temporary walls), the prior
        thermostat mode (whatever was active before, including 'off', via
        ``_snapshot_thermostat_state``/``_restore_thermostat_state``), and the
        prior ``time_step`` — even if the run raises.

        The substrate/real WCA epsilon set up by ``create_substrate``
        (``create_substrate_part`` or ``create_substrate_wall``) is
        temporarily softened to ``wall_epsilon`` for every 'real'
        monomer type — sigma is left untouched, so the equilibrium contact
        distance doesn't change, only how hard the substrate pushes back
        — and the original epsilon is restored in the same ``finally`` block
        , even if the run raises.

        :param n_iter: int (=100) | number of integrator steps to run
        :param time_step: float (=0.001) | time step to use during mixing
        :param kT: float (=1e-3) | Langevin temperature during mixing
        :param gamma: float (=10) | Langevin friction during mixing
        :param wall_epsilon: float (=10) | WCA epsilon of the temporary top/bottom walls and substrate
        :return: None
        :raises ValueError: if called before ``create_substrate``
        """
        if isinstance(self, list):
            raise ValueError("Must be used on Elastomer object type")

        # add initialization process, to get a nice random distribution before bonding
        old_time_step= float(self.sys.time_step)

        self.sys.time_step = time_step

        if self.substrate is None:
            raise ValueError("Substrate must be created before mix_elastomer_stuff().")

        # Add temporary wall (top and bottom only).
        types_M = tuple(typ for key, typ in self.part_types.items() if "real" in key)
        add_box_constraints_func(
            sides=['top', 'bottom'],
            top=self.params['box_E'][2],
            bottom=self._substrate_size,
            inter='wca',
            types_=types_M,
            wall_epsilon=wall_epsilon,
            sys=self.sys,
        )

        # Soften the substrate/real WCA epsilon (keep sigma) while mixing, if asked.
        substrate_epsilon_snapshot = {}
        for typ in types_M:
            wca_handle = self.sys.non_bonded_inter[self.part_types['substrate'], typ].wca
            substrate_epsilon_snapshot[typ] = wca_handle.epsilon
            wca_handle.set_params(epsilon=wall_epsilon, sigma=wca_handle.sigma)

        thermostat_snapshot = self._snapshot_thermostat_state()
        try:
            self.sys.thermostat.turn_off()
            self.sys.thermostat.set_langevin(kT=kT, gamma=gamma, seed=self.params['seed'])
            self.sys.integrator.run(n_iter)
        finally:
            # Remove temporary box particles
            remove_box_constraints_func(sys=self.sys)
            self._restore_thermostat_state(thermostat_snapshot)
            self.sys.time_step = old_time_step
            for typ, epsilon in substrate_epsilon_snapshot.items():
                wca_handle = self.sys.non_bonded_inter[self.part_types['substrate'], typ].wca
                wca_handle.set_params(epsilon=epsilon, sigma=wca_handle.sigma)

    def relax_overlaps(self, f_max=10., gamma=10, max_displacement_per_step=1e-3, steps_per_chunk=10,
                       max_displacement_factor=0.2, gap_tol=1e-4):
        """
        Removes leftover bead/substrate overlap (e.g. after mixing or hand
        placement) by steepest-descent, stopping as soon as the deepest network
        bead has cleared the substrate. "Network bead" means whatever
        ``network_beads`` returns, so an elastomer built out of associated
        objects relaxes its children's beads, not an empty set.

        ``build_Elastomer`` itself leaves no bead/substrate overlap (and at most
        15% bead/bead overlap, its fcc scaling floor 0.85); this is a small
        correction, not a repair of a badly packed network.

        The "gap" is the distance from the lowest network bead's centre
        to the wall plane ``_substrate_size + bead/2``: one contact radius
        (``bead = WCA_CONTACT_FACTOR * sigma``) above the substrate, the height
        ``build_Elastomer`` places the lowest beads at. The temporary bottom
        wall stops a bead exactly there (``add_box_constraints_func`` gives it
        half the bead type's WCA sigma), and so does a substrate lattice bead
        directly below, so a bead resting on the wall or on a lattice bead has
        gap 0. That holds because every network-bead type's own WCA sigma must
        equal the Elastomer's sigma. A negative gap means some bead centre is
        still below the wall plane (embedded in the substrate);
        ``gap > -gap_tol`` is the convergence criterion.

        Adds temporary top/bottom WCA walls exactly like
        ``mix_elastomer_stuff`` does — the bottom one, in particular, keeps
        beads from being squeezed sideways into the gaps between substrate
        lattice beads while the overlap is worked out. Then runs
        steepest-descent (``integrator.set_steepest_descent``) in
        ``ceil(max_steps / steps_per_chunk)`` chunks of ``steps_per_chunk``
        integrator steps, checking the gap after each chunk, until the gap
        clears ``-gap_tol`` AND the largest force on a network bead is below
        ``f_max``. Counting chunks, not the steps espresso reports, makes the
        budget a hard limit (espresso reports 0 steps when it stops at once).

        A bead moves at most ``max_displacement_per_step`` per step and
        at most ``max_displacement_factor * self._bead_size`` for all steps, so
        ``max_steps`` is calculated as
        ``ceil(max_displacement_factor * self._bead_size / max_displacement_per_step)``
        and any gaps deeper than ``max_displacement_factor * self._bead_size``
        is refused before any step is taken; a gap up to that depth clears
        within the budget. The walls are removed and the velocity-Verlet integrator
        restored in a ``finally`` block regardless of outcome, after which a
        fresh force recalculation reports the residual force.

        :param f_max: float (=10.) | steepest-descent force convergence criterion
        :param gamma: float (=10) | steepest-descent friction
        :param max_displacement_per_step: float (=1e-3) | steepest-descent max per-step displacement
        :param steps_per_chunk: int (=10) | integrator steps run between gap checks
        :param max_displacement_factor: float (=0.2) | upper bound on particle displacement
            on the full relaxation as a factor of the beads' size;
            0.2 means enough to move a bead 20% of its contact diameter
            ``WCA_CONTACT_FACTOR * sigma`` (200 for a diameter of 1)
        :param gap_tol: float (=1e-4) | tolerance on the substrate gap convergence criterion
        :return: dict with keys ``steps``, ``converged``, ``max_f_before``, ``max_f_after``, ``gap_before``, ``gap_after``
        :raises ValueError: if a network-bead type's own WCA sigma is not the Elastomer's ``sigma``
        :raises RuntimeError: if a network-bead type has no WCA interaction with itself, if the initial
            gap is deeper than ``max_displacement_factor``, or if the gap is still negative or the
            residual force still >= ``f_max`` after relaxation
        """
        if isinstance(self, list):
            raise ValueError("Must be used on Elastomer object type")

        for typ in sorted({hndl.type for hndl in self.network_beads()}):
            type_sigma = float(self.sys.non_bonded_inter[typ, typ].wca.sigma)
            if type_sigma <= 0.:
                raise RuntimeError(f"relax_overlaps needs the WCA interaction of network-bead type {typ} with "
                                   "itself set (set_steric) before it is called; the bottom wall takes its "
                                   "radius from it.")
            if type_sigma != self.params['sigma']:
                raise ValueError(f"relax_overlaps: the WCA sigma {type_sigma} of network-bead type {typ} differs "
                                 f"from the Elastomer's sigma {self.params['sigma']}; the bottom wall takes its "
                                 "radius from the former and the network was packed with the latter.")
        wall_plane = self._substrate_size + self._bead_size / 2  # where build_Elastomer puts the lowest beads
        max_displacement = max_displacement_factor * self._bead_size
        max_steps = int(np.ceil(max_displacement / max_displacement_per_step))
        max_travel = max_steps * max_displacement_per_step
        fix_the_start = ("Fix the initial configuration (packing, mixing, hand placement) before raising "
                         "max_steps.")

        def _gap():
            zs = [hndl.pos[2] for hndl in self.network_beads()]
            return float(min(zs) - wall_plane)

        def _max_f():
            return float(max(np.linalg.norm(hndl.f) for hndl in self.network_beads()))

        gap_before = _gap()  # positions only: checked before the temporary walls go in
        if gap_before < -max_travel:
            raise RuntimeError(
                f"relax_overlaps: the initial configuration is too overlapped: a bead sits {-gap_before} below "
                f"the wall plane, but the relaxation moves a bead at most max_steps * max_displacement = "
                f"{max_travel} (20% of a bead diameter by default). {fix_the_start}"
            )

        types_M = tuple(typ for key, typ in self.part_types.items() if "real" in key)
        add_box_constraints_func(
            sides=['top', 'bottom'],
            top=self.params['box_E'][2],
            bottom=self._substrate_size,
            inter='wca',
            types_=types_M,
            sys=self.sys,
        )

        steps = 0
        converged = False
        try:
            self.sys.integrator.run(0, recalc_forces=True)
            max_f_before = _max_f()
            for _ in range(int(np.ceil(max_steps / steps_per_chunk))):
                self.sys.integrator.set_steepest_descent(f_max=f_max, gamma=gamma, max_displacement=max_displacement)
                steps += self.sys.integrator.run(steps_per_chunk)
                self.sys.integrator.set_vv()
                self.sys.integrator.run(0, recalc_forces=True)
                if _gap() > -gap_tol and _max_f() < f_max:
                    converged = True
                    break
        finally:
            # Remove temporary box particles
            remove_box_constraints_func(sys=self.sys)
            self.sys.integrator.set_vv()

        self.sys.integrator.run(0, recalc_forces=True)
        max_f_after = _max_f()
        gap_after = _gap()

        if gap_after <= -gap_tol:
            raise RuntimeError(
                f"relax_overlaps: the gap {gap_after} is still negative after "
                f"relaxation (was {gap_before} before). The bottom layer of beads "
                "still overlaps the substrate: the initial configuration overlaps more than the "
                f"relaxation clears. {fix_the_start}"
            )

        if max_f_after >= f_max:
            raise RuntimeError(
                f"relax_overlaps: residual force {max_f_after} is still >= f_max={f_max} after "
                f"relaxation (was {max_f_before} before). The network may still be overlapping."
            )

        return dict(
            steps=steps,
            converged=converged,
            max_f_before=max_f_before,
            max_f_after=max_f_after,
            gap_before=gap_before,
            gap_after=gap_after,
        )

    def cure_elastomer(self, fold_coord=True):
        """
        Fixes the network in place: bonds the network beads into a permanent
        random-harmonic-bond network.

        If a substrate is set, every network bead (see ``network_beads``)
        sitting within a quarter bead size of the bottom packing plane
        ``_substrate_size + bead/2`` has its z motion fixed (``part.fix``).
        ``bead`` is the elastomer's own bead contact diameter
        ``WCA_CONTACT_FACTOR * sigma``, the one ``build_Elastomer`` packed
        the bottom layer with.

        Bonds are created via ``random_harmonic_bonds`` using
        ``bond_cutoff``/``max_bonds``/``bond_K_lims``, and any bead left
        with zero bonds is retried via ``bond_to_neighbors`` against its
        nearest neighbors.

        :param fold_coord: bool (=True) | if True, folds every network bead
            back into the primary periodic box before bonding
        :return: None
        """
        if isinstance(self, list):
            raise ValueError("Must be used on Elastomer object type")

        beads = self.network_beads()

        if fold_coord:
            for part in beads:
                part.pos = part.pos_folded

        if self.substrate is not None:
            # Stick the bottom layer of beads to the z=R_M plane
            #  (restrict movement in z direction)
            assert ( isinstance(self.substrate, espressomd.constraints.ShapeBasedConstraint) and isinstance(self.substrate.shape, espressomd.shapes.Wall) ) \
             or ( isinstance(self.substrate, list) and all([isinstance(part, espressomd.particle_data.ParticleHandle) for part in self.substrate]) ) \
            , "substrate must be None, an espresso wall constraint, or a list of particle handles. Use Elastomer.create_substrate to create valid substrate."
            # chose at which heights to capture Ms
            z_pin = self._substrate_size + self._bead_size / 2 + self._bead_size / 4
            n_pinned = 0
            for hndl in beads:
                if hndl.pos[2] < z_pin:
                    hndl.fix = [False, False, True]
                    n_pinned += 1
            logging.info("Elastomer.cure_elastomer: pinned %d of %d network beads below z=%g",
                         n_pinned, len(beads), z_pin)

        # Bond particles
        r_catch = self.params['bond_cutoff']
        max_bonds = self.params['max_bonds']
        bond_k = self.params['bond_K_lims']

        _, n_bonds_dict = self.random_harmonic_bonds(r_catch, bond_k, max_bonds, r_cut=BOND_R_CUT_NEVER_BREAK, std_scaling=6)
        lonely_M = [part_id for part_id, n_bonds in n_bonds_dict.items() if n_bonds == 0]
        if lonely_M:
            all_M = self.sys.part.by_ids(list(n_bonds_dict.keys()))
            # a lonely bead is allowed a slightly saturated partner, else it stays lonely
            self.bond_to_neighbors(parts=self.sys.part.by_ids(lonely_M), n_nghb=3, bond_k=bond_k,
                                   r_cut=BOND_R_CUT_NEVER_BREAK, r_catch=r_catch, std_scaling=6,
                                   candidate_parts=all_M, max_bonds=max_bonds+2, n_bonds_dict=n_bonds_dict)

    def _bond_network(self, parts, candidate_parts, select_partners, bond_k, r_catch, r_cut,
                      max_bonds, n_bonds_dict, std_scaling):
        """
        Shared body of ``random_harmonic_bonds`` and ``bond_to_neighbors``:
        builds the neighbour lists and adds the HarmonicBonds, leaving only the
        choice of *which* of a bead's free neighbours to bond to the caller.

        Spring constants are drawn from a normal distribution centred on the
        midpoint of ``bond_k`` with standard deviation
        ``(bond_k[1] - bond_k[0]) / std_scaling``, redrawn until it lands
        inside ``bond_k`` (a degenerate interval therefore gives a fixed k).
        Distances are computed under PBC, with z periodicity switched off while
        a substrate is in place. A pair is bonded at most once, and the bond is
        stored on one of the two particles only.

        :param parts: ParticleSlice | beads that may receive new bonds; bonds are only ever added FROM these
        :param candidate_parts: ParticleSlice | pool searched for partners (``parts`` itself for a same-lattice search)
        :param select_partners: callable(available_ids, n_free) -> iterable of ids | picks which of a
            bead's non-saturated neighbours to bond to, at most ``n_free`` of them
        :param bond_k: tuple(float, float) | (min, max) spring-constant interval
        :param r_catch: float | maximum distance allowed between two bonded beads
        :param r_cut: float | HarmonicBond break distance (see ``BOND_R_CUT_NEVER_BREAK``)
        :param max_bonds: int | a bead already at this degree (on either end) is skipped
        :param n_bonds_dict: defaultdict(int) | bond-count bookkeeping, updated in place
        :param std_scaling: float | higher values narrow the k distribution
        :return: int | number of bonds created
        """
        if not (np.ndim(bond_k) == 1 and len(bond_k) == 2 and bond_k[1] - bond_k[0] >= 0):
            raise ValueError("bond_k must be an interval of the form (min, max) with max >= min; "
                             f"got {bond_k}")
        _check_bond_r_cut(r_cut, r_catch)
        require_min_global_cut(self.sys, r_catch)  # bonds up to r_catch must not straddle rank domains
        k_mean = (bond_k[1] + bond_k[0]) / 2
        k_std = (bond_k[1] - bond_k[0]) / std_scaling

        if self.substrate is not None:
            old_periodicity = np.copy(self.sys.periodicity)
            self.sys.periodicity = [True, True, False]

        box_dim = self.sys.box_l + 2 * r_catch * ~np.array(self.sys.periodicity)  # pad > r_catch: no pair wraps across an open face

        parts_id_map = list(parts.id)
        if candidate_parts is parts or set(candidate_parts.id) == set(parts_id_map):
            candidates_id_map = parts_id_map
            neighbours_raw = get_neighbours(parts.pos, box_dim, cutoff=r_catch, sort=True)
        else:
            candidates_id_map = list(candidate_parts.id)
            neighbours_raw = get_neighbours_cross_lattice(parts.pos, candidate_parts.pos,
                                                            box_dim, cutoff=r_catch, sort=True)
        pairs = defaultdict(list)
        for idx, neigh in neighbours_raw.items():
            id1 = parts_id_map[idx]
            pairs[id1] = [candidates_id_map[j] for j in neigh if candidates_id_map[j] != id1]

        total_bonds = 0
        for id1 in parts_id_map:
            n_free = max_bonds - n_bonds_dict[id1]
            available = [id2 for id2 in pairs[id1] if n_bonds_dict[id2] < max_bonds]
            if n_free <= 0 or not available:
                continue
            particle = self.sys.part.by_id(id1)
            for id2 in select_partners(available, n_free):
                r_12 = self.sys.distance(p1=particle, p2=self.sys.part.by_id(id2))
                assert r_12 <= r_catch
                k_12 = bond_k[0] - 1.
                while k_12 < bond_k[0] or k_12 > bond_k[1]:
                    k_12 = np.random.normal(loc=k_mean, scale=k_std)
                elastic_bond = espressomd.interactions.HarmonicBond(r_0=r_12, k=k_12, r_cut=r_cut)
                self.sys.bonded_inter.add(elastic_bond)
                particle.add_bond((elastic_bond, id2))

                n_bonds_dict[id1] += 1
                n_bonds_dict[id2] += 1
                if id1 in pairs[id2]:
                    pairs[id2].remove(id1)  # bond every pair once, from one end only
                total_bonds += 1
        if self.substrate is not None:
            self.sys.periodicity = old_periodicity

        return total_bonds

    def random_harmonic_bonds(self, r_catch, bond_k=(0.001, 0.01), max_bonds=None,
                              r_cut=BOND_R_CUT_NEVER_BREAK, std_scaling=6):
        """
        Randomly bonds the network beads to each other with harmonic bonds.

        Every bead of the network (``network_beads``) is offered a random
        selection of the neighbours it has within ``r_catch``, up to the number
        of bonds it still has free. See ``_bond_network`` for the shared
        mechanics (spring constants, PBC, one bond per pair).

        :param r_catch: float | maximum distance allowed between two bonded beads
        :param bond_k: tuple(float, float) (=(0.001, 0.01)) | (min, max) spring-constant interval
        :param max_bonds: int (=None) | maximum number of bonds per bead; None means no cap
        :param r_cut: float (=BOND_R_CUT_NEVER_BREAK) | HarmonicBond break distance
        :param std_scaling: float (=6) | higher values narrow the k distribution
        :return: tuple(int, defaultdict(int)) | (bonds created, bond count per particle id
            with an entry for every network bead)
        """
        particles = self.sys.part.by_ids([hndl.id for hndl in self.network_beads()])
        if max_bonds is None:
            max_bonds = len(particles)
        n_bonds_dict = defaultdict(int)

        def pick_at_random(available, n_free):
            return np.random.choice(available, min(len(available), n_free), replace=False)

        total_bonds = self._bond_network(parts=particles, candidate_parts=particles,
                                         select_partners=pick_at_random, bond_k=bond_k,
                                         r_catch=r_catch, r_cut=r_cut, max_bonds=max_bonds,
                                         n_bonds_dict=n_bonds_dict, std_scaling=std_scaling)
        return total_bonds, n_bonds_dict

    def bond_to_neighbors(self, parts, n_nghb=3, bond_k=(0.001, 0.01), r_catch=None,
                          r_cut=BOND_R_CUT_NEVER_BREAK, std_scaling=6, candidate_parts=None,
                          max_bonds=None, n_bonds_dict=None):
        """
        Bonds ``parts`` to up to ``n_nghb`` of their neighbours within
        ``r_catch``, taken in neighbour-list order. Same mechanics as
        ``random_harmonic_bonds`` (see ``_bond_network``), different choice of
        partners: the first free neighbours instead of a random draw.

        :param parts: ParticleSlice | the beads that need NEW bonds; bonds are only ever added FROM these
        :param n_nghb: int (=3) | maximum number of new bonds per bead of ``parts``
        :param bond_k: tuple(float, float) (=(0.001, 0.01)) | (min, max) spring-constant interval
        :param r_catch: float (=None) | maximum bond length; None means half the free box height
        :param r_cut: float (=BOND_R_CUT_NEVER_BREAK) | HarmonicBond break distance
        :param std_scaling: float (=6) | higher values narrow the k distribution
        :param candidate_parts: ParticleSlice (=None) | pool searched for partners; None means ``parts`` itself
        :param max_bonds: int (=None) | a bead already holding this many bonds (on either end) is
            skipped; None means no cap
        :param n_bonds_dict: defaultdict(int) (=None) | bond-count bookkeeping shared with the caller
            (e.g. ``random_harmonic_bonds``' second return value). Updated in place and consulted for
            the ``max_bonds`` check; a local counter is used when not given.
        :return: None
        """
        if candidate_parts is None:
            candidate_parts = parts
        if n_bonds_dict is None:
            n_bonds_dict = defaultdict(int)
        if r_catch is None:
            r_catch = (self.sys.box_l[2] - self._substrate_size) / 2
        if max_bonds is None:
            max_bonds = len(candidate_parts)

        def pick_nearest(available, n_free):
            return available[:min(n_nghb, n_free)]

        self._bond_network(parts=parts, candidate_parts=candidate_parts,
                           select_partners=pick_nearest, bond_k=bond_k,
                           r_catch=r_catch, r_cut=r_cut, max_bonds=max_bonds,
                           n_bonds_dict=n_bonds_dict, std_scaling=std_scaling)
        lonely = [part_id for part_id in parts.id if n_bonds_dict[part_id] == 0]
        if lonely:
            warnings.warn(f"{len(lonely)} of {len(parts.id)} particles hold no bond at all after "
                          f"bond_to_neighbors (no neighbour within r_catch={r_catch}, or every "
                          f"neighbour already at max_bonds={max_bonds}): ids {lonely}. If this is a "
                          "negligible portion of the particles, closely monitor the simulation and it "
                          "should be fine. Otherwise consider increasing r_catch or n_nghb.")

    def create_substrate(self, geometry: str = 'part'):
        """
        Creates the elastomer's substrate, if none exists yet.

        :param geometry: str (='part') | 'wall' for an implicit WCA wall
            constraint (``create_substrate_wall``), anything else for an
            explicit lattice of fixed substrate particles
            (``create_substrate_part``)
        :return: None
        """
        if self.substrate is None:
            if geometry == 'wall':
                self.create_substrate_wall()
            else:
                self.create_substrate_part()
        else:
            warnings.warn("Substrate already set. Will ignore this call.")

    def remove_substrate(self, geometry: str = 'part'):
        """
        Removes the elastomer's substrate, if one exists.

        :param geometry: str (='part') | must match the geometry passed to
            ``create_substrate`` ('wall' or 'part')
        :return: None
        """
        if self.substrate is not None:
            if geometry == 'wall':
                self.remove_substrate_wall()
            else:
                self.remove_substrate_part()
        else:
            warnings.warn("Substrate not yet set. Will ignore this call.")

    def create_substrate_part(self):
        """
        Builds an explicit, fixed, sub-monomer-sized 'substrate' particle
        lattice covering the x/y footprint of the box at z=substrate_radius,
        and sets a strongly-repulsive WCA interaction between it and every
        'real' monomer type so monomers can't sink through it.

        :return: None
        """
        substrate_radius = self._substrate_size / 2.
        n_substrate_x = int(np.ceil(self.params['box_E'][0]))
        n_substrate_y = int(np.ceil(self.params['box_E'][1]))
        n_substrate= n_substrate_x * n_substrate_y
        pos_x, pos_y = np.meshgrid( np.linspace(substrate_radius, self.params['box_E'][0]-substrate_radius, n_substrate_x),
                                    np.linspace(substrate_radius, self.params['box_E'][1]-substrate_radius, n_substrate_y) )
        pos = np.column_stack((pos_x.ravel(), pos_y.ravel(), np.zeros(n_substrate) + substrate_radius))

        self.substrate = [self.add_particle(type_name="substrate", pos=pos[i],
                                            type=self.part_types['substrate'], fix=[True, True, True])
                          for i in range(n_substrate)]

        # The softened sigma keeps monomers sitting at their equilibrium
        # distance from the substrate instead of a stiff overlap.
        substrate_sigma_half = substrate_radius / WCA_CONTACT_FACTOR
        for key, typ in self.part_types.items():
            if "real" in key:
                sigma = self.params['sigma'] / 2 + substrate_sigma_half
                self.sys.non_bonded_inter[self.part_types['substrate'], typ].wca.set_params(epsilon=1e6, sigma=sigma)

    def remove_substrate_part(self):
        """
        Removes every substrate particle created by ``create_substrate_part``
        and deactivates the substrate/monomer WCA interaction.

        :return: None
        """
        for part in self.substrate:
            part.remove()
            self.type_part_dict['substrate'].remove(part)
        self.substrate = None

        for key, typ in self.part_types.items():
            if "real" in key:
                self.sys.non_bonded_inter[self.part_types['substrate'], typ].wca.deactivate()

    def create_substrate_wall(self):
        """
        Creates an implicit substrate as a bottom WCA wall constraint (see
        :func:`pressomancy.geometry.add_box_constraints_func`)
        against every 'real' monomer type.

        :return: None
        """
        types_M = tuple(typ for key, typ in self.part_types.items() if "real" in key)
        wall_constraints = add_box_constraints_func(bottom=self._substrate_size, wall_type=self.part_types['substrate'], inter='wca', types_=types_M, sys=self.sys)
        self.substrate = wall_constraints[0]

    def remove_substrate_wall(self):
        """
        Removes the bottom wall constraint created by ``create_substrate_wall``.

        :return: None
        """
        remove_box_constraints_func(wall_type=self.part_types['substrate'], sys=self.sys)
        self.substrate = None
