'''
``TelSeq``: folds a chain of ``brokenA``/``brokenB`` ``Quadriplex`` units into
a G-quadruplex telomeric-sequence topology (``parallel``/``antiparallel``/
``hybrid`` folds), via the ``_rule_maker``/``_telseq_block_offset``
corner-bonding rules and ``wrap_into_Tel``.
'''
import os
import espressomd
import numpy as np
import random
from pressomancy.object_classes.quadriplex_class import *
from pressomancy.object_classes.object_class import Simulation_Object, ObjectConfigParams
from pressomancy.object_classes.rigid_obj import GenericRigidObj
from pressomancy.helper_functions import RoutineWithArgs, make_centered_rand_orient_point_array, PartDictSafe, SinglePairDict, BondWrapper, get_orientation_vec, get_perpendicular, align_vectors, load_coord_file
import logging
import warnings

TELSEQ_RULES = {
    'quartet': {
        'block_size': 75,
        'top': [26, 45, 49, 30],
        'bottom': [51, 70, 74, 55],
    },
    'quartet_11x11': {
        'block_size': 363,
        'top': [122, 231, 241, 132],
        'bottom': [243, 352, 362, 253],
    },
}

#: Aliases whose TELSEQ_RULES entry has already been checked against geometry.
_VALIDATED_TELSEQ_ALIASES = set()


def _telseq_block_offset(alias, corner_particles):
    """
    Derive the particle-id offset of one monomer's TELSEQ_RULES block.

    ``TELSEQ_RULES`` indexes corners in a monomer-*local* space (0..block_size-1),
    while the ids handed to :func:`_rule_maker` are real ESPResSo particle ids. The
    two are related by a constant shift, which must be read off the monomer's *own*
    particles: object identity (``who_am_i``) is a monotonic, never-reset counter,
    whereas ``sys.part.clear()`` restarts espresso ids at 0, so the id spaces drift
    apart as soon as more than one TelSeq is built in a single process.
    """
    try:
        rule_params = TELSEQ_RULES[alias]
    except KeyError:
        raise ValueError(f"no TelSeq rule set is defined for alias '{alias}'; known aliases: {sorted(TELSEQ_RULES)}") from None
    _validate_telseq_rules(alias)
    local_ids = sorted(rule_params['top'] + rule_params['bottom'])
    actual_ids = sorted(part.id for part in corner_particles)
    if len(actual_ids) != len(local_ids):
        raise ValueError(f"alias '{alias}' expects {len(local_ids)} corner particles per monomer, got {len(actual_ids)}")
    offset = actual_ids[0] - local_ids[0]
    expected_ids = [local_id + offset for local_id in local_ids]
    if actual_ids != expected_ids:
        raise ValueError(f"corner particle ids {actual_ids} are not a rigid translation of the "
            f"'{alias}' rule corners {local_ids} (inferred offset={offset}, expected {expected_ids}); "
            "the monomer layout does not match the rule block")
    return offset


def _geometric_corner_local_ids(alias):
    """
    Independently recomputes which local indices of a Quartet's reference
    geometry are corner particles, using the same diagonal-distance rule
    ``Quartet.set_object`` uses to build ``corner_particles`` (both read the
    distance from ``Quartet.CORNER_DIAGONAL``, so there is one definition).

    Reads ``resources/<alias>.txt`` directly (via ``load_coord_file``, which
    staples the CoM particle in at local index 0, matching how
    ``GenericRigidObj``/``Quartet`` load it), rather than relying on a live
    ``Quartet`` instance, so no espresso system is needed.

    :param alias: str | a Quartet resource-file alias (e.g. 'quartet')
    :return: set(int) | local indices of the geometric corner particles
    """
    path = os.path.join(GenericRigidObj._resources_dir, f"{alias}.txt")
    sheet = load_coord_file(path)
    separations = np.linalg.norm(sheet[:, None, :] - sheet[None, :, :], axis=-1)
    rows, cols = np.nonzero(np.isclose(separations, Quartet.CORNER_DIAGONAL, atol=1e-6))
    return set(rows.tolist()) | set(cols.tolist())


def _validate_telseq_rules(alias):
    """
    Cross-checks one alias' TELSEQ_RULES entry against the corner geometry it
    claims to describe.

    TELSEQ_RULES and the resource-file geometry are two independently
    hand-derived facts, with nothing else tying them together. This catches a
    future edit to either one (a new/changed resource file, a typo in
    TELSEQ_RULES, a changed ``Quartet.CORNER_DIAGONAL``) that silently
    desynchronizes them, instead of only surfacing as a subtly wrong fold at
    simulation time.

    Called from :func:`_telseq_block_offset`, i.e. once TelSeq machinery is
    actually used, and memoized per alias — deliberately not run at import
    time, so an unrelated ``import pressomancy.simulation`` neither pays for
    the geometry check nor fails outright on a missing resource file.

    :param alias: str | a Quartet resource-file alias (e.g. 'quartet')
    :raises ValueError: if the alias' 'top'/'bottom' ids, once shifted back
        into single-Quartet-local space, don't match the corner particles
        actually present in that alias' resource-file geometry
    """
    if alias in _VALIDATED_TELSEQ_ALIASES:
        return
    rule_params = TELSEQ_RULES[alias]
    block_size = rule_params['block_size']
    if block_size % 3 != 0:
        raise ValueError(
            f"TELSEQ_RULES['{alias}']['block_size']={block_size} is not divisible "
            "by 3 (one Quadriplex monomer is always 3 Quartets)")
    quartet_size = block_size // 3
    expected = _geometric_corner_local_ids(alias)
    for side, block_index in (('top', 1), ('bottom', 2)):
        shift = block_index * quartet_size
        side_local = {idx - shift for idx in rule_params[side]}
        if side_local != expected:
            raise ValueError(
                f"TELSEQ_RULES['{alias}']['{side}'] does not match the corner particles "
                f"of resources/{alias}.txt: expected local ids {sorted(expected)} (i.e. "
                f"{side} ids {sorted(i + shift for i in expected)}), got "
                f"{sorted(rule_params[side])}. Either TELSEQ_RULES or the resource "
                f"file/Quartet.CORNER_DIAGONAL changed without updating the other.")
    _VALIDATED_TELSEQ_ALIASES.add(alias)


def _rule_maker(fold_type, choice_id, offset, n=3, alias=None):
    length = 4
    choice_local = choice_id - offset

    if alias is None:
        rule_sets = TELSEQ_RULES.values()
    else:
        try:
            rule_sets = (TELSEQ_RULES[alias],)
        except KeyError:
            raise ValueError(f"no TelSeq rule set is defined for alias '{alias}'; known aliases: {sorted(TELSEQ_RULES)}") from None

    for rule_params in rule_sets:
        top = rule_params['top']
        bottom = rule_params['bottom']
        if choice_local in bottom:
            i = bottom.index(choice_local)
            start_on_top = False
            break
        if choice_local in top:
            i = top.index(choice_local)
            start_on_top = True
            break
    else:
        raise ValueError(
            f"choice_id={choice_id} (local={choice_local}) is not a valid TelSeq corner id for offset={offset}"
        )

    diag_pairs = []
    across_pairs = []
    free_end = 0

    if fold_type == 'parallel':
        for _ in range(n):
            next_index = (i + 1) % length
            if start_on_top:
                diag_pairs.append((bottom[i] + offset, top[next_index] + offset))
                free_end = bottom[next_index] + offset
            else:
                diag_pairs.append((top[i] + offset, bottom[next_index] + offset))
                free_end = top[next_index] + offset
            i = next_index

    elif fold_type == 'hybrid':
        idx1 = (i + 1) % length
        idx2 = (i + 2) % length
        idx3 = (i + 3) % length
        idxm1 = (i - 1) % length
        if start_on_top:
            diag_pairs.append((bottom[i] + offset, top[idx1] + offset))
            across_pairs.append((bottom[idx1] + offset, bottom[idx2] + offset))
            across_pairs.append((top[idx2] + offset, top[idx3] + offset))
            free_end = bottom[idxm1] + offset
        else:
            diag_pairs.append((top[i] + offset, bottom[idx1] + offset))
            across_pairs.append((top[idx1] + offset, top[idx2] + offset))
            across_pairs.append((bottom[idx2] + offset, bottom[idx3] + offset))
            free_end = top[idxm1] + offset

    elif fold_type == 'antiparallel':
        idxm1 = (i - 1) % length
        idx1 = (i + 1) % length
        idx2 = (i + 2) % length
        if start_on_top:
            across_pairs.append((bottom[i] + offset, bottom[idxm1] + offset))
            diag_pairs.append((top[idxm1] + offset, top[idx1] + offset))
            across_pairs.append((bottom[idx1] + offset, bottom[idx2] + offset))
            free_end = top[idx2] + offset
        else:
            across_pairs.append((top[i] + offset, top[idxm1] + offset))
            diag_pairs.append((bottom[idxm1] + offset, bottom[idx1] + offset))
            across_pairs.append((top[idx1] + offset, top[idx2] + offset))
            free_end = bottom[idx2] + offset

    else:
        raise ValueError(f"unknown fold_type: {fold_type}")
    return diag_pairs, across_pairs, free_end

class TelSeq(metaclass=Simulation_Object):
    '''
    Class that contains TelSeq relevant parameters and methods. At construction one must pass an espresso handle because the class manages parameters that are both internal and external to espresso. It is assumed that in any simulation instance there will be only one type of a TelSeq. Therefore many relevant parameters are class specific, not instance specific.
    '''
    required_features=['MORSE',]
    numInstances = 0
    simulation_type=SinglePairDict('tel_seq', 37)
    part_types = PartDictSafe({'real': 1, 'virt': 2,'to_be_magnetized':3})
    config = ObjectConfigParams(
        bond_handle=BondWrapper(espressomd.interactions.FeneBond(k=0, r_0=0, d_r_max=0)),
        diag_bond_handle=BondWrapper(espressomd.interactions.FeneBond(k=0, r_0=0, d_r_max=0)),
        across_bond_handle=BondWrapper(espressomd.interactions.FeneBond(k=0, r_0=0, d_r_max=0)),
        spacing=None,
        type='parallel',
    )

    def __init__(self, config: ObjectConfigParams):
        '''
        Initialisation of a TelSeq object requires the specification of particle size, number of parts and a handle to the espresso system
        '''
        self.sys=config['espresso_handle']
        if not (config['type'] in ['parallel', 'antiparallel','hybrid']):
            raise ValueError('type must be either parallel, antiparallel or hybrid!!!')
        self.params=config
        if self.params['associated_objects'] is None:
            warnings.warn('no associated_objects have been passed explicitly. Creating objects required to initialise object implicitly!')
            configuration=Quartet.config.specify(espresso_handle=self.sys,type='brokenA')
            quartets=[Quartet(config=configuration) for _ in range(3*self.params['n_parts'])]
            grouped_quartets = [quartets[i:i+3]
                    for i in range(0, len(quartets), 3)]
            quadriplex_config_list = [Quadriplex.config.specify(associated_objects=elem, espresso_handle=self.sys) for elem in grouped_quartets]
            self.params['associated_objects']= [Quadriplex(config=elem) for elem in quadriplex_config_list]
        self.associated_objects=self.params['associated_objects']

        self.build_function=RoutineWithArgs(func=make_centered_rand_orient_point_array,num_monomers=self.params['n_parts'],spacing=config['spacing'])
        self.orientor = np.empty(shape=3, dtype=float)
        self.type_part_dict=PartDictSafe({key: [] for key in TelSeq.part_types.keys()})
        TelSeq.numInstances += 1

    def _choose_antiparallel_phi(self, chain_dir, n_phi=720):
        chain_dir = np.asarray(chain_dir, dtype=float)
        chain_dir /= np.linalg.norm(chain_dir)

        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 1.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])

        best_phi = 0.0
        best_score = -np.inf
        for idx in range(n_phi):
            phi = 2.0 * np.pi * idx / n_phi
            side_axis = get_perpendicular(chain_dir, phi=phi)
            rotation_matrix = align_vectors(z_axis, side_axis)
            x_world = rotation_matrix @ x_axis
            y_world = rotation_matrix @ y_axis
            score = max(np.abs(np.dot(x_world, chain_dir)), np.abs(np.dot(y_world, chain_dir)))
            if score > best_score + 1e-08:
                best_score = score
                best_phi = phi
        return best_phi

    def set_object(self,  pos, ori):
        '''
        Sets a n_parts sequence of particles in espresso, asserting that the dimensionality of the pos parameter passed is commensurate with n_part.Using a generator object with the particle enumeration logic, and a try catch paradigm. Particles created here are treated as real, non_magnetic, with enabled rotations. Indices of added particles stored in self.realz_indices.append attribute. Orientation of TelSeq stored in self.orientor = self.get_orientation_vec()

        :param pos: np.array() | float, list of positions
        :return: None

        '''
        pos=np.atleast_2d(pos)
        assert len(
            pos) == self.params['n_parts'], 'there is a missmatch between the pos lenth and TelSeq n_parts'
        self.orientor = get_orientation_vec(pos)

        if not (self.params['n_parts'] == len(
            self.associated_objects)):
            raise ValueError(" there doesn't seem to be enough monomers stored!!! ")
        if not (all([x.simulation_type==self.associated_objects[0].simulation_type for x in self.associated_objects[1:]])):
            raise ValueError('all objects must have the same simulation type!')
        local_orientor = self.orientor
        if self.params['type'] == 'antiparallel':
            phi_opt = self._choose_antiparallel_phi(self.orientor)
            local_orientor = get_perpendicular(self.orientor, phi=phi_opt)
        for obj_el, pos_el in zip(self.associated_objects, pos):
            _=obj_el.set_object(pos_el, local_orientor)
        return self

    def wrap_into_Tel(self):
        '''
        associated_objects contains monomer objects (assume quadriplex). We add corner particles in each quadriplex pair to a pool of candidate corners: candidate1 and candidate2. Finally checks which corner pairs have a distance self.params['sigma']-2*fene_r0. Relies on np.isclose().
        :return: None

        '''
        for iid in range(len(self.associated_objects)):
            monomer = self.associated_objects[iid]
            candidates1 = []
            candidates1.extend(monomer.associated_objects[1].corner_particles)
            candidates1.extend(monomer.associated_objects[2].corner_particles)
            if monomer == self.associated_objects[0]:
                start_part_id = random.choice(candidates1).id
            alias = monomer.associated_objects[0].params['alias']
            offset = _telseq_block_offset(alias, candidates1)
            logging.debug(
                "wrap_into_Tel step=%s monomer_id=%s start_part_id=%s offset=%s",
                iid,
                monomer.who_am_i,
                start_part_id,
                offset,
            )
            diag_pairs, across_pairs, free_end = _rule_maker(
                self.params['type'], start_part_id, offset, alias=alias
            )
            logging.debug(
                "rule_result fold_type=%s choice_local=%s diag_pairs=%s across_pairs=%s free_end=%s",
                self.params['type'],
                start_part_id - offset,
                diag_pairs,
                across_pairs,
                free_end,
            )
            for id1, id2 in diag_pairs:
                self.bond_owned_part_pair(self.sys.part.by_id(id1), self.sys.part.by_id(id2), bond_handle=self.params['diag_bond_handle'])
            for id1, id2 in across_pairs:
                self.bond_owned_part_pair(self.sys.part.by_id(id1), self.sys.part.by_id(id2), bond_handle=self.params['across_bond_handle'])

            candidates2 = []

            try:
                monomer = self.associated_objects[iid+1]
                candidates2.extend(
                    monomer.associated_objects[1].corner_particles)
                candidates2.extend(
                    monomer.associated_objects[2].corner_particles)
                candidate_pos = np.array([x.pos for x in candidates2])

                pair_distances_free = np.linalg.norm(
                    candidate_pos - self.sys.part.by_id(free_end).pos, axis=-1
                )
                pair_distances_start = np.linalg.norm(
                    candidate_pos - self.sys.part.by_id(start_part_id).pos, axis=-1
                )
                combined_distances = np.column_stack((pair_distances_free, pair_distances_start))
                index, ref_index = np.unravel_index(np.argmin(combined_distances), combined_distances.shape)
                ref_part_id = free_end if ref_index == 0 else start_part_id
                logging.debug(
                    "handoff_choice next_monomer_id=%s index=%s ref=%s min_dist=%s",
                    monomer.who_am_i,
                    index,
                    "free_end" if ref_index == 0 else "start_part_id",
                    combined_distances[index, ref_index],
                )
                self.bond_owned_part_pair(candidates2[index], self.sys.part.by_id(ref_part_id))

                start_part_id = candidates2[index].id
                logging.debug("new_start_part_id=%s", start_part_id)
            except IndexError:
                logging.info('end of chain reached')
                continue
