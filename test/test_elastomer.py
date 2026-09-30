import signal
from pressomancy.simulation import Elastomer, PointDipolePermanent
from pressomancy.geometry import WCA_CONTACT_FACTOR
from .create_system import sim_inst, BaseTestCase, BoxTestCase
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

class ElastomerConfigTest(BoxTestCase):
    box_dim = (8, 8, 16)  # the default Elastomer fills the bottom quarter: ~80 beads here

    def test_size_is_refused(self):
        """An Elastomer lays itself out; its beads come from sigma, never from a placement size."""
        with self.assertRaisesRegex(ValueError, "Elastomer takes no size"):
            Elastomer(config=Elastomer.config.specify(size=1., espresso_handle=sim_inst.sys))

    def test_default_builds_beads_of_sigma_one(self):
        """The default Elastomer (sigma=1) fills the bottom quarter of the box with its inferred
        n_parts, and its lowest beads rest one bead radius, WCA_CONTACT_FACTOR / 2, above the substrate."""
        # bond_cutoff is lowered due to small box. It doe snot affect the purpose of the test
        instance = Elastomer(config=Elastomer.config.specify(espresso_handle=sim_inst.sys, seed=sim_inst.seed,
                                                              bond_cutoff=2.))
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])
        beads = np.array([p.pos for p in instance.network_beads()])
        # n_parts is inferred from box_E's volume at 0.3 volume fraction
        self.assertEqual(instance.params['n_parts'], 77)
        self.assertEqual(len(beads), instance.params['n_parts'])
        self.assertAlmostEqual(beads[:, 2].min(), instance._substrate_size + WCA_CONTACT_FACTOR / 2)

        # substrate/real WCA sigma
        substrate_real_sigma = sim_inst.sys.non_bonded_inter[
            sim_inst.part_types['substrate'], sim_inst.part_types['real']].wca.sigma
        self.assertAlmostEqual(substrate_real_sigma,
                                instance.params['sigma'] / 2 + 0.5 / WCA_CONTACT_FACTOR)

        # cure_elastomer pins any bead sitting below z_pin = substrate_size + bead/2 + bead/4
        # (cure_elastomer) to move only in x/y.
        instance.cure_elastomer()
        z_pin = instance._substrate_size + WCA_CONTACT_FACTOR / 2 + WCA_CONTACT_FACTOR / 4
        for hndl in instance.network_beads():
            self.assertEqual(bool(hndl.fix[2]), bool(hndl.pos[2] < z_pin))


class ElastomerTest(BoxTestCase):
    box_E = [3, 3, 9]
    # x and y: twice the default bond_cutoff (5), the neighbour search's minimum-image limit;
    # z: box_E's height
    box_dim = (10, 10, 9)
    part_size = 1.  # a bead's contact diameter
    sigma = part_size / WCA_CONTACT_FACTOR
    conf_magn = PointDipolePermanent.config.specify(dipm=1., espresso_handle=sim_inst.sys)

    def _elastomer(self, n_parts=10, associated=True):
        """A stored and placed Elastomer in box_E, its beads PointDipolePermanent children if `associated`."""
        mag_part = [PointDipolePermanent(config=self.conf_magn) for _ in range(n_parts)] if associated else None
        instance = Elastomer(config=Elastomer.config.specify(
            box_E=self.box_E, n_parts=n_parts, sigma=self.sigma, espresso_handle=sim_inst.sys,
            seed=sim_inst.seed, associated_objects=mag_part))
        if associated:
            sim_inst.store_objects(mag_part)
        sim_inst.store_objects([instance])
        sim_inst.set_objects([instance])
        return instance

    @staticmethod
    def _network_bonds(instance):
        """(bond, bead, partner) for every bond the network holds, and each bead's degree counted from both ends."""
        bonds = [(bond, hndl, sim_inst.sys.part.by_id(partner))
                 for hndl in instance.network_beads() for bond, partner in hndl.bonds]
        degree = Counter({hndl.id: 0 for hndl in instance.network_beads()})
        for _, hndl, partner in bonds:
            degree[hndl.id] += 1
            degree[partner.id] += 1
        return bonds, degree

    def test_cure_against_either_substrate(self):
        """Curing against a substrate lattice or a wall bonds each pair once, near max_bonds, in cutoff and k range."""
        substrate_pos = {(x, y, 0.5) for y in (0.5, 1.5, 2.5) for x in (0.5, 1.5, 2.5)}

        def part(instance):
            self.assertEqual(set(map(tuple, (p.pos for p in instance.substrate))), substrate_pos)

        def wall(instance):
            instance.remove_substrate()
            self.assertIsNone(instance.substrate)
            self.assertEqual(len(sim_inst.sys.part.select(type=sim_inst.part_types["substrate"])), 0)
            instance.create_substrate(geometry="wall")
            self.assertEqual(len(sim_inst.sys.part.select(type=sim_inst.part_types["substrate"])), 0)

        for label, substrate in [('part substrate', part), ('wall substrate', wall)]:
            with self.subTest(label):
                BaseTestCase.cleanup(self.box_dim)
                instance = self._elastomer()
                substrate(instance)
                instance.cure_elastomer()

                bonds, degree = self._network_bonds(instance)
                max_bonds = instance.params['max_bonds']
                self.assertEqual(len(degree), instance.params['n_parts'])
                self.assertEqual(len(bonds), len({frozenset((hndl.id, partner.id)) for _, hndl, partner in bonds}))
                self.assertGreater(min(degree.values()), 0)
                # a lonely bead's retry may take partners to max_bonds + 2; it bonds <= n_nghb=3 of them,
                # so one retry's worth of beads ends up above max_bonds
                self.assertLessEqual(max(degree.values()), max_bonds + 2)
                self.assertLessEqual(sum(d > max_bonds for d in degree.values()), 3)
                mean_degree = np.mean(list(degree.values()))
                self.assertTrue(max_bonds - 2 <= mean_degree <= max_bonds, mean_degree)
                # unfolded: bonds are made with z periodicity off, and sys.distance would re-fold z here
                self.assertLessEqual(max(np.linalg.norm(hndl.pos - partner.pos) for _, hndl, partner in bonds),
                                     instance.params['bond_cutoff'])
                k = [bond.params['k'] for bond, _, _ in bonds]
                self.assertGreaterEqual(min(k), instance.params['bond_K_lims'][0])
                self.assertLessEqual(max(k), instance.params['bond_K_lims'][1])

    def test_thermostat_state_is_preserved(self):
        """The mixing run and the snapshot/restore pair hand the thermostat back as it was."""
        instance = self._elastomer()

        # langevin: through the mixing run, which swaps the thermostat out and back
        sim_inst.sys.thermostat.set_langevin(kT=0.7, gamma=3.5, seed=41)
        instance.mix_elastomer_stuff(n_iter=0)

        self.assertFalse(sim_inst.thermostat_is_off())
        self.assertTrue(sim_inst.sys.thermostat.langevin.is_active)
        self.assertAlmostEqual(sim_inst.sys.thermostat.kT, 0.7)
        np.testing.assert_allclose(np.copy(sim_inst.sys.thermostat.langevin.gamma), 3.5)
        self.assertEqual(sim_inst.sys.thermostat.langevin.seed, 41)

        # brownian: through the snapshot/restore pair directly
        sim_inst.sys.thermostat.turn_off()
        sim_inst.sys.thermostat.set_brownian(kT=0.9, gamma=2.5, seed=43)
        snapshot = instance._snapshot_thermostat_state()
        sim_inst.sys.thermostat.turn_off()
        instance._restore_thermostat_state(snapshot)

        self.assertFalse(sim_inst.thermostat_is_off())
        self.assertTrue(sim_inst.sys.thermostat.brownian.is_active)
        self.assertAlmostEqual(sim_inst.sys.thermostat.kT, 0.9)
        np.testing.assert_allclose(np.copy(sim_inst.sys.thermostat.brownian.gamma), 2.5)
        self.assertEqual(sim_inst.sys.thermostat.brownian.seed, 43)

    def _four_beads(self):
        """An Elastomer of four beads well inside r_catch=0.5 of each other."""
        instance = self._elastomer(n_parts=4, associated=False)
        positions = [[0.90, 1.00, 1.6], [1.10, 1.00, 1.6], [1.00, 0.90, 1.6], [1.00, 1.10, 1.6]]
        for part, pos in zip(instance.type_part_dict["real"], positions):
            part.pos = pos
        return instance

    def test_bond_to_neighbors(self):
        """Four beads well inside r_catch of each other bond every pair exactly once."""
        instance = self._four_beads()
        instance.bond_to_neighbors(parts=sim_inst.sys.part.select(type=sim_inst.part_types["real"]), n_nghb=3,
                                   bond_k=(0.04, 0.06), r_catch=0.5)

        bonds, degree = self._network_bonds(instance)
        self.assertEqual(sorted(degree.values()), [3, 3, 3, 3])
        self.assertEqual(sorted(frozenset((hndl.id, partner.id)) for _, hndl, partner in bonds),
                         sorted(frozenset(pair) for pair in combinations(degree, 2)))

    def test_bond_to_neighbors_retry_skips_saturated_partners(self):
        """cure_elastomer's retry: a lonely bead bonds, from its own end, only to partners below the raised cap."""
        instance = self._four_beads()
        lonely, full, near, free = (hndl.id for hndl in instance.network_beads())
        m = instance.params['max_bonds']
        n_bonds_dict = defaultdict(int, {full: m + 2, near: m})
        instance.bond_to_neighbors(parts=sim_inst.sys.part.by_ids([lonely]),
                                   candidate_parts=sim_inst.sys.part.select(type=sim_inst.part_types["real"]),
                                   max_bonds=m + 2, n_bonds_dict=n_bonds_dict, bond_k=(0.04, 0.06), r_catch=0.5,
                                   n_nghb=3)

        bonds, _ = self._network_bonds(instance)
        self.assertEqual(sorted((hndl.id, partner.id) for _, hndl, partner in bonds),
                         [(lonely, near), (lonely, free)])
        self.assertEqual(dict(n_bonds_dict), {lonely: 2, full: m + 2, near: m + 1, free: 1})

    def test_relax_overlaps(self):
        """A bead below the wall plane is lifted back; a foreign WCA sigma and a too-deep overlap are refused."""
        rows = [  # (label, associated, steric keys, steric sigma, push, refusal)
            ("own beads", False, ("real",), self.sigma, 0.15, None),
            ("associated objects' beads", True, ("pdp_real",), self.sigma, 0.15, None),
            ("another sigma", False, ("real",), self.part_size, 0.15,
             (ValueError, "differs from the Elastomer's sigma")),
            ("deeper than the budget", False, ("real",), self.sigma, 0.3, (RuntimeError, "too overlapped")),
        ]
        for label, associated, keys, steric_sigma, push, refusal in rows:
            with self.subTest(label):
                BaseTestCase.cleanup(self.box_dim)
                instance = self._elastomer(associated=associated)
                sim_inst.set_steric(keys, sigma=steric_sigma)  # every network-bead type's own WCA sigma
                if associated:  # the beads are the children's, so an elastomer-only lookup would find none
                    self.assertEqual(len(instance.network_beads()), 10)
                    self.assertEqual(len(instance.type_part_dict["real"]), 0)
                # the lowest bead, straight down from the wall plane (one contact radius above the substrate):
                # it overlaps the substrate only, never another bead, and clears in push / max_displacement steps
                lowest = min(instance.network_beads(), key=lambda hndl: hndl.pos[2])
                lowest.pos = [lowest.pos[0], lowest.pos[1], instance._substrate_size + instance._bead_size / 2 - push]
                if refusal:
                    with self.assertRaisesRegex(*refusal):
                        instance.relax_overlaps()
                    continue

                result = instance.relax_overlaps()

                self.assertEqual(set(result.keys()), {"steps", "converged", "max_f_before", "max_f_after",
                                                          "gap_before", "gap_after"})
                self.assertAlmostEqual(result["gap_before"], -push, places=6)
                self.assertTrue(result["converged"])
                self.assertGreater(result["gap_after"], -1e-4)

    def test_relax_overlaps_budget_is_a_hard_limit(self):
        """A pinned bead under a large force makes steepest descent report 0 steps; relaxing still ends at once."""
        instance = self._elastomer(associated=False)
        sim_inst.set_steric(("real",), sigma=self.sigma)
        pinned = max(instance.network_beads(), key=lambda hndl: hndl.pos[2])
        pinned.fix = [True] * 3
        pinned.ext_force = [0., 0., 100.]

        def hung(signum, frame):
            raise TimeoutError("relax_overlaps did not return: the step budget is not a hard limit")
        self.addCleanup(signal.signal, signal.SIGALRM, signal.signal(signal.SIGALRM, hung))
        self.addCleanup(signal.alarm, 0)
        signal.alarm(10)  # the loop returns to Python every chunk, so the alarm can interrupt a hang
        with self.assertRaisesRegex(RuntimeError, "residual force"):
            instance.relax_overlaps()

    def test_bond_arguments_are_validated(self):
        """random_harmonic_bonds refuses a negative r_cut, an r_cut inside r_catch and a single bond_k."""
        instance = self._elastomer()
        rows = [  # (label, kwargs, fragment)
            ("negative r_cut", dict(r_catch=2., bond_k=(0.04, 0.06), r_cut=-1), "negative"),
            ("r_cut inside r_catch", dict(r_catch=2., bond_k=(0.04, 0.06), r_cut=1.), "larger than any bond length"),
            ("single bond_k", dict(r_catch=2., bond_k=0.05), "interval"),
        ]
        for label, kwargs, fragment in rows:
            with self.subTest(label), self.assertRaisesRegex(ValueError, fragment):
                instance.random_harmonic_bonds(**kwargs)
