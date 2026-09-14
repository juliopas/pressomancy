from .create_system import BaseTestCase, sim_inst
import numpy as np
from pressomancy.helper_functions import (get_perpendicular, partition_cuboid_volume, get_neighbours,
                                          get_neighbours_cross_lattice, fcc_lattice, min_img_dist,
                                          align_vectors, make_centered_rand_orient_point_array,
                                          get_orientation_vec, get_cross_lattice_nonintersecting_volumes,
                                          PartDictSafe, generate_random_unit_vectors,
                                          random_nested_3d_vectors_like, RoutineWithArgs,
                                          check_free_cuboid, calculate_pair_distances)

class HelperFunctionsTest(BaseTestCase):

    box_dim=np.array([2.5, 2.5, 2.5])
    num_vol_all=14
    num_vol_side=5

    sph_diam=1
    sph_rad=0.5*sph_diam

    def test_get_perpendicular_random_path_returns_unit_perpendicular(self):
        vec = np.array([0.3, -0.4, 0.5])
        unit_vec = vec / np.linalg.norm(vec)
        results = []
        for _ in range(8):
            perp = get_perpendicular(vec, phi=None)
            results.append(perp)
            self.assertTrue(np.isclose(np.linalg.norm(perp), 1.0))
            self.assertTrue(np.isclose(np.dot(perp, unit_vec), 0.0, atol=1e-10))
        # With uniform random phi, at least one sample should differ from the first.
        self.assertTrue(any(not np.allclose(results[0], sample) for sample in results[1:]))

    def test_get_perpendicular_fixed_phi_is_deterministic(self):
        vec = np.array([0.2, 0.6, -0.3])
        perp_1 = get_perpendicular(vec, phi=np.pi / 3.0)
        perp_2 = get_perpendicular(vec, phi=np.pi / 3.0)
        unit_vec = vec / np.linalg.norm(vec)
        self.assertTrue(np.allclose(perp_1, perp_2))
        self.assertTrue(np.isclose(np.linalg.norm(perp_1), 1.0))
        self.assertTrue(np.isclose(np.dot(perp_1, unit_vec), 0.0, atol=1e-10))

    def test_get_perpendicular_phi_zero_matches_expected_base_projection(self):
        vec_z = np.array([0.0, 0.0, 1.0])
        perp_z = get_perpendicular(vec_z, phi=0.0)
        self.assertTrue(np.allclose(perp_z, np.array([1.0, 0.0, 0.0])))

        vec_x = np.array([1.0, 0.0, 0.0])
        perp_x = get_perpendicular(vec_x, phi=0.0)
        self.assertTrue(np.allclose(perp_x, np.array([0.0, 1.0, 0.0])))


class RegressionTest(BaseTestCase):
    """Claude found some bugs (mostly minor), so I had him make tests that would catch them."""

    def test_align_vectors_handles_every_antiparallel_axis(self):
        """C1: the -x case used to project the reference onto zero and return NaN."""
        axes = [np.array(v, dtype=float) for v in
                ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]
        for v1 in axes:
            v2 = -v1
            rot = align_vectors(v1, v2)
            with self.subTest(v1=v1.tolist()):
                self.assertTrue(np.all(np.isfinite(rot)), msg=f"non-finite rotation for {v1}")
                self.assertTrue(np.allclose(rot @ v1, v2, atol=1e-12))
                # A rotation, not a reflection.
                self.assertAlmostEqual(float(np.linalg.det(rot)), 1.0, places=12)

    def test_random_orientation_is_uniform_on_the_sphere(self):
        """P1: phi was drawn uniformly, which oversampled the poles."""
        rng_backup = np.random.get_state()
        try:
            np.random.seed(20240601)
            n_draws = 20000
            z = np.array([
                make_centered_rand_orient_point_array(
                    center=np.zeros(3), sphere_radius=1.0, num_monomers=2, spacing=1.0)[0][0][2]
                for _ in range(n_draws)
            ])
        finally:
            np.random.set_state(rng_backup)

        # Uniform on the sphere => |cos(phi)| is uniform on [0, 1], mean 0.5.
        self.assertAlmostEqual(float(np.abs(z).mean()), 0.5, delta=0.02)
        self.assertAlmostEqual(float((np.abs(z) > 0.9).mean()), 0.10, delta=0.02)

    def test_coincident_lattices_are_never_reported_free(self):
        """P2: zero-distance pairs were filtered out before the separation test."""
        box = np.array([30.0, 30.0, 30.0])
        centers = fcc_lattice(radius=3.0, box_dim=box, mode="pack")
        res = get_cross_lattice_nonintersecting_volumes(
            current_lattice_centers=centers, current_lattice_grouped_part_pos=centers,
            current_lattice_diam=6.0, other_lattice_centers=centers,
            other_lattice_grouped_part_pos=centers, other_lattice_diam=6.0,
            box_lengths=box)
        free = [key for key, val in res.items() if all(val)]
        self.assertEqual(free, [], msg="a lattice superposed on itself must not be free")

    def test_orientation_vec_is_real_and_sign_pinned(self):
        """P3: eig could return a complex type, and the sign was arbitrary."""
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.05],
                        [2.0, 0.2, 0.10], [3.0, 0.3, 0.15]])
        axis = get_orientation_vec(pos)
        self.assertEqual(axis.dtype.kind, 'f')
        self.assertTrue(np.all(np.isfinite(axis)))
        self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=12)
        # Deterministic, and oriented first -> last.
        self.assertTrue(np.allclose(axis, get_orientation_vec(pos)))
        self.assertGreater(float(np.dot(axis, pos[-1] - pos[0])), 0.0)
        self.assertTrue(np.allclose(get_orientation_vec(pos[::-1]), -axis, atol=1e-10))

    def test_part_dict_safe_strict_mode(self):
        """C2: an integer-valued map must not auto-create a list for a typo."""
        strict = PartDictSafe({'real': 1}, default_factory=None)
        with self.assertRaises(KeyError):
            strict['nonmang']
        self.assertEqual(strict['real'], 1)
        strict['virt'] = 2
        self.assertEqual(dict(strict), {'real': 1, 'virt': 2})
        # The list-valued bookkeeping keeps its default, which objects rely on.
        lenient = PartDictSafe({'real': []})
        self.assertEqual(lenient['virt'], [])

    def test_random_unit_vectors_honour_the_supplied_rng(self):
        """C14: the rng argument was accepted and then ignored."""
        first = generate_random_unit_vectors(5, rng=np.random.default_rng(1234))
        same = generate_random_unit_vectors(5, rng=np.random.default_rng(1234))
        other = generate_random_unit_vectors(5, rng=np.random.default_rng(9999))
        self.assertTrue(np.allclose(first, same))
        self.assertFalse(np.allclose(first, other))
        nested_a = random_nested_3d_vectors_like(np.zeros((4, 3)), rng=np.random.default_rng(7))
        nested_b = random_nested_3d_vectors_like(np.zeros((4, 3)), rng=np.random.default_rng(7))
        self.assertTrue(np.allclose(nested_a, nested_b))

    def test_impossible_placement_raises_instead_of_hanging(self):
        """C11: the retry loop had no cap and would spin forever."""
        routine = RoutineWithArgs(func=make_centered_rand_orient_point_array,
                                  num_monomers=4, monomer_size=100.0, spacing=1.0)
        with self.assertRaises(ValueError) as ctx:
            partition_cuboid_volume(box_lengths=np.array([30.0, 30.0, 30.0]),
                                    num_spheres=20, sphere_diameter=6.0,
                                    routine_per_volume=routine)
        self.assertIn("without overlapping its neighbours", str(ctx.exception))

    def test_partition_cuboid_volume_handles_a_non_cubic_box(self):
        # Dense on purpose: at 8 volumes of diameter 5 the nearest inter-volume
        # separation is ~3-4, so the invariant would hold even with the overlap
        # rejection removed and the test would prove nothing. At these settings the
        # worst separation lands at ~1.00-1.07, i.e. the rejection is load-bearing.
        n_vol, n_mon, mono_size = 100, 4, 1.0
        for box in (np.array([30.0, 30.0, 30.0]), np.array([30.0, 10.0, 60.0])):
            routine = RoutineWithArgs(func=make_centered_rand_orient_point_array,
                                      num_monomers=n_mon, monomer_size=mono_size,
                                      spacing=mono_size)
            rng_backup = np.random.get_state()
            try:
                np.random.seed(4242)
                _, positions, orientations = partition_cuboid_volume(
                    box_lengths=box, num_spheres=n_vol, sphere_diameter=3.0,
                    routine_per_volume=routine)
            finally:
                np.random.set_state(rng_backup)

            label = "cubic" if box[0] == box[1] == box[2] else "non-cubic"
            with self.subTest(box=label):
                self.assertEqual(np.asarray(positions).shape, (n_vol, n_mon, 3))
                self.assertEqual(np.asarray(orientations).shape, (n_vol, n_mon, 3))
                flat = np.asarray(positions).reshape(-1, 3)
                owner = np.repeat(np.arange(n_vol), n_mon)
                worst, worst_pair = np.inf, None
                for i in range(len(flat)):
                    for j in range(i + 1, len(flat)):
                        if owner[i] == owner[j]:
                            continue
                        d = float(np.linalg.norm(
                            min_img_dist(flat[i], flat[j], box_dim=box)))
                        if d < worst:
                            worst, worst_pair = d, (i, j)
                self.assertGreater(
                    worst, mono_size - 1e-9,
                    msg=(f"{label} box: monomers from different volumes are "
                         f"{worst:.4f} apart (< {mono_size}) at {worst_pair}"))

    def test_check_free_cuboid_folds_drifted_positions(self):
        """P4: espresso stores unfolded positions, so a drifted particle read as outside."""
        cuboid = np.array([10.0, 10.0, 10.0])
        try:
            probe = sim_inst.sys.part.add(pos=[5.0, 5.0, 5.0], type=99)
            self.assertFalse(check_free_cuboid(sim_inst.sys, cuboid),
                             msg="a particle sitting inside the cuboid must be seen")
            probe.pos = np.array([5.0, 5.0, 5.0]) + np.asarray(sim_inst.sys.box_l)
            self.assertFalse(check_free_cuboid(sim_inst.sys, cuboid),
                             msg="the same particle displaced by one box must still be seen")
            probe.pos = [25.0, 25.0, 25.0]
            self.assertTrue(check_free_cuboid(sim_inst.sys, cuboid),
                            msg="a particle genuinely outside must leave the cuboid free")
        finally:
            self.cleanup()


    def test_calculate_pair_distances_matches_index_pair_construction(self):
        """3.4.2: broadcasting must reproduce the explicit N*M index list exactly."""
        from itertools import product
        from pressomancy.helper_functions import min_img_dist
        rng = np.random.default_rng(11)
        box = np.array([30.0, 25.0, 40.0])
        for n, m in ((1, 1), (5, 5), (3, 7), (12, 4)):
            a = rng.random((n, 3)) * box
            b = rng.random((m, 3)) * box
            combos = np.array(list(product(range(n), range(m))))
            reference = np.linalg.norm(
                min_img_dist(a[combos[:, 0]], b[combos[:, 1]], box_dim=box), axis=-1)
            with self.subTest(shape=(n, m)):
                got = calculate_pair_distances(a, b, box_lengths=box)
                self.assertEqual(got.shape, reference.shape)
                np.testing.assert_allclose(got, reference, rtol=0, atol=0)

    def test_get_orientation_vec_matches_component_wise_tensor(self):
        """3.4.3: the covariance product must reproduce the six component means."""
        rng = np.random.default_rng(12)
        for n in (3, 10, 50):
            pos = np.cumsum(rng.random((n, 3)), axis=0)
            cm = pos.mean(axis=0)
            comp = np.array([[np.mean([(p[i] - cm[i]) * (p[j] - cm[j]) for p in pos])
                              for j in range(3)] for i in range(3)])
            res, egiv = np.linalg.eigh(comp)
            expected = egiv[:, np.argmax(res)]
            expected = expected / np.linalg.norm(expected)
            if np.dot(expected, pos[-1] - pos[0]) < 0:
                expected = -expected
            with self.subTest(n=n):
                np.testing.assert_allclose(get_orientation_vec(pos), expected, atol=1e-12)


class PartitioningTest(BaseTestCase):

    box_dim=np.array([2.5, 2.5, 2.5])
    num_vol_all=14
    num_vol_side=5

    sph_diam=1
    sph_rad=0.5*sph_diam
    rect_box = np.array([10.0, 20.0, 30.0])

    def test_get_neighbours(self):
        control= {0: [2,], 1: [2,],2: [0, 1, 3, 4,], 3: [2, ], 4: [2,]}
        sphere_centers_short, _,_=partition_cuboid_volume(self.box_dim,self.num_vol_side,self.sph_diam, flag='norand')
        neigh=get_neighbours(sphere_centers_short,self.box_dim,cutoff=self.sph_diam)
        neigh_sets = {key: set(val) for key, val in neigh.items()}
        control_sets = {key: set(val) for key, val in control.items()}
        self.assertEqual(neigh_sets,control_sets,'the get_neighbour method failed to reproduce correct neighbour pairs for a single face of an fcc lattice')

    def test_get_neighbours_cross_lattice(self):

        control={0: [0, 2, 5, 6], 1: [1, 2, 5, 7], 2: [0, 1, 2, 3, 4, 5, 6, 7, 8], 3: [2, 3, 6, 8], 4: [2, 4, 7, 8]}
        sphere_centers_long, _,_=partition_cuboid_volume(self.box_dim,self.num_vol_all,self.sph_diam,flag='norand')

        sphere_centers_short, _,_=partition_cuboid_volume(self.box_dim,self.num_vol_side,self.sph_diam,flag='norand')

        neigh=get_neighbours_cross_lattice(sphere_centers_short,sphere_centers_long,self.box_dim,cutoff=self.sph_diam)
        self.assertEqual(neigh,control,'the get_neighbour method failed to reproduce correct neighbour pairs for a single face of an fcc lattice')

    def test_get_neighbours_rectangular(self):
        box = self.rect_box
        cut = 2.0
        points = np.array(
            [
                # Across x-boundary (wrap), within cutoff.
                [0.5, 10.0, 15.0],
                [9.7, 10.0, 15.0],
                # Central point should not be within cutoff of either.
                [5.0, 10.0, 15.0],
            ]
        )
        neigh = get_neighbours(points, box, cutoff=cut)

        self.assertIn(1, neigh[0])
        self.assertIn(0, neigh[1])
        self.assertNotIn(2, neigh[0])
        self.assertNotIn(2, neigh[1])

    def test_get_neighbours_cross_lattice_rectangular(self):
        box = self.rect_box
        cut = 2.0
        lattice_a = np.array(
            [
                # Nearest neighbor only across x-boundary.
                [0.5, 10.0, 15.0],
                # Nearest neighbor only inside the box.
                [5.0, 10.0, 15.0],
            ]
        )
        lattice_b = np.array(
            [
                [9.7, 10.0, 15.0],
                [5.9, 10.0, 15.0],
                # Far in y; should be excluded.
                [5.0, 18.5, 15.0],
            ]
        )
        neigh = get_neighbours_cross_lattice(lattice_a, lattice_b, box, cutoff=cut)

        expected = {0: [0], 1: [1]}
        self.assertEqual(neigh, expected)

    def test_get_neighbours_multiple_rectangular(self):
        box = self.rect_box
        cut = 0.35
        points = np.array(
            [
                # Point at origin corner.
                [0.2, 0.2, 0.2],
                # Within cutoff across x-boundary.
                [9.9, 0.2, 0.2],
                # Within cutoff across y-boundary.
                [0.2, 19.9, 0.2],
                # Within cutoff across z-boundary.
                [0.2, 0.2, 29.9],
                # Far center point.
                [5.0, 10.0, 15.0],
            ]
        )
        neigh = get_neighbours(points, box, cutoff=cut)
        expected = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0], 4: []}
        neigh_sets = {key: set(val) for key, val in neigh.items()}
        expected_sets = {key: set(val) for key, val in expected.items()}
        self.assertEqual(neigh_sets, expected_sets)

    def test_get_neighbours_cross_lattice_multiple_rectangular(self):
        box = self.rect_box
        cut = 0.35
        lattice_a = np.array(
            [
                # Should match two neighbors across x/y boundaries.
                [0.2, 0.2, 0.2],
                # Should match only the nearby internal point.
                [5.0, 10.0, 15.0],
            ]
        )
        lattice_b = np.array(
            [
                [9.9, 0.2, 0.2],
                [0.2, 19.9, 0.2],
                [5.1, 10.0, 15.0],
                # Far away; should be excluded.
                [8.0, 8.0, 8.0],
            ]
        )
        neigh = get_neighbours_cross_lattice(lattice_a, lattice_b, box, cutoff=cut)

        expected = {0: [0, 1], 1: [2]}
        self.assertEqual(neigh, expected)

# control={0: [0, 1, 2, 3, 5, 6, 9],
        #      1: [ 0,  1,  2,  4,  5,  7, 10],
        #      2: [ 0,  1,  2,  3,  4,  5,  6,  7,  8, 11],
        #      3: [ 0,  2,  3,  4,  6,  8, 12],
        #      4:[ 1,  2,  3,  4,  7,  8, 13]}


class FccLatticeTest(BaseTestCase):
    """Checks that fcc_lattice actually produces an FCC arrangement of
    touching spheres."""

    # (radius, box_dim, scalling_factor)
    cases = [
        (0.5, np.array([6.0, 6.0, 6.0]), 1.0),
        (0.3, np.array([5.0, 5.0, 5.0]), 1.0),
        (0.5, np.array([10.0, 6.0, 8.0]), 1.0),
        (0.5, np.array([7.0, 7.0, 7.0]), 0.5),
        (1.0, np.array([9.0, 9.0, 12.0]), 1.0),
    ]
    cubic = [c for c in cases if len(set(c[1].tolist())) == 1]
    tol = 1e-6

    def _build(self, radius, box, sf, mode):
            return fcc_lattice(radius, box, scaling_factor=sf, mode=mode)

    def _axis_steps(self, points, decimals=8):
        out = []
        for d in range(3):
            uniq = np.unique(np.round(points[:, d], decimals))
            out.append(np.diff(uniq))
        return out

    def _min_img_dist_self(self, points, box_dim):
        dists = min_img_dist(points[:, None, :], points[None, :, :], box_dim=box_dim)
        dists = np.linalg.norm(dists, axis=-1)
        np.fill_diagonal(dists, np.inf) # exclude self distances
        return dists

    def test_spacing_pack_touches_crystal_only_expands(self):
        """Neither mode may place a sphere, or a periodic image of one, closer
        than 2r. Pack mode sits exactly at that bound; crystal mode may only
        push neighbours apart -- never pull them in -- and does so by widening
        the sub-lattice pitch, never by dropping or adding sites."""
        for radius, box, sf in self.cases:
            tight = self._build(radius, box, sf, mode="pack")
            loose = self._build(radius, box, sf, mode="crystal")
            touch = 2 * radius * sf
            nn = {}
            for mode, points in (("pack", tight), ("crystal", loose)):
                with self.subTest(radius=radius, box=box, sf=sf, mode=mode):
                    nn[mode] = self._min_img_dist_self(points, box).min()
                    self.assertGreaterEqual(
                        nn[mode], touch - self.tol,
                        f"{mode}: closest pair (incl. periodic images) is "
                        f"{nn[mode]}, below the touching distance {touch}")
            with self.subTest(radius=radius, box=box, sf=sf, mode="compare"):
                self.assertAlmostEqual(nn["pack"], touch, delta=self.tol)
                self.assertGreaterEqual(nn["crystal"], nn["pack"] - self.tol)
                # thight may have an extra "half-lattice-step"
                self.assertGreaterEqual(len(tight), len(loose))
                for d, (st, sl) in enumerate(zip(self._axis_steps(tight), self._axis_steps(loose))):
                    self.assertGreaterEqual(sl.min(), st.min() - self.tol,
                                            f"axis {d} compressed")

    def test_crystal_tiles_the_box_exactly(self):
            """In crystal mode the pitch divides the box into an even number of
            half-steps, so the seam gap equals the bulk spacing: no void, and the
            wrap partner is always of opposite (i+j+k) parity."""
            for radius, box, sf in self.cases:
                with self.subTest(radius=radius, box=box, sf=sf):
                    points = self._build(radius, box, sf, mode="crystal")
                    for d, steps in enumerate(self._axis_steps(points)):
                        pitch = steps[0]
                        self.assertTrue(np.allclose(steps, pitch, atol=self.tol),
                                        f"axis {d} spacing is not uniform: {np.unique(steps)}")
                        n_half = box[d] / pitch
                        self.assertAlmostEqual(n_half, round(n_half), delta=self.tol,
                                               msg=f"axis {d} does not tile the box")
                        seam = box[d] - points[:, d].max()
                        self.assertAlmostEqual(seam, pitch, delta=self.tol,
                                               msg=f"axis {d} seam gap != bulk pitch")

    def test_coordination_number_is_twelve(self):
        """12 nearest neighbours is a signature of FCC. Crystal cubic boxes
        tile seamlessly so every site has its full shell; in pack mode the
        seam void breaks the shell, so only sites far from it are checked."""
        for radius, box, sf in self.cubic:
            for mode in ("pack", "crystal"):
                with self.subTest(radius=radius, box=box, sf=sf, mode=mode):
                    points = self._build(radius, box, sf, mode)
                    dists = self._min_img_dist_self(points, box)
                    counts = (dists <= dists.min() + self.tol).sum(axis=1)
                    if mode == "crystal":
                        keep = np.ones(len(points), dtype=bool)
                    else:
                        keep = np.all((points > points.min(axis=0)) & (points < points.max(axis=0)), axis=1)
                        self.assertGreater(keep.sum(), 0, "box too small for this check")
                    self.assertTrue(
                        np.all(counts[keep] == 12),
                        f"coordination numbers were {sorted(set(counts[keep].tolist()))}, "
                        "expected all 12",
                    )

    def test_max_points_per_side_caps_the_lattice(self):
        """The cap is on distinct coordinates per axis; exceeding it must widen
        the lattice constant rather than emit more points."""
        for mode in ("pack", "crystal"):
            for cap in (4, 10):
                with self.subTest(mode=mode, cap=cap):
                    points = fcc_lattice(0.5, [60.0, 60.0, 60.0],
                                            max_points_per_side=cap, mode=mode)
                    for d in range(3):
                        n_uniq = len(np.unique(np.round(points[:, d], 9)))
                        self.assertLessEqual(n_uniq, cap, f"axis {d} exceeded the cap")

    def test_rejects_bad_input(self):
        for kwargs in ({"radius": -1.0}, {"radius": 0.0}, {"scaling_factor": 0.0}):
            with self.subTest(**kwargs):
                call = {"radius": 0.5, "box_dim": [6.0, 6.0, 6.0], **kwargs}
                self.assertRaises(ValueError, fcc_lattice, **call)
        # box too small to hold a single conventional cell
        self.assertRaises(ValueError, fcc_lattice, 5.0, [1.0, 1.0, 1.0])
