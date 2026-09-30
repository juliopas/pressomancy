from .create_system import BaseTestCase, sim_inst
import unittest
from itertools import product
from unittest import mock
import numpy as np
from pressomancy import geometry
from pressomancy.infra import RoutineWithArgs
from pressomancy.geometry import get_perpendicular, partition_cuboid_volume, partition_cubic_volume_oriented_rectangles, get_neighbours, get_neighbours_cross_lattice, fcc_lattice, fold_coords, min_img_dist, align_vectors, make_centered_rand_orient_point_array, get_orientation_vec, get_cross_lattice_nonintersecting_volumes, generate_random_unit_vectors, random_nested_3d_vectors_like, check_free_cuboid, calculate_pair_distances, require_min_global_cut

class HelperFunctionsTest(unittest.TestCase):

    def test_get_perpendicular(self):
        """Random phi gives varying unit perpendiculars; phi=0 gives the documented base projection."""
        with self.subTest(case='random_path_returns_unit_perpendicular'):
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

        with self.subTest(case='phi_zero_matches_expected_base_projection'):
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
                self.assertTrue(np.all(np.isfinite(rot)))
                self.assertTrue(np.allclose(rot @ v1, v2, atol=1e-12))
                # A rotation, not a reflection.
                self.assertAlmostEqual(float(np.linalg.det(rot)), 1.0, places=12)

    def test_random_orientation_is_uniform_on_the_sphere(self):
        """P1: phi was drawn uniformly, which oversampled the poles."""
        rng_backup = np.random.get_state()
        try:
            np.random.seed(20240601)
            n_draws = 5000
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
            current_lattice_centers=centers, current_lattice_diam=6.0,
            other_lattice_centers=centers, other_lattice_diam=6.0,
            box_dim=box)
        free = [key for key, val in res.items() if all(val)]
        self.assertEqual(free, [])

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
            partition_cuboid_volume(box_dim=np.array([30.0, 30.0, 30.0]),
                                    num_spheres=20, sphere_diameter=6.0,
                                    routine_per_volume=routine)
        self.assertIn("without overlapping its neighbours", str(ctx.exception))

    def test_partition_cuboid_volume_handles_a_non_cubic_box(self):
        """Monomers of different volumes stay a monomer size apart, in a cubic and a non-cubic box."""
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
                    box_dim=box, num_spheres=n_vol, sphere_diameter=3.0,
                    routine_per_volume=routine)
            finally:
                np.random.set_state(rng_backup)

            label = "cubic" if box[0] == box[1] == box[2] else "non-cubic"
            with self.subTest(box=label):
                self.assertEqual(np.asarray(positions).shape, (n_vol, n_mon, 3))
                self.assertEqual(np.asarray(orientations).shape, (n_vol, n_mon, 3))
                flat = np.asarray(positions).reshape(-1, 3)
                owner = np.repeat(np.arange(n_vol), n_mon)
                dist = calculate_pair_distances(flat, flat, box_dim=box).reshape(len(flat), len(flat))
                dist[owner[:, None] == owner[None, :]] = np.inf
                worst_pair = np.unravel_index(np.argmin(dist), dist.shape)
                worst = float(dist[worst_pair])
                self.assertGreater(
                    worst, mono_size - 1e-9,
                    msg=(f"{label} box: monomers from different volumes are "
                         f"{worst:.4f} apart (< {mono_size}) at {worst_pair}"))

    def test_check_free_cuboid_folds_drifted_positions(self):
        """P4: espresso stores unfolded positions, so a drifted particle read as outside."""
        cuboid = np.array([10.0, 10.0, 10.0])
        probe = sim_inst.sys.part.add(pos=[5.0, 5.0, 5.0], type=99)
        self.assertFalse(check_free_cuboid(sim_inst.sys, cuboid))
        probe.pos = np.array([5.0, 5.0, 5.0]) + np.asarray(sim_inst.sys.box_l)
        self.assertFalse(check_free_cuboid(sim_inst.sys, cuboid))
        probe.pos = (cuboid + np.asarray(sim_inst.sys.box_l)) / 2  # between the cuboid and the box edge, in any box
        self.assertTrue(check_free_cuboid(sim_inst.sys, cuboid))

    def test_calculate_pair_distances_matches_index_pair_construction(self):
        """3.4.2: broadcasting must reproduce the explicit N*M index list exactly."""
        rng = np.random.default_rng(11)
        box = np.array([30.0, 25.0, 40.0])
        for n, m in ((1, 1), (5, 5), (3, 7), (12, 4)):
            a = rng.random((n, 3)) * box
            b = rng.random((m, 3)) * box
            combos = np.array(list(product(range(n), range(m))))
            reference = np.linalg.norm(
                min_img_dist(a[combos[:, 0]], b[combos[:, 1]], box_dim=box), axis=-1)
            with self.subTest(shape=(n, m)):
                got = calculate_pair_distances(a, b, box_dim=box)
                self.assertEqual(got.shape, reference.shape)
                np.testing.assert_allclose(got, reference, rtol=0, atol=0)

    def test_get_orientation_vec_matches_component_wise_tensor(self):
        """P3 + 3.4.3: a real, deterministic, first -> last unit axis equal to the component-wise tensor's."""
        rng = np.random.default_rng(12)
        # P3: eig could return a complex type, and the sign was arbitrary.
        nearly_straight = np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.05],
                                    [2.0, 0.2, 0.10], [3.0, 0.3, 0.15]])
        for pos in (nearly_straight, *(np.cumsum(rng.random((n, 3)), axis=0) for n in (3, 10, 50))):
            cm = pos.mean(axis=0)
            comp = np.array([[np.mean([(p[i] - cm[i]) * (p[j] - cm[j]) for p in pos])
                              for j in range(3)] for i in range(3)])
            res, egiv = np.linalg.eigh(comp)
            expected = egiv[:, np.argmax(res)]
            expected = expected / np.linalg.norm(expected)
            if np.dot(expected, pos[-1] - pos[0]) < 0:
                expected = -expected
            with self.subTest(n=len(pos)):
                axis = get_orientation_vec(pos)
                self.assertEqual(axis.dtype.kind, 'f')
                self.assertTrue(np.all(np.isfinite(axis)))
                self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=12)
                self.assertTrue(np.allclose(axis, get_orientation_vec(pos)))
                self.assertGreater(float(np.dot(axis, pos[-1] - pos[0])), 0.0)
                self.assertTrue(np.allclose(get_orientation_vec(pos[::-1]), -axis, atol=1e-10))
                np.testing.assert_allclose(axis, expected, atol=1e-12)


class PartitioningTest(unittest.TestCase):

    geo_box = np.array([2.5, 2.5, 2.5])  # a geometry input, not the simulation box
    num_vol_side=5
    sph_diam=1
    rect_box = np.array([10.0, 20.0, 30.0])

    def test_get_neighbours(self):
        """Neighbour pairs of one fcc face of touching spheres."""
        control= {0: [2,], 1: [2,],2: [0, 1, 3, 4,], 3: [2, ], 4: [2,]}
        sphere_centers_short, _,_=partition_cuboid_volume(self.geo_box,self.num_vol_side,self.sph_diam, flag='norand')
        neigh=get_neighbours(sphere_centers_short,self.geo_box,cutoff=self.sph_diam)
        neigh_sets = {key: set(val) for key, val in neigh.items()}
        control_sets = {key: set(val) for key, val in control.items()}
        self.assertEqual(neigh_sets,control_sets)

    def test_get_neighbours_multiple_rectangular(self):
        """A corner point finds a partner across each of the three periodic faces; a lone point maps to []."""
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

    def test_neighbour_search_matches_brute_force(self):
        """Both cell-list searches reproduce the min_img_dist pair list, on the regular and on the capped grid."""
        # (box, n_points, cutoffs up to L/2, grid): a 'capped' row has more cells per cutoff than _choose_grid
        # allows (~8 per point), so it takes the bisection fallback (cells wider than the cutoff on a capped
        # axis); a tuple pins the grid a row must use.
        cases = (
            (np.array([12.0, 12.0, 12.0]), 600, (0.8, 2.5, 6.0), None),
            (np.array([10.0, 14.0, 22.0]), 600, (0.9, 2.5, 5.0), None),
            (np.array([20.0, 16.0, 24.0]), 800, (1.0,), 'capped'),
            # 3 cells per axis: the "every distinct cell" stencil branch, where only the periodic minimum keeps offset 2
            (np.array([12.0, 12.0, 12.0]), 200, (3.5,), (3, 3, 3)),
        )
        rng = np.random.default_rng(31)
        choose_grid, grids = geometry._choose_grid, []

        def choose_grid_spy(*args, **kwargs):
            grid = choose_grid(*args, **kwargs)
            grids.append(grid[0])
            return grid

        for box, n_points, cutoffs, grid in cases:
            # Unfolded on purpose (both searches fold); the tail of `other` coincides with `points`,
            # and the cross search must report those pairs at distance zero.
            points = rng.uniform(-1, 2, (n_points, 3)) * box
            other = np.concatenate([rng.uniform(-1, 2, (n_points - 5, 3)) * box, points[:5]])
            folded, folded_other = fold_coords(points, box), fold_coords(other, box)
            dist_self = np.linalg.norm(min_img_dist(points[:, None], points[None, :], box_dim=box), axis=-1)
            np.fill_diagonal(dist_self, np.inf)
            dist_cross = np.linalg.norm(min_img_dist(points[:, None], other[None, :], box_dim=box), axis=-1)
            for cutoff in cutoffs:
                with self.subTest(box=box.tolist(), cutoff=cutoff):
                    grids.clear()
                    with mock.patch.object(geometry, "_choose_grid", side_effect=choose_grid_spy):
                        got_self = get_neighbours(points, box, cutoff=cutoff)
                        got_cross = get_neighbours_cross_lattice(points, other, box, cutoff=cutoff)
                    for got, dist, target in ((got_self, dist_self, folded), (got_cross, dist_cross, folded_other)):
                        close = dist <= cutoff
                        i, j = np.nonzero(close)
                        # Some pair crosses every periodic face, so the wrap is exercised on each axis.
                        self.assertTrue(np.all((np.abs(folded[i] - target[j]) > box / 2).any(axis=0)))
                        # exact lists: no duplicate partner, ascending order (sort=True; elastomer bonding relies on both)
                        self.assertEqual(got, {k: np.flatnonzero(row).tolist() for k, row in enumerate(close)})
                    capped = grid == 'capped'
                    self.assertEqual([bool(np.any(n < np.floor(box / cutoff))) for n in grids], [capped, capped])
                    if isinstance(grid, tuple):
                        self.assertEqual([tuple(n.tolist()) for n in grids], [grid, grid])

    def test_partition_cubic_volume_oriented_rectangles(self):
        """Centres are distinct small-box grid sites; each volume is a centred rod of evenly spaced points."""
        box, small, n_vol, n_mon = np.array([12.0, 15.0, 16.0]), np.array([2.0, 3.0, 4.0]), 20, 5
        radius = small[2] / 2
        rng_backup = np.random.get_state()
        try:
            np.random.seed(515)
            centers, points = partition_cubic_volume_oriented_rectangles(box, n_vol, small, n_mon)
            with self.assertRaisesRegex(ValueError, "enough possible volumes"):  # the 6 x 5 x 4 grid holds 120
                partition_cubic_volume_oriented_rectangles(box, 121, small, n_mon)
        finally:
            np.random.set_state(rng_backup)

        self.assertEqual(centers.shape, (n_vol, 3))
        self.assertEqual(len(np.unique(centers, axis=0)), n_vol)
        self.assertTrue(np.all((centers > 0) & (centers < box)))
        site = centers / small - 0.5
        np.testing.assert_allclose(site, np.round(site), atol=1e-12)
        self.assertEqual(points.shape, (n_vol, n_mon, 3))
        for center, rod in zip(centers, points):
            with self.subTest(center=center.tolist()):
                np.testing.assert_allclose(np.linalg.svd(rod - rod.mean(axis=0), compute_uv=False)[1:], 0, atol=1e-12)
                np.testing.assert_allclose(rod.mean(axis=0), center, atol=1e-12)
                np.testing.assert_allclose(np.linalg.norm(np.diff(rod, axis=0), axis=1), 2 * radius / n_mon, atol=1e-12)
                self.assertTrue(np.all(np.linalg.norm(rod - center, axis=1) <= radius))


class FccLatticeTest(unittest.TestCase):
    """fcc_lattice produces an FCC arrangement of touching spheres."""

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
        """No sphere or image sits closer than 2r: pack exactly at it, crystal only widening the pitch."""
        for radius, box, sf in self.cases:
            tight = self._build(radius, box, sf, mode="pack")
            loose = self._build(radius, box, sf, mode="crystal")
            touch = 2 * radius * sf
            nn = {}
            for mode, points in (("pack", tight), ("crystal", loose)):
                with self.subTest(radius=radius, box=box, sf=sf, mode=mode):
                    nn[mode] = self._min_img_dist_self(points, box).min()
                    self.assertGreaterEqual(nn[mode], touch - self.tol)
            with self.subTest(radius=radius, box=box, sf=sf, mode="compare"):
                self.assertAlmostEqual(nn["pack"], touch, delta=self.tol)
                self.assertGreaterEqual(nn["crystal"], nn["pack"] - self.tol)
                # thight may have an extra "half-lattice-step"
                self.assertGreaterEqual(len(tight), len(loose))
                for d, (st, sl) in enumerate(zip(self._axis_steps(tight), self._axis_steps(loose))):
                    self.assertGreaterEqual(sl.min(), st.min() - self.tol, f"axis {d}")

    def test_crystal_tiles_the_box_exactly(self):
        """Crystal mode tiles each axis with an even number of half-steps: the seam gap is the bulk pitch."""
        for radius, box, sf in self.cases:
            points = self._build(radius, box, sf, mode="crystal")
            for d, steps in enumerate(self._axis_steps(points)):
                with self.subTest(radius=radius, box=box, sf=sf, axis=d):
                    pitch = steps[0]
                    self.assertTrue(np.allclose(steps, pitch, atol=self.tol))
                    n_half = box[d] / pitch
                    self.assertAlmostEqual(n_half, round(n_half), delta=self.tol)
                    seam = box[d] - points[:, d].max()
                    self.assertAlmostEqual(seam, pitch, delta=self.tol)

    def test_coordination_number_is_twelve(self):
        """Every site has 12 nearest neighbours (in pack mode only sites away from the seam void)."""
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
                    self.assertEqual(set(counts[keep].tolist()), {12})

    def test_max_points_per_side_caps_the_lattice(self):
        """Exceeding the per-axis cap widens the lattice constant instead of emitting more points."""
        for mode in ("pack", "crystal"):
            for cap in (4, 10):
                points = fcc_lattice(0.5, [60.0, 60.0, 60.0], max_points_per_side=cap, mode=mode)
                for d in range(3):
                    with self.subTest(mode=mode, cap=cap, axis=d):
                        self.assertLessEqual(len(np.unique(np.round(points[:, d], 9))), cap)

    def test_rejects_bad_input(self):
        """Non-positive radius or scaling factor, or a box smaller than one cell, raise ValueError."""
        for kwargs in ({"radius": -1.0}, {"radius": 0.0}, {"scaling_factor": 0.0}):
            with self.subTest(**kwargs):
                call = {"radius": 0.5, "box_dim": [6.0, 6.0, 6.0], **kwargs}
                self.assertRaises(ValueError, fcc_lattice, **call)
        # box too small to hold a single conventional cell
        self.assertRaises(ValueError, fcc_lattice, 5.0, [1.0, 1.0, 1.0])


class RequireMinGlobalCutTest(unittest.TestCase):
    """The ghost-layer guard bites only on more than one rank, at 1.5 x the bond length."""

    class _FakeSys:
        def __init__(self, n_nodes, min_global_cut):
            self.min_global_cut = min_global_cut
            self.cell_system = type("CellSystem", (), {"get_state": lambda _self: {"n_nodes": n_nodes}})()

    def test_require_min_global_cut(self):
        """One rank always passes; several need min_global_cut >= 1.5 x cutoff, and the error names the fix."""
        require_min_global_cut(self._FakeSys(1, 0.1), cutoff=2.)
        require_min_global_cut(self._FakeSys(4, 3.), cutoff=2.)
        with self.assertRaises(RuntimeError) as ctx:
            require_min_global_cut(self._FakeSys(4, 2.9), cutoff=2.)
        self.assertIn("set_sys(min_global_cut=3.0)", str(ctx.exception))
