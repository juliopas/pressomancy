"""Equivalence tests for the HDF5 read planner.

``read_h5_selection`` reads ``dataset[indices]`` with NumPy outer-product
semantics while pulling only the selected data off disk. To do that it picks
between several strategies using a chunk-aware cost model:

* zero or one fancy axis            -> handed straight to h5py
* an empty fancy axis               -> shortcut to an empty array
* two fancy axes, cheap over-read   -> one read of a bounding span, with the
                                       fancy index kept on whichever axis is
                                       cheaper
* two fancy axes, costly over-read  -> one read per position along the shorter
                                       fancy axis

All of them must return exactly the same array. Nothing checked that before, and
the failure mode is a silently wrong array rather than an exception -- these
reads feed the analysis layer and, through it, pyanal.

The tests below force each branch (``max_overread_factor`` steers the
span-vs-loop choice; the relative spread and size of the index arrays steer the
sub-variants), assert the branch actually taken via a spy, and compare against a
NumPy reference built with plain ``np.take``.
"""
import os
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

from pressomancy.analysis import h5_helper_functions as hh
from pressomancy.analysis.h5_helper_functions import read_h5_selection


def numpy_outer_reference(array, indices):
    """``array[indices]`` with outer-product semantics, via successive takes.

    NumPy would broadcast two list indices *pairwise* (the diagonal), which is
    exactly what read_h5_selection does not do, so the reference cannot simply be
    ``array[indices]``. Taking one axis at a time gives the outer product, and
    drops the axis for an integer index just as NumPy does.
    """
    out = array
    axis = 0
    for index in indices:
        if isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool)):
            out = np.take(out, int(index), axis=axis)
            continue
        if isinstance(index, slice):
            out = out[(slice(None),) * axis + (index,)]
            axis += 1
            continue
        positions = np.asarray(index)
        if positions.dtype == bool:
            positions = np.flatnonzero(positions)
        out = np.take(out, positions.astype(np.intp), axis=axis)
        axis += 1
    return out


class ReadSelectionTest(unittest.TestCase):
    """Every read strategy must agree with the NumPy outer product."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmpdir.name, "selection.h5")
        cls.file = h5py.File(cls.path, "w")

        # A distinct value per element, so any misordering shows up as a
        # mismatch rather than coincidentally matching.
        cls.shape = (12, 9, 4)
        data = np.arange(int(np.prod(cls.shape)), dtype=np.int64).reshape(cls.shape)
        cls.data = data
        cls.chunked = cls.file.create_dataset("chunked", data=data, chunks=(1, 9, 4))
        cls.contiguous = cls.file.create_dataset("contiguous", data=data)

        # The geometry the writer actually produces: (frames, particles, dim).
        cls.frame_shape = (7, 20, 3)
        frames = np.arange(int(np.prod(cls.frame_shape)), dtype=np.float64)
        frames = frames.reshape(cls.frame_shape)
        cls.frame_data = frames
        cls.frames = cls.file.create_dataset(
            "frames", data=frames, chunks=(1, 20, 3))

    @classmethod
    def tearDownClass(cls):
        cls.file.close()
        cls.tmpdir.cleanup()

    def assert_matches_reference(self, dataset, array, indices, **kwargs):
        got = read_h5_selection(dataset, *indices, **kwargs)
        expected = numpy_outer_reference(array, indices)
        self.assertEqual(got.shape, expected.shape,
                         msg=f"shape mismatch for indices {indices!r}")
        np.testing.assert_array_equal(got, expected,
                                      err_msg=f"value mismatch for indices {indices!r}")
        return got

    # -- single / zero fancy axis -----------------------------------------
    def test_zero_or_one_fancy_axis_goes_straight_to_h5py(self):
        cases = [
            (slice(None), slice(None), slice(None)),
            (slice(2, 9, 2), slice(None), slice(None)),
            ([0, 3, 7], slice(None), slice(None)),
            (slice(None), [1, 4, 8], slice(None)),
            (3, slice(None), slice(None)),
            (3, [0, 2], slice(None)),
        ]
        for indices in cases:
            with self.subTest(indices=indices):
                with mock.patch.object(hh, "_read_two_fancy_axes_with_bounding_span") as span, \
                     mock.patch.object(hh, "_read_two_fancy_axes_in_chunks") as loop:
                    self.assert_matches_reference(self.chunked, self.data, indices)
                    span.assert_not_called()
                    loop.assert_not_called()

    def test_empty_fancy_axis_returns_a_correctly_shaped_empty_array(self):
        for indices in [([], slice(None), slice(None)),
                        ([1, 2, 3], [], slice(None)),
                        (2, [], slice(None))]:
            with self.subTest(indices=indices):
                got = self.assert_matches_reference(self.chunked, self.data, indices)
                self.assertEqual(got.size, 0)
                self.assertEqual(got.dtype, self.chunked.dtype)

    # -- two fancy axes: the bounding-span branch --------------------------
    def test_bounding_span_branch_matches_reference(self):
        """A huge over-read budget forces the single-read strategy every time."""
        cases = [
            ([0, 5, 11], [1, 4, 8]),
            ([1, 2, 3], [0, 1]),
            ([0, 11], [0, 8]),
        ]
        for a, b in cases:
            for dataset, label in ((self.chunked, "chunked"),
                                   (self.contiguous, "contiguous")):
                with self.subTest(a=a, b=b, dataset=label):
                    with mock.patch.object(
                            hh, "_read_two_fancy_axes_with_bounding_span",
                            wraps=hh._read_two_fancy_axes_with_bounding_span) as span, \
                         mock.patch.object(hh, "_read_two_fancy_axes_in_chunks") as loop:
                        self.assert_matches_reference(
                            dataset, self.data, (a, b, slice(None)),
                            max_overread_factor=1e9)
                        span.assert_called_once()
                        loop.assert_not_called()

    def test_both_bounding_span_variants_are_exercised(self):
        """The fancy index is kept on axis a or axis b, whichever costs less.

        Uses the contiguous dataset because on the writer's (1, N, dim) chunking
        this choice is degenerate: axis 1's chunk spans the whole axis, so a
        fancy read and a span read there cost the same, which makes
        ``cost_fancy_a <= cost_fancy_b`` unconditionally true. Worth knowing --
        real frame datasets only ever take the keep-on-a variant.
        """
        seen = set()
        candidates = [
            ([0, 1, 2], [0, 8]),
            ([0, 11], [3, 4, 5]),
            ([0, 6, 11], [0, 1, 2]),
            ([4, 5], [0, 4, 8]),
        ]
        for a, b in candidates:
            arr_a = np.asarray(a, dtype=np.intp)
            arr_b = np.asarray(b, dtype=np.intp)
            _, cost_a, cost_b = hh._estimate_bounding_costs(
                arr_a, arr_b,
                hh._axis_chunk_size(self.contiguous, 0),
                hh._axis_chunk_size(self.contiguous, 1))
            expected_keep_on_a = cost_a <= cost_b
            with self.subTest(a=a, b=b, keep_fancy_on_a=expected_keep_on_a):
                with mock.patch.object(
                        hh, "_read_two_fancy_axes_with_bounding_span",
                        wraps=hh._read_two_fancy_axes_with_bounding_span) as span:
                    self.assert_matches_reference(
                        self.contiguous, self.data, (a, b, slice(None)),
                        max_overread_factor=1e9)
                    self.assertEqual(span.call_args.kwargs["keep_fancy_on_a"],
                                     expected_keep_on_a)
                seen.add(expected_keep_on_a)
        self.assertEqual(seen, {True, False},
                         msg="both keep_fancy_on_a variants must be covered")

    def test_chunked_frame_geometry_always_keeps_fancy_on_axis_a(self):
        """Documents the degeneracy the previous test works around."""
        for a, b in (([0, 1, 2], [0, 19]), ([0, 6], [0, 1, 2]), ([3], [5, 15])):
            _, cost_a, cost_b = hh._estimate_bounding_costs(
                np.asarray(a, dtype=np.intp), np.asarray(b, dtype=np.intp),
                hh._axis_chunk_size(self.frames, 0),
                hh._axis_chunk_size(self.frames, 1))
            with self.subTest(a=a, b=b):
                self.assertLessEqual(cost_a, cost_b)

    # -- two fancy axes: the chunked-loop branch ---------------------------
    def test_chunked_loop_branch_matches_reference(self):
        """A tiny over-read budget forces the per-position loop every time."""
        cases = [
            ([0, 5, 11], [1, 4, 8]),
            ([0, 11], [0, 8]),
            ([2, 3], [2, 3]),
        ]
        for a, b in cases:
            for dataset, label in ((self.chunked, "chunked"),
                                   (self.contiguous, "contiguous")):
                with self.subTest(a=a, b=b, dataset=label):
                    with mock.patch.object(
                            hh, "_read_two_fancy_axes_in_chunks",
                            wraps=hh._read_two_fancy_axes_in_chunks) as loop, \
                         mock.patch.object(hh, "_read_two_fancy_axes_with_bounding_span") as span:
                        self.assert_matches_reference(
                            dataset, self.data, (a, b, slice(None)),
                            max_overread_factor=1e-9)
                        loop.assert_called_once()
                        span.assert_not_called()

    def test_both_chunked_loop_variants_are_exercised(self):
        """The loop runs over whichever fancy axis is shorter."""
        seen = set()
        for a, b in (([0, 4, 9, 11], [2, 6]), ([1, 5], [0, 3, 5, 8])):
            expected_loop_axis = 1 if len(b) <= len(a) else 0
            with self.subTest(a=a, b=b, loop_axis=expected_loop_axis):
                with mock.patch.object(
                        hh, "_read_two_fancy_axes_in_chunks",
                        wraps=hh._read_two_fancy_axes_in_chunks) as loop:
                    self.assert_matches_reference(
                        self.chunked, self.data, (a, b, slice(None)),
                        max_overread_factor=1e-9)
                    self.assertEqual(loop.call_args.args[2], expected_loop_axis)
                seen.add(expected_loop_axis)
        self.assertEqual(seen, {0, 1},
                         msg="both loop-axis variants must be covered")

    # -- the two strategies against each other -----------------------------
    def test_span_and_loop_agree_across_many_selections(self):
        """The core invariant: strategy choice must never change the answer."""
        rng = np.random.default_rng(20260907)
        for _ in range(60):
            n_a = int(rng.integers(1, 7))
            n_b = int(rng.integers(1, 7))
            a = sorted(rng.choice(self.shape[0], size=n_a, replace=False).tolist())
            b = sorted(rng.choice(self.shape[1], size=n_b, replace=False).tolist())
            indices = (a, b, slice(None))
            expected = numpy_outer_reference(self.data, indices)
            via_span = read_h5_selection(self.chunked, *indices, max_overread_factor=1e9)
            via_loop = read_h5_selection(self.chunked, *indices, max_overread_factor=1e-9)
            default = read_h5_selection(self.chunked, *indices)
            with self.subTest(a=a, b=b):
                np.testing.assert_array_equal(via_span, expected)
                np.testing.assert_array_equal(via_loop, expected)
                np.testing.assert_array_equal(default, expected)

    # -- writer geometry ---------------------------------------------------
    def test_writer_frame_geometry(self):
        """(frames, particles, dim) with (1, N, dim) chunks -- what H5Writer makes."""
        cases = [
            ([0, 3, 6], [0, 5, 19], slice(None)),
            ([1, 2], [3], slice(None)),
            (slice(None), [0, 10], slice(None)),
            (-1, [0, 1, 2], slice(None)),
            ([0, 6], slice(2, 15, 3), slice(None)),
        ]
        for indices in cases:
            for factor in (1e9, 1e-9, None):
                kwargs = {} if factor is None else {"max_overread_factor": factor}
                with self.subTest(indices=indices, factor=factor):
                    self.assert_matches_reference(
                        self.frames, self.frame_data, indices, **kwargs)

    # -- axis bookkeeping --------------------------------------------------
    def test_integer_index_drops_its_axis_and_slice_does_not(self):
        """_apply_post_indices handles this per strategy, so check it per strategy."""
        for factor in (1e9, 1e-9):
            with self.subTest(factor=factor):
                dropped = read_h5_selection(self.chunked, 3, [1, 2], slice(None),
                                            max_overread_factor=factor)
                self.assertEqual(dropped.shape, (2, self.shape[2]))
                kept = read_h5_selection(self.chunked, [3], [1, 2], slice(None),
                                         max_overread_factor=factor)
                self.assertEqual(kept.shape, (1, 2, self.shape[2]))
                np.testing.assert_array_equal(dropped, kept[0])

    def test_negative_and_unsorted_and_repeated_positions(self):
        """h5py needs sorted unique fancy indices; the caller should not have to care."""
        cases = [
            ([11, 0, 5], [8, 1]),
            ([-1, -12, 4], [-1, 0]),
            ([3, 3, 1], [2, 2]),
        ]
        for a, b in cases:
            for factor in (1e9, 1e-9):
                with self.subTest(a=a, b=b, factor=factor):
                    self.assert_matches_reference(
                        self.chunked, self.data, (a, b, slice(None)),
                        max_overread_factor=factor)

    def test_boolean_masks_and_negative_step_slices(self):
        mask_a = np.zeros(self.shape[0], dtype=bool)
        mask_a[[0, 4, 9]] = True
        mask_b = np.zeros(self.shape[1], dtype=bool)
        mask_b[[2, 7]] = True
        for factor in (1e9, 1e-9):
            with self.subTest(factor=factor):
                self.assert_matches_reference(
                    self.chunked, self.data, (mask_a, mask_b, slice(None)),
                    max_overread_factor=factor)
        # A negative step is normalized into a position list, so it becomes fancy.
        self.assert_matches_reference(
            self.chunked, self.data, (slice(None, None, -1), [1, 3], slice(None)))

    # -- rejected inputs ---------------------------------------------------
    def test_rejected_selections(self):
        with self.assertRaises(ValueError):
            read_h5_selection(self.chunked, [0], [0], max_overread_factor=0)
        with self.assertRaises(IndexError):
            read_h5_selection(self.chunked, 0, 0, 0, 0)
        with self.assertRaises(NotImplementedError):
            read_h5_selection(self.chunked, [0, 1], [0, 1], [0, 1])
        with self.assertRaises(TypeError):
            read_h5_selection(self.chunked, Ellipsis)
        with self.assertRaises(TypeError):
            read_h5_selection(self.chunked, None)
        with self.assertRaises(IndexError):
            read_h5_selection(self.chunked, self.shape[0])


if __name__ == "__main__":
    unittest.main()
