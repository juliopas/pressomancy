"""Read side of the HDF5 layer (``pressomancy.io.read``), without espresso.

Two things live here: equivalence tests of the read planner (every strategy
``read_h5_selection`` might pick must return exactly the same array as a NumPy
outer-product reference -- a wrong pick is a silently wrong array feeding
straight into the analysis script, not an exception) and selector/lookup tests
on a fixture written by hand in the writer's own layout, so they run without espresso.
"""
import os
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

from pressomancy.io import read as hh
from pressomancy.io import (H5DataSelector, H5ObservableSelector, read_h5_selection,
                            stored_steps, frame_of_step, frame_of_time, has_step)


def numpy_outer_reference(array, indices):
    """``array[indices]`` with outer-product semantics, via successive ``np.take``.

    NumPy broadcasts two list indices *pairwise* (the diagonal) instead, so the
    reference cannot simply be ``array[indices]``.
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

    @classmethod
    def tearDownClass(cls):
        cls.file.close()
        cls.tmpdir.cleanup()

    def assert_matches_reference(self, dataset, array, indices, **kwargs):
        got = read_h5_selection(dataset, *indices, **kwargs)
        expected = numpy_outer_reference(array, indices)
        self.assertEqual(got.shape, expected.shape)
        np.testing.assert_array_equal(got, expected)
        return got

    # -- the two strategies against each other -----------------------------
    def test_span_and_loop_agree_across_many_selections(self):
        """Both strategies -- and every one of their sub-variants -- must run
        and agree with the NumPy reference, on both datasets' chunk geometry."""
        rng = np.random.default_rng(20260907)
        n_cases = 30
        with mock.patch.object(
                hh, "_read_two_fancy_axes_with_bounding_span",
                wraps=hh._read_two_fancy_axes_with_bounding_span) as span, \
             mock.patch.object(
                hh, "_read_two_fancy_axes_in_chunks",
                wraps=hh._read_two_fancy_axes_in_chunks) as loop:
            for dataset, label in ((self.chunked, "chunked"), (self.contiguous, "contiguous")):
                for _ in range(n_cases):
                    n_a = int(rng.integers(1, 7))
                    n_b = int(rng.integers(1, 7))
                    a = sorted(rng.choice(self.shape[0], size=n_a, replace=False).tolist())
                    b = sorted(rng.choice(self.shape[1], size=n_b, replace=False).tolist())
                    indices = (a, b, slice(None))
                    with self.subTest(dataset=label, a=a, b=b):
                        expected = numpy_outer_reference(self.data, indices)
                        for factor in (1e9, 1e-9, None):
                            kwargs = {} if factor is None else {"max_overread_factor": factor}
                            got = read_h5_selection(dataset, *indices, **kwargs)
                            np.testing.assert_array_equal(got, expected, err_msg=f"max_overread_factor={factor}")

        keep_variants = {call.kwargs["keep_fancy_on_a"] for call in span.call_args_list}
        loop_variants = {call.args[2] for call in loop.call_args_list}
        self.assertEqual(keep_variants, {True, False},
                         msg="both keep_fancy_on_a variants must be covered")
        self.assertEqual(loop_variants, {0, 1},
                         msg="both loop-axis variants must be covered")

    # -- axis bookkeeping --------------------------------------------------
    def test_index_forms_agree_under_both_read_strategies(self):
        """Every index form -- zero/one/two fancy axes, empty, boolean, negative
        step -- must agree with the NumPy reference; h5py needs sorted unique
        fancy indices while the caller should not have to care, so the
        two-fancy-axis rows are checked under both the span and the loop
        strategy."""
        # -- zero or one fancy axis: handled directly by h5py -------------
        for indices in [(slice(None), slice(None), slice(None)),
                        (slice(2, 9, 2), slice(None), slice(None)),
                        ([0, 3, 7], slice(None), slice(None)),
                        (slice(None), [1, 4, 8], slice(None)),
                        (3, slice(None), slice(None)),
                        (3, [0, 2], slice(None)),
                        (-1, [0, 1, 2], slice(None)),
                        ([0, 6], slice(2, 15, 3), slice(None))]:
            with self.subTest(indices=indices):
                self.assert_matches_reference(self.chunked, self.data, indices)

        # -- an empty fancy axis: a correctly shaped and typed empty array -
        for indices in [([], slice(None), slice(None)),
                        ([1, 2, 3], [], slice(None)),
                        (2, [], slice(None))]:
            with self.subTest(indices=indices):
                got = self.assert_matches_reference(self.chunked, self.data, indices)
                self.assertEqual(got.size, 0)
                self.assertEqual(got.dtype, self.chunked.dtype)

        # -- two fancy axes: every index form, forced onto each strategy --
        for factor in (1e9, 1e-9):
            with self.subTest(kind="integer_vs_slice_axis", factor=factor):
                dropped = read_h5_selection(self.chunked, 3, [1, 2], slice(None),
                                            max_overread_factor=factor)
                self.assertEqual(dropped.shape, (2, self.shape[2]))
                kept = read_h5_selection(self.chunked, [3], [1, 2], slice(None),
                                         max_overread_factor=factor)
                self.assertEqual(kept.shape, (1, 2, self.shape[2]))
                np.testing.assert_array_equal(dropped, kept[0])

            for a, b in (([11, 0, 5], [8, 1]), ([-1, -12, 4], [-1, 0]), ([3, 3, 1], [2, 2])):
                with self.subTest(a=a, b=b, factor=factor):
                    self.assert_matches_reference(
                        self.chunked, self.data, (a, b, slice(None)),
                        max_overread_factor=factor)

            mask_a = np.zeros(self.shape[0], dtype=bool)
            mask_a[[0, 4, 9]] = True
            mask_b = np.zeros(self.shape[1], dtype=bool)
            mask_b[[2, 7]] = True
            with self.subTest(kind="boolean_masks", factor=factor):
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



def _write_layout(h5_file):
    """The attr every particle-data reader checks before it reads anything."""
    h5_file.require_group("parameters/pressomancy").attrs["layout"] = hh.LAYOUT


def _write_particle_group(h5_file, group, ids, types, steps, times):
    """A minimal written particle group, in the ``h5md-1`` layout.

    Elements carry their H5MD names (``position``, ``species``), scalars are
    ``[F, N]`` and vectors ``[F, N, 3]``, and the group's one ``step``/``time``
    pair is created under the first element and hard-linked into the others.
    """
    grp = h5_file.require_group(f"particles/{group}")
    n_frames = len(steps)
    # The writer's own dtypes: id int32, type int16, pos float64; time float32.
    shared = None
    for attr, column, dim, dtype in (
            ("pos", np.arange(len(ids) * 3).reshape(len(ids), 3), 3, np.float64),
            ("id", ids, None, np.int32), ("type", types, None, np.int16)):
        element = grp.require_group(hh.element_name(attr))
        per_particle = () if dim is None else (dim,)
        value = np.tile(np.asarray(column, dtype=dtype).reshape(1, len(ids), *per_particle),
                        (n_frames, *(1,) * (1 + len(per_particle))))
        element.create_dataset("value", data=value)
        if shared is None:
            shared = (element.create_dataset("step", data=np.asarray(steps, dtype=np.int32)),
                      element.create_dataset("time", data=np.asarray(times, dtype=np.float32)))
        else:
            element["step"], element["time"] = shared
    box = grp.require_group("box")
    box.attrs["dimension"] = 3
    box.attrs["boundary"] = np.array(["periodic"] * 3, dtype=h5py.string_dtype("ascii"))
    # the box is a time-dependent element on the group's shared timeline
    edges = box.require_group("edges")
    edges["step"], edges["time"] = shared
    edges.create_dataset("value", data=np.full((n_frames, 3), 10.0, dtype=np.float64))


def _write_bonds(h5_file, group, particle_ids, links, params):
    """The link tables of ``io/bonds.py::write_bonds``: one per arity, holding
    COLUMN INDICES into ``particles/<group>/id/value``, plus the row-aligned id
    arrays and class tables under ``pressomancy/<group>/bond_params``.

    ``links`` is ``[(bond_id, owner_id, [partner_id, ...]), ...]``; ``params`` is
    ``{type_name: (dtype, [row, ...])}``.
    """
    columns = {int(pid): index for index, pid in enumerate(particle_ids)}
    conn = h5_file.require_group(f"connectivity/{group}")
    params_grp = h5_file.require_group(f"pressomancy/{group}/bond_params")
    by_arity = {1: []}
    for bond_id, owner, partners in links:
        by_arity.setdefault(len(partners), []).append(
            (bond_id, [columns[owner]] + [columns[p] for p in partners]))
    for arity, rows in sorted(by_arity.items()):
        table_name, id_name = hh.LINK_TABLES[arity]
        table = conn.create_dataset(
            table_name,
            data=np.array([row for _, row in rows], dtype=np.int32).reshape(len(rows), 1 + arity))
        table.attrs["particles_group"] = f"/particles/{group}"
        params_grp.create_dataset(
            id_name, data=np.array([bond_id for bond_id, _ in rows], dtype=np.int32))
    conn["bonds"].attrs["n_links"] = len(links)
    conn["bonds"].attrs["captured_at_time"] = 0.0
    for type_name, (dtype, table_rows) in params.items():
        params_grp.create_dataset(type_name, data=np.array(table_rows, dtype=dtype))


def _write_observable(h5_file, name, steps, times, values):
    """An observable group in the layout ``H5Writer`` writes."""
    grp = h5_file.require_group(f"observables/{name}")
    grp.create_dataset("step", data=np.asarray(steps, dtype=np.int32))
    grp.create_dataset("time", data=np.asarray(times, dtype=np.float64))
    grp.create_dataset("value", data=np.asarray(values, dtype=np.float64))


class SelectionAndStepLookupTest(unittest.TestCase):
    """``by_particle``, ``select_particles_by_type``, the connectivity lookups,
    the observable selector and the step/time <-> frame lookups, on a fixture
    written by hand in the writer's own layout (so these tests run without
    espresso)."""

    IDS = [10, 11, 12, 13, 14]
    TYPES = [0, 0, 1, 1, 2]
    STEPS = [0, 100, 200]
    TIMES = [0.0, 1.0, 2.0]
    OBSERVABLE = "magnetic_dipole_moment"

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmpdir.name, "selection_bonds.h5")
        cls.file = h5py.File(cls.path, "w")
        _write_layout(cls.file)
        _write_particle_group(cls.file, "Elastomer", cls.IDS, cls.TYPES,
                              cls.STEPS, cls.TIMES)
        # particle 10 owns a pair bond and an angle bond, particle 12 a pair bond,
        # 11/13/14 own none -- by_particle must still give them a key.
        _write_bonds(
            cls.file, "Elastomer", cls.IDS,
            links=[(0, 10, [11]), (1, 10, [11, 12]), (0, 12, [13])],
            params={
                "HarmonicBond": (np.dtype([("bond_id", "<i4"), ("k", "<f8"),
                                           ("r_0", "<f8")]),
                                 [(0, 2.5, 1.0)]),
                "AngleHarmonic": (np.dtype([("bond_id", "<i4"), ("bend", "<f8"),
                                            ("phi0", "<f8")]),
                                  [(1, 7.0, 3.14)]),
            })
        # A second group whose step 100 was written twice -> ambiguous lookup.
        _write_particle_group(cls.file, "Dup", [0, 1], [0, 0],
                              [0, 100, 100], [0.0, 1.0, 1.0])
        cls.observable_values = np.arange(len(cls.STEPS) * 3, dtype=np.float64).reshape(-1, 3)
        _write_observable(cls.file, cls.OBSERVABLE, cls.STEPS, cls.TIMES,
                          cls.observable_values)
        cls.file.flush()

    @classmethod
    def tearDownClass(cls):
        cls.file.close()
        cls.tmpdir.cleanup()

    def selector(self):
        return H5DataSelector(self.file, "Elastomer")

    # -- bonds -------------------------------------------------------------
    def test_by_particle_keys_every_selected_particle_and_follows_the_slice(self):
        by_particle = self.selector().bonds.by_particle()
        self.assertEqual(list(by_particle), self.IDS)
        self.assertEqual([len(v) for v in by_particle.values()], [2, 0, 1, 0, 0])
        pair, angle = by_particle[10]
        self.assertEqual((pair.bond_id, pair.bond_type, pair.partners),
                         (0, "HarmonicBond", (11,)))
        self.assertEqual(pair.params, {"k": 2.5, "r_0": 1.0})
        # The -1 padding of the pair link's row must not leak into the partners.
        self.assertEqual((angle.bond_id, angle.bond_type, angle.partners),
                         (1, "AngleHarmonic", (11, 12)))
        self.assertEqual(by_particle[12][0].partners, (13,))

        sliced = self.selector().particles[2:].bonds.by_particle()
        self.assertEqual(list(sliced), [12, 13, 14])
        self.assertEqual([len(v) for v in sliced.values()], [1, 0, 0])

    # -- names & misuse ------------------------------------------------------
    def test_the_selector_speaks_espresso_names_and_the_file_h5md_ones(self):
        """`pos` reads `position/value`; the H5MD name itself is not an
        attribute, so a caller cannot reach the dataset by spelling its path
        name. Also the selector API misuse cases: an unknown particle group is
        a ValueError, and indexing/iterating/len of an un-sliced selector is a
        TypeError, on both H5DataSelector and H5ObservableSelector -- neither
        selector is a sequence itself; `.timestep`/`.particles` are."""
        data = self.selector()
        np.testing.assert_array_equal(
            data.timestep[0].pos, self.file["particles/Elastomer/position/value"][0])
        with self.assertRaises(AttributeError) as caught:
            data.position
        self.assertIn("'pos'", str(caught.exception))
        with self.assertRaises(KeyError):
            data.get_property("species")

        with self.assertRaises(ValueError):
            H5DataSelector(self.file, particle_group="DangerNoodle")
        with self.assertRaises(TypeError):
            data[-1]
        with self.assertRaises(TypeError):
            iter(data)
        with self.assertRaises(TypeError):
            len(data)

        observable_selector = H5ObservableSelector(self.file, observable_name=self.OBSERVABLE)
        with self.assertRaises(TypeError):
            observable_selector[0]
        with self.assertRaises(TypeError):
            iter(observable_selector)
        with self.assertRaises(TypeError):
            len(observable_selector)

    # -- type selection ----------------------------------------------------
    def test_select_particles_by_type_composes_and_names_empty_results(self):
        data = self.selector()
        self.assertEqual(data.select_particles_by_type([1]).id.reshape(-1).tolist(),
                         [12, 13] * len(self.STEPS))
        # Composition: particle 12 is dropped by the slice, not by the type.
        composed = data.particles[3:].select_particles_by_type([0, 1])
        self.assertEqual(np.unique(composed.id).tolist(), [13])
        # The timestep slice survives type selection ('id' is a scalar element,
        # and an integer timestep index drops the frame axis).
        self.assertEqual(data.timestep[1].select_particles_by_type([2]).id.shape,
                         (1,))
        with self.assertRaises(ValueError) as caught:
            data.select_particles_by_type([7, 9])
        self.assertIn("[0, 1, 2]", str(caught.exception))

    # -- object relations --------------------------------------------------
    def test_a_missing_connectivity_table_raises_naming_it(self):
        """An absent relation is a KeyError naming the dataset."""
        selector = self.selector()
        for call in (lambda: selector.get_connectivity_map("Elastomer", "Nothing"),
                     lambda: selector.get_child_ids("Elastomer", "Nothing", 0),
                     lambda: selector.get_parent_ids("Elastomer", "Nothing", 0)):
            with self.assertRaises(KeyError) as caught:
                call()
            self.assertIn("pressomancy/Elastomer/ownership/Elastomer_to_Nothing",
                          str(caught.exception))

    # -- observables -------------------------------------------------------
    def test_observable_selector_reads_and_slices_the_frame_axis(self):
        selector = H5ObservableSelector(self.file, observable_name=self.OBSERVABLE)
        self.assertEqual(selector.common_dims, (len(self.STEPS),))
        np.testing.assert_array_equal(selector.step, self.STEPS)
        np.testing.assert_allclose(selector.time, self.TIMES)
        np.testing.assert_allclose(selector.value, self.observable_values)
        sliced = selector.timestep[1:]
        self.assertEqual(len(sliced.timestep), len(self.STEPS) - 1)
        np.testing.assert_array_equal(sliced.step, self.STEPS[1:])
        np.testing.assert_allclose(sliced.time, self.TIMES[1:])
        np.testing.assert_allclose(sliced.value, self.observable_values[1:])
        frames = [frame for frame in selector.timestep]
        self.assertEqual(len(frames), len(self.STEPS))
        np.testing.assert_array_equal([frame.step for frame in frames], self.STEPS)
        with self.assertRaises(ValueError) as caught:
            H5ObservableSelector(self.file, observable_name="not_an_observable")
        self.assertIn(self.OBSERVABLE, str(caught.exception))

    # -- frame lookups -------------------------------------------------------
    def test_frame_lookups_agree_and_reject_missing_ambiguous_and_paths(self):
        """`step` and `time` are two ways of naming a frame; `time` is the only
        one matched approximately, since stored times are floats a caller
        re-types by hand."""
        self.assertEqual(stored_steps(self.file, "Elastomer").tolist(), self.STEPS)
        self.assertEqual(frame_of_step(self.file, "Elastomer", 100), 1)
        self.assertTrue(has_step(self.file, "Elastomer", 200))
        self.assertFalse(has_step(self.file, "Elastomer", 150))
        with self.assertRaises(KeyError) as missing:
            frame_of_step(self.file, "Elastomer", 150)
        self.assertIn("[0, 100, 200]", str(missing.exception))
        with self.assertRaises(ValueError) as ambiguous:
            frame_of_step(self.file, "Dup", 100)
        self.assertIn("ambiguous", str(ambiguous.exception))
        with self.assertRaises(KeyError):
            stored_steps(self.file, "NoSuchGroup")
        # A path is not an accepted input form: one way of doing this.
        with self.assertRaises(TypeError):
            stored_steps(self.path, "Elastomer")

        for frame, time in enumerate(self.TIMES):
            self.assertEqual(frame_of_time(self.file, "Elastomer", time), frame)
        # inside the relative tolerance of 1e-6, and outside it
        self.assertEqual(frame_of_time(self.file, "Elastomer", 2.0 * (1 + 1e-9)), 2)
        with self.assertRaises(KeyError) as caught:
            frame_of_time(self.file, "Elastomer", 1.5)
        self.assertIn("nearest stored time", str(caught.exception))
        with self.assertRaises(KeyError):
            frame_of_time(self.file, "NoSuchGroup", 0.0)


if __name__ == "__main__":
    unittest.main()
