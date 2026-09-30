import numpy as np
import espressomd
from .create_system import sim_inst, BaseTestCase
from pressomancy.simulation import (Filament, Quartet, Quadriplex, Crowder, Elastomer,
                                    PointDipolePermanent, PointDipoleMagnetizable)
from pressomancy.infra import BondWrapper, api_agnostic_feature_check
from pressomancy.geometry import WCA_CONTACT_FACTOR, min_img_dist
from pressomancy.magnetodynamics import required_features_for
from pressomancy.io import (H5DataSelector, H5ObservableSelector, stored_steps,
                            CHECKPOINT_PROPERTIES, checkpoint_properties)
from pressomancy.io.read import LAYOUT, element_name
import h5py
import importlib.util
import tempfile
import os
import shutil
import unittest
from unittest.mock import patch
import pressomancy.io.write as write_module
from pressomancy.io.bonds import verify_bond_params, write_bonds, read_bonds, read_bond_params

#: Box of the SourceFixture classes and the bond-heavy ones: a cell-grid rebuild costs ~16 ms in the default
#: 20^3, ~4 ms here. 16 is the smallest side a SourceFixture filament (size 8) fits: set_objects needs L/2 >= size.
SMALL_BOX = (16, 16, 16)


class IOTestCase(BaseTestCase):
    """A class tmpdir on top of BaseTestCase's per-class box and per-test reset."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmpdir.cleanup)       # also when setUpClass fails
        super().setUpClass()


class CommonH5DataSelectorTests:
    """A class-scope fixture: setUpClass writes one file that every test reads, so the system is reset once
    per class (on entry and on exit), never per test; the concrete classes are plain unittest.TestCase."""

    box_dim = BaseTestCase.box_dim  # the default box; a fixture that needs another declares it
    runner_script="repo/project/script.py"
    runner_script_repo ="main@abc1234-dirty"
    library_vers= "main@def5678"
    lib_path="/some/path/"
    author="dungeonwitch"
    email='dungeonwitch@dungeon.com'
    kT = 0.75           # not the Simulation default (1.), so the stored kT attr means something

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        BaseTestCase.cleanup(cls.box_dim)   # builds a new Simulation: author and kT (1.) are set after it
        sim_inst.set_author(cls.author, cls.email)
        sim_inst.kT = cls.kT
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmpdir.cleanup)
        cls.addClassCleanup(BaseTestCase.cleanup, cls.box_dim)
        cls.h5_filename = os.path.join(cls.tmpdir.name, "testfile.h5")
        cls.written_steps = [10, 20, 30, 40]
        cls.written_times = []
        cls.build_fixture()
        properties = sim_inst.io_dict["properties"]
        #: {group: {prop: array [F, N(, dim)] in the stored dtype}}, columns in ascending id like the file's
        cls.snapshots = {group_type.__name__: {prop: [] for prop, _dim, _dtype in properties}
                         for group_type in cls.group_types}
        if hasattr(cls, "observable_name"):
            cls.observable_value = np.zeros(3, dtype=np.float64)
            cls.observable_values = []
        with patch.object(write_module, "get_submission_creator_info", return_value=(cls.runner_script, cls.runner_script_repo)), patch.object(write_module, "get_repo_context", return_value=(cls.lib_path, cls.library_vers)):
            sim_inst.inscribe_part_group_to_h5(group_type=cls.group_types, h5_data_path=cls.h5_filename)
            if hasattr(cls, "observable_name"):
                sim_inst.inscribe_observable_group_to_h5(
                    observable_defs=[(cls.observable_name, 3, np.float64, cls.observable_value)],
                    h5_data_path=cls.h5_filename,
                    mode='NEW',
                )
        for frame_index, GLOBAL_COUNTER in enumerate(cls.written_steps):
            sim_inst.sys.integrator.run(1)
            if hasattr(cls, "observable_name"):
                cls.observable_value[:] = np.array([GLOBAL_COUNTER, frame_index + 1, -GLOBAL_COUNTER], dtype=np.float64)
                cls.observable_values.append(cls.observable_value.copy())
            sim_inst.write_registered_to_h5(step=GLOBAL_COUNTER)
            cls.written_times.append(sim_inst.sys.time)
            for group_type in cls.group_types:
                parts = sorted((part for obj in sim_inst.objects if isinstance(obj, group_type)
                                for part in obj.get_owned_part()[0]), key=lambda part: part.id)
                for prop, _dim, dtype in properties:
                    cls.snapshots[group_type.__name__][prop].append(
                        np.array([getattr(part, prop) for part in parts], dtype=dtype))
        cls.snapshots = {group: {prop: np.asarray(frames) for prop, frames in snapshot.items()}
                         for group, snapshot in cls.snapshots.items()}
        BaseTestCase.reset_io_state()

    def tearDown(self):
        BaseTestCase.reset_io_state()
        super().tearDown()

    def check_selection(self, view, snapshot, time_slice=None, types=None):
        """dtype, shape and values of every stored property of ``view`` against the snapshot's frames
        ``time_slice`` (None: all) and its columns of ``types`` (None: all), selected frame by frame."""
        keep = np.ones(snapshot['type'].shape, bool) if types is None else np.isin(snapshot['type'], types)
        for prop, _dim, dtype in sim_inst.io_dict['properties']:
            want = np.stack([frame[mask] for frame, mask in zip(snapshot[prop], keep)])
            if time_slice is not None:
                want = want[time_slice]         # an integer frame index drops the frame axis
            got = getattr(view, prop)
            with self.subTest(prop=prop, time_slice=time_slice, types=types):
                self.assertEqual(got.dtype, np.dtype(dtype))
                self.assertEqual(got.shape, want.shape)
                np.testing.assert_allclose(got, want, rtol=1e-05, atol=1e-08)

    def check_version_signing(self, h5_file):
        """Provenance attrs, read from the file itself (the selector keeps no metadata tree)."""
        np.testing.assert_array_equal(h5_file["h5md"].attrs["version"], np.array([1, 1], dtype=np.int32))
        self.assertEqual(h5_file["h5md/creator"].attrs["name"], self.runner_script)
        self.assertEqual(h5_file["h5md/creator"].attrs["version"], self.runner_script_repo)
        self.assertEqual(h5_file["parameters/pressomancy"].attrs["version"], self.library_vers)
        self.assertEqual(h5_file["parameters/pressomancy"].attrs["layout"], LAYOUT)
        expected_part_types = {key: int(value) for key, value in sim_inst.part_types.items()
                               if isinstance(value, (int, np.integer))}
        observed_part_types = {key: int(value)
                               for key, value in h5_file["parameters/pressomancy/part_types"].attrs.items()}
        self.assertEqual(observed_part_types, expected_part_types)
        self.assertEqual(h5_file["h5md/author"].attrs["name"], self.author)
        self.assertEqual(h5_file["h5md/author"].attrs["email"], self.email)

    def test_file_layout(self):
        """Provenance, `/pressomancy/system` attrs, and per group the box, dims, H5MD elements and one shared step/time."""
        n_frames = len(self.written_steps)
        with h5py.File(self.h5_filename, "r") as h5_file:
            self.check_version_signing(h5_file)
            system = h5_file["pressomancy/system"]
            self.assertEqual(int(system.attrs["seed"]), sim_inst.seed)
            self.assertEqual(float(system.attrs["kT"]), self.kT)
            self.assertEqual(float(system.attrs["time_step"]), sim_inst.sys.time_step)
            np.testing.assert_allclose(system.attrs["box_l"], np.array(self.box_dim, dtype=float))
            np.testing.assert_array_equal(system.attrs["periodicity"], np.array(sim_inst.sys.periodicity))

            for group_type in self.group_types:
                group_name = group_type.__name__
                n_particles = sum(len(obj.get_owned_part()[0])
                                  for obj in sim_inst.objects if isinstance(obj, group_type))
                dataview = H5DataSelector(h5_file, particle_group=group_name)
                box = dataview.get_box()
                self.assertEqual(box["dimension"], len(self.box_dim))
                np.testing.assert_equal(box["boundary"], ("periodic", "periodic", "periodic"))
                # the box is a time-dependent element: edges is [F, D], one row per frame
                np.testing.assert_allclose(box["edges"], np.tile(np.array(self.box_dim, dtype=float), (n_frames, 1)))
                np.testing.assert_equal(dataview.common_dims, (n_frames, n_particles))
                self.assertEqual(len(dataview.timestep), n_frames)
                self.assertEqual(len(dataview.particles), n_particles)

                group = h5_file[f"particles/{group_name}"]
                # The H5MD names spelled out: element_name() below agrees with any table, a wrong one too.
                for h5md_name in ("id", "species", "position", "image", "box/edges"):
                    self.assertIn(h5md_name, group)
                for espresso_name in ("pos", "type", "image_box"):
                    self.assertNotIn(espresso_name, group)
                # a scalar (dim None) is [F, N], a vector [F, N, dim]; step/time are the *same* HDF5 objects
                step, time = (group[f"{element_name('pos')}/step"],
                              group[f"{element_name('pos')}/time"])
                for attr, dim, dtype in sim_inst.io_dict['properties']:
                    expected = (n_frames, n_particles) if dim is None else (n_frames, n_particles, dim)
                    with self.subTest(group=group_name, attr=attr):
                        element = group[element_name(attr)]
                        self.assertEqual(element["value"].shape, expected)
                        self.assertEqual(element["value"].dtype, np.dtype(dtype))
                        self.assertEqual(element["step"], step)
                        self.assertEqual(element["time"], time)
                self.assertEqual(group["box/edges/value"].shape,
                                 (n_frames, int(group["box"].attrs["dimension"])))
                self.assertEqual(group["box/edges/value"].dtype, np.dtype(np.float64))
                self.assertEqual(group["box/edges/step"], step)
                self.assertEqual(group["box/edges/time"], time)
                self.assertEqual(time.dtype, np.float32)
                np.testing.assert_array_equal(stored_steps(h5_file, group_name), self.written_steps)

            if hasattr(self, "observable_name"):
                selector = H5ObservableSelector(h5_file, observable_name=self.observable_name)
                self.assertEqual(len(selector.timestep), n_frames)
                np.testing.assert_array_equal(selector.step, self.written_steps)
                np.testing.assert_allclose(selector.time, self.written_times)
                np.testing.assert_allclose(selector.value, np.array(self.observable_values))
            else:
                self.assertNotIn("observables", h5_file)

    def test_load_modes(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            for mode in ("LOAD_NEW", "LOAD"):
                h5_filename = os.path.join(tmpdirname, f"{mode}.h5")
                shutil.copy2(self.h5_filename, h5_filename)
                try:
                    GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(
                        group_type=self.group_types, h5_data_path=h5_filename, mode=mode)
                    self.assertEqual(GLOBAL_COUNTER, len(self.written_steps))
                    for group_type in self.group_types:
                        group_name = group_type.__name__
                        expected_ids = sorted(part.id for obj in sim_inst.objects if isinstance(obj, group_type)
                                              for part in obj.get_owned_part()[0])
                        reconstructed_ids = [part.id for part in sim_inst.io_dict['flat_part_view'][group_name]]
                        np.testing.assert_array_equal(reconstructed_ids, expected_ids, err_msg=f"{mode} {group_name}")
                    if hasattr(self, "observable_name"):
                        observable_counter = sim_inst.inscribe_observable_group_to_h5(
                            observable_defs=[(self.observable_name, self.observable_value.shape, self.observable_value.dtype, self.observable_value)],
                            h5_data_path=h5_filename,
                            mode=mode,
                        )
                        self.assertEqual(observable_counter, len(self.written_steps))
                        registered = sim_inst.io_dict['registered_observables']
                        self.assertIn(self.observable_name, registered)
                        self.assertEqual(registered[self.observable_name]['shape'], self.observable_value.shape)
                        self.assertEqual(registered[self.observable_name]['dtype'], self.observable_value.dtype)
                        self.assertIs(registered[self.observable_name]['value'], self.observable_value)
                        selector = H5ObservableSelector(sim_inst.io_dict['h5_file'], observable_name=self.observable_name)
                        np.testing.assert_array_equal(selector.step, self.written_steps)
                        np.testing.assert_allclose(selector.time, self.written_times)
                        np.testing.assert_allclose(selector.value, np.array(self.observable_values))
                finally:
                    BaseTestCase.reset_io_state()

    def test_select_particles_by_object(self):
        with h5py.File(self.h5_filename, "r") as h5_file:
            for group_type in self.group_types:
                object_name = group_type.__name__
                snapshot = self.snapshots[object_name]
                dataview = H5DataSelector(h5_file, particle_group=object_name)
                self.assertEqual(len(dataview.timestep), len(self.written_steps))
                self.assertEqual(dataview.timestep[-1].step, self.written_steps[-1])
                # particle times are stored as float32, like the observables'
                self.assertEqual(dataview.timestep[-1].time, np.float32(self.written_times[-1]))
                objects = [obj for obj in sim_inst.objects if isinstance(obj, group_type)]
                connectivity_value = np.array([obj.who_am_i for obj in objects], dtype=int)
                types = np.unique(snapshot['type']).tolist()
                np.testing.assert_array_equal(dataview.get_connectivity_values(object_name), connectivity_value)

                connected_objects = sim_inst._collect_instances_recursively(objects)
                for class_name in sorted({type(obj).__name__ for obj in connected_objects}):
                    members = [obj for obj in connected_objects if type(obj).__name__ == class_name]
                    for predicate_type in types:
                        control_ids = [obj.who_am_i for obj in members
                                       if all(part.type == predicate_type for part in obj.get_owned_part()[0])]
                        selected_ids = dataview.get_connectivity_values(
                            class_name,
                            predicate=lambda subset, t=predicate_type: np.all(subset.timestep[-1].type == t))
                        np.testing.assert_array_equal(control_ids, np.array(selected_ids, dtype=int),
                                                      err_msg=f"{class_name} type {predicate_type}")

                for time_slice in (None, -1, 0, slice(0, 2)):
                    source = dataview if time_slice is None else dataview.timestep[time_slice]
                    with self.subTest(group=object_name, form="per particle"):
                        self.check_selection(source.select_particles_by_object(
                            object_name=object_name, connectivity_value=connectivity_value), snapshot, time_slice)
                        for predicate_type in types:
                            self.check_selection(source.select_particles_by_object(
                                object_name=object_name, connectivity_value=connectivity_value,
                                predicate=lambda p, t=predicate_type: p.type == t),
                                snapshot, time_slice, [predicate_type])
                for predicate_type in types:
                    with self.subTest(group=object_name, form="timestep[-1]"):
                        selection = dataview.select_particles_by_object(
                            object_name=object_name, connectivity_value=connectivity_value,
                            predicate=lambda subset, t=predicate_type: subset.timestep[-1].type == t)
                        self.assertEqual(len(selection.timestep), len(dataview.timestep))
                        self.check_selection(selection, snapshot, None, [predicate_type])

    def test_object_relations(self):
        with h5py.File(self.h5_filename, "r") as h5_file:
            for group_type in self.group_types:
                dataview = H5DataSelector(h5_file, particle_group=group_type.__name__)
                parents = sim_inst._collect_instances_recursively(
                    [obj for obj in sim_inst.objects if isinstance(obj, group_type)])
                for parent in parents:
                    if not getattr(parent, "associated_objects", None):
                        continue
                    parent_key = parent.__class__.__name__
                    child_key = parent.associated_objects[0].__class__.__name__
                    np.testing.assert_array_equal(
                        dataview.get_child_ids(parent_key, child_key, parent.who_am_i),
                        [child.who_am_i for child in parent.associated_objects],
                        err_msg=f"{parent_key} {parent.who_am_i}")
                    for child in parent.associated_objects:
                        expected_parent_ids = [obj.who_am_i for obj in sim_inst.objects
                                               if child in (getattr(obj, "associated_objects", None) or [])]
                        np.testing.assert_array_equal(
                            dataview.get_parent_ids(parent_key, child_key, child.who_am_i),
                            expected_parent_ids, err_msg=f"{child_key} {child.who_am_i}")

class ElastomerFixture(CommonH5DataSelectorTests, unittest.TestCase):
    layer_height = 4
    n_part = 20
    observable_name = "magnetic_dipole_moment"

    @classmethod
    def build_fixture(cls):
        conf_point_dipole = PointDipolePermanent.config.specify(dipm=1., espresso_handle=sim_inst.sys)
        point_dipoles = [PointDipolePermanent(config=conf_point_dipole) for _ in range(cls.n_part)]
        config_E = Elastomer.config.specify(
            layer_height=cls.layer_height, n_parts=cls.n_part, associated_objects=point_dipoles, espresso_handle=sim_inst.sys, seed=sim_inst.seed)
        elastomer=Elastomer(config=config_E)
        sim_inst.store_objects([elastomer])
        sim_inst.set_objects([elastomer])
        cls.group_types = [Elastomer, PointDipolePermanent]

    @unittest.skipUnless(importlib.util.find_spec("MDAnalysis"), "needs MDAnalysis")
    def test_mdanalysis_reads_positions_and_box(self):
        """An external H5MD reader sees what was written. MDAnalysis reads only the
        first group of `/particles` in h5py's (alphabetical) order, so the Universe
        holds the Elastomer group; the file carries no units, hence convert_units=False."""
        import MDAnalysis
        with h5py.File(self.h5_filename, "r") as h5_file:
            self.assertEqual(list(h5_file["particles"])[0], "Elastomer",
                             msg="MDAnalysis would read another group than the one compared here")
        # the default preset stores pos as float32, which is what MDAnalysis holds
        positions = self.snapshots["Elastomer"]["pos"].astype(np.float32)
        universe = MDAnalysis.Universe.empty(positions.shape[1], trajectory=True)
        universe.load_new(self.h5_filename, format='H5MD', convert_units=False)
        try:
            self.assertEqual(len(universe.trajectory), len(self.written_steps))
            for frame, want in zip(universe.trajectory, positions):
                with self.subTest(frame=frame.frame):
                    np.testing.assert_array_equal(frame.positions, want)
                    np.testing.assert_allclose(frame.dimensions, [*sim_inst.sys.box_l, 90., 90., 90.])
        finally:
            universe.trajectory.close()

class FilamentFixture(CommonH5DataSelectorTests, unittest.TestCase):

    box_dim = (75.6, 75.6, 75.6)
    no_obj=30
    sheets_per_quad = 3
    part_per_filament = 2
    no_crowders=10

    @classmethod
    def build_fixture(cls):
        quartet_configuration = Quartet.config.specify(espresso_handle=sim_inst.sys)
        quartets = [Quartet(config=quartet_configuration) for _ in range(cls.no_obj)]
        sim_inst.store_objects(quartets)

        bond_quad = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2*1.5))
        grouped_quartets = [quartets[i:i+cls.sheets_per_quad]
                            for i in range(0, len(quartets), cls.sheets_per_quad)]
        quadriplex_configuration_list = [
            Quadriplex.config.specify(size=6., espresso_handle=sim_inst.sys, bond_handle=bond_quad, associated_objects=elem)
            for elem in grouped_quartets
        ]

        quadriplexes = [Quadriplex(config=configuration) for configuration in quadriplex_configuration_list]
        sim_inst.store_objects(quadriplexes)
        bond_pass = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=2*1.5))
        grouped_quadriplexes = [quadriplexes[i:i+cls.part_per_filament:]
                                for i in range(0, len(quadriplexes), cls.part_per_filament)]
        filament_configuration_list = [
            Filament.config.specify(sigma=6, size=6*cls.part_per_filament, n_parts=cls.part_per_filament, espresso_handle=sim_inst.sys, bond_handle=bond_pass, associated_objects=elem)
            for elem in grouped_quadriplexes
        ]
        filaments = [Filament(config=configuration) for configuration in filament_configuration_list]
        sim_inst.store_objects(filaments)
        sim_inst.set_objects(filaments)

        crowder_configuration=Crowder.config.specify(size=1., espresso_handle=sim_inst.sys)
        crowders = [Crowder(config=crowder_configuration) for _ in range(cls.no_crowders)]
        sim_inst.store_objects(crowders)
        sim_inst.set_objects(crowders)

        cls.group_types = [Filament, Crowder]


class BondTopologyIOTest(IOTestCase):
    """Every mode records `bond_links`; a stray bond is refused before any element of the frame is appended."""

    box_dim = (20, 20, 20)      # set_objects needs L/2 >= the filament size (10)
    n_parts = 5
    n_filaments = 3

    def test_every_mode_records_the_links_and_refuses_a_stray_bond(self):
        bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=3.))
        filaments = [Filament(config=Filament.config.specify(
            sigma=2., size=2. * self.n_parts, n_parts=self.n_parts, espresso_handle=sim_inst.sys,
            bond_handle=bond)) for _ in range(self.n_filaments)]
        sim_inst.store_objects(filaments)
        sim_inst.set_objects(filaments)
        for filament in filaments:
            filament.bond_center_to_center(type_name='real')
        live_links = sum(len(part.bonds) for part in sim_inst.sys.part.all())
        self.assertGreater(live_links, 0, msg="fixture built no bonds")
        path = os.path.join(self.tmpdir.name, "bond_topology.h5")
        handles = filaments[0].type_part_dict['real']
        stray = (filaments[0].params['bond_handle'].get_raw_handle(), handles[-1].id)
        # Each mode resumes the file the previous one left: one frame per mode.
        for n_frames, mode in enumerate(("NEW", "LOAD", "LOAD_NEW")):
            with self.subTest(mode=mode):
                sim_inst.io_dict["bonds"] = True
                try:
                    counter = sim_inst.inscribe_part_group_to_h5(
                        group_type=[Filament], mode=mode, h5_data_path=path)
                    self.assertEqual(counter, n_frames)
                    self.assertEqual(sim_inst.io_dict["bond_links"]["Filament"], live_links)
                    h5_file = sim_inst.io_dict["h5_file"]
                    # angles/dihedrals tables are written only when such links exist
                    self.assertNotIn("angles", h5_file["connectivity/Filament"])
                    sim_inst.write_part_group_to_h5(step=n_frames)

                    handles[0].add_bond(stray)
                    try:
                        with self.assertRaises(RuntimeError) as ctx:
                            sim_inst.write_part_group_to_h5(step=n_frames + 1)
                        self.assertIn("changed after inscription", str(ctx.exception))
                    finally:
                        handles[0].delete_bond(stray)
                    group = h5_file["particles/Filament"]
                    for element in [element_name(attr) for attr, _dim, _dtype in sim_inst.io_dict['properties']] + ["box/edges"]:
                        for dataset in ("value", "step"):
                            self.assertEqual(group[f"{element}/{dataset}"].shape[0], n_frames + 1,
                                             msg=f"{element}/{dataset}")
                finally:
                    self.reset_io_state()     # what a fresh process starts from


class BulkFrameReadTest(IOTestCase):
    """`_capture_frame` equals a per-particle loop on gapped and custom ids, and refuses a disordered view.

    A ParticleSlice built from non-monotonic ids disagrees with itself about row order: the writer slices ascending ids only."""

    box_dim = (20, 20, 20)      # holds the particles placed in [0, 20)^3

    def test_bulk_read_matches_per_particle_loop_for_gapped_ids(self):
        rng = np.random.default_rng(4242)
        for _ in range(60):
            part = sim_inst.sys.part.add(pos=rng.random(3) * 20.0, type=int(rng.integers(0, 4)))
            part.dip = rng.random(3) + 0.5
        sim_inst.sys.part.add(id=5000, pos=[1.0, 2.0, 3.0], type=1)          # a custom id
        for stray in list(sim_inst.sys.part.all())[3:60:7]:                   # punch holes in the id range
            stray.remove()
        sim_inst.sys.integrator.run(5)

        handles = sorted(sim_inst.sys.part.all(), key=lambda p: p.id)
        group = "GappedProbe"
        sim_inst.io_dict['flat_part_view'][group] = handles
        writer = sim_inst._h5_writer
        for prop, dim, dtype in sim_inst.io_dict['properties']:
            # dim None is a scalar column [N]; a vector one is [N, dim]
            expected = np.array([getattr(h, prop) for h in handles], dtype=dtype)
            got = writer._capture_frame(group, prop, dim, dtype)
            with self.subTest(prop=prop):
                self.assertEqual(got.shape, expected.shape)
                np.testing.assert_array_equal(got, expected)
        for bad, label in ((handles[::-1], "unsorted"), (handles[:1] * 2, "duplicate")):
            sim_inst.io_dict['flat_part_view'][group] = bad
            writer._slice_cache.pop(group, None)
            with self.subTest(label), self.assertRaises(ValueError) as ctx:
                writer._capture_frame(group, 'pos', 3, np.float64)
            self.assertIn("strictly ascending particle ids", str(ctx.exception))


class SourceFixture(IOTestCase):
    """Build a tree, write it to a source file, rebuild the tree, read the file back onto the new one.

    Seeding needs the file's who_am_i set (a per-class construction counter) to equal the seeded objects';
    `rebuild()` rewinds the counters as a restart would. Seeded objects are stored, not set: placement adds
    their particles (hence `place=`). Source files use `SOURCE_PROPERTIES` (floats float64, dip stored)."""

    n_parts = 4
    n_filaments = 2
    box_dim = SMALL_BOX
    SOURCE_PROPERTIES = ([('id', None, np.int32), ('type', None, np.int16), ('pos', 3, np.float64)]
                         + ([('director', 3, np.float64)] if api_agnostic_feature_check('ROTATION') else [])
                         + ([('dip', 3, np.float64)] if api_agnostic_feature_check('DIPOLES') else []))

    def setUp(self):
        super().setUp()
        self.pin()

    # -- building ----------------------------------------------------------
    def rebuild(self):
        """Empty the system and rewind the who_am_i counters: what a restart hands a script."""
        BaseTestCase.cleanup(self.box_dim)
        self.pin()

    def pin(self):
        """Re-apply the properties list `cleanup()` resets to the writer's default."""
        sim_inst.io_dict['properties'] = list(self.SOURCE_PROPERTIES)

    def build_filaments(self, n_filaments=None, place=True, one_bond_handle=True,
                        with_dipoles=False, with_anchors=False, bonded=False):
        """Store (and by default place) filaments; ``one_bond_handle=False`` gives each its own, equal FeneBond."""
        shared = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=3.))
        filaments = [Filament(config=Filament.config.specify(
            sigma=2., size=2. * self.n_parts, n_parts=self.n_parts,
            espresso_handle=sim_inst.sys,
            bond_handle=shared if one_bond_handle else BondWrapper(
                espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=3.))))
            for _ in range(self.n_filaments if n_filaments is None else n_filaments)]
        sim_inst.store_objects(filaments)
        if place:
            sim_inst.set_objects(filaments)
            if with_anchors:
                for filament in filaments:
                    filament.add_anchors(type_name='real')
            if with_dipoles:
                for index, part in enumerate(sim_inst.sys.part.all()):
                    part.dip = np.array([1.0, 0.5, 0.25]) * (index + 1)
            if bonded:
                for filament in filaments:
                    filament.bond_center_to_center(type_name='real')
        return filaments

    def build_elastomer(self, place=True):
        elastomer = Elastomer(config=Elastomer.config.specify(
            box_E=[3, 3, 9], n_parts=10, sigma=1. / WCA_CONTACT_FACTOR, espresso_handle=sim_inst.sys,
            seed=sim_inst.seed))
        sim_inst.store_objects([elastomer])
        if place:
            sim_inst.set_objects([elastomer])
        return elastomer

    # -- writing -----------------------------------------------------------
    def write_source(self, name, group_type=Filament, bonds=False, steps=(0,), advance=None):
        """Write one frame per entry of ``steps`` (``advance()`` before each) and return the path."""
        path = os.path.join(self.tmpdir.name, name)
        sim_inst.io_dict["bonds"] = bonds
        sim_inst.inscribe_part_group_to_h5(group_type=[group_type], h5_data_path=path,
                                           mode="NEW")
        for step in steps:
            if advance is not None:
                advance()
            sim_inst.write_part_group_to_h5(step=step)
        self.reset_io_state()
        return path

    def edited_copy(self, path, name, edit):
        """A copy of ``path`` named ``name``, with ``edit(h5_file)`` applied to it."""
        copy = os.path.join(self.tmpdir.name, name)
        shutil.copy2(path, copy)
        with h5py.File(copy, "r+") as h5_file:
            edit(h5_file)
        return copy

    # -- reading the live system -------------------------------------------
    @staticmethod
    def typed_values(objects, type_name, attr='pos'):
        """``{who_am_i: (N, dim) array}`` of ``attr`` over the objects' ``type_name`` particles, in ascending id."""
        return {obj.who_am_i: np.array([np.copy(getattr(part, attr))
                                        for part in sorted(obj.type_part_dict[type_name], key=lambda p: p.id)])
                for obj in objects}

    @classmethod
    def real_positions(cls, objects):
        return cls.typed_values(objects, 'real')

    @staticmethod
    def live_links(particles):
        """``(owner, partner ids, bond class, bond params)`` of every bond on ``particles``, sorted by the first three."""
        return sorted(((int(part.id), tuple(int(x) for x in entry[1:]), type(entry[0]).__name__,
                        entry[0].get_params())
                       for part in particles for entry in part.bonds), key=lambda link: link[:3])

    @staticmethod
    def n_live_links():
        return sum(len(part.bonds) for part in sim_inst.sys.part.all())

    @staticmethod
    def n_registered_bonds():
        return sum(1 for _ in sim_inst.sys.bonded_inter)


class SourceSeedingTest(SourceFixture):
    """What the source supplies (`get_pos_ori_from_src`), what it copies (`set_prop_from_src`), and the pairing."""

    def test_non_contiguous_ids_round_trip_in_ascending_id_order(self):
        """Gaps and custom ids in the source do not disturb the zip onto a tree with contiguous ids."""
        first = self.build_filaments(n_filaments=1)
        sim_inst.sys.part.add(id=700, pos=[1.0, 1.0, 1.0], type=0)       # custom id -> the next ids start at 701
        second = self.build_filaments(n_filaments=1)
        sim_inst.sys.part.by_id(700).remove()                            # leaves a hole below the 2nd filament
        filaments = first + second
        written = self.real_positions(filaments)
        path = self.write_source("gapped.h5")
        with h5py.File(path, 'r') as f:
            ids = np.asarray(f['particles/Filament/id/value'][-1])
        self.assertTrue(np.all(np.diff(ids) > 0), ids)
        self.assertGreater(int(ids[self.n_parts]) - int(ids[self.n_parts - 1]), 1, "fixture left no id gap")

        self.rebuild()
        filaments = self.build_filaments(n_filaments=2)                  # contiguous ids this time
        sim_inst.load_from_src(filaments, path, src_to_loc={('real', 'real'): [('pos', 'pos')]})
        for who, positions in self.real_positions(filaments).items():
            np.testing.assert_allclose(positions, written[who], err_msg=f"filament {who}")

    def test_a_dip_becomes_a_director_unless_it_is_zero(self):
        """The source dip is normalised onto the local director; a zero dip has no direction to give."""
        filaments = self.build_filaments(with_dipoles=True)
        dips = self.typed_values(filaments, 'real', 'dip')
        path = self.write_source("dip_to_director.h5")
        for part in sim_inst.sys.part.all():
            part.dipm = 0.
        zero_path = self.write_source("zero_dip.h5")          # the same tree, dip zero

        self.rebuild()
        filaments = self.build_filaments()
        for part in sim_inst.sys.part.all():
            part.director = [0., 0., 1.]
        dip_to_director = {('real', 'real'): [('dip', 'director')]}
        sim_inst._h5_init.set_init_src(path, place_from=['real'], src_to_loc=dip_to_director)
        sim_inst._h5_init.set_prop_from_src(filaments)
        for who, directors in self.typed_values(filaments, 'real', 'director').items():
            want = dips[who] / np.linalg.norm(dips[who], axis=1, keepdims=True)
            np.testing.assert_allclose(directors, want, rtol=0, atol=1e-12)

        sim_inst._h5_init.set_init_src(zero_path, place_from=['real'], src_to_loc=dip_to_director)
        with self.assertRaises(ValueError) as ctx:
            sim_inst._h5_init.set_prop_from_src(filaments)
        self.assertIn("dip moment magnitude is 0", str(ctx.exception))

    def test_each_type_pair_copies_its_own_property_list(self):
        """A grouped key shares pos between 'real' and 'virt'; dip is copied for 'real' only, 'virt' keeps zeros."""
        filaments = self.build_filaments(with_anchors=True, with_dipoles=True)   # anchors: a second type
        written = {(name, attr): self.typed_values(filaments, name, attr)
                   for name in ('real', 'virt') for attr in ('pos', 'dip')}
        self.assertTrue(any(np.any(dips) for dips in written[('virt', 'dip')].values()),
                        msg="source virt dips are zero")
        path = self.write_source("per_pair_props.h5")

        self.rebuild()
        filaments = self.build_filaments(with_anchors=True)
        for part in sim_inst.sys.part.all():
            part.pos = part.pos + np.array([4.0, 1.0, -3.0])
        for name in ('real', 'virt'):
            for who, positions in self.typed_values(filaments, name).items():
                self.assertFalse(np.allclose(positions, written[(name, 'pos')][who]),
                                 msg=f"the fixture did not move {name}")

        sim_inst._h5_init.set_init_src(path, place_from=['real'], src_to_loc={
            (('real', 'real'), ('virt', 'virt')): [('pos', 'pos')], ('real', 'real'): [('dip', 'dip')]})
        sim_inst._h5_init.set_prop_from_src(filaments)
        for name in ('real', 'virt'):
            for who, positions in self.typed_values(filaments, name).items():
                np.testing.assert_allclose(positions, written[(name, 'pos')][who], rtol=1e-10, atol=1e-10)
        for who, dips in self.typed_values(filaments, 'real', 'dip').items():
            np.testing.assert_allclose(dips, written[('real', 'dip')][who], rtol=0, atol=1e-12)
        for who, dips in self.typed_values(filaments, 'virt', 'dip').items():
            np.testing.assert_array_equal(dips, np.zeros((2 * self.n_parts, 3)))

    def test_the_pairing_is_strictly_one_to_one(self):
        """Every type pair is zipped one-to-one (no `min()`, no skip), and a mismatch assigns or places nothing."""
        filaments = self.build_filaments()
        path = self.write_source("pairing.h5")

        with self.subTest(case="4 source reals vs 8 local virts"):
            self.rebuild()
            filaments = self.build_filaments(with_anchors=True)
            before = self.typed_values(filaments, 'virt')
            sim_inst._h5_init.set_init_src(path, place_from=['real'],
                                          src_to_loc={('real', 'virt'): [('pos', 'pos')]})
            with self.assertRaises(ValueError) as ctx:
                sim_inst._h5_init.set_prop_from_src(filaments)
            self.assertIn(f"{self.n_parts} source particles vs {2 * self.n_parts} local particles",
                          str(ctx.exception))
            for who, positions in self.typed_values(filaments, 'virt').items():
                np.testing.assert_array_equal(positions, before[who])

        # One unplaced tree for the rest: its first n_filaments carry the source's who_am_i.
        self.rebuild()
        larger = self.build_filaments(n_filaments=self.n_filaments + 1, place=False)
        filaments = larger[:self.n_filaments]
        with self.subTest(case="an unplaced tree without place_from"):
            with self.assertRaises(ValueError) as ctx:
                sim_inst.load_from_src(filaments, path,
                                       src_to_loc={('real', 'real'): [('pos', 'pos')]})
            self.assertIn(f"{self.n_parts} source particles vs 0 local particles", str(ctx.exception))
            self.assertEqual(len(sim_inst.sys.part), 0)
        with self.subTest(case="an empty property list still pairs"):
            sim_inst._h5_init.set_init_src(path, place_from=['real'], src_to_loc={('real', 'virt'): []})
            with self.assertRaises(ValueError) as ctx:
                sim_inst._h5_init.set_prop_from_src(filaments)
            self.assertIn(f"{self.n_parts} source particles vs 0 local particles", str(ctx.exception))
        sim_inst._h5_init.set_init_src(path, place_from=['real'])
        for label, subset in (("a who_am_i subset", larger[:1]), ("a who_am_i superset", larger)):
            with self.subTest(case=label):
                with self.assertRaises(ValueError) as ctx:
                    sim_inst._h5_init.get_pos_ori_from_src(subset)
                self.assertIn("who_am_i", str(ctx.exception))
        self.assertEqual(len(sim_inst.sys.part), 0)

    def test_a_malformed_declaration_raises_naming_the_offender(self):
        """Every shape the `src_to_loc` grammar or `place_from` rejects: a ValueError, and nothing declared."""
        self.build_filaments()
        path = self.write_source("malformed_declaration.h5")
        pos = {('real', 'real'): [('pos', 'pos')]}
        cases = {   # label: (src_to_loc, place_from, fragment of the raise)
            "a pair twice in one list": (
                {('real', 'real'): [('pos', 'pos'), ('pos', 'pos')]}, ['real'], "listed twice"),
            "a pair once under each of two keys": (
                {('real', 'real'): [('pos', 'pos')], (('real', 'real'), ('virt', 'virt')): [('pos', 'pos')]},
                ['real'], "listed twice"),
            "a pair twice through one grouped key": (
                {(('real', 'real'), ('real', 'real')): [('pos', 'pos')]}, ['real'], "listed twice"),
            "a key that is not a tuple": ({'real': [('pos', 'pos')]}, ['real'], "'real'"),
            "an empty key": ({(): [('pos', 'pos')]}, ['real'], "()"),
            "a type triple": ({('real', 'real', 'virt'): []}, ['real'], "('real', 'real', 'virt')"),
            "a non-string type name": ({('real', 1): []}, ['real'], "('real', 1)"),
            "a value that is not a list": ({('real', 'real'): 'pos'}, ['real'], "'pos'"),
            "a property triple": ({('real', 'real'): [('pos', 'pos', 'pos')]}, ['real'], "('pos', 'pos', 'pos')"),
            "a mapping that is not a dict": ([('real', 'real')], ['real'], "[('real', 'real')]"),
            # bool is an int subclass: True must not pass as type 1
            "a bool source type": ({(True, 'real'): []}, ['real'], "(True, 'real')"),
            "a float place_from entry": (pos, [1.5], "[1.5]"),
            "a bool place_from entry": (pos, [True], "[True]"),
        }
        for label, (src_to_loc, place_from, fragment) in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError) as ctx:
                    sim_inst._h5_init.set_init_src(path, src_to_loc=src_to_loc, place_from=place_from)
                self.assertIn(fragment, str(ctx.exception))
                self.assertEqual(sim_inst._h5_init.src_to_loc, {})
                self.assertIsNone(sim_inst._h5_init.place_from)

    def test_get_prop_from_src_returns_the_written_values_per_object(self):
        """Read a stored column back per object on an unplaced tree, naming a missing dataset."""
        filaments = self.build_filaments(with_dipoles=True)
        written = self.real_positions(filaments)
        path = self.write_source("get_prop.h5")

        self.rebuild()
        filaments = self.build_filaments(place=False)
        got = sim_inst.get_prop_from_src(filaments, path, src_type='real', prop='pos')
        self.assertEqual(len(got), len(filaments))
        for filament, positions in zip(filaments, got):
            self.assertEqual(positions.shape, (self.n_parts, 3))
            np.testing.assert_array_equal(positions, written[filament.who_am_i])
        self.assertEqual(len(sim_inst.sys.part), 0)

        with self.assertRaises(KeyError) as ctx:
            sim_inst.get_prop_from_src(filaments, path, src_type='real', prop='no_such_prop')
        self.assertIn("particles/Filament/no_such_prop/value", str(ctx.exception))
        with self.assertRaises(KeyError):
            sim_inst.get_prop_from_src(filaments, path, src_type='no_such_type', prop='pos')

    def test_a_reader_out_of_order_is_a_state_error(self):
        """No source declared, a missing file, or no `place_from` for a placing reader: refused, nothing placed."""
        self.build_filaments()
        path = self.write_source("state_errors.h5")

        self.rebuild()
        filaments = self.build_filaments(place=False)
        self.assertIsNone(sim_inst._h5_init.src_path_h5)
        for label, call in (("set_prop_from_src", lambda: sim_inst._h5_init.set_prop_from_src(filaments)),
                            ("get_pos_ori_from_src", lambda: sim_inst._h5_init.get_pos_ori_from_src(filaments))):
            with self.subTest(case="no source declared", method=label):
                with self.assertRaises(RuntimeError):
                    call()

        missing = os.path.join(self.tmpdir.name, "no_such_source.h5")
        with self.subTest(case="a missing source file"):
            with self.assertRaises(FileNotFoundError) as ctx:
                sim_inst._h5_init.set_init_src(missing, src_to_loc={('real', 'real'): [('pos', 'pos')]},
                                              place_from=['real'])
            self.assertIn(missing, str(ctx.exception))
            self.assertIsNone(sim_inst._h5_init.src_path_h5)

        sim_inst._h5_init.set_init_src(path, src_to_loc={('real', 'real'): [('pos', 'pos')]})
        self.assertIsNone(sim_inst._h5_init.place_from)
        for label, call in (("get_pos_ori_from_src",
                             lambda: sim_inst._h5_init.get_pos_ori_from_src(filaments)),
                            ("set_objects_from_src",
                             lambda: sim_inst._h5_init.set_objects_from_src(filaments))):
            with self.subTest(case="no place_from", method=label):
                with self.assertRaises(RuntimeError) as ctx:
                    call()
                self.assertIn("place_from", str(ctx.exception))
        self.assertEqual(len(sim_inst.sys.part), 0)

    # -- placement is optional (place_from) --------------------------------
    def test_load_from_src_without_place_from_copies_state_onto_an_existing_tree(self):
        """No `place_from` (`samples/poly_BRACO.py`): state and topology land on the script's own tree, no particle added."""
        filaments = self.build_filaments(bonded=True)
        written = self.real_positions(filaments)
        n_links = self.n_live_links()
        self.assertGreater(n_links, 0, msg="fixture built no bonds")
        path = self.write_source("existing_tree.h5", bonds=True)

        self.rebuild()
        filaments = self.build_filaments()          # built and placed locally, no bonds yet
        for part in sim_inst.sys.part.all():
            part.pos = part.pos + np.array([2.0, -1.0, 0.5])
        n_particles = len(sim_inst.sys.part)
        self.assertGreater(n_particles, 0, msg="the tree was not placed")

        n_added = sim_inst.load_from_src(filaments, path,
                                         src_to_loc={('real', 'real'): [('pos', 'pos')]},
                                         bonds=True)
        self.assertEqual(len(sim_inst.sys.part), n_particles)
        for who, positions in self.real_positions(filaments).items():
            np.testing.assert_allclose(positions, written[who], rtol=1e-10, atol=1e-10)
        self.assertEqual(n_added, n_links)
        self.assertEqual(self.n_live_links(), n_links)

    def test_a_stored_type_number_reads_a_file_without_a_type_table(self):
        """A numeric source type is the espresso type stored in the file, so a file with no type table still seeds."""
        filaments = self.build_filaments()
        written = self.real_positions(filaments)
        real_type = int(sim_inst.part_types['real'])
        no_table = self.edited_copy(self.write_source("type_resolution.h5"), "no_type_table.h5",
                                    lambda f: f.pop("parameters/pressomancy/part_types"))

        self.rebuild()
        filaments = self.build_filaments(place=False)
        sim_inst.load_from_src(filaments, no_table,
                               src_to_loc={(real_type, 'real'): [('pos', 'pos')]},
                               place_from=[real_type])
        for who, positions in self.real_positions(filaments).items():
            np.testing.assert_allclose(positions, written[who], rtol=1e-10, atol=1e-10)


class SeededObjectsAreObstaclesTest(SourceFixture):
    """Objects placed by `load_from_src` used to be invisible to a later `set_objects`."""

    def test_set_objects_avoids_seeded_filaments(self):
        """Of the size-1 sites in the 16^3 box, ~19% lie within reach of the two filaments (size 8),
        so 200 random ones miss them by chance with probability ~1e-18."""
        self.build_filaments()
        path = self.write_source("seed.h5")

        self.rebuild()
        filaments = self.build_filaments(place=False)
        sim_inst.load_from_src(filaments, path, src_to_loc={('real', 'real'): [('pos', 'pos')]},
                               place_from=['real'])
        seeded = [np.mean([part.pos for part in fil.get_owned_part()[0]], axis=0) for fil in filaments]
        crowders = [Crowder(config=Crowder.config.specify(size=1., espresso_handle=sim_inst.sys))
                    for _ in range(200)]
        sim_inst.store_objects(crowders)
        sim_inst.set_objects(crowders)
        centres = np.array([crowder.get_owned_part()[0][0].pos for crowder in crowders])
        dist = np.linalg.norm(min_img_dist(centres[:, None], np.asarray(seeded)[None], sim_inst.sys.box_l), axis=-1)
        self.assertGreaterEqual(dist.min(), 0.5 * (filaments[0].params['size'] + 1.) - 1e-9)


class OldLayoutFileTest(SourceFixture):
    """A file without `parameters/pressomancy/layout` is pre-H5MD: every particle-data entry refuses it."""

    def test_a_file_without_the_layout_attr_is_refused_with_the_convert_message(self):
        self.build_filaments()
        path = self.write_source("old_layout.h5")
        with h5py.File(path, "r+") as h5_file:
            del h5_file["parameters/pressomancy"].attrs["layout"]

        self.rebuild()
        filaments = self.build_filaments(place=False)
        with h5py.File(path, "r") as handle:
            entries = {
                "H5DataSelector": lambda: H5DataSelector(handle, particle_group="Filament"),
                "stored_steps": lambda: stored_steps(handle, "Filament"),
            }
            for name, call in entries.items():
                with self.subTest(entry=name):
                    with self.assertRaises(RuntimeError) as ctx:
                        call()
                    self.assertIn("old pressomancy HDF5 layout", str(ctx.exception))
        with self.assertRaises(RuntimeError) as ctx:
            sim_inst.load_from_src(filaments, path,
                                   src_to_loc={('real', 'real'): [('pos', 'pos')]},
                                   place_from=['real'])
        self.assertIn("old pressomancy HDF5 layout", str(ctx.exception))
        self.assertEqual(len(sim_inst.sys.part), 0)


class LoadNewPartTypesTest(SourceFixture):
    """LOAD_NEW checks the file's type table against the live one instead of restoring it."""

    def inscribe_load_new(self, path):
        return sim_inst.inscribe_part_group_to_h5(group_type=[Filament], h5_data_path=path,
                                                  mode='LOAD_NEW')

    def test_the_type_table_is_checked_not_restored(self):
        """A shared name with another number refuses; a file-only name is adopted; no table still guards the columns."""
        filaments = self.build_filaments()
        source = self.write_source("part_types.h5", steps=(0, 1))
        live_type = int(sim_inst.part_types['real'])
        table = "parameters/pressomancy/part_types"

        with self.subTest(case="a name declared with another number"):
            path = self.edited_copy(source, "types_mismatch.h5",
                                    lambda f: f[table].attrs.create('real', live_type + 100))
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    self.inscribe_load_new(path)
            finally:
                self.reset_io_state()
            self.assertIn("part_types mismatch", str(ctx.exception))

        with self.subTest(case="a name only the file declares"):
            path = self.edited_copy(source, "ghost_type.h5", lambda f: f[table].attrs.create('ghost', 999))
            self.assertNotIn('ghost', sim_inst.part_types)
            try:
                self.assertEqual(self.inscribe_load_new(path), 2)
                self.assertEqual(int(sim_inst.part_types['ghost']), 999)
            finally:
                sim_inst.part_types.pop('ghost', None)
                self.reset_io_state()

        with self.subTest(case="a file without the table"):
            path = self.edited_copy(source, "types_absent.h5", lambda f: f.pop(table))
            try:
                self.assertEqual(self.inscribe_load_new(path), 2)
            finally:
                self.reset_io_state()
            moved = filaments[0].type_part_dict['real'][0]
            moved.pos = np.asarray(moved.pos) + np.array([1.0, 0.0, 0.0])
            with self.assertRaises(RuntimeError) as ctx:
                self.inscribe_load_new(path)
            self.assertIn("refusing to append", str(ctx.exception))


_SM_FEATURES = sorted(set(required_features_for('langevin')) | set(Elastomer.required_features))


class SourceBondRestoreTest(SourceFixture):
    """`set_bonds_from_src` onto a rebuilt tree: the topology it restores and the guards that fire before it attaches."""

    def assert_links_match(self, got, want, exact_params=(), float32_params=()):
        """Same (owner, partners, bond class) rows; the named parameters equal exactly or as float32 (stored)."""
        self.assertEqual([link[:3] for link in got], [link[:3] for link in want])
        for (*_, got_params), (*_, want_params) in zip(got, want):
            for name in exact_params:
                self.assertEqual(got_params[name], want_params[name], msg=name)
            for name in float32_params:
                self.assertEqual(np.float32(got_params[name]), np.float32(want_params[name]),
                                 msg=name)

    def test_elastomer_round_trip_overrides_r_cut(self):
        elastomer = self.build_elastomer()
        # A positive stored break distance, so the override is observable.
        elastomer.random_harmonic_bonds(r_catch=2.5, bond_k=(0.01, 0.1), max_bonds=4, r_cut=3.0)
        written = self.live_links(elastomer.get_owned_part()[0])
        self.assertGreater(len(written), 0, msg="fixture built no bonds")
        path = self.write_source("elastomer_bonds.h5", group_type=Elastomer, bonds=True)
        with h5py.File(path, "r") as h5_file:
            n_links = int(h5_file["connectivity/Elastomer/bonds"].attrs["n_links"])
        self.assertEqual(n_links, len(written))

        for r_cut_override, want_r_cut in ((0.0, 0.0), (None, 3.0)):
            with self.subTest(r_cut_override=r_cut_override):
                self.rebuild()
                elastomer = self.build_elastomer(place=False)
                n_added = sim_inst.load_from_src([elastomer], path,
                                                 src_to_loc={('real', 'real'): []}, bonds=True,
                                                 place_from=['real'],
                                                 r_cut_override=r_cut_override)
                self.assertEqual(n_added, n_links)
                restored = self.live_links(elastomer.get_owned_part()[0])
                self.assert_links_match(restored, written, float32_params=("k", "r_0"))
                self.assertEqual({params["r_cut"] for *_, params in restored}, {want_r_cut})
                # every link carries its own random k, so every parameter set is distinct
                self.assertEqual(self.n_registered_bonds(), n_links)
                sim_inst.sys.integrator.run(0)
                self.assertLess(abs(sim_inst.sys.analysis.energy()["bonded"]), 1e-10)

    def test_a_filament_round_trip_restores_pairs_and_angles_and_copies_nothing(self):
        """Pairs and angles (own tables) come back once per parameter set; `[]` zips for the bonds but copies no dip."""
        filaments = self.build_filaments(one_bond_handle=False, bonded=True, with_dipoles=True)
        for filament in filaments:
            filament.add_bending_potential(
                type_name='real',
                bond_handle=espressomd.interactions.AngleHarmonic(bend=3., phi0=np.pi))
        written = self.live_links([part for filament in filaments for part in filament.get_owned_part()[0]])
        n_pairs = self.n_filaments * (self.n_parts - 1)
        n_angles = self.n_filaments * (self.n_parts - 2)
        self.assertEqual(len(written), n_pairs + n_angles)
        self.assertTrue(all(np.any(dips) for dips in self.typed_values(filaments, 'real', 'dip').values()),
                        msg="source dips are zero")
        path = self.write_source("filament_bonds.h5", bonds=True)
        with h5py.File(path, "r") as h5_file:
            for bond_class in ("FeneBond", "AngleHarmonic"):     # one stored parameter set per filament
                self.assertEqual(len(h5_file[f"pressomancy/Filament/bond_params/{bond_class}"]),
                                 self.n_filaments, msg=bond_class)
            self.assertEqual(h5_file["connectivity/Filament/bonds"].shape, (n_pairs, 2))
            self.assertEqual(h5_file["connectivity/Filament/angles"].shape, (n_angles, 3))
            stored = sorted((owner, partners) for owner, partners, _ in
                            read_bonds(h5_file, "Filament"))
        self.assertEqual(stored, sorted((owner, partners) for owner, partners, *_ in written))

        self.rebuild()
        filaments = self.build_filaments(place=False)
        n_added = sim_inst.load_from_src(filaments, path, src_to_loc={('real', 'real'): []},
                                         bonds=True, place_from=['real'])
        self.assertEqual(n_added, len(written))
        restored = self.live_links([part for filament in filaments for part in filament.get_owned_part()[0]])
        self.assertEqual([link[:3] for link in restored], [link[:3] for link in written])
        # FeneBond has no r_cut, so the default override leaves its parameters exact
        self.assert_links_match([link for link in restored if link[2] == 'FeneBond'],
                                [link for link in written if link[2] == 'FeneBond'],
                                exact_params=("k", "r_0", "d_r_max"))
        self.assertEqual(self.n_registered_bonds(), 2)       # one FeneBond, one AngleHarmonic handle
        for who, dips in self.typed_values(filaments, 'real', 'dip').items():
            np.testing.assert_array_equal(dips, np.zeros((self.n_parts, 3)))

    def test_every_bond_guard_fires_before_a_bond_is_attached(self):
        """Every refusal leaves no live link and no new registration.

        The dangling-bond guard is unreachable by seeding part of a group (the who_am_i set check forbids it):
        a bond that leaves the mapped *types* ('real' to an anchor 'virt') reaches it, in both directions."""
        filaments = self.build_filaments(with_anchors=True, bonded=True)
        bondless = self.write_source("bondless.h5")
        bonded = self.write_source("bonded.h5", bonds=True)
        real, virt = filaments[0].type_part_dict['real'][0], filaments[0].type_part_dict['virt'][0]
        filaments[0].bond_owned_part_pair(real, virt)
        partner_outside = self.write_source("partner_outside.h5", bonds=True)
        real.delete_bond((filaments[0].params['bond_handle'].get_raw_handle(), virt.id))
        filaments[0].bond_owned_part_pair(virt, real)
        owner_outside = self.write_source("owner_outside.h5", bonds=True)

        self.rebuild()
        filaments = self.build_filaments(place=False)
        registered_before = self.n_registered_bonds()
        n_placed = self.n_filaments * self.n_parts

        def restore_bonds(path, src_to_loc=None):
            sim_inst._h5_init.set_init_src(path, place_from=['real'],
                                          src_to_loc={('real', 'real'): []} if src_to_loc is None else src_to_loc)
            return sim_inst._h5_init.set_bonds_from_src(filaments)

        def restore_bonds_onto_a_moved_tree():
            for part in sim_inst.sys.part.all():
                part.pos = part.pos + np.array([1.0, 0., 0.])
            return restore_bonds(bonded)

        # In order: the first two see an unplaced tree, the third places it from the file.
        cases = [   # (label, error, fragment of the raise, call, particles afterwards)
            ("no source declared", RuntimeError, "no source declared",
             lambda: sim_inst._h5_init.set_bonds_from_src(filaments), 0),
            ("bonds without a mapping", ValueError, "src_to_loc is empty",
             lambda: sim_inst.load_from_src(filaments, bonded, None, bonds=True, place_from=['real']), 0),
            ("partner outside the mapping", ValueError, "dangling bond",
             lambda: sim_inst.load_from_src(filaments, partner_outside, {('real', 'real'): []},
                                            bonds=True, place_from=['real']), n_placed),
            ("owner outside the mapping", ValueError, "dangling bond",
             lambda: restore_bonds(owner_outside), n_placed),
            ("an unknown local type", KeyError, "no_such_type",
             lambda: restore_bonds(bonded, {('real', 'no_such_type'): []}), n_placed),
            ("a source type the file does not declare", KeyError, "no_such_type",
             lambda: restore_bonds(bonded, {('no_such_type', 'real'): []}), n_placed),
            ("a bondless source", KeyError, "no bond topology",
             lambda: restore_bonds(bondless), n_placed),
            # the zip is cross-checked against the source positions
            ("objects placed elsewhere", ValueError, "positions differ",
             restore_bonds_onto_a_moved_tree, n_placed),
        ]
        for label, error, fragment, call, n_particles in cases:
            with self.subTest(case=label):
                with self.assertRaises(error) as ctx:
                    call()
                self.assertIn(fragment, str(ctx.exception))
                self.assertEqual(len(sim_inst.sys.part), n_particles)
                self.assertEqual(self.n_live_links(), 0)
                self.assertEqual(self.n_registered_bonds(), registered_before)

    @unittest.skipIf(not all(api_agnostic_feature_check(f) for f in _SM_FEATURES),
                     f"needs espresso features {_SM_FEATURES}")
    def test_a_bare_sample_seeds_a_magnetizable_network(self):
        """A bare Elastomer ('real') seeds a PointDipoleMagnetizable network ('pdm_real') in one `load_from_src`.

        Local ids differ from source ids (each PDM owns a real and a virtual particle), so the
        bond partners are compared through positions, independently of the zip convention."""
        box_E, n_parts, sigma = [4., 4., 4.], 16, 1. / WCA_CONTACT_FACTOR
        elastomer = Elastomer(config=Elastomer.config.specify(
            box_E=box_E, n_parts=n_parts, sigma=sigma, bond_cutoff=2., max_bonds=4,
            espresso_handle=sim_inst.sys, seed=sim_inst.seed))
        sim_inst.store_objects([elastomer])
        sim_inst.set_objects([elastomer])
        # A mixed network leaves the build lattice's z range; source-driven placement must accept that.
        real = elastomer.type_part_dict['real']
        z = np.array([p.pos[2] for p in real])
        low, high = real[int(np.argmin(z))], real[int(np.argmax(z))]
        low.pos = low.pos - [0., 0., 0.2]
        high.pos = high.pos + [0., 0., 0.2]
        elastomer.cure_elastomer()
        written_who_am_i = elastomer.who_am_i
        sim_inst.io_dict['properties'] = list(self.SOURCE_PROPERTIES) + [('fix', 3, np.bool_)]
        path = self.write_source("sample.h5", group_type=Elastomer, bonds=True)
        with h5py.File(path, "r") as h5_file:
            grp = h5_file["particles/Elastomer"]
            types = grp[f"{element_name('type')}/value"][-1]
            is_real = types == Elastomer.part_types['real']
            src_ids = grp[f"{element_name('id')}/value"][-1][is_real]
            src_pos = grp[f"{element_name('pos')}/value"][-1][is_real]
            src_fix = grp["fix/value"][-1][is_real]     # a custom element keeps its own name
            n_links = int(h5_file["connectivity/Elastomer/bonds"].attrs["n_links"])
            params = read_bond_params(h5_file, "Elastomer")
            # read_bonds speaks particle ids; the file stores column indices.
            stored = {(owner, partners[0], float(params[bond_id][1]["k"]),
                       float(params[bond_id][1]["r_0"]))
                      for owner, partners, bond_id in read_bonds(h5_file, "Elastomer")}

        self.rebuild()
        pdm = [PointDipoleMagnetizable(config=PointDipoleMagnetizable.config.specify(
            dipm_sat=1., mag_susc_0=0.1, magnetization_model='langevin', espresso_handle=sim_inst.sys))
            for _ in range(n_parts)]
        elastomer = Elastomer(config=Elastomer.config.specify(
            box_E=box_E, n_parts=n_parts, sigma=sigma, associated_objects=pdm,
            espresso_handle=sim_inst.sys, seed=sim_inst.seed))
        sim_inst.store_objects([elastomer])
        n_added = sim_inst.load_from_src([elastomer], path,
                                         src_to_loc={('real', 'pdm_real'): [('fix', 'fix')]},
                                         bonds=True, place_from=['real'])
        handles = [p for p in elastomer.get_owned_part()[0] if p.type == sim_inst.part_types['pdm_real']]
        ids = [int(p.id) for p in handles]
        pos = np.array([p.pos for p in handles])
        bonds = [(int(p.id), int(partner), bond.k, bond.r_0, bond.r_cut)
                 for p in elastomer.get_owned_part()[0] for bond, partner in p.bonds]

        self.assertEqual(elastomer.who_am_i, written_who_am_i)
        # the source really leaves the strict build range, so the relaxed check was exercised
        self.assertLess(src_pos[:, 2].min(), 1. + elastomer._bead_size / 2)

        # positions and fix, both restored by the single load_from_src call, bit-exact in column order
        self.assertEqual(len(ids), n_parts)
        np.testing.assert_array_equal(pos, src_pos)
        np.testing.assert_array_equal(np.array([p.fix for p in handles]), src_fix)
        self.assertTrue(src_fix.any(), msg="cure pinned no bead; fix copy untested")
        self.assertNotEqual(ids, src_ids.tolist(), msg="identity mapping hides mapping bugs")

        # bonds: count, r_cut as cured (never-break 0; the override is pinned by the r_cut round trip)
        self.assertEqual(n_added, n_links)
        self.assertEqual(len(bonds), n_links)
        self.assertEqual(self.n_registered_bonds(), n_links)
        self.assertEqual({b[4] for b in bonds}, {0.0})

        # partners through positions: local link -> source link, with its own k and r_0
        src_by_pos = {tuple(p): int(sid) for sid, p in zip(src_ids, src_pos)}
        loc_pos = dict(zip(ids, map(tuple, pos)))
        mapped = {(src_by_pos[loc_pos[o]], src_by_pos[loc_pos[p]], float(np.float32(k)), float(np.float32(r0)))
                  for o, p, k, r0, _ in bonds}
        self.assertEqual(mapped, stored)


def write_bond_tables(path, group_name, parts, sys):
    """The smallest file `io/bonds.py` reads back: the group's id column (rows are column indices) and its tables."""
    with h5py.File(path, "w") as handle:
        handle.create_dataset(f"particles/{group_name}/{element_name('id')}/value",
                              data=np.array([[int(part.id) for part in parts]], dtype=np.int32))
        return write_bonds(handle, group_name, particles=parts, sys=sys)


class BondVerificationTest(IOTestCase):
    """`verify_bond_params` compares each stored link with the bond attached to that particle, not by registration id."""

    box_dim = SMALL_BOX
    GROUP = "Probe"

    @staticmethod
    def attach(links):
        """Register and attach `links` = [(owner, partner, k)] in the given order, one handle each."""
        for owner, partner, k in links:
            bond = espressomd.interactions.HarmonicBond(k=k, r_0=1.)
            sim_inst.sys.bonded_inter.add(bond)
            owner.add_bond((bond, partner.id))

    @staticmethod
    def restart(parts):
        """Drop every bond and the whole registry, as a new process would."""
        for part in parts:
            part.delete_all_bonds()
        sim_inst.sys.bonded_inter.clear()

    def test_reordered_registration_is_silent_and_a_drifted_k_warns(self):
        parts = [sim_inst.sys.part.add(pos=[1.0 + i, 1.0, 1.0], type=0) for i in range(4)]
        links = [(parts[0], parts[1], 1.0), (parts[1], parts[2], 2.0), (parts[2], parts[3], 3.0)]
        self.attach(links)
        path = os.path.join(self.tmpdir.name, "bonds.h5")
        write_bond_tables(path, self.GROUP, parts, sim_inst.sys)

        self.restart(parts)
        self.attach(list(reversed(links)))      # same topology, opposite registration order
        with h5py.File(path, "r") as handle, self.assertNoLogs(level="WARNING"):
            verify_bond_params(handle, self.GROUP, sim_inst.sys)

        self.restart(parts)
        self.attach([links[0], (parts[1], parts[2], 99.0), links[2]])     # same link, different stiffness
        with h5py.File(path, "r") as handle, self.assertLogs(level="WARNING") as captured:
            verify_bond_params(handle, self.GROUP, sim_inst.sys)
        text = "\n".join(captured.output)
        self.assertRegex(text, rf"""different parameters \(e\.g\. \[["']particle {parts[1].id} -> """)
        self.assertIn("not registered in sys.bonded_inter", text)     # the registry check (a separate warning)

class SourceFrameSelectionTest(SourceFixture):
    """`frame` is a file index, `step` a stored step value, `time` a stored time: three numbers for one frame.

    Every test gets a three-frame source and a rebuilt, unplaced tree, so a wrong frame is a wrong position."""

    steps = (0, 5, 10)

    def setUp(self):
        super().setUp()
        filaments = self.build_filaments(bonded=True)
        self.n_links = self.n_live_links()
        self.frames, self.directors = [], []

        def advance():
            # A distinct, known displacement and time per frame.
            for part in sim_inst.sys.part.all():
                part.pos = part.pos + np.array([0.5, 0., 0.])
            sim_inst.sys.time = sim_inst.sys.time + 1.0
            self.frames.append(self.real_positions(filaments))
            self.directors.append(self.typed_values(filaments, 'real', 'director'))

        self.path = self.write_source("frames.h5", bonds=True, steps=self.steps,
                                      advance=advance)
        with h5py.File(self.path, "r") as h5_file:
            self.times = h5_file[f"particles/Filament/{element_name('pos')}/time"][...].tolist()

        self.rebuild()
        self.filaments = self.build_filaments(place=False)
        sim_inst._h5_init.set_init_src(self.path, place_from=['real'],
                                       src_to_loc={('real', 'real'): [('pos', 'pos')]})

    def assert_at_frame(self, frame_index):
        for who, positions in self.real_positions(self.filaments).items():
            np.testing.assert_allclose(positions, self.frames[frame_index][who],
                                       rtol=1e-10, atol=1e-10)

    def test_every_selector_form_reads_the_same_frame(self):
        """frame, step and time of one frame agree; no selector is the last frame; frame=-1 and agreeing selectors work."""
        last = len(self.steps) - 1
        cases = [(f"frame {index} by {name}", {name: value}, index)
                 for index, step in enumerate(self.steps)
                 for name, value in (("frame", index), ("step", step), ("time", self.times[index]))]
        cases += [("no selector", {}, last),
                  ("frame=-1", dict(frame=-1), last),
                  ("frame, step and time agreeing", dict(frame=0, step=self.steps[0], time=self.times[0]), 0)]
        for label, selector, index in cases:
            with self.subTest(case=label):
                positions, orientations = sim_inst._h5_init.get_pos_ori_from_src(self.filaments, **selector)
                self.assertEqual(len(positions), len(self.filaments))
                for filament, pos, ori in zip(self.filaments, positions, orientations):
                    np.testing.assert_allclose(pos, self.frames[index][filament.who_am_i], rtol=0, atol=1e-12)
                    np.testing.assert_allclose(ori, self.directors[index][filament.who_am_i], rtol=0, atol=1e-12)

    def test_every_reader_honours_the_selector(self):
        """Objects and bonds from step 5 (not the last frame; bonds cross-check positions), then props at every frame."""
        sim_inst._h5_init.set_objects_from_src(self.filaments, step=self.steps[1])
        self.assert_at_frame(1)
        n_added = sim_inst._h5_init.set_bonds_from_src(self.filaments, step=self.steps[1])
        self.assertEqual(n_added, self.n_links)
        self.assertEqual(self.n_live_links(), self.n_links)
        for index, step in enumerate(self.steps):
            for label, selector in (("frame", dict(frame=index)),
                                    ("step", dict(step=step)),
                                    ("time", dict(time=self.times[index]))):
                with self.subTest(frame=index, selector=label):
                    for part in sim_inst.sys.part.all():
                        part.pos = part.pos + np.array([7.0, 0., 0.])
                    sim_inst._h5_init.set_prop_from_src(self.filaments, **selector)
                    self.assert_at_frame(index)

    def test_a_bad_selector_raises_before_anything_is_placed(self):
        # without the range check a frame index would silently wrap around (frame % n_frames)
        with self.subTest(case="an out-of-range frame"), self.assertRaises(IndexError):
            sim_inst._h5_init.get_pos_ori_from_src(self.filaments, frame=len(self.steps))
        conflicting = {
            "get_pos_ori_from_src": lambda: sim_inst._h5_init.get_pos_ori_from_src(
                self.filaments, frame=0, step=self.steps[1]),
            "set_prop_from_src": lambda: sim_inst._h5_init.set_prop_from_src(
                self.filaments, frame=0, step=self.steps[1]),
            "set_bonds_from_src": lambda: sim_inst._h5_init.set_bonds_from_src(
                self.filaments, frame=0, step=self.steps[1]),
            "set_objects_from_src": lambda: sim_inst._h5_init.set_objects_from_src(
                self.filaments, frame=0, time=self.times[1]),
        }
        for name, call in conflicting.items():
            with self.subTest(method=name):
                with self.assertRaises(ValueError) as ctx:
                    call()
                self.assertIn("select different frames", str(ctx.exception))
        self.assertEqual(len(sim_inst.sys.part), 0)


class BondSerializationTest(IOTestCase):
    """`io/bonds.py` on its own: which table a link lands in, as column indices, and cross-group links."""

    GROUP = "Probe"
    box_dim = SMALL_BOX

    @staticmethod
    def add_particles(n, first_id=5):
        """Ascending, gapped, non-zero-based ids: a column index never equals the id it stands for."""
        return [sim_inst.sys.part.add(id=first_id + 2 * i, pos=[1.0 + i, 1.0, 1.0], type=0)
                for i in range(n)]

    def write(self, parts, name):
        """``(path, n_links)`` of a file holding only this group's id column and tables."""
        path = os.path.join(self.tmpdir.name, name)
        return path, write_bond_tables(path, self.GROUP, parts, sim_inst.sys)

    def test_links_are_stored_per_arity_as_column_indices(self):
        """Rows are column indices in the table of their partner count (`bonds` even when empty); read_bonds maps back."""
        from pressomancy.io.bonds import _bond_id_of
        fene = espressomd.interactions.FeneBond(k=10., r_0=1., d_r_max=2.)
        angle = espressomd.interactions.AngleHarmonic(bend=1.0, phi0=np.pi)
        sim_inst.sys.bonded_inter.add(fene)
        sim_inst.sys.bonded_inter.add(angle)
        mixed, angle_only = self.add_particles(4), self.add_particles(3, first_id=21)
        for a, b in zip(mixed, mixed[1:]):
            a.add_bond((fene, b.id))
        for parts in (mixed, angle_only):
            parts[1].add_bond((angle, parts[0].id, parts[2].id))

        with self.subTest(case="pairs and an angle, gapped ids"):
            path, n_links = self.write(mixed, "mixed.h5")
            self.assertEqual(n_links, 4)
            with h5py.File(path, "r") as handle:
                conn = handle[f"connectivity/{self.GROUP}"]
                np.testing.assert_array_equal(conn["bonds"][...], [[0, 1], [1, 2], [2, 3]])
                np.testing.assert_array_equal(conn["angles"][...], [[1, 0, 2]])
                self.assertEqual(int(conn["bonds"].attrs["n_links"]), 4)
                self.assertEqual(conn["bonds"].attrs["particles_group"], f"/particles/{self.GROUP}")
                np.testing.assert_array_equal(
                    handle[f"pressomancy/{self.GROUP}/bond_params/bond_id"][...],
                    [_bond_id_of(fene)] * 3)
                recovered = sorted((pid, tuple(partners), bond)
                                   for pid, partners, bond in read_bonds(handle, self.GROUP))
            ids = [part.id for part in mixed]
            self.assertEqual(recovered, sorted(
                [(ids[i], (ids[i + 1],), _bond_id_of(fene)) for i in range(3)]
                + [(ids[1], (ids[0], ids[2]), _bond_id_of(angle))]))

        with self.subTest(case="an angle only"):
            path, n_links = self.write(angle_only, "angle.h5")
            self.assertEqual(n_links, 1)
            with h5py.File(path, "r") as handle:
                conn = handle[f"connectivity/{self.GROUP}"]
                np.testing.assert_array_equal(conn["angles"][...], [[1, 0, 2]])
                self.assertEqual(conn["bonds"].shape, (0, 2))
                self.assertEqual(int(conn["bonds"].attrs["n_links"]), 1)   # all tables

    def test_a_partner_outside_the_group_is_not_implemented(self):
        """A column index cannot name a foreign particle: a cross-group bond raises NotImplementedError."""
        inside = self.add_particles(2)
        outside = sim_inst.sys.part.add(pos=[9.0, 9.0, 9.0], type=1)
        fene = espressomd.interactions.FeneBond(k=10., r_0=1., d_r_max=2.)
        sim_inst.sys.bonded_inter.add(fene)
        inside[0].add_bond((fene, outside.id))

        with self.assertRaises(NotImplementedError) as ctx:
            self.write(inside, "dangling.h5")
        self.assertIn(f"partner {outside.id}", str(ctx.exception))


# ``fix`` needs EXTERNAL_FORCES; it is the trajectory extra whose non-float dtype a
# checkpoint must keep and that a restart copies through ``src_to_loc``. Without the
# feature the same checks fall back on ``image_box`` (int32, always stored).
_FIX_PROPERTY = [('fix', 3, np.bool_)] if api_agnostic_feature_check('EXTERNAL_FORCES') else []
_FIX_PAIRS = {('real', 'real'): [('fix', 'fix')]} if _FIX_PROPERTY else None
#: The per-particle state a checkpoint restores in this build, the contract's comparison list.
_STATE_PROPERTIES = [attr for attr, _dim, feature in CHECKPOINT_PROPERTIES
                     if feature is None or api_agnostic_feature_check(feature)]


class PropertiesListTest(SourceFixture):
    """`io_dict['properties']` is validated at inscription and must describe the file exactly to resume."""

    def inscribe(self, path, mode='NEW', **kwargs):
        return sim_inst.inscribe_part_group_to_h5(group_type=[Filament], h5_data_path=path,
                                                  mode=mode, **kwargs)

    def test_a_malformed_properties_list_is_refused_before_the_file_is_touched(self):
        self.build_filaments()
        head = [('id', None, np.int32), ('type', None, np.int16), ('pos', 3, np.float64)]
        cases = {
            'first three out of order': [head[1], head[0], head[2]],
            'pos not floating': head[:2] + [('pos', 3, np.int32)],
            'pos a scalar': head[:2] + [('pos', None, np.float64)],
            'only two entries': head[:2],
            'a duplicated attr': head + [('f', 3, np.float64), ('f', 3, np.float64)],
            'a non-positive dim': head + [('v', 0, np.float64)],
            'an unresolvable dtype': head + [('v', 3, 'nonsense')],
            'not a list': tuple(head),
        }
        path = os.path.join(self.tmpdir.name, "malformed.h5")
        for label, properties in cases.items():
            with self.subTest(case=label):
                sim_inst.io_dict['properties'] = properties
                with self.assertRaises(ValueError):
                    self.inscribe(path)
                self.assertIsNone(sim_inst.io_dict['h5_file'])
                self.assertFalse(os.path.exists(path))

    def test_the_properties_list_must_describe_the_file_to_resume(self):
        """Missing, extra, retyped or reshaped: RuntimeError naming the element; the matching list resumes."""
        self.build_filaments()
        stored_properties = list(self.SOURCE_PROPERTIES) + [('fix', 3, np.bool_)]
        sim_inst.io_dict['properties'] = list(stored_properties)
        path = self.write_source("properties_match.h5", steps=(0, 1))
        cases = {
            'element missing from the list': (
                [entry for entry in stored_properties if entry[0] != 'fix'], r"not in the list: \['fix'\]"),
            'element extra in the list': (
                stored_properties + [('v', 3, np.float64)], r"missing from the file: \['v'\]"),
            'dtype mismatch': (list(self.SOURCE_PROPERTIES) + [('fix', 3, np.int8)], r"'fix' is stored as bool"),
            'dim mismatch': (list(self.SOURCE_PROPERTIES) + [('fix', 1, np.bool_)], r"'fix' is stored as .* \(3,\)"),
        }
        for mode in ('LOAD', 'LOAD_NEW'):
            for label, (properties, message) in cases.items():
                with self.subTest(mode=mode, case=label):
                    sim_inst.io_dict['properties'] = list(properties)
                    with self.assertRaisesRegex(RuntimeError, message):
                        self.inscribe(path, mode)
                    self.reset_io_state()
            sim_inst.io_dict['properties'] = list(stored_properties)
            self.assertEqual(self.inscribe(path, mode), 2, mode)
            self.reset_io_state()


class TruncationTest(SourceFixture):
    """`rewind_to_step`/`force_resize_to_size` cut every stream of two groups (a cut reaching only the first fails)."""

    observable_name = "probe"
    GROUPS = (Filament, Crowder)

    def setUp(self):
        super().setUp()
        self.build_filaments()
        crowders = [Crowder(config=Crowder.config.specify(size=1., espresso_handle=sim_inst.sys))
                    for _ in range(3)]
        sim_inst.store_objects(crowders)
        sim_inst.set_objects(crowders)
        self.observable_value = np.zeros(3, dtype=np.float64)
        self.path = os.path.join(self.tmpdir.name, f"{self.id().rsplit('.', 1)[-1]}.h5")
        self.inscribe('particles', 'NEW')
        self.inscribe('observables', 'NEW')
        self.times = []
        for step in range(3):
            sim_inst.sys.time = 0.5 * step          # exact in the float32 time dataset
            self.times.append(float(sim_inst.sys.time))
            self.observable_value[:] = step
            sim_inst.write_registered_to_h5(step=step)
        self.reset_io_state()

    def inscribe(self, stream, mode, path=None, **kwargs):
        """Inscribe one stream ('particles' or 'observables') of the setUp file, or of ``path``."""
        path = self.path if path is None else path
        if stream == 'particles':
            return sim_inst.inscribe_part_group_to_h5(group_type=list(self.GROUPS), h5_data_path=path,
                                                      mode=mode, **kwargs)
        return sim_inst.inscribe_observable_group_to_h5(
            observable_defs=[(self.observable_name, 3, np.float64, self.observable_value)],
            h5_data_path=path, mode=mode, **kwargs)

    def frames_per_stream(self, h5_file):
        """``{dataset path: frame count}`` of every element and box of every group, and of the observable."""
        counts = {}
        for group in self.GROUPS:
            where = f"particles/{group.__name__}"
            data_grp = h5_file[where]
            counts.update({f"{where}/{name}": member["value"].shape[0]
                           for name, member in data_grp.items() if name != 'box'})
            counts[f"{where}/box/edges"] = data_grp["box/edges/value"].shape[0]
            counts[f"{where}/step"] = data_grp[f"{element_name('pos')}/step"].shape[0]
            counts[f"{where}/time"] = data_grp[f"{element_name('pos')}/time"].shape[0]
        obs_group = h5_file["observables"][self.observable_name]
        counts.update({f"observables/{name}": obs_group[name].shape[0]
                       for name in ("step", "time", "value")})
        return counts

    def test_truncation_cuts_every_stream_and_the_run_appends(self):
        """Both arguments, both resume modes: every stream keeps its first two frames, then takes the next one."""
        n_particles = {group.__name__: sum(len(obj.get_owned_part()[0]) for obj in sim_inst.objects
                                           if isinstance(obj, group))
                       for group in self.GROUPS}
        for truncation in (dict(rewind_to_step=1), dict(force_resize_to_size=2)):
            for mode in ('LOAD_NEW', 'LOAD'):
                with self.subTest(mode=mode, **truncation):
                    path = os.path.join(self.tmpdir.name, f"{mode}_{next(iter(truncation))}.h5")
                    shutil.copy2(self.path, path)
                    sim_inst.sys.time = self.times[1]
                    try:
                        kept = [self.inscribe(stream, mode, path, **truncation)
                                for stream in ('particles', 'observables')]
                        self.assertEqual(kept, [2, 2])
                        h5_file = sim_inst.io_dict['h5_file']
                        for stream, count in self.frames_per_stream(h5_file).items():
                            self.assertEqual(count, 2, msg=stream)
                        for group, count in n_particles.items():
                            np.testing.assert_array_equal(stored_steps(h5_file, group), [0, 1])
                            dataview = H5DataSelector(h5_file, particle_group=group)
                            np.testing.assert_allclose(dataview.time, self.times[:2])
                            np.testing.assert_equal(dataview.common_dims, (2, count))
                        observable = H5ObservableSelector(h5_file, observable_name=self.observable_name)
                        np.testing.assert_array_equal(observable.step, [0, 1])
                        np.testing.assert_allclose(observable.time, self.times[:2])
                        np.testing.assert_allclose(observable.value, [[0., 0., 0.], [1., 1., 1.]])

                        sim_inst.sys.time = 1.5
                        self.observable_value[:] = 7.
                        sim_inst.write_registered_to_h5(step=2)
                        for group in n_particles:
                            np.testing.assert_array_equal(stored_steps(h5_file, group), [0, 1, 2])
                        for stream, count in self.frames_per_stream(h5_file).items():
                            self.assertEqual(count, 3, msg=stream)
                    finally:
                        self.reset_io_state()

    def test_a_refused_truncation_truncates_nothing(self):
        # A numpy integer passes the type check, so its rows reach the step lookup
        # and the size check; a bool is an int subclass and is refused explicitly
        # (True would rewind to step 1 / keep one frame).
        cases = {
            'an unknown step': (KeyError, self.times[1], 'LOAD_NEW', dict(rewind_to_step=99)),
            'an unknown numpy-integer step': (KeyError, self.times[1], 'LOAD_NEW',
                                              dict(rewind_to_step=np.int64(99))),
            'a numpy-integer size beyond the file': (ValueError, self.times[1], 'LOAD_NEW',
                                                     dict(force_resize_to_size=np.int64(4))),
            'a time mismatch': (RuntimeError, self.times[2], 'LOAD_NEW', dict(rewind_to_step=1)),
            'both truncation arguments': (ValueError, self.times[1], 'LOAD_NEW',
                                          dict(rewind_to_step=1, force_resize_to_size=2)),
            'NEW mode': (ValueError, self.times[1], 'NEW', dict(rewind_to_step=1)),
            'a non-integer step': (TypeError, self.times[1], 'LOAD_NEW', dict(rewind_to_step=1.0)),
            'a bool step': (TypeError, self.times[1], 'LOAD_NEW', dict(rewind_to_step=True)),
            'a bool size': (TypeError, self.times[1], 'LOAD_NEW', dict(force_resize_to_size=True)),
        }
        for label, (error, live_time, mode, kwargs) in cases.items():
            for stream in ('particles', 'observables'):
                with self.subTest(case=label, stream=stream):
                    sim_inst.sys.time = live_time
                    try:
                        with self.assertRaises(error):
                            self.inscribe(stream, mode, **kwargs)
                    finally:
                        self.reset_io_state()     # a row that failed must not hand on its open file
        with h5py.File(self.path, "r") as h5_file:
            for group in self.GROUPS:
                np.testing.assert_array_equal(stored_steps(h5_file, group.__name__), [0, 1, 2])
            for stream, count in self.frames_per_stream(h5_file).items():
                self.assertEqual(count, 3, msg=stream)

    def test_a_repeated_step_points_at_rewind(self):
        """A step that does not strictly increase is a RuntimeError on both streams, pointing at rewind_to_step."""
        sim_inst.sys.time = self.times[2]
        self.inscribe('particles', 'LOAD_NEW')
        self.inscribe('observables', 'LOAD_NEW')
        for stream, call in (('particles', lambda: sim_inst.write_part_group_to_h5(step=2)),
                             ('observables', lambda: sim_inst.write_observable_group_to_h5(step=1))):
            with self.subTest(stream=stream), self.assertRaisesRegex(RuntimeError, "rewind_to_step"):
                call()
        for group in self.GROUPS:
            np.testing.assert_array_equal(stored_steps(sim_inst.io_dict['h5_file'], group.__name__), [0, 1, 2])


class CheckpointWriteTest(SourceFixture):
    """`write_checkpoint` puts one verified float64 frame beside the open trajectory, or leaves a `.tmp`."""

    thermostat_seed = 7

    def setUp(self):
        super().setUp()
        self.build_filaments(bonded=True)
        sim_inst.io_dict['properties'] = list(self.SOURCE_PROPERTIES) + _FIX_PROPERTY
        if _FIX_PROPERTY:
            sorted(sim_inst.sys.part.all(), key=lambda part: part.id)[0].fix = [True, False, False]
        sim_inst.io_dict['bonds'] = True
        self.traj = os.path.join(self.tmpdir.name, f"{self.id().rsplit('.', 1)[-1]}_traj.h5")
        self.ckpt = os.path.join(self.tmpdir.name, f"{self.id().rsplit('.', 1)[-1]}_ckpt.h5")
        sim_inst.inscribe_part_group_to_h5(group_type=[Filament], h5_data_path=self.traj, mode='NEW')
        sim_inst.sys.thermostat.set_langevin(kT=1., gamma=1., seed=self.thermostat_seed)
        sim_inst.write_part_group_to_h5(step=0)
        sim_inst.sys.integrator.run(5)
        sim_inst.write_part_group_to_h5(step=5)

    def test_a_checkpoint_holds_one_float64_frame_of_the_restorable_state(self):
        self.assertEqual(sim_inst.write_checkpoint([Filament], self.ckpt, step=5), 5)
        self.assertFalse(os.path.exists(self.ckpt + ".tmp"))
        expected = checkpoint_properties(sim_inst.io_dict['properties'])
        with h5py.File(self.ckpt, "r") as h5_file:
            data_grp = h5_file["particles/Filament"]
            np.testing.assert_array_equal(stored_steps(h5_file, "Filament"), [5])
            self.assertEqual({name for name in data_grp if name != 'box'},
                             {element_name(attr) for attr, _dim, _dtype in expected})
            # v and f are never feature-gated: their standard H5MD names, spelled out, since
            # every other check here goes through element_name() and cannot see a wrong table
            self.assertLessEqual({"position", "velocity", "force"}, set(data_grp))
            for attr, dim, dtype in expected:
                value = data_grp[f"{element_name(attr)}/value"]
                self.assertEqual(value.dtype, np.dtype(dtype), msg=attr)
                self.assertEqual(value.shape[0], 1, msg=attr)
                self.assertEqual(tuple(value.shape[2:]), () if dim is None else (dim,), msg=attr)
                if np.issubdtype(value.dtype, np.floating):
                    self.assertEqual(value.dtype, np.float64, msg=attr)
            # the trajectory's non-float extras (image_box, fix) keep their dtype
            for attr, _dim, dtype in sim_inst.io_dict['properties']:
                if not np.issubdtype(np.dtype(dtype), np.floating):
                    self.assertEqual(data_grp[f"{element_name(attr)}/value"].dtype, np.dtype(dtype), msg=attr)
            # feature-gated state is there exactly when this build has the feature
            for attr, _dim, feature in CHECKPOINT_PROPERTIES:
                self.assertEqual(element_name(attr) in data_grp,
                                 feature is None or api_agnostic_feature_check(feature), msg=attr)
            attrs = h5_file["pressomancy/checkpoint"].attrs
            self.assertEqual(attrs['time'].dtype, np.float64)
            self.assertEqual(float(attrs['time']), float(sim_inst.sys.time))
            self.assertEqual(int(attrs['step']), 5)
            self.assertEqual(int(attrs['langevin_philox_counter']),
                             int(sim_inst.sys.thermostat.langevin.philox_counter))
            self.assertEqual(int(h5_file["connectivity/Filament/bonds"].attrs['n_links']),
                             sim_inst.io_dict['bond_links']['Filament'])
        # the throwaway writer left the open trajectory alone
        sim_inst.sys.integrator.run(5)
        sim_inst.write_part_group_to_h5(step=10)
        np.testing.assert_array_equal(stored_steps(sim_inst.io_dict['h5_file'], "Filament"), [0, 5, 10])

    def test_a_refused_checkpoint_leaves_every_file_as_it_was(self):
        """The trajectory's own path is refused; a failed verification keeps its `.tmp`, the old checkpoint intact."""
        with self.assertRaisesRegex(ValueError, "trajectory"):
            sim_inst.write_checkpoint([Filament], self.traj, step=5)
        sim_inst.write_checkpoint([Filament], self.ckpt, step=5)
        with open(self.ckpt, "rb") as handle:
            before = handle.read()
        sim_inst.sys.integrator.run(5)
        with patch.object(write_module, "_check_load_new_columns", side_effect=RuntimeError("forced")):
            with self.assertRaises(RuntimeError):
                sim_inst.write_checkpoint([Filament], self.ckpt, step=10)
        self.assertTrue(os.path.exists(self.ckpt + ".tmp"))
        with open(self.ckpt, "rb") as handle:
            self.assertEqual(handle.read(), before)
        sim_inst.write_part_group_to_h5(step=10)
        np.testing.assert_array_equal(stored_steps(sim_inst.io_dict['h5_file'], "Filament"), [0, 5, 10])


class CheckpointRestartTest(SourceFixture):
    """N steps == N/2 + `write_checkpoint` + rebuild + `restart_from_checkpoint` + `run(N/2, reuse_forces=True)`."""

    n_steps = 40
    thermostat_seed = 11

    def path(self, name):
        return os.path.join(self.tmpdir.name, f"{self.id().rsplit('.', 1)[-1]}_{name}.h5")

    def set_thermostat(self):
        sim_inst.sys.thermostat.set_langevin(kT=1., gamma=1., seed=self.thermostat_seed)

    @staticmethod
    def state():
        """The restorable state of every live particle, in ascending id."""
        parts = sorted(sim_inst.sys.part.all(), key=lambda part: part.id)
        return {attr: np.array([getattr(part, attr) for part in parts]) for attr in _STATE_PROPERTIES}

    def assert_state_matches(self, reference):
        state = self.state()
        for attr in _STATE_PROPERTIES:
            np.testing.assert_allclose(state[attr], reference[attr], rtol=0, atol=1e-10, err_msg=attr)

    def test_a_restart_continues_the_uninterrupted_run_and_its_trajectory(self):
        """The restart restores step, time, Philox counter and extras; the trajectory rewinds; the run matches."""
        half, past = self.n_steps // 2, 10
        traj, ckpt = self.path("traj"), self.path("ckpt")
        properties = list(self.SOURCE_PROPERTIES) + _FIX_PROPERTY     # the trajectory's list

        self.build_filaments(bonded=True)
        sim_inst.io_dict['properties'] = list(properties)
        if _FIX_PROPERTY:
            sorted(sim_inst.sys.part.all(), key=lambda part: part.id)[0].fix = [True, False, False]
        sim_inst.io_dict['bonds'] = True
        sim_inst.inscribe_part_group_to_h5(group_type=[Filament], h5_data_path=traj, mode='NEW')
        self.set_thermostat()
        sim_inst.write_part_group_to_h5(step=0)
        sim_inst.sys.integrator.run(half)
        sim_inst.write_part_group_to_h5(step=half)
        self.assertEqual(sim_inst.write_checkpoint([Filament], ckpt, step=half), half)
        checkpoint_time = float(sim_inst.sys.time)
        checkpoint_counter = int(sim_inst.sys.thermostat.langevin.philox_counter)
        sim_inst.sys.integrator.run(past)
        sim_inst.write_part_group_to_h5(step=half + past)     # a frame past the checkpoint
        sim_inst.sys.integrator.run(half - past)
        reference, reference_time = self.state(), float(sim_inst.sys.time)
        reference_counter = int(sim_inst.sys.thermostat.langevin.philox_counter)
        self.reset_io_state()

        self.rebuild()
        filaments = self.build_filaments(place=False)
        self.set_thermostat()
        step = sim_inst.restart_from_checkpoint(filaments, ckpt, src_to_loc=_FIX_PAIRS,
                                                bonds=True, place_from=['real'])
        self.assertEqual(step, half)
        self.assertEqual(float(sim_inst.sys.time), checkpoint_time)
        self.assertEqual(int(sim_inst.sys.thermostat.langevin.philox_counter), checkpoint_counter)
        if _FIX_PROPERTY:
            self.assertEqual(list(sorted(sim_inst.sys.part.all(), key=lambda p: p.id)[0].fix),
                             [True, False, False])

        # rebuild() re-pinned SOURCE_PROPERTIES only; resuming needs the trajectory's own list
        sim_inst.io_dict['properties'] = list(properties)
        sim_inst.io_dict['bonds'] = True
        kept = sim_inst.inscribe_part_group_to_h5(group_type=[Filament], h5_data_path=traj,
                                                  mode='LOAD_NEW', rewind_to_step=step)
        self.assertEqual(kept, 2)       # the frame past the checkpoint is dropped
        h5_file = sim_inst.io_dict['h5_file']
        np.testing.assert_array_equal(stored_steps(h5_file, "Filament"), [0, half])

        sim_inst.sys.integrator.run(half, reuse_forces=True)
        sim_inst.write_part_group_to_h5(step=2 * half)
        self.assert_state_matches(reference)
        self.assertEqual(int(sim_inst.sys.thermostat.langevin.philox_counter), reference_counter)
        self.assertAlmostEqual(float(sim_inst.sys.time), reference_time, places=9)
        np.testing.assert_array_equal(stored_steps(h5_file, "Filament"), [0, half, 2 * half])
        dataview = H5DataSelector(h5_file, particle_group="Filament")
        np.testing.assert_allclose(dataview.timestep[-1].pos,
                                   [part.pos for part in sorted(sim_inst.sys.part.all(), key=lambda p: p.id)])

    def test_restart_refuses_what_it_cannot_continue(self):
        """An inactive thermostat, a trajectory, and three tampered checkpoints: refused before anything is placed."""
        traj, ckpt = self.path("traj"), self.path("ckpt")
        self.build_filaments(bonded=True)
        sim_inst.io_dict['bonds'] = True
        sim_inst.inscribe_part_group_to_h5(group_type=[Filament], h5_data_path=traj, mode='NEW')
        self.set_thermostat()
        sim_inst.write_part_group_to_h5(step=0)
        sim_inst.write_checkpoint([Filament], ckpt, step=0)
        self.reset_io_state()
        tampered = {   # label: (edit of a copy of the checkpoint, fragment of the raise)
            "a missing restorable element": (
                lambda f: f["particles/Filament"].pop(element_name('v')), "fewer features"),
            "a stored type the table does not name": (
                lambda f: f["parameters/pressomancy/part_types"].attrs.pop('real'), "does not name"),
            "the counter of a thermostat this build lacks": (
                lambda f: f["pressomancy/checkpoint"].attrs.create('imaginary_philox_counter', 0),
                "compiled without the imaginary thermostat"),
        }
        cases = [   # (label, thermostat set, path, error, fragment of the raise)
            ("an inactive thermostat", False, ckpt, RuntimeError, "thermostat"),
            ("a trajectory", True, traj, ValueError, "load_from_src")]
        cases += [(label, True, self.edited_copy(ckpt, f"tampered_{index}.h5", edit), RuntimeError, fragment)
                  for index, (label, (edit, fragment)) in enumerate(tampered.items())]

        self.rebuild()
        filaments = self.build_filaments(place=False)
        for label, thermostat, path, error, fragment in cases:
            with self.subTest(case=label):
                if thermostat:
                    self.set_thermostat()
                else:
                    sim_inst.sys.thermostat.turn_off()
                with self.assertRaisesRegex(error, fragment):
                    sim_inst.restart_from_checkpoint(filaments, path, bonds=True, place_from=['real'])
                self.assertEqual(len(sim_inst.sys.part), 0)
