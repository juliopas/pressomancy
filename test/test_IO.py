import numpy as np
import espressomd
from .create_system import sim_inst , BaseTestCase
from pressomancy.simulation import Filament, Quartet, Quadriplex, Crowder, Elastomer, PointDipolePermanent
from pressomancy.helper_functions import BondWrapper
from pressomancy.analysis import H5DataSelector, H5ObservableSelector
import h5py
import tempfile
import os
import logging
import shutil
from unittest.mock import patch
import pressomancy.simulation as simulation_module
import pressomancy.io.h5_writer as h5_writer_module
import warnings

class cestica():
    pass

def capture_particle_snapshot(parts, custom_prop=None):
    snap=[]
    for part in parts:
        new=cestica()
        for prop, _, _ in sim_inst.io_dict["properties"]:
            setattr(new, prop, getattr(part, prop))
        if custom_prop is not None:
            setattr(new, custom_prop, getattr(part, custom_prop))
        snap.append(new)
    return snap

def check_prop_dim_dtype(dataview, ref_parts, prop_shape_dtype, time_slice=-1, expected_types=None):
    prop, shape, _dtype = prop_shape_dtype
    property_data_h5df=getattr(dataview,prop)
    # dtype / shape checks on the stored data, before any squeeze
    assert property_data_h5df.dtype == np.dtype(_dtype), \
        f"'{prop}': stored dtype {property_data_h5df.dtype}, expected {np.dtype(_dtype)}"
    assert property_data_h5df.shape[-1] == shape, \
        f"'{prop}': stored last dim {property_data_h5df.shape[-1]}, expected {shape}"
    property_data=[]
    time_part_slice=np.atleast_2d(ref_parts if time_slice is None else ref_parts[time_slice])
    for snap in time_part_slice:
        if expected_types is not None:
            property_data.append([getattr(part,prop) for part in snap if part.type in expected_types])
        else:
            property_data.append([getattr(part,prop) for part in snap])
    if shape == 1:
        property_data_h5df=np.squeeze(property_data_h5df, axis=-1)
    assert np.allclose(property_data, property_data_h5df, rtol=1e-05, atol=1e-08), \
        f'The vectors differ!, {property_data}, {property_data_h5df}'

def get_and_check_complete_object(dataview, object_grp_name, identity, ref_parts, expected_types, time_slice):

    selection_source = dataview if time_slice is None else dataview.timestep[time_slice]
    selection=selection_source.select_particles_by_object(object_name=object_grp_name, connectivity_value=identity)
    for prop,shape,_dtpye in sim_inst.io_dict['properties']:
        check_prop_dim_dtype(selection, ref_parts, (prop, shape, _dtpye), time_slice=time_slice, expected_types=expected_types)
    for predicate_type in expected_types:
        selection=selection_source.select_particles_by_object(object_name=object_grp_name, connectivity_value=identity,predicate=lambda p:p.type==predicate_type)
        for prop,shape,_dtype in sim_inst.io_dict['properties']:
            check_prop_dim_dtype(selection, ref_parts, (prop, shape, _dtype), time_slice=time_slice, expected_types=[predicate_type])

class CommonH5DataSelectorTests:

    runner_script="repo/project/script.py"
    runner_script_repo ="main@abc1234-dirty"
    library_vers= "main@def5678"
    lib_path="/some/path/"
    author="dungeonwitch"
    email='dungeonwitch@dungeon.com'

    @classmethod
    def setUpClass(cls):
        sim_inst.set_author(cls.author, cls.email)
        super().setUpClass()
        sim_inst.sys.box_l = cls.box_dim
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.h5_filename = os.path.join(cls.tmpdir.name, "testfile.h5")
        cls.written_steps = [10, 20, 30, 40]
        cls.written_times = []
        cls.build_fixture()
        cls.part_snapshots = {group_type.__name__: [] for group_type in cls.group_types}
        if hasattr(cls, "observable_name"):
            cls.observable_value = np.zeros(3, dtype=np.float64)
            cls.observable_values = []
        with patch.object(h5_writer_module, "get_submission_creator_info", return_value=(cls.runner_script, cls.runner_script_repo)), patch.object(h5_writer_module, "get_repo_context", return_value=(cls.lib_path, cls.library_vers)):
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
            sim_inst.write_registered_to_h5(time_step=GLOBAL_COUNTER)
            cls.written_times.append(sim_inst.sys.time)
            for group_type in cls.group_types:
                parts = []
                for obj in sim_inst.objects:
                    if isinstance(obj, group_type):
                        owned_parts, _ = obj.get_owned_part()
                        parts.extend(owned_parts)
                cls.part_snapshots[group_type.__name__].append(capture_particle_snapshot(parts))
        cls.reset_io_state()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmpdir"):
            cls.tmpdir.cleanup()
            cls.tmpdir = None
        cls.reset_io_state()
        BaseTestCase.cleanup()
        assert len(sim_inst.sys.part) == 0
        super().tearDownClass()

    def tearDown(self):
        self.reset_io_state()
        super().tearDown()

    @staticmethod
    def reset_io_state():
        h5_file = sim_inst.io_dict.get("h5_file")
        if h5_file is not None:
            h5_file.flush()
            h5_file.close()
        sim_inst.io_dict["h5_file"] = None
        sim_inst.io_dict["flat_part_view"].clear()
        sim_inst.io_dict["registered_observables"] = {}
        sim_inst.io_dict["registered_group_type"] = None
        sim_inst.io_dict["bonds"] = False
        sim_inst.io_dict["bond_links"] = {}

    @staticmethod
    def check_box_data(dataview, expected_edges, expected_boundary=("periodic", "periodic", "periodic")):
        box = dataview.get_box()
        expected_edges = np.array(expected_edges, dtype=float, copy=True)
        np.testing.assert_equal(box["dimension"], len(expected_edges), err_msg="Box dimension from selector does not match!")
        np.testing.assert_equal(box["boundary"], expected_boundary, err_msg="Box boundary from selector does not match!")
        np.testing.assert_allclose(box["edges"], expected_edges, err_msg="Box edges from selector do not match!")

    def check_version_signing(self, dataview):
        np.testing.assert_array_equal(dataview.metadata["h5md"]["_meta"]["attributes"]["version"], np.array([1, 0], dtype=np.int32))
        self.assertEqual(dataview.metadata["h5md"]["creator"]["_meta"]["attributes"]["name"], self.runner_script)
        self.assertEqual(dataview.metadata["h5md"]["creator"]["_meta"]["attributes"]["version"], self.runner_script_repo)
        self.assertEqual(dataview.metadata["parameters"]["pressomancy"]["_meta"]["attributes"]["version"], self.library_vers)
        expected_part_types = {
            key: int(value)
            for key, value in sim_inst.part_types.items()
            if isinstance(value, (int, np.integer))
        }
        observed_part_types = {
            key: int(value)
            for key, value in dataview.metadata["parameters"]["pressomancy"]["part_types"]["_meta"]["attributes"].items()
        }
        self.assertEqual(observed_part_types, expected_part_types)
        self.assertEqual(dataview.metadata["h5md"]["author"]["_meta"]["attributes"]["name"], self.author)
        self.assertEqual(dataview.metadata["h5md"]["author"]["_meta"]["attributes"]["email"], self.email)

    @staticmethod
    def get_and_check_connectivity_predicate(dataview, object_grp_name, particle_type, control_ids):
        selected_ids = np.array(
            dataview.get_connectivity_values(
                object_grp_name,
                predicate=lambda subset: np.all(subset.timestep[-1].type == particle_type),
            ),
            dtype=int,
        )
        np.testing.assert_array_equal(control_ids, selected_ids, err_msg=f"{object_grp_name} predicate-filtered IDs do not match!")

    def check_expected_metadata(self, dataview, h5_file, particle_group=None):

        def metadata_node(metadata, path):
            node = metadata
            for key in path.split("/"):
                node = node[key]
            return node

        metadata = dataview.metadata
        particle_group = next(iter(h5_file["particles"])) if particle_group is None else particle_group
        self.assertEqual(metadata["_meta"]["type"], "Group")
        self.assertEqual(set(metadata["_meta"]["members"]), set(h5_file.keys()))
        for group_name in ("h5md", "parameters", "particles", "connectivity"):
            self.assertIn(group_name, h5_file)
            self.assertIn(group_name, metadata)
        if hasattr(self, "observable_name"):
            self.assertIn("observables", h5_file)
            self.assertIn("observables", metadata)

        group_paths = [
            "h5md",
            "parameters/pressomancy",
            f"particles/{particle_group}",
            f"connectivity/{particle_group}",
        ]
        if hasattr(self, "observable_name"):
            group_paths.append(f"observables/{self.observable_name}")
        dataset_paths = [f"particles/{particle_group}/id/value",
                         f"particles/{particle_group}/pos/value",
                         f"particles/{particle_group}/box/edges",
                         *(f"connectivity/{particle_group}/{dataset_name}" for dataset_name in h5_file[f"connectivity/{particle_group}"]),]
        if hasattr(self, "observable_name"):
                dataset_paths.extend([f"observables/{self.observable_name}/step", f"observables/{self.observable_name}/time", f"observables/{self.observable_name}/value"])

        for group_path in group_paths:
            h5_group = h5_file[group_path]
            meta_node = metadata_node(metadata, group_path)
            self.assertEqual(meta_node["_meta"]["type"], "Group")
            self.assertEqual(set(meta_node["_meta"]["members"]), set(h5_group.keys()))
            actual_attrs = {key: h5_group.attrs[key] for key in h5_group.attrs}
            observed_attrs = meta_node["_meta"]["attributes"]
            self.assertEqual(set(observed_attrs), set(actual_attrs))
            for key, value in actual_attrs.items():
                np.testing.assert_equal(observed_attrs[key], value)

        for dataset_path in dataset_paths:
            h5_dataset = h5_file[dataset_path]
            meta_node = metadata_node(metadata, dataset_path)
            self.assertEqual(meta_node["type"], "Dataset")
            self.assertEqual(meta_node["shape"], h5_dataset.shape)
            self.assertEqual(meta_node["dtype"], str(h5_dataset.dtype))
            actual_attrs = {key: h5_dataset.attrs[key] for key in h5_dataset.attrs}
            observed_attrs = meta_node["attributes"]
            self.assertEqual(set(observed_attrs), set(actual_attrs))
            for key, value in actual_attrs.items():
                np.testing.assert_equal(observed_attrs[key], value)

    def test_observables(self):
        if not hasattr(self, "observable_name"):
            return
        with h5py.File(self.h5_filename, "r") as h5_file:
            selector = H5ObservableSelector(h5_file, observable_name=self.observable_name)
            np.testing.assert_equal(len(selector.timestep), len(self.written_steps), err_msg="Observable selector length is incorrect!")
            np.testing.assert_array_equal(selector.step, self.written_steps, err_msg="Observable frame counters are incorrect!")
            np.testing.assert_allclose(selector.time, self.written_times, err_msg="Observable times do not match fixture write times.")
            np.testing.assert_allclose(selector.value, np.array(self.observable_values), err_msg="Observable values do not match fixture payloads.")
            sliced = selector.timestep[0:2]
            np.testing.assert_equal(len(sliced.timestep), 2, err_msg="Observable timestep slicing did not preserve frame count.")
            np.testing.assert_array_equal(sliced.step, self.written_steps[0:2], err_msg="Observable timestep slicing did not preserve steps.")
            np.testing.assert_allclose(sliced.time, self.written_times[0:2], err_msg="Observable timestep slicing did not preserve times.")
            np.testing.assert_allclose(sliced.value, np.array(self.observable_values[0:2]), err_msg="Observable timestep slicing did not preserve values.")
            frames = [frame for frame in selector.timestep]
            np.testing.assert_equal(len(frames), len(self.written_steps), err_msg="Observable timestep iteration did not yield every frame.")
            np.testing.assert_array_equal([frame.step for frame in frames], self.written_steps, err_msg="Observable timestep iteration did not preserve steps.")

    def test_metadata(self):
        with h5py.File(self.h5_filename, "r") as h5_file:
            for group_type in self.group_types:
                particle_group = group_type.__name__
                expected_particles = 0
                for obj in sim_inst.objects:
                    if isinstance(obj, group_type):
                        parts, _ = obj.get_owned_part()
                        expected_particles += len(parts)
                dataview = H5DataSelector(h5_file, particle_group=particle_group)
                self.check_box_data(dataview, self.box_dim)
                self.check_version_signing(dataview)
                self.check_expected_metadata(dataview, h5_file, particle_group=particle_group)
                expected_timesteps = len(self.written_steps)
                np.testing.assert_equal(
                    dataview.common_dims,
                    (expected_timesteps, expected_particles),
                    err_msg=f"{particle_group} selector common dimensions do not match!",
                )
                np.testing.assert_equal(len(dataview.timestep), expected_timesteps, err_msg=f"{particle_group} timestep length does not match!")
                np.testing.assert_equal(len(dataview.particles), expected_particles, err_msg=f"{particle_group} particle count does not match!")

    def test_trigger_exceptions_smoke(self):
        with h5py.File(self.h5_filename, "r") as h5_file:
            dataview = H5DataSelector(
                h5_file, particle_group=self.group_types[0].__name__
                )
            tests = [
                (lambda: H5DataSelector(h5_file, particle_group="DangerNoodle"), ValueError),
                (lambda: dataview[-1], TypeError),
                (lambda: iter(dataview), TypeError),
                (lambda: len(dataview), TypeError),
            ]
            if hasattr(self, "observable_name"):
                observable_selector = H5ObservableSelector(h5_file, observable_name=self.observable_name)
                tests.extend([
                    (lambda: H5ObservableSelector(h5_file, observable_name="missing_observable"), ValueError),
                    (lambda: observable_selector[0], TypeError),
                    (lambda: iter(observable_selector), TypeError),
                    (lambda: len(observable_selector), TypeError),
                ])
            for fn, exc in tests:
                with self.assertRaises(exc):
                    fn()

    def test_mk_src_file(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            dst_filename = os.path.join(tmpdirname, "dst.h5")
            sim_inst.mk_src_file(self.h5_filename, dst_filename)

            with h5py.File(dst_filename, "r") as h5_file:
                for group_type in self.group_types:
                    particle_group = group_type.__name__
                    dataview = H5DataSelector(h5_file, particle_group=particle_group)
                    self.check_box_data(dataview, self.box_dim)
                    self.check_version_signing(dataview)
                    self.check_expected_metadata(dataview, h5_file, particle_group=particle_group)
                    np.testing.assert_array_equal(dataview.step, np.array([self.written_steps[-1]], dtype=np.int32))
                    check_prop_dim_dtype(dataview, [self.part_snapshots[particle_group][-1]], ("pos", 3, np.float64), time_slice=0)

    def test_load_modes(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            for mode in ("LOAD_NEW", "LOAD"):
                h5_filename = os.path.join(tmpdirname, f"{mode}.h5")
                shutil.copy2(self.h5_filename, h5_filename)
                saved_part_types = dict(sim_inst.part_types)
                try:
                    if mode == "LOAD_NEW":
                        sim_inst.part_types.clear()
                    GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(
                        group_type=self.group_types,
                        h5_data_path=h5_filename,
                        mode=mode,
                    )
                    self.assertEqual(GLOBAL_COUNTER, len(self.written_steps))
                    if mode == "LOAD_NEW":
                        self.assertEqual(dict(sim_inst.part_types), saved_part_types)
                    for group_type in self.group_types:
                        group_name = group_type.__name__
                        expected_ids = []
                        for obj in sim_inst.objects:
                            if isinstance(obj, group_type):
                                parts, _ = obj.get_owned_part()
                                expected_ids.extend(part.id for part in parts)
                        reconstructed_ids = [part.id for part in sim_inst.io_dict['flat_part_view'][group_name]]
                        np.testing.assert_array_equal(
                            reconstructed_ids,
                            expected_ids,
                            err_msg=f"{mode} flat_part_view for {group_name} does not match live object order.",
                        )
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
                    sim_inst.part_types.clear()
                    sim_inst.part_types.update(saved_part_types)
                    self.reset_io_state()

    def test_load_modes_force_resize(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            for mode in ("LOAD_NEW", "LOAD"):
                h5_filename = os.path.join(tmpdirname, f"{mode}_resized.h5")
                shutil.copy2(self.h5_filename, h5_filename)
                GLOBAL_COUNTER = sim_inst.inscribe_part_group_to_h5(
                    group_type=self.group_types,
                    h5_data_path=h5_filename,
                    mode=mode,
                    force_resize_to_size=2,
                )
                self.assertEqual(GLOBAL_COUNTER, 2)
                for group_type in self.group_types:
                    particle_group = group_type.__name__
                    dataview = H5DataSelector(sim_inst.io_dict['h5_file'], particle_group=particle_group)
                    np.testing.assert_array_equal(dataview.step,
                                                  self.written_steps[:2])
                    np.testing.assert_allclose(dataview.time,
                                               self.written_times[:2])
                    np.testing.assert_equal(dataview.common_dims, (2, len(self.part_snapshots[particle_group][0])))

                if hasattr(self, "observable_name"):
                    observable_counter = sim_inst.inscribe_observable_group_to_h5(
                        observable_defs=[(self.observable_name, self.observable_value.shape, self.observable_value.dtype, self.observable_value)],
                        h5_data_path=h5_filename,
                        mode=mode,
                        force_resize_to_size=2,
                    )
                    selector = H5ObservableSelector(sim_inst.io_dict['h5_file'], observable_name=self.observable_name)
                    self.assertEqual(observable_counter, 2)
                    np.testing.assert_array_equal(selector.step, self.written_steps[:2])
                    np.testing.assert_allclose(selector.time, self.written_times[:2])
                    np.testing.assert_allclose(selector.value,
                    np.array(self.observable_values[:2]))
                self.reset_io_state()

    def test_select_particles_by_object(self):
        with h5py.File(self.h5_filename, "r") as h5_file:
            for group_type in self.group_types:
                object_name = group_type.__name__
                dataview = H5DataSelector(h5_file, particle_group=object_name)
                np.testing.assert_equal(len(dataview.timestep), len(self.written_steps), err_msg=f"{object_name} stored frame count does not match fixture writes.")
                self.assertEqual(dataview.timestep[-1].step, self.written_steps[-1])
                self.assertEqual(dataview.timestep[-1].time, self.written_times[-1])
                objects = [obj for obj in sim_inst.objects if isinstance(obj, group_type)]
                connectivity_value = np.array([obj.who_am_i for obj in objects], dtype=int)
                expected_types = sorted({
                    int(part.type)
                    for obj in objects
                    for part in obj.get_owned_part()[0]
                })
                ids = dataview.get_connectivity_values(object_name)
                np.testing.assert_array_equal(ids, connectivity_value, err_msg=f"{object_name} connectivity IDs do not match fixture object ids!")
                connected_objects = sim_inst._collect_instances_recursively(objects)
                connected_object_names = sorted({obj.__class__.__name__ for obj in connected_objects})
                for connected_object_name in connected_object_names:
                    connected_class_objects = [obj for obj in connected_objects if obj.__class__.__name__ == connected_object_name]
                    for predicate_type in expected_types:
                        control_ids = [obj.who_am_i for obj in connected_class_objects if all(part.type == predicate_type for part in obj.get_owned_part()[0])]
                        self.get_and_check_connectivity_predicate(dataview, connected_object_name, predicate_type, control_ids)
                for time_slice in [None, -1, 0, slice(0, 2, 1)]:
                    get_and_check_complete_object(dataview, object_name, connectivity_value, self.part_snapshots[object_name], expected_types, time_slice=time_slice)
                for predicate_type in expected_types:
                    selection = dataview.select_particles_by_object(
                        object_name=object_name,
                        connectivity_value=connectivity_value,
                        predicate=lambda subset, predicate_type=predicate_type: subset.timestep[-1].type == predicate_type,
                    )
                    np.testing.assert_equal(len(selection.timestep), len(dataview.timestep), err_msg="Predicate selection changed timestep context!")
                    expected_ids = [[part.id for part in snap if part.type == predicate_type] for snap in self.part_snapshots[object_name]]
                    np.testing.assert_allclose(selection.id.squeeze(axis=-1), expected_ids)

    def test_object_relations(self):
        with h5py.File(self.h5_filename, "r") as h5_file:
            for group_type in self.group_types:
                particle_group = group_type.__name__
                dataview = H5DataSelector(h5_file, particle_group=particle_group)
                parents = sim_inst._collect_instances_recursively(
                    [obj for obj in sim_inst.objects if isinstance(obj, group_type)]
                    )

                for parent in parents:
                    if not getattr(parent, "associated_objects", None):
                        continue
                    parent_key = parent.__class__.__name__
                    child_key = parent.associated_objects[0].__class__.__name__
                    child_ids = [child.who_am_i for child in parent.associated_objects]
                    np.testing.assert_array_equal(
                        dataview.get_child_ids(parent_key, child_key, parent.who_am_i),
                        child_ids,
                        err_msg=f"{parent_key}_to_{child_key} child IDs do not match for parent {parent.who_am_i}!",
                    )
                    for child in parent.associated_objects:
                        expected_parent_ids = [
                            obj.who_am_i
                            for obj in sim_inst.objects
                            if child in (getattr(obj, "associated_objects", None) or [])
                        ]
                        np.testing.assert_array_equal(
                            dataview.get_parent_ids(parent_key, child_key, child.who_am_i),
                            expected_parent_ids,
                            err_msg=f"{parent_key}_to_{child_key} parent IDs do not match for child {child.who_am_i}!",
                        )

class ElastomerFixture(CommonH5DataSelectorTests, BaseTestCase):
    box_dim = [5,5,20]
    layer_height = 4
    n_part = 20
    observable_name = "magnetic_dipole_moment"

    @classmethod
    def build_fixture(cls):
        sim_inst.sys.box_l = cls.box_dim
        conf_point_dipole = PointDipolePermanent.config.specify(dipm=1., espresso_handle=sim_inst.sys)
        point_dipoles = [PointDipolePermanent(config=conf_point_dipole) for _ in range(cls.n_part)]
        config_E = Elastomer.config.specify(
            layer_height=cls.layer_height, n_parts=cls.n_part, associated_objects=point_dipoles, espresso_handle=sim_inst.sys, seed=sim_inst.seed)
        elastomer=Elastomer(config=config_E)
        sim_inst.store_objects([elastomer])
        sim_inst.set_objects([elastomer])
        cls.group_types = [Elastomer, PointDipolePermanent]

class FilamentFixture(CommonH5DataSelectorTests, BaseTestCase):

    N_avog = 6.02214076e23
    sigma = 1.
    rho_si = 0.6*N_avog
    no_obj=30
    N = no_obj/3
    vol = N/rho_si
    box_l = pow(vol, 1/3)
    _box_l = box_l/0.4e-09
    box_dim = _box_l*np.ones(3)
    _rho = N/pow(_box_l, 3)

    sheets_per_quad = 3
    part_per_filament = 2
    no_crowders=10
    part_per_ligand=2

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

        crowder_configuration=Crowder.config.specify(sigma=1., size=1., espresso_handle=sim_inst.sys)
        crowders = [Crowder(config=crowder_configuration) for _ in range(cls.no_crowders)]
        sim_inst.store_objects(crowders)
        sim_inst.set_objects(crowders)

        cls.group_types = [Filament, Crowder]

    def test_crowder_missing_relation_smoke(self):
        with h5py.File(self.h5_filename, "r") as h5_file:
            crowder = next(obj for obj in sim_inst.objects if isinstance(obj, Crowder))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                missing_children = H5DataSelector(
                    h5_file,
                    particle_group="Crowder",
                ).get_child_ids(
                    "Crowder",
                    "Quadriplex",
                    crowder.who_am_i,
                )
            self.assertIsNone(missing_children)
            self.assertGreaterEqual(len(caught), 1)


class BondTopologyIOTest(BaseTestCase):

    n_parts = 5
    n_filaments = 3

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.h5_filename = os.path.join(cls.tmpdir.name, "bond_topology.h5")

        bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=3.))
        configs = [Filament.config.specify(sigma=2., size=2. * cls.n_parts,
                                           n_parts=cls.n_parts,
                                           espresso_handle=sim_inst.sys,
                                           bond_handle=bond)
                   for _ in range(cls.n_filaments)]
        cls.filaments = [Filament(config=cfg) for cfg in configs]
        sim_inst.store_objects(cls.filaments)
        sim_inst.set_objects(cls.filaments)
        for filament in cls.filaments:
            filament.bond_center_to_center(type_name='real')
        cls.live_links = sum(len(part.bonds) for part in sim_inst.sys.part.all())

    @classmethod
    def tearDownClass(cls):
        cls.close_h5()
        if getattr(cls, "tmpdir", None) is not None:
            cls.tmpdir.cleanup()
            cls.tmpdir = None
        sim_inst.io_dict["bonds"] = False
        sim_inst.io_dict["bond_links"] = {}
        sim_inst.io_dict["flat_part_view"].clear()
        sim_inst.io_dict["registered_group_type"] = None
        BaseTestCase.cleanup()
        super().tearDownClass()

    @staticmethod
    def close_h5():
        handle = sim_inst.io_dict.get("h5_file")
        if handle is not None:
            handle.flush()
            handle.close()
        sim_inst.io_dict["h5_file"] = None

    @staticmethod
    def reopen(mode):
        """Drop the in-memory view the way a fresh process would, then inscribe."""
        sim_inst.io_dict["flat_part_view"].clear()
        sim_inst.io_dict["bond_links"] = {}
        return sim_inst.inscribe_part_group_to_h5(
            group_type=[Filament], h5_data_path=BondTopologyIOTest.h5_filename, mode=mode)

    def test_bond_topology_round_trip(self):
        self.assertGreater(self.live_links, 0, msg="fixture built no bonds to write")
        sim_inst.io_dict["bonds"] = True

        # --- NEW -----------------------------------------------------------
        counter = self.reopen('NEW')
        self.assertEqual(counter, 0)
        self.assertEqual(sim_inst.io_dict["bond_links"]["Filament"], self.live_links)
        for step in range(3):
            sim_inst.write_part_group_to_h5(step=step)
        self.close_h5()

        with h5py.File(self.h5_filename, "r") as h5_file:
            bonds_grp = h5_file["connectivity/Filament/bonds"]
            self.assertIn("links", bonds_grp)
            self.assertIn("offsets", bonds_grp)
            self.assertIn("particle_ids", bonds_grp)
            self.assertEqual(int(bonds_grp.attrs["n_links"]), self.live_links)
            self.assertTrue(bool(bonds_grp.attrs["static_topology"]))
            self.assertEqual(h5_file["particles/Filament/pos/value"].shape[0], 3)

        # --- LOAD ----------------------------------------------------------
        self.assertEqual(self.reopen('LOAD'), 3)
        self.assertEqual(sim_inst.io_dict["bond_links"]["Filament"], self.live_links)
        sim_inst.write_part_group_to_h5(step=3)
        self.close_h5()

        # --- LOAD_NEW ------------------------------------------------------
        self.assertEqual(self.reopen('LOAD_NEW'), 4)
        self.assertEqual(sim_inst.io_dict["bond_links"]["Filament"], self.live_links)
        self.close_h5()

    def test_topology_change_after_inscription_is_caught(self):
        """The guard only works because bond_links is populated at inscription."""
        sim_inst.io_dict["bonds"] = True
        self.reopen('NEW')
        sim_inst.write_part_group_to_h5(step=0)

        # Add a bond the file knows nothing about; topology has no time axis.
        handles = self.filaments[0].type_part_dict['real']
        handles[0].add_bond((self.filaments[0].params['bond_handle'].get_raw_handle(),
                             handles[-1].id))
        try:
            with self.assertRaises(RuntimeError) as ctx:
                sim_inst.write_part_group_to_h5(step=1)
            self.assertIn("changed after inscription", str(ctx.exception))
        finally:
            handles[0].delete_bond((self.filaments[0].params['bond_handle'].get_raw_handle(),
                                    handles[-1].id))
            self.close_h5()


class BulkFrameReadTest(BaseTestCase):
    """A ParticleSlice built from a non-monotonic id list is not self-consistent
    about row order -- espresso routes `type`/`q`/`pos`/`pos_folded` through an
    optimised path and everything else through a per-id loop, and those two
    disagreed before espresso commit 45376706e. H5Writer sidesteps it by always
    slicing on sorted ids and inverting the permutation. This pins that.
    """

    @classmethod
    def tearDownClass(cls):
        BaseTestCase.cleanup()
        super().tearDownClass()

    def test_bulk_read_matches_per_particle_loop_for_shuffled_ids(self):
        rng = np.random.default_rng(4242)
        for _ in range(120):
            part = sim_inst.sys.part.add(pos=rng.random(3) * 20.0, type=int(rng.integers(0, 4)))
            part.dip = rng.random(3) + 0.5
        sim_inst.sys.integrator.run(5)

        ids = [int(x) for x in rng.permutation([p.id for p in sim_inst.sys.part.all()])]
        handles = [sim_inst.sys.part.by_id(i) for i in ids]
        group = "ShuffledProbe"
        sim_inst.io_dict['flat_part_view'][group] = handles
        try:
            writer = sim_inst._h5_writer
            for prop, dim, dtype in sim_inst.io_dict['properties']:
                expected = np.array(
                    [np.atleast_1d(getattr(h, prop)) for h in handles], dtype=dtype)
                got = writer._read_frame(group, prop, dim, dtype)
                with self.subTest(prop=prop):
                    self.assertEqual(got.shape, expected.shape)
                    np.testing.assert_array_equal(got, expected)
        finally:
            sim_inst.io_dict['flat_part_view'].pop(group, None)
            sim_inst._h5_writer._slice_cache.pop(group, None)

    def test_duplicate_ids_are_rejected(self):
        """The espresso-side reorder is keyed on id, so duplicates must not pass."""
        part = sim_inst.sys.part.add(pos=[1.0, 1.0, 1.0], type=0)
        group = "DupProbe"
        sim_inst.io_dict['flat_part_view'][group] = [part, part]
        try:
            with self.assertRaises(ValueError) as ctx:
                sim_inst._h5_writer._read_frame(group, 'pos', 3, np.float64)
            self.assertIn("duplicate particle ids", str(ctx.exception))
        finally:
            sim_inst.io_dict['flat_part_view'].pop(group, None)
            sim_inst._h5_writer._slice_cache.pop(group, None)


class SourceSeedingTest(BaseTestCase):
    """Only `set_prop_from_src` with an identity type map and pos->pos was covered
    before, by a bare assert inside samples/poly_BRACO.py. `get_pos_ori_from_src`
    (the `set_objects(mode='INIT_SRC')` path) had no exercise at all.

    Note these tests read back onto the *same* objects that wrote the file.
    Selection is keyed on `who_am_i`, which the metaclass allocates monotonically
    and never resets, so a freshly constructed object can never carry a source
    id within one process -- INIT_SRC across a real restart works because
    numInstances starts from zero in a new interpreter.
    """

    n_parts = 4
    n_filaments = 2

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmpdir = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.close_open_file()
        if getattr(cls, "tmpdir", None) is not None:
            cls.tmpdir.cleanup()
            cls.tmpdir = None
        BaseTestCase.cleanup()
        super().tearDownClass()

    @staticmethod
    def close_open_file():
        handle = sim_inst.io_dict.get("h5_file")
        if handle is not None:
            handle.flush()
            handle.close()
        sim_inst.io_dict["h5_file"] = None
        sim_inst.io_dict["flat_part_view"].clear()
        sim_inst.io_dict["registered_group_type"] = None

    def tearDown(self):
        self.close_open_file()
        BaseTestCase.cleanup()
        super().tearDown()

    def build_filaments(self, with_dipoles=False, with_anchors=False):
        bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=3.))
        configs = [Filament.config.specify(
            sigma=2., size=2. * self.n_parts, n_parts=self.n_parts,
            espresso_handle=sim_inst.sys, bond_handle=bond)
            for _ in range(self.n_filaments)]
        filaments = [Filament(config=cfg) for cfg in configs]
        sim_inst.store_objects(filaments)
        sim_inst.set_objects(filaments)
        if with_anchors:
            for filament in filaments:
                filament.add_anchors(type_name='real')
        if with_dipoles:
            for index, part in enumerate(sim_inst.sys.part.all()):
                part.dip = np.array([1.0, 0.5, 0.25]) * (index + 1)
        return filaments

    def write_source(self, filaments, name, properties=None):
        """Write one frame and return its path, optionally with a custom schema."""
        path = os.path.join(self.tmpdir.name, name)
        original = sim_inst.io_dict["properties"]
        if properties is not None:
            sim_inst.io_dict["properties"] = properties
        try:
            sim_inst.inscribe_part_group_to_h5(
                group_type=[Filament], h5_data_path=path, mode="NEW")
            sim_inst.write_part_group_to_h5(step=0)
            sim_inst.io_dict["h5_file"].flush()
        finally:
            if properties is not None:
                sim_inst.io_dict["properties"] = original
        return path

    # -- get_pos_ori_from_src ---------------------------------------------
    def test_get_pos_ori_returns_source_positions_and_directors(self):
        filaments = self.build_filaments()
        path = self.write_source(filaments, "geometry.h5")
        expected = {f.who_am_i: ([p.pos.copy() for p in f.type_part_dict['real']],
                                 [p.director.copy() for p in f.type_part_dict['real']])
                    for f in filaments}
        self.close_open_file()

        sim_inst.set_init_src(path=path, pos_ori_src_type=['real'])
        positions, orientations = sim_inst._get_pos_ori_from_src(filaments)

        self.assertEqual(len(positions), len(filaments))
        for filament, pos, ori in zip(filaments, positions, orientations):
            want_pos, want_ori = expected[filament.who_am_i]
            np.testing.assert_allclose(pos, np.array(want_pos), rtol=0, atol=1e-12)
            np.testing.assert_allclose(ori, np.array(want_ori), rtol=0, atol=1e-12)

    def test_set_objects_init_src_uses_the_source_reader(self):
        filaments = self.build_filaments()
        path = self.write_source(filaments, "routing.h5")
        self.close_open_file()
        sim_inst.set_init_src(path=path, pos_ori_src_type=['real'])

        with patch.object(sim_inst._h5_init, "get_pos_ori_from_src",
                          wraps=sim_inst._h5_init.get_pos_ori_from_src) as reader:
            with patch.object(type(sim_inst.instance), "place_objects") as place:
                sim_inst.set_objects(filaments, mode='INIT_SRC')
            reader.assert_called_once()
            place.assert_called_once()

    def test_orientation_falls_back_to_normalised_dip(self):
        """With no `director` column stored, orientation comes from `dip`."""
        filaments = self.build_filaments(with_dipoles=True)
        no_director = [entry for entry in sim_inst.io_dict["properties"]
                       if entry[0] != "director"]
        path = self.write_source(filaments, "dip_only.h5", properties=no_director)
        expected = {f.who_am_i: [p.dip.copy() for p in f.type_part_dict['real']]
                    for f in filaments}
        self.close_open_file()

        sim_inst.set_init_src(path=path, pos_ori_src_type=['real'])
        _, orientations = sim_inst._get_pos_ori_from_src(filaments)
        for filament, ori in zip(filaments, orientations):
            dips = np.array(expected[filament.who_am_i])
            want = dips / np.linalg.norm(dips, axis=1, keepdims=True)
            np.testing.assert_allclose(ori, want, rtol=0, atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(ori, axis=1), 1.0, atol=1e-12)

    def test_zero_dip_raises_rather_than_producing_nan(self):
        filaments = self.build_filaments()          # dip left at zero
        no_director = [entry for entry in sim_inst.io_dict["properties"]
                       if entry[0] != "director"]
        path = self.write_source(filaments, "zero_dip.h5", properties=no_director)
        self.close_open_file()

        sim_inst.set_init_src(path=path, pos_ori_src_type=['real'])
        with self.assertRaises(ValueError) as ctx:
            sim_inst._get_pos_ori_from_src(filaments)
        self.assertIn("dip moment magnitude is 0", str(ctx.exception))

    # -- set_prop_from_src -------------------------------------------------
    def test_set_prop_from_src_restores_positions(self):
        filaments = self.build_filaments()
        path = self.write_source(filaments, "props.h5")
        written = sim_inst.sys.part.all().pos.copy()
        self.close_open_file()

        # Move everything, then put it back from the file.
        for part in sim_inst.sys.part.all():
            part.pos = part.pos + np.array([3.0, -2.0, 1.5])
        self.assertFalse(np.allclose(sim_inst.sys.part.all().pos, written))

        sim_inst.set_init_src(path=path,
                              type_to_type_map=[('real', 'real')],
                              prop_to_prop_map=[('pos', 'pos')])
        sim_inst.set_prop_from_src(filaments)
        np.testing.assert_allclose(sim_inst.sys.part.all().pos, written,
                                   rtol=1e-10, atol=1e-10)

    def test_non_identity_type_map_touches_only_the_target_type(self):
        """poly_BRACO only ever maps a type onto itself, so this path was untested."""
        filaments = self.build_filaments(with_anchors=True)
        path = self.write_source(filaments, "remap.h5")
        source_real = {f.who_am_i: [p.pos.copy() for p in f.type_part_dict['real']]
                       for f in filaments}
        self.close_open_file()

        virt_before = {p.id: p.pos.copy()
                       for f in filaments for p in f.type_part_dict['virt']}
        # Copy the *real* particles' stored positions onto the *virt* particles.
        sim_inst.set_init_src(path=path,
                              type_to_type_map=[('real', 'virt')],
                              prop_to_prop_map=[('pos', 'pos')])
        sim_inst.set_prop_from_src(filaments)

        for filament in filaments:
            want = source_real[filament.who_am_i]
            got = [p.pos for p in filament.type_part_dict['virt'][:len(want)]]
            np.testing.assert_allclose(np.array(got), np.array(want),
                                       rtol=1e-10, atol=1e-10)
        moved = sum(1 for f in filaments for p in f.type_part_dict['virt']
                    if not np.allclose(p.pos, virt_before[p.id]))
        self.assertGreater(moved, 0, msg="the remap did not touch the target type")

    # -- guards ------------------------------------------------------------
    def test_mismatched_map_lengths_raise(self):
        filaments = self.build_filaments()
        path = self.write_source(filaments, "mismatch.h5")
        self.close_open_file()
        sim_inst.set_init_src(path=path,
                              type_to_type_map=[('real', 'real'), ('real', 'real')],
                              prop_to_prop_map=[('pos', 'pos')])
        # ValueError, not AssertionError: the length guard is caller-facing, and a
        # typed exception distinguishes it from the source-type validation that runs
        # first and also used to raise AssertionError here.
        with self.assertRaises(ValueError) as ctx:
            sim_inst.set_prop_from_src(filaments)
        self.assertIn("same length", str(ctx.exception))

    def test_reading_before_declaring_a_source_raises(self):
        filaments = self.build_filaments()
        self.assertFalse(sim_inst.src_params_set)
        # RuntimeError: using the reader before declaring a source is a state error.
        with self.assertRaises(RuntimeError):
            sim_inst.set_prop_from_src(filaments)
        with self.assertRaises(RuntimeError):
            sim_inst._get_pos_ori_from_src(filaments)


class BondSerializationTest(BaseTestCase):
    """Everything here was previously reached only through H5Writer, so the CSR
    construction and the derived parameter schema were only ever exercised on
    one shape of input: two-body FeneBonds, every partner inside the group.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmpdir = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "tmpdir", None) is not None:
            cls.tmpdir.cleanup()
            cls.tmpdir = None
        BaseTestCase.cleanup()
        super().tearDownClass()

    def tearDown(self):
        BaseTestCase.cleanup()
        super().tearDown()

    @staticmethod
    def add_particles(n):
        return [sim_inst.sys.part.add(pos=[1.0 + i, 1.0, 1.0], type=0) for i in range(n)]

    # -- CSR construction --------------------------------------------------
    def test_csr_invariants_for_two_body_bonds(self):
        from pressomancy.io.bonds import collect_bond_links
        parts = self.add_particles(4)
        fene = espressomd.interactions.FeneBond(k=10., r_0=1., d_r_max=2.)
        sim_inst.sys.bonded_inter.add(fene)
        for a, b in zip(parts, parts[1:]):
            a.add_bond((fene, b.id))

        particle_ids, offsets, links, max_partners = collect_bond_links(parts)

        np.testing.assert_array_equal(particle_ids, [p.id for p in parts])
        self.assertEqual(len(offsets), len(parts) + 1)
        self.assertEqual(offsets[0], 0)
        self.assertTrue(np.all(np.diff(offsets) >= 0), msg="offsets must be monotonic")
        self.assertEqual(int(offsets[-1]), links.shape[0])
        self.assertEqual(int(offsets[-1]), 3)
        self.assertEqual(max_partners, 1)
        self.assertEqual(links.shape[1], 2 + max_partners)
        # Every row: (bond_id, n_partners, partner...)
        for row in links:
            self.assertEqual(int(row[1]), 1)
            self.assertIn(int(row[2]), [p.id for p in parts])

    def test_multi_partner_bond_is_laid_out_and_padded(self):
        """Angle bonds carry two partners; the fixtures elsewhere never do."""
        from pressomancy.io.bonds import collect_bond_links
        parts = self.add_particles(3)
        angle = espressomd.interactions.AngleHarmonic(bend=1.0, phi0=np.pi)
        sim_inst.sys.bonded_inter.add(angle)
        parts[1].add_bond((angle, parts[0].id, parts[2].id))

        _, offsets, links, max_partners = collect_bond_links(parts)
        self.assertEqual(max_partners, 2)
        self.assertEqual(links.shape[1], 4)
        self.assertEqual(int(offsets[-1]), 1)
        row = links[0]
        self.assertEqual(int(row[1]), 2)
        self.assertEqual({int(row[2]), int(row[3])}, {parts[0].id, parts[2].id})

    def test_padding_uses_minus_one_when_partner_counts_differ(self):
        from pressomancy.io.bonds import collect_bond_links
        parts = self.add_particles(3)
        fene = espressomd.interactions.FeneBond(k=10., r_0=1., d_r_max=2.)
        angle = espressomd.interactions.AngleHarmonic(bend=1.0, phi0=np.pi)
        sim_inst.sys.bonded_inter.add(fene)
        sim_inst.sys.bonded_inter.add(angle)
        parts[0].add_bond((fene, parts[1].id))
        parts[1].add_bond((angle, parts[0].id, parts[2].id))

        _, _, links, max_partners = collect_bond_links(parts)
        self.assertEqual(max_partners, 2)
        one_partner = [row for row in links if int(row[1]) == 1]
        self.assertEqual(len(one_partner), 1)
        # The unused partner slot is padded, not left as a stale id.
        self.assertEqual(int(one_partner[0][3]), -1)

    def test_partner_outside_the_group_is_kept_as_a_raw_id(self):
        """The docstring promises a dangling partner stays visible."""
        from pressomancy.io.bonds import collect_bond_links
        inside = self.add_particles(2)
        outside = sim_inst.sys.part.add(pos=[9.0, 9.0, 9.0], type=1)
        fene = espressomd.interactions.FeneBond(k=10., r_0=1., d_r_max=2.)
        sim_inst.sys.bonded_inter.add(fene)
        inside[0].add_bond((fene, outside.id))

        _, _, links, _ = collect_bond_links(inside)
        self.assertEqual(links.shape[0], 1)
        self.assertEqual(int(links[0][2]), outside.id)
        self.assertNotIn(outside.id, [p.id for p in inside])

    # -- parameter schema round trip ---------------------------------------
    def test_bond_params_round_trip(self):
        from pressomancy.io.bonds import (h5_dtype_for, write_bond_params,
                                          read_bond_params, _bond_id_of)
        fene = espressomd.interactions.FeneBond(k=11.5, r_0=1.25, d_r_max=2.5)
        harmonic = espressomd.interactions.HarmonicBond(k=3.75, r_0=0.5)
        sim_inst.sys.bonded_inter.add(fene)
        sim_inst.sys.bonded_inter.add(harmonic)

        dtype = h5_dtype_for(fene)
        self.assertIn("bond_id", dtype.names)
        for name in ("k", "r_0", "d_r_max"):
            self.assertIn(name, dtype.names)

        path = os.path.join(self.tmpdir.name, "params.h5")
        with h5py.File(path, "w") as handle:
            write_bond_params(handle.require_group("bonds"), sim_inst.sys)
        with h5py.File(path, "r") as handle:
            table = read_bond_params(handle["bonds"])

        for original in (fene, harmonic):
            cls, kw = table[_bond_id_of(original)]
            self.assertIs(cls, type(original))
            live = original.get_params()
            for name, value in kw.items():
                self.assertAlmostEqual(float(value), float(live[name]), places=5,
                                       msg=f"{type(original).__name__}.{name}")
            rebuilt = cls(**kw)
            self.assertIsInstance(rebuilt, type(original))

    # -- read_bonds --------------------------------------------------------
    def test_read_bonds_recovers_topology(self):
        from pressomancy.io.bonds import write_bonds, read_bonds, _bond_id_of
        parts = self.add_particles(3)
        fene = espressomd.interactions.FeneBond(k=10., r_0=1., d_r_max=2.)
        angle = espressomd.interactions.AngleHarmonic(bend=1.0, phi0=np.pi)
        sim_inst.sys.bonded_inter.add(fene)
        sim_inst.sys.bonded_inter.add(angle)
        parts[0].add_bond((fene, parts[1].id))
        parts[1].add_bond((angle, parts[0].id, parts[2].id))

        path = os.path.join(self.tmpdir.name, "topology.h5")
        with h5py.File(path, "w") as handle:
            n_links = write_bonds(handle.require_group("connectivity"),
                                  particles=parts, sys=sim_inst.sys, step=0)
        self.assertEqual(n_links, 2)

        with h5py.File(path, "r") as handle:
            recovered = list(read_bonds(handle["connectivity/bonds"]))
            live = list(read_bonds(handle["connectivity/bonds"], instantiate=True))

        self.assertEqual(len(recovered), 2)
        by_particle = {pid: (tuple(partners), bond) for pid, partners, bond in recovered}
        self.assertEqual(by_particle[parts[0].id][0], (parts[1].id,))
        self.assertEqual(by_particle[parts[0].id][1], _bond_id_of(fene))
        self.assertEqual(set(by_particle[parts[1].id][0]), {parts[0].id, parts[2].id})
        self.assertEqual(by_particle[parts[1].id][1], _bond_id_of(angle))

        # instantiate=True must hand back live espresso objects, not ids.
        kinds = {type(bond) for _, _, bond in live}
        self.assertEqual(kinds, {espressomd.interactions.FeneBond,
                                 espressomd.interactions.AngleHarmonic})
