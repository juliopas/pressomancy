IO And Analysis
===============

Pressomancy's IO layer is useful for the same reason its object model is
useful: it preserves structure. Instead of writing only a flat list of
particle properties, the HDF5 writer stores those properties together with the
ownership and connectivity information that explains which simulation object
each particle came from. That makes the saved file useful in three different
roles at once: as a trajectory, as an analysis source, and as the starting
point for a new simulation state.

Why Use The IO Stack
--------------------

For each registered object family, pressomancy writes particle data under
``/particles/<Group>/<property>`` using the H5MD-style triplet of ``value``,
``step``, and ``time`` datasets. The corresponding property tensors have shape
``(T, N, D)``, where ``T`` is the number of stored frames, ``N`` is the number
of particles in the registered flat view for that family, and ``D`` is the
dimensionality of the property. In parallel, the writer stores object-context
information under ``/connectivity`` so that the file can answer questions a
plain trajectory cannot, such as which particles belong to a given
:class:`~pressomancy.object_classes.filament_class.Filament` or which child
objects belong to a given parent object.

That combined layout is the real benefit of using the built-in IO stack. The
particle-property part is close enough to H5MD to be familiar and efficient,
while the connectivity extension makes the result meaningful inside a
pressomancy workflow. The payoff is practical rather than theoretical: one
format supports long-run storage, object-aware analysis, and source-driven
reconstruction without format conversion in between.

Write And Restart A Trajectory
------------------------------

The usual write workflow starts by telling the simulation which object
families should be persisted. That happens through
:meth:`~pressomancy.simulation.Simulation.inscribe_part_group_to_h5`, which
creates the HDF5 layout and records the flat particle views that later writes
will use. Concretely, the simulation stores those particle handles in
``sim.io_dict['flat_part_view']``. That view is the ordered particle list used
by :meth:`~pressomancy.simulation.Simulation.write_part_group_to_h5` when each
new frame is appended. In other words, ``flat_part_view`` is the bridge
between the simulation-object hierarchy and the flat per-group particle layout
written to HDF5.

.. code-block:: python

   step_index = sim.inscribe_part_group_to_h5(
       group_type=[Filament, Crowder],
       h5_data_path="run.h5",
       mode="NEW",
   )

   for _ in range(n_steps):
       sim.sys.integrator.run(1)
       sim.write_part_group_to_h5(step=step_index)
       step_index += 1

On the current branch,
:meth:`~pressomancy.simulation.Simulation.inscribe_part_group_to_h5` supports
three modes. In practice, they are not interchangeable and it is worth being
explicit about what each one assumes.

- ``NEW`` creates a fresh HDF5 file and registers the current flat particle
  view for later writes. This is also the mode used when a run starts fresh
  from a source file: seeding positions, orientations, properties, and bonds
  from a previous trajectory (see "Use Output As A Source" below) is
  orthogonal to the writer mode used for the new run's own output.
- ``LOAD_NEW`` reopens an existing HDF5 file and rebuilds the flat particle
  view from the connectivity already stored in that file. In practice this is
  the preferred continuation mode when you have both the ESPResSo binary
  checkpoint and the HDF5 file available, because it does not require you to
  reconstruct the full Python-side object graph before continuing IO.
- ``LOAD`` reopens an existing HDF5 file, but it assumes that the simulation
  setup has already been reconstructed in memory and that the current objects
  already own the right particles. This mode therefore depends only on the
  ESPResSo checkpoint for particle state, but it requires the whole simulation
  script to rebuild the object hierarchy before the IO layer can safely resume.

In day-to-day restart work, ``LOAD_NEW`` is usually the strongest option.

New files also carry optional metadata that complements the particle and
connectivity layout. The writer records H5MD-style root metadata under
``/h5md`` together with pressomancy-specific metadata under
``/parameters/pressomancy``. In practice, that metadata serves two roles.
First, it stores lightweight provenance for the submission script and the
pressomancy checkout that produced the file. Second, it stores the current
``part_types`` map so that ``LOAD_NEW`` can restore symbolic particle-type
bookkeeping directly when that information is available. The
connectivity tables in the HDF5 file give pressomancy enough information to
rebuild the flat particle view directly from saved object ownership, which is
why it offers extra functionality compared with ``LOAD``.

This is also a good place to keep the role of ESPResSo checkpointing in
perspective. The binary checkpoint mechanism is powerful and robust, but it
relies on binary state and pickle-based reconstruction. In practice, that
means it is less portable across systems, package versions, ESPResSo versions,
and runtime layouts such as MPI rank counts. The HDF5 files are less tied to
that execution environment and can, in principle, be used to reconstruct a
simulation state while sidestepping binary checkpoint portability limits. That
does not remove the need for care, but it is one of the reasons the HDF5 path
is so valuable in longer-lived workflows.

Keep Checkpoints And HDF5 In Sync
---------------------------------

Long simulations often save two independent kinds of state: something to
restart the live system from, and an HDF5 trajectory for analysis. If a job is
interrupted between those two writes, the last valid restart state and the
last valid HDF5 frame may no longer correspond to one another. Pressomancy's
own checkpoint mechanism,
:meth:`~pressomancy.simulation.Simulation.write_checkpoint` and
:meth:`~pressomancy.simulation.Simulation.restart_from_checkpoint`, together
with ``rewind_to_step``, is the pressomancy-native way to keep the two in
sync without leaning on ESPResSo's own binary checkpoint.

``write_checkpoint(group_type, path, step)`` writes an ordinary one-frame
HDF5 file to ``path``: the full per-particle state needed to resume exactly
(position, velocity, force, and, where the espresso build has the feature,
quaternion, angular velocity, lab-frame torque and dipole moment, all in
float64), the bond topology, and a small metadata group recording the
simulation time and every active thermostat's Philox counter. The file is
written to ``path + ".tmp"``, reopened and verified, and only then moved onto
``path`` with an atomic replace; a failed verification leaves the ``.tmp`` in
place and never touches a previous checkpoint at ``path`` -- the mechanism
never deletes, so keeping several checkpoints around is a matter of using
several paths.

.. code-block:: python

   step_index = sim.write_checkpoint([Filament, Crowder], "run.ckpt.h5", step=step_index)

On the restart side, ``restart_from_checkpoint(objects, path, src_to_loc=None,
bonds=False, place_from=None, r_cut_override=0.0)`` is the mirror: it is a
``load_from_src`` call with the checkpoint's own recipe filled in (every
stored type paired with the local type of the same name, restoring the fixed
per-particle state in the order the espresso setters require), plus setting
``sys.time`` and overriding the recorded thermostat counters -- so the
thermostat must already be set, with its seed, before the call. It returns
the checkpoint's stored step.

.. code-block:: python

   step_index = sim.restart_from_checkpoint(objects, "run.ckpt.h5", place_from=["real"])
   sim.sys.integrator.run(1, reuse_forces=True)   # first call after a restart: reuse the restored forces

That first ``reuse_forces=True`` call matters: the checkpoint restores the
force and the thermostat's Philox counter together, and recomputing forces
before integrating would draw fresh noise from that counter instead of
continuing the same stream. Done this way, the continuation matches an
uninterrupted run to roundoff, not bitwise. The LB fluid state is not covered
by a checkpoint; ``write_checkpoint`` refuses to write one while the LB
thermostat is active.

Once the live system is back at the checkpoint's step, the HDF5 trajectory
itself needs to be rewound to match it before writing resumes, which is what
``rewind_to_step`` on ``inscribe_part_group_to_h5``/``inscribe_observable_group_to_h5``
is for: it looks up the frame stored at that step, requires its recorded time
to equal the live ``sys.time`` (which the restart above just set) as a
sanity check, and truncates the trajectory to end there.

.. code-block:: python

   step_index = sim.inscribe_part_group_to_h5(
       group_type=[Filament, Crowder], h5_data_path="run.h5", mode="LOAD_NEW",
       rewind_to_step=step_index,
   )

``force_resize_to_size`` still exists alongside ``rewind_to_step`` -- mutually
exclusive, and resolved before the usual ``LOAD_NEW`` checks either way -- for
the case where what you have is a frame *count* to truncate to rather than a
stored step value, for instance a counter tracked independently of any
checkpoint.

Analysis API
------------

Reading follows the same explicit style. The entry point is
:class:`~pressomancy.io.read.H5DataSelector`. The important
idea is that the selector never asks you to guess which axis an index belongs
to. Timesteps and particles are separate iteration contexts, exposed through
``.timestep`` and ``.particles``.

.. code-block:: python

   with h5py.File("run.h5", "r") as h5_file:
       data = H5DataSelector(h5_file, particle_group="Filament")

       frame = data.timestep[-1]
       particle_window = data.particles[100:150]
       composed_a = data.timestep[-1].particles[100:150]
       composed_b = data.particles[100:150].timestep[-1]

Because the selector stores one timestep slice and one particle slice
internally, timestep and particle selection can be composed in either order.
That gives you unambiguous iteration contexts for both axes and makes slicing
behavior robust even in longer chained expressions. If you want to iterate
frame by frame, iterate over ``data.timestep``. If you want to iterate over a
particle subset, iterate over ``data.particles``. If you want both, compose
the two contexts first and then access the properties.

Predicates are the next important layer. They let you refine a selected view
without leaving the selector API.

.. code-block:: python

   with h5py.File("run.h5", "r") as h5_file:
       data = H5DataSelector(h5_file, particle_group="Filament")
       frame = data.timestep[-1]

       filament_zero = frame.select_particles_by_object(
           object_name="Filament",
           connectivity_value=0,
           predicate=lambda subset: subset.type == sim.part_types["real"],
       )

       magnetic_filaments = frame.get_connectivity_values(
           "Filament",
           predicate=lambda subset: np.any(subset.dip[..., 2] > 0.0),
       )

This is useful because it lets you express selection logic in the same
coordinate system in which the data are stored. You can first reduce the view
by timestep, then by object ownership, then by a property predicate, without
having to manually reconstruct index arrays outside the API.

The object-context helpers are what make the selector especially useful for
pressomancy-generated data. ``select_particles_by_object`` gives you the
particle subset belonging to one saved object instance, such as one
:class:`~pressomancy.object_classes.filament_class.Filament` or one
:class:`~pressomancy.object_classes.quadriplex_class.Quadriplex`.
:meth:`~pressomancy.io.read.H5DataSelector.get_connectivity_values`
lets you enumerate object IDs, optionally filtered by a predicate, and raises
``KeyError`` naming the missing mapping rather than returning ``None``.
:meth:`~pressomancy.io.read.H5DataSelector.get_child_ids`,
:meth:`~pressomancy.io.read.H5DataSelector.get_parent_ids`, and
:meth:`~pressomancy.io.read.H5DataSelector.get_connectivity_map`
then let you move up and down the saved object graph, raising the same way on
a missing mapping. In practice, this enables
workflows such as selecting all particles of one
:class:`~pressomancy.object_classes.filament_class.Filament` at one frame,
querying which
:class:`~pressomancy.object_classes.quadriplex_class.Quadriplex` objects belong
to that filament, or filtering object IDs based on per-object property tests
without re-deriving connectivity from raw coordinates.

Use Output As A Source
----------------------

Source-driven initialization is best understood as a workflow for building a
new simulation from the structural state of an older one. A representative
example is a long run of fixed-point dipole chains used to reach an
equilibrium self-assembly picture. A later run can then take a chosen snapshot
from that older trajectory, reuse its positions and orientations, and
substitute a different local particle model such as
:class:`~pressomancy.object_classes.egg_model_part.EGGPart` while keeping the
larger assembled structure.

The single public entry point is
:meth:`~pressomancy.simulation.Simulation.load_from_src`. It seeds ``objects``
from ``path`` in one call, at one resolved frame, and returns the number of
bond links added (``0`` when ``bonds`` is left false).

.. code-block:: python

   sim.store_objects(objects)
   sim.load_from_src(
       objects,
       path="run.h5",
       src_to_loc={(("real", "real"), ("virt", "virt")): [("dip", "director")]},
       bonds=True,
       place_from=["real"],
   )

Placement is optional, and ``place_from`` is the switch. Give it (a list of
*source* type names) when the file should place your objects, which requires one
stored particle of those types per monomer; leave it out when you built the tree
yourself -- compound objects, or a running system whose state is being restored.
Without ``place_from`` the call skips placement entirely and only copies
``src_to_loc`` (and the bonds, when asked); the placing readers
``get_pos_ori_from_src`` and ``set_objects_from_src`` then raise ``RuntimeError``,
and a tree that was never placed fails loudly in the one-to-one count check,
with zero local particles to pair.

``src_to_loc`` is one dict that says both *which* particles pair up and *what*
is copied onto them. Every tuple in it is ordered ``(source, local)``. A key is
one type pair, ``("real", "real")``, or a tuple of type pairs sharing one
property list; the value is that key's list of ``(src_prop, loc_prop)`` pairs,
and ``[]`` is allowed -- it means "pair these particles, copy nothing", which is
what a bond-only or placement-only type needs. The same type pair may appear
under several keys and its lists are concatenated; repeating a property pair for
one type pair, or writing a malformed entry, raises ``ValueError`` naming it.

The type pairs of ``src_to_loc`` are also the pairing bond restoration uses, so
the mapping must be non-empty whenever ``bonds`` is true. Source type names,
those in ``place_from`` and the source half of every type pair, are
resolved through the source file's own recorded type table rather than the live
simulation's, so a name the file never wrote raises ``KeyError`` naming what
the file does declare.

.. code-block:: python

   # checkpoint restart of simple objects: the file places them, identity over every
   # owned type, the saved state, bonds back
   STATE = [("pos", "pos"), ("director", "director"), ("v", "v"),
            ("omega_lab", "omega_lab"), ("fix", "fix")]
   sim.load_from_src(objects, "run.h5", bonds=True, place_from=["real"], step=k,
                     src_to_loc={(("real", "real"), ("virt", "virt")): STATE,
                                 ("substrate", "substrate"): []})

   # re-typed start: new object classes at the saved places, only what both share
   sim.load_from_src(new_objects, "run.h5", place_from=["pdp_real"],
                     src_to_loc={("pdp_real", "real"): [("pos", "pos"), ("director", "director")]})

   # compound objects / an existing tree: build and place it as usual, then copy the state
   # (no place_from; bonds=True only for objects that do not build their own bonds)
   sim.load_from_src(objects, "run.h5", src_to_loc={(("real", "real"),
                                                     ("virt", "virt")): STATE})

   # anything that is not a 1:1 copy: fetch it, then let the object apply it
   dips = sim.get_prop_from_src(objects, "run.h5", src_type="pdp_real", prop="dip", step=k)

:meth:`~pressomancy.simulation.Simulation.get_prop_from_src` is the read-only
counterpart: it returns one ``(N_i, dim)`` array per object -- the stored values
of that property for the object's ``src_type`` particles, in stored order -- and
changes nothing in the system. An unknown property raises ``KeyError`` naming
the dataset that is missing.

``load_from_src`` chains, all at the same resolved frame: declaring the
source and the mapping, reading positions and orientations for ``place_from``
and placing the objects when one was given, copying each type pair's
properties when ``src_to_loc`` is non-empty, and re-creating the stored bonds when
``bonds`` is true. Positions and orientations are special in this workflow:
with ``place_from``, top-level objects, for example
:class:`~pressomancy.object_classes.filament_class.Filament` instances, are
created directly at the positions and orientations read from the HDF5 file,
rather than through the normal placement step.

.. code-block:: python

   sim.load_from_src(objects, "run.h5", STATE, place_from=["real"])            # last frame
   sim.load_from_src(objects, "run.h5", STATE, place_from=["real"], frame=0)    # frame INDEX
   sim.load_from_src(objects, "run.h5", STATE, place_from=["real"], step=250)   # stored step VALUE
   sim.load_from_src(objects, "run.h5", STATE, place_from=["real"], time=12.5)  # nearest stored time

``frame``, ``step`` and ``time`` select the same frame through three
different coordinate systems. Passing more than one is only valid when they
agree on the same frame; disagreeing selectors raise ``ValueError``. ``time``
matches the nearest stored time within a relative tolerance of ``1e-6`` and
raises ``KeyError`` when nothing is close enough.

``load_from_src`` is the only public seeding entry point; declaring the
source, placing the objects, copying properties, and restoring bonds happen
together in that one call rather than as separate steps a script chains
itself.

Performance And Common Mistakes
-------------------------------

The HDF5 writer stores each ``value`` dataset with timestep-shaped chunks,
namely ``(1, N, D)``, and uses moderate gzip compression. This is not just a
storage detail. It means a read such as ``data.timestep[-1]`` naturally
touches one frame at a time rather than encouraging whole-trajectory reads.
For long runs, that keeps memory use under control and makes frame-local,
object-aware analysis practical in an interactive workflow.

The most common IO failures are mapping failures rather than storage failures.
If a source-driven workflow breaks, the first things to check are whether
``load_from_src`` was given the right source path and whether the declared
type names exist both in the local simulation and in the source file's own
type table. The next thing to check is ``src_to_loc``: a type pair that does not
pair source and local particles one-to-one raises rather than guessing, and a
property listed under the wrong type pair is copied onto the wrong particles or
not at all. Placement has its own requirement: a ``place_from`` type must resolve to
exactly one particle per monomer of the object being placed, so a deeply nested
tree may have to be built and placed locally and only have its state restored,
by a ``load_from_src`` call without ``place_from`` (``samples/poly_BRACO.py``
shows that case).

The other recurring mistake is to postpone the structured IO path until the
analysis stage. By that point, the important ownership information may simply
never have been written. If you expect object-level analysis or source-driven
restarts later, the built-in HDF5 path is worth using from the first run.

For a case where hierarchy, placement, and IO all matter at once, continue to
:doc:`g_quadruplex`.
