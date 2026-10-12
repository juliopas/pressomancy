"""Kill-and-resume scan of the trajectory lifecycle (run by test_IO.SwmrKillTest; not a test module).

Usage: LD_PRELOAD=<killshim.so> python swmr_kill_harness.py <killshim.so> <workdir>

One process (one espresso start-up); every kill is a forked child, since the shim's
SIGKILL at the k-th ``pwrite`` is per process (see ``killshim.c``). The scans, each
sweeping k from 1 until the child completes without a kill:

- ``append``: for 1 and 65 stored frames, a child resumes the file (``LOAD_NEW``: an
  installed copy reopened in SWMR mode) unarmed, then writes the next frame and is killed
  at its k-th pwrite. Then (parent) the crashed file must open with ``open_output`` with
  every earlier frame readable, and ``LOAD_NEW`` with ``rewind_to_step`` at the last
  complete frame (the checkpoint's step) + one append + a full read of every element must
  succeed and match. (A copy keeps its source's index type, so the 65-frame scan also
  covers the index of a NEW-created file.)
- ``append_new``: the same at 1 frame on a file the child itself created: ``NEW`` + the
  stored frame unarmed, then the killed append (a fresh run's first chunks).
- ``resume``: a child is killed at its k-th pwrite anywhere inside ``LOAD_NEW`` (the copy,
  the truncation, the close of ``.new``, the reopen). The file at the real name must be
  either byte-identical to the original or the complete installed copy; a plain
  ``LOAD_NEW`` + append + full read must then succeed.

The last stdout line is a JSON report.
"""
import ctypes
import json
import os
import shutil
import signal
import sys
import warnings

import h5py
import numpy as np

warnings.filterwarnings("ignore", message="This process .* is multi-threaded")

shim_path, workdir = sys.argv[1], sys.argv[2]
shim = ctypes.CDLL(shim_path)      # the LD_PRELOADed copy: dlopen of the same path returns it
shim.arm.argtypes = [ctypes.c_long]

import espressomd                                            # noqa: E402
import pressomancy                                           # noqa: E402
from pressomancy.simulation import Simulation, Filament      # noqa: E402
from pressomancy.infra import BondWrapper                    # noqa: E402
from pressomancy.io import open_output                       # noqa: E402
from pressomancy.io.read import element_name, superblock_version   # noqa: E402

sim = Simulation(box_dim=[16., 16., 16.])
sim.set_sys(min_global_cut=1.)
bond = BondWrapper(espressomd.interactions.FeneBond(k=10., r_0=2., d_r_max=3.))
filaments = [Filament(config=Filament.config.specify(sigma=2., size=8., n_parts=4, espresso_handle=sim.sys,
                                                     bond_handle=bond)) for _ in range(2)]
sim.store_objects(filaments)
sim.set_objects(filaments)
sim.io_dict['bonds'] = True
GROUP = [Filament]
ATTRS = {'parameters/run': {'H': 0.6}, 'parameters/run/starts/0': {'host': 'harness'}}
RESTART_ATTRS = {'parameters/run/starts/1': {'host': 'harness', 'kind': 'restart'}}
ELEMENTS = [element_name(attr) for attr, _dim, _dtype in sim.io_dict['properties']]
MAX_KILL_POINTS = 2000


def time_of(step):
    return 0.5 * step          # exact in the float32 time column


def close():
    """Back to 'nothing inscribed' (what test/create_system.reset_io_state does)."""
    sim._h5_writer._discard()


def write_frames(path, n_frames):
    """NEW + ``n_frames`` frames (steps 0..n-1) with the run's attrs, left open in SWMR mode."""
    sim.inscribe_part_group_to_h5(group_type=GROUP, h5_data_path=path, mode='NEW', attrs=ATTRS)
    for step in range(n_frames):
        sim.sys.time = time_of(step)
        sim.write_part_group_to_h5(step=step)


def write_base(path, n_frames):
    """A cleanly closed trajectory of ``n_frames`` frames (steps 0..n-1) with the run's attrs."""
    write_frames(path, n_frames)
    close()


def read_all(h5_file):
    """``{element: array}`` of every element, the box edges and the timeline: a full read."""
    data_grp = h5_file["particles/Filament"]
    out = {element: data_grp[f"{element}/value"][...] for element in ELEMENTS}
    out['box/edges'] = data_grp["box/edges/value"][...]
    out['step'] = data_grp["position/step"][...]
    out['time'] = data_grp["position/time"][...]
    return out


def links_shared(h5_file):
    data_grp = h5_file["particles/Filament"]
    return all(data_grp[f"{element}/step"] == data_grp["position/step"]
               and data_grp[f"{element}/time"] == data_grp["position/time"]
               for element in ELEMENTS + ["box/edges"])


def live_frame():
    parts = sorted(sim.sys.part.all(), key=lambda part: part.id)
    return {'id': np.array([part.id for part in parts]), 'position': np.array([part.pos for part in parts])}


def run_killed(k, prepare, action):
    """Fork; the child runs ``prepare()`` unarmed, arms the shim at ``k`` and runs ``action()``.
    Returns 'killed', 'done' (the child finished: k is past the last pwrite) or an error string."""
    pid = os.fork()
    if pid == 0:
        try:
            prepare()
            shim.arm(k)
            action()
            shim.arm(0)
            os._exit(0)
        except BaseException as exc:          # noqa: BLE001 -- reported through the exit code
            shim.arm(0)
            sys.stdout.write(f"child error at k={k}: {exc!r}\n")
            sys.stdout.flush()
            os._exit(3)
    _, status = os.waitpid(pid, 0)
    if os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL:
        return 'killed'
    return 'done' if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 else f"child exit {status}"


def resume_append_read(path, n_frames, rewind_step, expected_base):
    """The parent's restart: LOAD_NEW rewound to ``rewind_step`` + one append + full read, compared."""
    sim.sys.time = time_of(rewind_step)
    kept = sim.inscribe_part_group_to_h5(group_type=GROUP, h5_data_path=path, mode='LOAD_NEW',
                                         rewind_to_step=rewind_step, attrs=RESTART_ATTRS)
    if kept != rewind_step + 1:
        raise AssertionError(f"kept {kept} frames, expected {rewind_step + 1}")
    sim.sys.time = time_of(rewind_step + 1)
    sim.write_part_group_to_h5(step=rewind_step + 1)
    h5_file = sim.io_dict['h5_file']
    got = read_all(h5_file)
    if not links_shared(h5_file):
        raise AssertionError("step/time are not shared after the copy")
    if h5_file["parameters/run/starts/1"].attrs['kind'] != 'restart' or h5_file["parameters/run"].attrs['H'] != 0.6:
        raise AssertionError("attrs missing after the resume")
    for name, want in expected_base.items():
        if not np.array_equal(got[name][:rewind_step + 1], want[:rewind_step + 1]):
            raise AssertionError(f"frames 0..{rewind_step} of {name} differ from the base")
    live = live_frame()
    if not (np.array_equal(got['id'][-1], live['id']) and np.allclose(got['position'][-1], live['position'])
            and got['step'].tolist() == list(range(rewind_step + 2))):
        raise AssertionError("the appended frame does not match the live system")
    if min(len(v) for v in got.values()) != rewind_step + 2:
        raise AssertionError(f"ragged after the append: {[len(v) for v in got.values()]}")
    close()


def read_crashed(path, n_frames, expected_base):
    """Every earlier frame of a crashed file is readable through open_output."""
    with open_output(path) as h5_file:
        got = read_all(h5_file)
        for name, want in expected_base.items():
            if got[name].shape[0] < n_frames or not np.array_equal(got[name][:n_frames], want):
                raise AssertionError(f"{name}: frames 0..{n_frames - 1} lost or changed after the kill")


def scan_append(n_frames, created_by_new=False):
    """``created_by_new``: the child writes the file itself (NEW + frames) instead of resuming a copy."""
    tag = f"append{'_new' if created_by_new else ''}_{n_frames}"
    base, path = os.path.join(workdir, f"{tag}_base.h5"), os.path.join(workdir, f"{tag}.h5")
    write_base(base, n_frames)
    with h5py.File(base, 'r') as h5_file:
        expected = read_all(h5_file)
    report = {'kill_points': 0, 'ok': 0, 'failures': [], 'flags': {}}

    def prepare():
        if created_by_new:
            write_frames(path, n_frames)
        else:
            sim.inscribe_part_group_to_h5(group_type=GROUP, h5_data_path=path, mode='LOAD_NEW')
        sim.sys.time = time_of(n_frames)

    def action():
        sim.write_part_group_to_h5(step=n_frames)

    for k in range(1, MAX_KILL_POINTS):
        if not created_by_new:
            shutil.copy(base, path)
        outcome = run_killed(k, prepare, action)
        if outcome == 'done':
            break
        if outcome != 'killed':
            report['failures'].append((k, outcome))
            continue
        report['kill_points'] += 1
        flag = open(path, 'rb').read(12)[11]
        report['flags'][str(flag)] = report['flags'].get(str(flag), 0) + 1
        try:
            read_crashed(path, n_frames, expected)
            resume_append_read(path, n_frames, n_frames - 1, expected)
            report['ok'] += 1
        except Exception as exc:      # noqa: BLE001
            close()
            report['failures'].append((k, f"{type(exc).__name__}: {str(exc)[:160]}"))
    return report


def check_installed_or_intact(path, original_bytes, n_kept):
    """After a kill inside LOAD_NEW: the real name holds the original bytes, or the complete copy."""
    with open(path, 'rb') as handle:
        if handle.read() == original_bytes:
            return 'intact'
    with open_output(path) as h5_file:
        got = read_all(h5_file)
        if {len(v) for v in got.values()} != {n_kept}:
            raise AssertionError(f"installed copy is not complete: {[len(v) for v in got.values()]}")
        if h5_file["parameters/run/starts/1"].attrs['kind'] != 'restart':
            raise AssertionError("installed copy lacks the restart attrs")
    return 'installed'


def scan_resume(n_frames):
    base, path = os.path.join(workdir, "resume_base.h5"), os.path.join(workdir, "resume.h5")
    write_base(base, n_frames)
    with h5py.File(base, 'r') as h5_file:
        expected = read_all(h5_file)
    original = open(base, 'rb').read()
    rewind_step = n_frames - 2
    report = {'kill_points': 0, 'ok': 0, 'failures': [], 'intact': 0, 'installed': 0}

    def prepare():
        sim.sys.time = time_of(rewind_step)

    def action():
        sim.inscribe_part_group_to_h5(group_type=GROUP, h5_data_path=path, mode='LOAD_NEW',
                                      rewind_to_step=rewind_step, attrs=RESTART_ATTRS)

    for k in range(1, MAX_KILL_POINTS):
        shutil.copy(base, path)
        outcome = run_killed(k, prepare, action)
        if outcome == 'done':
            break
        if outcome != 'killed':
            report['failures'].append((k, outcome))
            continue
        report['kill_points'] += 1
        try:
            report[check_installed_or_intact(path, original, rewind_step + 1)] += 1
            resume_append_read(path, n_frames, rewind_step, expected)
            report['ok'] += 1
        except Exception as exc:      # noqa: BLE001
            close()
            report['failures'].append((k, f"{type(exc).__name__}: {str(exc)[:160]}"))
    return report


report = {'append': {str(n): scan_append(n) for n in (1, 65)},
          'append_new': {'1': scan_append(1, created_by_new=True)},
          'resume': scan_resume(6),
          'superblock_version': superblock_version(os.path.join(workdir, "resume_base.h5")),
          'pressomancy_file': pressomancy.__file__}
print(json.dumps(report))
