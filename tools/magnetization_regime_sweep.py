#!/usr/bin/env pypresso
"""Map the parameter regime where pressomancy's mutual-magnetization scheme is usable.

Espresso evaluates the magnetization models inside the integrator loop, responding
to the dipolar field computed in the *previous* force calculation. The moments
therefore approach their self-consistent value over several timesteps, and only if
the map ``m <- m(H_ext + H_dip(m))`` is a contraction. There is no quantitative
criterion for that anywhere in the source, so this tool measures one.

Two rules make the measurement meaningful:

* **A small contraction ratio alone is not a pass.** Because the moment saturates,
  the differential susceptibility falls off with field, so the map also contracts
  around a spuriously saturated fixed point. A point counts as usable only if the
  ratio is below 1 *and* the moment is well clear of saturation.
* **Measure near zero field.** Saturation damps the response, so large-field
  measurements flatter the scheme. The stability boundary lives in the linear
  regime, where the differential susceptibility is largest and the map is least
  contractive. Measuring there removes the false positive by construction rather
  than trying to detect it after the fact.

Two geometries are reported:

``chain``
    A touching head-to-tail line -- the worst case, where dipolar fields reinforce
    maximally. Gives a conservative bound.
``mae``
    The real magneto-active elastomer configuration: an ``Elastomer`` with embedded
    ``PointDipoleMagnetizable`` particles, as in ``samples/mae-BoS.py``. A dense 3D
    arrangement partially cancels, so this is the operative number.

Run with, e.g.::

    /path/to/pypresso tools/magnetization_regime_sweep.py --geometry chain
    /path/to/pypresso tools/magnetization_regime_sweep.py --geometry mae --n-parts 60

Note for anyone extending this: you cannot seed an initial magnetization to study
branch selection. ``dip`` is overwritten from ``H_tot`` before ``dip_fld`` is ever
computed from it, so a seeded value never generates a field and the measurement is
silently meaningless. Reaching a magnetized state requires an external field.
"""
import argparse
import logging

import numpy as np

import espressomd
from espressomd.magnetostatics import DipolarDirectSum

from pressomancy.simulation import Simulation, Elastomer, PointDipoleMagnetizable

#: Field used for every measurement. Small on purpose -- see the module docstring.
PROBE_FIELD = 0.01

#: A moment above this fraction of dipm_sat is treated as saturated, so a small
#: contraction ratio there is the documented false positive rather than a pass.
SATURATION_CUTOFF = 0.5

#: Amplification above which the settled moment is dominated by the mutual field
#: rather than the applied one. Not unphysical -- chains really do amplify -- but
#: the fixed point is then highly parameter-sensitive.
AMPLIFICATION_CUTOFF = 10.0

SIZE_PART = 2.0 ** (1.0 / 6.0)


def _isolated_moment(chi0, field, m_sat=1.0):
    """Langevin moment of a single particle -- the no-mutual-interaction baseline."""
    alpha = 3.0 * chi0 * field / m_sat
    if alpha < 1e-12:
        return 0.0
    return m_sat * (1.0 / np.tanh(alpha) - 1.0 / alpha)


def _settle(sim, virt, n_iter, chi0):
    """Advance the fixed point at frozen positions -> (ratio, m/m_sat, amplification).

    ``amplification`` is the settled moment divided by what a single isolated
    particle would reach at the same field. It is the most direct read-out of how
    much of the result is mutual magnetization rather than the applied field, and
    unlike the contraction ratio it grows monotonically into the unstable regime
    instead of folding back once the moments saturate.
    """
    ratios = sim.probe_magnetization_convergence(virt, n_iter=n_iter)
    moments = np.array([float(np.linalg.norm(p.dip)) for p in virt])
    mean_moment = float(moments.mean())
    baseline = _isolated_moment(chi0, PROBE_FIELD)
    amplification = mean_moment / baseline if baseline > 0 else float('nan')
    return (float(ratios[-1]) if len(ratios) else 0.0), mean_moment, amplification


def measure_chain(sim, chi0, prefactor, spacing, n_particles=6, n_iter=40):
    """One point of the worst-case geometry: a touching head-to-tail chain."""
    sim.reinitialize_instance()
    sim.sys.box_l = (40., 40., 40.)
    cfg = PointDipoleMagnetizable.config.specify(
        magnetization_model='langevin', dipm_sat=1., mag_susc_0=chi0,
        espresso_handle=sim.sys)
    objs = [PointDipoleMagnetizable(config=cfg) for _ in range(n_particles)]
    sim.store_objects(objs)
    centre = 20.0 - 0.5 * spacing * (n_particles - 1)
    sim.place_objects(
        objs,
        [np.array([20., 20., centre + i * spacing]) for i in range(n_particles)],
        [np.array([0., 0., 1.]) for _ in range(n_particles)])
    sim.init_magnetic_inter(DipolarDirectSum(prefactor=prefactor))
    sim.set_H_ext(H=(0, 0, PROBE_FIELD))
    virt = [p for o in objs for p in o.get_owned_part()[0]
            if int(p.type) == PointDipoleMagnetizable.part_types['pdm_virt']]
    return _settle(sim, virt, n_iter, chi0)


def measure_mae(sim, chi0, prefactor, n_parts=60, density=0.3, n_iter=40):
    """One point of the operative geometry: elastomer with embedded magnetizables."""
    sim.reinitialize_instance()
    r_part = SIZE_PART / 2
    box_xy = np.cbrt(n_parts * 4 / 3 * np.pi / density) * r_part * 2.0
    layer = 6 * SIZE_PART
    sim.sys.box_l = [box_xy, box_xy, 4 * box_xy]
    sim.sys.periodicity = (True, True, True)

    cfg = PointDipoleMagnetizable.config.specify(
        magnetization_model='langevin', dipm_sat=1., mag_susc_0=chi0,
        espresso_handle=sim.sys)
    dipoles = [PointDipoleMagnetizable(config=cfg) for _ in range(n_parts)]
    cfg_e = Elastomer.config.specify(
        layer_height=layer, n_parts=n_parts, associated_objects=dipoles,
        bond_K_lims=(0.01, 0.1), size=SIZE_PART, sigma=1.0,
        espresso_handle=sim.sys, seed=sim.seed)
    elastomer = Elastomer(config=cfg_e)
    sim.store_objects([elastomer])
    sim.set_objects([elastomer])
    sim.init_magnetic_inter(DipolarDirectSum(prefactor=prefactor))
    sim.set_H_ext(H=(0, 0, PROBE_FIELD))
    virt = [p for p in sim.sys.part.all()
            if int(p.type) == PointDipoleMagnetizable.part_types['pdm_virt']]
    return _settle(sim, virt, n_iter, chi0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--geometry', choices=('chain', 'mae'), default='chain')
    ap.add_argument('--chi0', type=float, nargs='+',
                    default=[0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    ap.add_argument('--prefactor', type=float, nargs='+', default=[1.0])
    ap.add_argument('--spacing', type=float, nargs='+', default=[SIZE_PART],
                    help='chain geometry only, in sigma units')
    ap.add_argument('--n-parts', type=int, default=60, help='mae geometry only')
    ap.add_argument('--n-iter', type=int, default=40)
    args = ap.parse_args()

    logging.disable(logging.CRITICAL)
    sim = Simulation(box_dim=(40., 40., 40.))
    sim.set_sys(timestep=0.001)

    print(f"# geometry={args.geometry}  probe field H={PROBE_FIELD}  "
          f"saturation cutoff={SATURATION_CUTOFF}*m_sat")
    print(f"{'chi0':>7} {'prefac':>7} {'spacing':>8} {'ratio':>9} {'m/m_sat':>9} {'amplif':>8}  verdict")
    for prefactor in args.prefactor:
        spacings = args.spacing if args.geometry == 'chain' else [float('nan')]
        for spacing in spacings:
            for chi0 in args.chi0:
                try:
                    if args.geometry == 'chain':
                        ratio, moment, amp = measure_chain(sim, chi0, prefactor, spacing,
                                                           n_iter=args.n_iter)
                    else:
                        ratio, moment, amp = measure_mae(sim, chi0, prefactor,
                                                         n_parts=args.n_parts,
                                                         n_iter=args.n_iter)
                except Exception as exc:                       # keep the sweep going
                    print(f"{chi0:7.3f} {prefactor:7.2f} {spacing:8.3f} "
                          f"{'--':>9} {'--':>9} {'--':>8}  FAILED: {type(exc).__name__}: {str(exc)[:40]}")
                    continue
                if moment >= SATURATION_CUTOFF:
                    verdict = "SATURATED (ratio not meaningful)"
                elif ratio >= 1.0:
                    verdict = "DIVERGES"
                elif amp >= AMPLIFICATION_CUTOFF:
                    # Converges and is not saturated, but the moment is almost all
                    # mutual field. Physically a real strong-coupling regime; a fixed
                    # point this close to the instability is very sensitive to the
                    # parameters, so it is not somewhere to run production blind.
                    verdict = "MARGINAL (near instability)"
                else:
                    verdict = "usable"
                print(f"{chi0:7.3f} {prefactor:7.2f} {spacing:8.3f} "
                      f"{ratio:9.4f} {moment:9.4f} {amp:8.1f}  {verdict}")


if __name__ == '__main__':
    main()
