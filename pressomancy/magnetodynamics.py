'''
Per-particle magnetodynamics: the single seam between pressomancy and the
magnetization models implemented natively in ESPResSo.

ESPResSo evaluates these models inside the integrator loop, so they are
configured **once** through flat particle properties rather than driven by a
per-step python loop. All models share the same setup contract:

* the moment lives on a **virtual site** bound to a real anchor via
  ``vs_auto_relate_to``, propagated as
  ``Propagation.TRANS_VS_RELATIVE | Propagation.ROT_VS_INDEPENDENT``;
* a particle may enable **at most one** model;
* the driving field is ``H_ext + dip_fld``, i.e. the sum of all
  ``HomogeneousMagneticField`` constraints plus the local dipolar field.

Two models are currently exposed, both parameterised by the saturation moment
``dipm_sat`` (:math:`m_{sat}`, must be > 0) and the initial susceptibility
``mag_susc_0`` (:math:`\\chi_0`, must be >= 0):

``langevin``
    :math:`m = m_{sat} L(\\alpha) \\hat{H}` with :math:`\\alpha = 3 \\chi_0 |H| / m_{sat}`.
``froelich_kennelly``
    :math:`m = \\chi_0 m_{sat} / (m_{sat} + \\chi_0 |H|) H`.

.. note::
    The retired ``Simulation.magnetize`` used :math:`\\alpha = m |H|`. To reproduce
    it with the ``langevin`` model set ``mag_susc_0 = dipm_sat**2 / 3``. The retired
    ``Simulation.magnetize_froelich_kennelly`` maps one to one, with its ``Xi``
    becoming ``mag_susc_0`` and its ``dip_magnitude`` becoming ``dipm_sat``.
'''
import numpy as np

import espressomd
import espressomd.version

from pressomancy.helper_functions import MissingFeature, api_agnostic_feature_check

if espressomd.version.major() == 5:
    import espressomd.propagation
    Propagation = espressomd.propagation.Propagation

#: Features every magnetodynamics model needs, whichever model is picked.
COMMON_FEATURES = ['DIPOLES', 'DIPOLE_FIELD_TRACKING', 'VIRTUAL_SITES_RELATIVE']

#: model name -> (espresso feature, per-particle enable flag)
MAGNETIZATION_MODELS = {
    'langevin': ('LANGEVIN_MAGNETIZATION', 'langevin_magnetization_is_enabled'),
    'froelich_kennelly': ('FROELICH_KENNELLY', 'froelich_kennelly_is_enabled'),
}


def validate_model(model):
    '''
    Checks that a magnetization model name is one this module knows about.

    :param model: str | key of MAGNETIZATION_MODELS
    :return: str | the validated model name
    :raises ValueError: if the name is unknown
    '''
    if model not in MAGNETIZATION_MODELS:
        raise ValueError(
            f"Unknown magnetization model '{model}'. "
            f"Valid models: {sorted(MAGNETIZATION_MODELS.keys())}.")
    return model


def susceptibility_from_kT(dipm, kT):
    '''
    Initial susceptibility of a thermally fluctuating point dipole.

    A superparamagnetic particle whose moment has fixed magnitude ``dipm`` and
    reorients freely against thermal noise follows the classical Langevin law

    .. math::
        m(H) = \\mathrm{dipm} \\cdot L(\\alpha), \\qquad \\alpha = \\frac{\\mathrm{dipm} \\cdot H}{k_B T}

    Espresso instead parameterises its Langevin model by the initial
    susceptibility, :math:`\\alpha = 3 \\chi_0 H / m_{sat}`. Matching the two
    exponents of the field gives the slope of the curve at vanishing field,

    .. math::
        \\chi_0 = \\left. \\frac{\\mathrm{d}m}{\\mathrm{d}H} \\right|_{H \\to 0}
                = \\frac{\\mathrm{dipm}^2}{3 k_B T}

    The moment enters squared: one factor turns the field into an energy that
    competes with :math:`k_B T`, the other turns the resulting alignment back
    into a moment.

    :param dipm: float | magnitude of the point dipole, must be > 0
    :param kT: float | thermal energy, must be > 0

    :return: float | the initial susceptibility to hand to espresso as mag_susc_0

    :raises ValueError: if dipm or kT is not strictly positive
    '''
    if dipm <= 0.:
        raise ValueError(f"dipm must be > 0, got {dipm}.")
    if kT <= 0.:
        raise ValueError(f"kT must be > 0, got {kT}.")
    return dipm * dipm / (3. * kT)


def required_features_for(model):
    '''
    Features an object must declare to use a given magnetization model.

    :param model: str | key of MAGNETIZATION_MODELS
    :return: list(str)
    '''
    feature, _ = MAGNETIZATION_MODELS[validate_model(model)]
    return COMMON_FEATURES + [feature]


def configure_magnetization(part_hndl, model, dipm_sat, mag_susc_0, anchor=None):
    '''
    Makes a particle magnetizable under one of ESPResSo's magnetization models.

    Sets up the virtual-site binding and propagation mode if an anchor is given,
    then writes the model parameters and enables exactly one model. Any other
    model already enabled on the particle is switched off first, so the
    one-model-per-particle rule is enforced here rather than surfacing as a
    deferred runtime error when the integrator starts.

    :param part_hndl: ParticleHandle | the virtual site that carries the moment
    :param model: str | key of MAGNETIZATION_MODELS
    :param dipm_sat: float | saturation moment, must be > 0
    :param mag_susc_0: float | initial susceptibility, must be >= 0
    :param anchor: ParticleHandle (=None) | real particle to bind to. If None the
        particle is assumed to be virtual already and is left bound as it is.

    :return: None

    :raises ValueError: on an unknown model or out of range parameters
    :raises MissingFeature: if the ESPResSo build lacks the required features
    '''
    validate_model(model)
    feature, enable_flag = MAGNETIZATION_MODELS[model]

    missing = [f for f in required_features_for(model)
               if not api_agnostic_feature_check(f)]
    if missing:
        raise MissingFeature(
            f"Magnetization model '{model}' requires features: "
            f"{required_features_for(model)}.\n"
            f"Missing required features: {', '.join(missing)}.")

    if dipm_sat <= 0.:
        raise ValueError(
            f"dipm_sat must be > 0, got {dipm_sat}. A particle with no "
            f"saturation moment cannot be magnetized.")
    if mag_susc_0 < 0.:
        raise ValueError(f"mag_susc_0 must be >= 0, got {mag_susc_0}.")

    if anchor is not None:
        part_hndl.vs_auto_relate_to(anchor)
    if espressomd.version.major() == 5:
        part_hndl.propagation = (Propagation.TRANS_VS_RELATIVE |
                                 Propagation.ROT_VS_INDEPENDENT)

    for other_model, (other_feature, other_flag) in MAGNETIZATION_MODELS.items():
        if other_model != model and api_agnostic_feature_check(other_feature):
            setattr(part_hndl, other_flag, False)

    part_hndl.dipm_sat = dipm_sat
    part_hndl.mag_susc_0 = mag_susc_0
    setattr(part_hndl, enable_flag, True)


def contraction_ratio(system, part_list, n_iter=50, tol=1e-12):
    '''
    Measures how fast espresso's mutual magnetization iteration contracts.

    Particles magnetize each other explicitly: the dipolar field a particle
    responds to is the one computed in the *previous* force calculation, so the
    moments approach their self consistent value over several timesteps rather
    than within one. Whether they approach it at all depends on whether the map
    m <- m(H_ext + H_dip(m)) is a contraction.

    ``system.integrator.run(0, recalc_forces=True)`` performs exactly one such
    iterate at frozen positions: it runs the pre loop block, which updates the
    virtual sites, the magnetization and the forces that refill dip_fld, and
    then skips the integration loop because zero steps were asked for. Calling
    it repeatedly therefore advances the fixed point iteration without advancing
    the simulation, and the ratio of successive moment increments is the
    empirical amplification factor of the map.

    Ratios that settle below 1 and shrink indicate convergence. A ratio that
    plateaus at or above 1, or moments that keep creeping towards saturation,
    mean the one iterate per timestep scheme is not resolving the physics for
    this combination of susceptibility, density and dipolar prefactor.

    .. warning::
        A small ratio on its own does **not** prove the result is the physical
        branch. Because the moment saturates, the differential susceptibility
        falls off with field, so the map also contracts around a spuriously
        saturated fixed point. Complement this with a field up and down sweep
        through the same values: if the two directions disagree at equal field,
        the branch was chosen by numerical accident rather than by physics.

    :param system: espressomd.System | the system holding the particles
    :param part_list: iterable(ParticleHandle) | the magnetizable particles to watch
    :param n_iter: int (=50) | number of iterates, must be at least 2
    :param tol: float (=1e-12) | increments at or below this are treated as
        converged. The series is truncated there, because once the moments stop
        changing to machine precision the ratio of two successive increments is
        numerical noise and drifts towards 1, which would read as a stalled
        iteration when in fact it converged.

    :return: np.ndarray | successive ratios over the resolvable part of the
        series. Empty if the fixed point was reached within the first iterate,
        which leaves no ratio to form.

    :raises ValueError: if n_iter < 2, tol is not positive, or part_list is empty
    '''
    if n_iter < 2:
        raise ValueError(f"n_iter must be at least 2 to form a ratio, got {n_iter}.")
    if tol <= 0.:
        raise ValueError(f"tol must be positive, got {tol}.")
    parts = list(part_list)
    if not parts:
        raise ValueError("part_list is empty, there is nothing to watch.")

    increments = []
    previous = None
    for _ in range(n_iter):
        system.integrator.run(0, recalc_forces=True)
        moments = np.array([p.dipm for p in parts])
        if previous is not None:
            increments.append(np.linalg.norm(moments - previous))
        previous = moments

    increments = np.asarray(increments)
    # drop everything from the first converged increment onwards, so the ratios
    # are formed only from increments that are still numerically resolvable
    converged = np.flatnonzero(increments <= tol)
    cut = int(converged[0]) if converged.size else len(increments)
    increments = increments[:cut]
    if len(increments) < 2:
        return np.empty(0)
    return increments[1:] / increments[:-1]
