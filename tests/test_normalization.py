"""
Regression tests for the normalization of the alignment models in alignment_functions/modeling.py
(compute_rel_e_model, get_rel_e_model, compute_auto_model), the k_z spline cache, and tau_to_AIA.

Until 2026-11 the E+ models were pi too high and the E++ model -4 pi (IA/parity/validation/norm_check/:
review/REPORT.md, external/report.md). These tests pin the corrected normalization to an external reference:
the finite-Pi_max Singh, Mandelbaum & More (2015) Eq. 9 kernel (and its II extension), computed by check 2's
external_refs.py on the L24 bins with the bundled AbacusSummit c000 z = 0.8 P(k) (tests/data/s15_reference.json,
written by tests/data/make_s15_reference.py, which records its provenance).

Fast tests use reduced k_z-spline sampling (n_samples = 220 log nodes over 1e-4 < K < 4 h/Mpc, where the bundled
P(k) is nonzero) and a 1% tolerance. The slow test uses the production defaults (n_samples = 1000, K and k_z up to
200) and a 0.1% tolerance. Splines and model evaluations run in a process pool (SPECIA_TEST_WORKERS, default
min(16, cpu count)).

HOW TO RUN (from anywhere)
--------------------------
    pytest tests/test_normalization.py -v                # fast tests only (< 1 min)
    pytest tests/test_normalization.py -v --runslow      # also the converged-sampling tests (a few minutes)
Set SPECIA_NORM_RESULTS=/path/results.json to also write every ratio computed here to that file.
"""
import concurrent.futures, json, multiprocessing, os, pickle, sys

import numpy as np
import pytest
from astropy.table import Table

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from alignment_functions import modeling as m

with open(os.path.join(REPO, 'tests', 'data', 's15_reference.json')) as f:
    REF = json.load(f)
CASES = ('L24', 'const200')
Z = 0.8                                     # z of the bundled P(k); z = PS_z, so no growth rescaling
B_GAL = 1.99                                # L24 LRG bias (only used by the sign test)
FAST_SPLINES = dict(PS_min=1e-4, PS_max=4.0, n_samples=220)
SLOW_SPLINES = dict(n_samples=1000)         # production defaults otherwise: PS_min = 1e-4, PS_max = 200
# normalization at spec-IA 2a92b45 (before the fix); the old compute_auto_model also returned -tau instead of tau^2
OLD_CONSTANTS = dict(_K_MEASURE=1.0, _LOS_PREFACTOR=1/np.pi, _II_ANGULAR=2.0)
N_WORKERS = int(os.environ.get('SPECIA_TEST_WORKERS', min(16, os.cpu_count() or 1)))


def record(name, values):
    """Append computed values to $SPECIA_NORM_RESULTS (if set), for reporting."""
    path = os.environ.get('SPECIA_NORM_RESULTS')
    if not path:
        return
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data[name] = np.asarray(values, float).tolist()
    with open(path, 'w') as f:
        json.dump(data, f, indent=1)


def case_arrays(case):
    c = REF['cases'][case]
    return (np.asarray(c['R_bin_min']), np.asarray(c['R_bin_max']), np.asarray(c['pimax']),
            np.asarray(c['unit_gp']), np.asarray(c['unit_pp']))


def tables(case, wp=None):
    lo, hi, pm = case_arrays(case)[:3]
    meas = Table(); meas['R_bin_min'], meas['R_bin_max'], meas['pimax'] = lo, hi, pm
    wpt = Table(); wpt['R_bin_min'], wpt['R_bin_max'] = lo, hi
    wpt['wp'] = np.zeros(len(lo)) if wp is None else wp
    return meas, wpt


# ------------------------------------------------------------------------------------------- process-pool jobs
def _spline_job(kind, pimax, directory, kw):
    fn = m.precompute_kz_integral if kind == 'gI' else m.precompute_kz_integral_II
    fn([pimax], m.abacus_ps_nl, directory, warning_handling='ignore', **kw)
    return kind, pimax, os.path.join(directory, m.kz_spline_filename(pimax, kind))


def _model_job(fn_name, args, kwargs, constants=None):
    """Call modeling.<fn_name>(*args, **kwargs) with optional normalization constants swapped in."""
    saved = {k: getattr(m, k) for k in (constants or {})}
    try:
        for k, v in (constants or {}).items():
            setattr(m, k, v)
        out = getattr(m, fn_name)(*args, warning_handling='ignore', **kwargs)
    finally:
        for k, v in saved.items():
            setattr(m, k, v)
    return out[1] if isinstance(out, tuple) else out


def _pool():
    ctx = multiprocessing.get_context('fork' if sys.platform.startswith('linux') else 'spawn')
    return concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx)


def make_splines(pool, directory, spline_kw):
    """Splines for every Pi_max of both cases -> {case: {'gI': [paths], 'II': [paths]}}."""
    jobs = {(kind, float(p)) for case in CASES for p in case_arrays(case)[2] for kind in ('gI', 'II')}
    done = {}
    for kind, p, path in pool.map(_spline_job, *zip(*[(k, p, directory, spline_kw) for k, p in sorted(jobs)])):
        done[(kind, p)] = path
    return {case: {kind: sorted({done[(kind, float(p))] for p in case_arrays(case)[2]}) for kind in ('gI', 'II')}
            for case in CASES}


def run_models(spline_kw, directory, extra=False):
    """Evaluate the models used by the tests, all in parallel. Returns {name: model array}."""
    with _pool() as pool:
        paths = make_splines(pool, directory, spline_kw)
        jobs = {}
        for case in CASES:
            lo, hi, pm = case_arrays(case)[:3]
            meas, wpt = tables(case)
            jobs[('E+', case)] = ('get_rel_e_model', (lo, hi, pm, np.zeros(len(lo)), 1.0),
                                  dict(z=Z, PS_z=Z, precomputed_kz_integral_paths=paths[case]['gI']), None)
            jobs[('E++', case)] = ('compute_auto_model', (meas, wpt),
                                   dict(z=Z, PS_z=Z, precomputed_kz_integral_paths=paths[case]['II']), None)
        if extra:
            lo, hi, pm = case_arrays('L24')[:3]
            wp = 60.0*(0.5*(lo + hi)/10.0)**-0.9        # any nonzero w_p, to exercise the wbar_p path too
            meas, wpt = tables('L24', wp=wp)
            gI = paths['L24']['gI']
            jobs[('get_wp', 'L24')] = ('get_rel_e_model', (lo, hi, pm, wp, 1.0),
                                       dict(z=Z, PS_z=Z, precomputed_kz_integral_paths=gI), None)
            jobs[('compute_wp', 'L24')] = ('compute_rel_e_model', (meas, wpt, pm, 1.0),
                                           dict(z=Z, PS_z=Z, PS_max=100, precomputed_kz_integral_paths=gI), None)
            jobs[('E+ old', 'L24')] = jobs[('E+', 'L24')][:3] + (OLD_CONSTANTS,)
            jobs[('E++ old', 'L24')] = jobs[('E++', 'L24')][:3] + (OLD_CONSTANTS,)
        futures = {key: pool.submit(_model_job, *job) for key, job in jobs.items()}
        out = {key: np.asarray(f.result()) for key, f in futures.items()}
    out['paths'] = paths
    return out


@pytest.fixture(scope='module')
def fast(tmp_path_factory):
    return run_models(FAST_SPLINES, str(tmp_path_factory.mktemp('kz_fast')), extra=True)


@pytest.fixture(scope='module')
def slow(tmp_path_factory):
    return run_models(SLOW_SPLINES, str(tmp_path_factory.mktemp('kz_slow')))


def ratios_to_s15(results, case):
    """spec-IA template / S15 template, per bin, for E+ and E++ (tau = 1, b = 1, wbar_p = 0)."""
    lo, hi, pm, gp, pp = case_arrays(case)
    return results[('E+', case)] / (-gp/(2*pm)), results[('E++', case)] / (pp/(2*pm))


# ------------------------------------------------------------------------------------------- fast tests
@pytest.mark.parametrize('case', CASES)
def test_eplus_matches_s15(fast, case):
    r_plus, _ = ratios_to_s15(fast, case)
    record(f'fast/{case}/E+_over_S15', r_plus)
    assert np.all(np.abs(r_plus - 1) < 0.01), r_plus


@pytest.mark.parametrize('case', CASES)
def test_eplusplus_matches_s15(fast, case):
    _, r_pp = ratios_to_s15(fast, case)
    record(f'fast/{case}/E++_over_S15', r_pp)
    assert np.all(np.abs(r_pp - 1) < 0.01), r_pp


def test_old_over_new_is_pi(fast):
    """Old normalization (constants of 2a92b45) / new, same splines and code path otherwise: pi exactly for E+,
    and -4 pi for E++ (the old E++ returned -tau x template, so at tau = 1 it is minus the patched template)."""
    r_plus = fast[('E+ old', 'L24')] / fast[('E+', 'L24')]
    r_pp = -fast[('E++ old', 'L24')] / fast[('E++', 'L24')]
    record('fast/L24/old_over_new_E+', r_plus)
    record('fast/L24/old_over_new_E++', r_pp)
    np.testing.assert_allclose(r_plus, np.pi, rtol=1e-9, atol=0)
    np.testing.assert_allclose(r_pp, -4*np.pi, rtol=1e-9, atol=0)


def test_get_rel_e_model_equals_compute_rel_e_model(fast):
    a, b = fast[('get_wp', 'L24')], fast[('compute_wp', 'L24')]
    record('fast/L24/get_over_compute', a/b)
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=0)


def test_sign_positive_for_radial_alignment(fast):
    """A_IA > 0 (radial alignment) -> tau < 0 -> E+ > 0 in every bin; E++ > 0 for any tau."""
    tau = m.AIA_to_tau(1.0, Z, cosmology=m.cosmo_c000)
    assert tau < 0
    for case in CASES:
        assert np.all(tau * B_GAL * fast[('E+', case)] > 0)
        assert np.all(tau**2 * fast[('E++', case)] > 0)


def test_splines_are_versioned_and_start_at_kz_zero(fast):
    for path in fast['paths']['L24']['gI'] + fast['paths']['L24']['II']:
        assert '_' + m._KZ_TAG + '_pimax_' in os.path.basename(path)
        with np.load(path, allow_pickle=False) as d:
            assert int(d['version']) == m.KZ_SPLINE_VERSION
            assert float(d['kz_min']) == 0.0
            assert int(d['n_samples']) == FAST_SPLINES['n_samples']


def test_stale_and_mismatched_splines_are_rejected(fast, tmp_path):
    lo, hi, pm = case_arrays('L24')[:3]
    meas, wpt = tables('L24')
    good_gI, good_II = fast['paths']['L24']['gI'], fast['paths']['L24']['II']
    # a legacy (version-1) spline: a pickled tck in a *_pimax_<P>.npy file, the format before the fix
    legacy = str(tmp_path / ('kz_integral_NL_spl_pimax_' + str(round(float(pm[0]), 2)) + '.npy'))
    with np.load(good_gI[0], allow_pickle=False) as d:
        tck = (d['t'], d['c'], int(d['k']))
    with open(legacy, 'wb') as f:
        pickle.dump(tck, f)
    bundled = sorted(str(p) for p in m.kz_spline_parent_paths.glob('kz_integral_NL_spl_pimax_*.npy'))[:3]
    calls = [lambda paths: m.get_rel_e_model(lo, hi, pm, np.zeros(len(lo)), 1.0, precomputed_kz_integral_paths=paths),
             lambda paths: m.compute_rel_e_model(meas, wpt, pm, 1.0, precomputed_kz_integral_paths=paths)]
    for call in calls:
        for paths in ([legacy] + good_gI, bundled, good_II, []):
            with pytest.raises(ValueError):
                call(paths)
    for paths in ([legacy] + good_II, good_gI):
        with pytest.raises(ValueError):
            m.compute_auto_model(meas, wpt, precomputed_kz_integral_paths=paths)
    # the module default never picks up legacy files
    assert all('_' + m._KZ_TAG + '_pimax_' in str(p) for p in m.kz_spline_paths)


# ------------------------------------------------------------------------------------------- tau <-> A_IA
def test_tau_of_unit_AIA_standard():
    tau = m.AIA_to_tau(1.0, Z, cosmology=m.cosmo_c000)               # C1 rho_crit = 0.0134, Omega_m,0, D(0) = 1
    record('tau_AIA1/standard', [tau])
    assert abs(tau / -0.006359 - 1) < 1e-4, tau
    # same thing written out
    assert abs(tau - (-0.0134 * m.cosmo_c000.Om0 / m.growth_factor(Z, m.cosmo_c000))) < 1e-15
    # C1 rho_crit is overridable (CCL / CosmoSIS: 5e-14 rho_crit = 0.013877)
    tau_ccl = m.AIA_to_tau(1.0, Z, cosmology=m.cosmo_c000, C1rho_crit=0.013877)
    assert abs(tau_ccl / tau - 0.013877/0.0134) < 1e-12


def test_tau_of_unit_AIA_L24():
    tau = m.AIA_to_tau(1.0, Z, convention='L24')
    record('tau_AIA1/L24', [tau])
    assert abs(tau / -0.018703 - 1) < 1e-4, tau


def test_tau_AIA_roundtrip_and_argument_checks():
    for kw in (dict(cosmology=m.cosmo_c000), dict(convention='L24'), dict(convention='L24', norm_at_z0=True)):
        z = np.array([0.1, 0.5, 1.3])
        np.testing.assert_allclose(m.tau_to_AIA(m.AIA_to_tau(np.array([1.0, 5.0, -2.0]), z, **kw), z, **kw),
                                   [1.0, 5.0, -2.0], rtol=1e-12)
    for kw in (dict(), dict(cosmology=m.cosmo_c000, norm_at_z0=True), dict(convention='L24', cosmology=m.cosmo_c000),
               dict(convention='L24', C1rho_crit=0.0134), dict(convention='other', cosmology=m.cosmo_c000)):
        with pytest.raises(ValueError):
            m.tau_to_AIA(-0.1, Z, **kw)


def test_growth_factor_c000():
    assert m.growth_factor(0.0, m.cosmo_c000) == pytest.approx(1.0, abs=1e-12)
    assert m.growth_factor(Z, m.cosmo_c000) == pytest.approx(0.66419, abs=1e-5)    # check-2: matches the Abacus headers


# ------------------------------------------------------------------------------------------- slow test
@pytest.mark.slow
@pytest.mark.parametrize('case', CASES)
def test_eplus_matches_s15_converged(slow, case):
    r_plus, r_pp = ratios_to_s15(slow, case)
    record(f'slow/{case}/E+_over_S15', r_plus)
    record(f'slow/{case}/E++_over_S15', r_pp)
    assert np.all(np.abs(r_plus - 1) < 1e-3), r_plus


@pytest.mark.slow
@pytest.mark.parametrize('case', CASES)
def test_eplusplus_matches_s15_converged(slow, case):
    _, r_pp = ratios_to_s15(slow, case)
    assert np.all(np.abs(r_pp - 1) < 1e-3), r_pp
