### FROM CLAUDE ###

"""Regression test: the two alignment code paths should compute the same
estimator.

The estimator is

    <E * cos(2 Phi)>_w  =  sum(w * E * cos(2 Phi)) / sum(w)

where the average is over tracer-shape pairs in a projected-separation bin,
E is the shape magnitude (E_ABS), Phi is the angle between the shape
orientation and the projected separation vector, and w is the tracer weight.

Two code paths exist that must agree on this quantity:

  Path A: rel_angle_regions -> bin_region_results -> bin_results
          (the default path, used by get_multiplet_alignment)

  Path B: rel_angle_regions_binned -> calculate_rel_ang_cartesian_binAverage
          (the memory-saver path, used by get_multiplet_alignment with
          early_binning=True and by get_3D_MIA_from3D_autocorr)

When E_ABS=1 for all shapes (stick model), these paths agree out of the box.
When E_ABS != 1 (full-shape alignment), Path A silently drops E_ABS -- which
is the bug this test guards against.

HOW TO RUN
----------
With pytest:
    pytest test_estimator_consistency.py -v

As a plain script:
    python test_estimator_consistency.py

Expected behavior
-----------------
* test_stick_model_agreement   -- should ALWAYS pass (no E_ABS involvement)
* test_full_shape_agreement    -- FAILS before the rel_angle_regions fix,
                                  PASSES after
"""
import sys, os
sys.path.insert(0, '/global/homes/c/clamman/IA/spec-IA')

import numpy as np
from astropy.table import Table

from alignment_functions.gal_multiplets import make_group_catalog
from alignment_functions.basic_alignment import (
    rel_angle_regions,
    bin_region_results,
    rel_angle_regions_binned,
)
from geometry_functions.coordinate_functions import get_cosmo_points

# Path to the small example catalog that ships with the repo.
# Adjust if your test runner uses a different working directory.
EXAMPLE_CATALOG = os.environ.get(
    'SPEC_IA_TEST_CATALOG', '/global/homes/c/clamman/IA/spec-IA/example_catalogs/abacus_simpleMock.fits',
)

# Tolerance. Path A and Path B use slightly different region slicings and
# treat pairs on region boundaries differently, so agreement is at the
# statistical level (~3-4 decimal places on the example catalog), not
# bit-identical. 1e-3 catches any real divergence while staying robust to
# boundary-pair differences.
ATOL = 1e-3


# ---- fixture-like helpers (plain functions so pytest isn't required) ----

_cached_catalog = None

def _get_test_catalog():
    """Load and cache the example group + tracer catalogs."""
    global _cached_catalog
    if _cached_catalog is not None:
        return _cached_catalog

    cat = Table.read(EXAMPLE_CATALOG)
    cat['center_loc'] = get_cosmo_points(cat)
    group_catalog = make_group_catalog(cat)

    tracer_positions = get_cosmo_points(cat)
    tracer_weights = np.asarray(cat['WEIGHT'])

    _cached_catalog = (group_catalog, tracer_positions, tracer_weights)
    return _cached_catalog


def _run_path_A(group_catalog, tracer_positions, tracer_weights, R_bins,
                pimax, n_regions):
    """Default path: rel_angle_regions -> bin_region_results.

    This wraps the current API. After the proposed fix, if group_catalog
    contains an 'E_ABS' column, it will be used automatically; if not,
    E_ABS=1 is assumed.
    """
    all_proj, all_pa, all_w = rel_angle_regions(
        group_catalog,
        loc_tracers=tracer_positions,
        tracer_weights=tracer_weights,
        n_regions=n_regions,
        pimax=pimax,
        max_proj_sep=R_bins[-1],
    )
    _, pa_av, _, _ = bin_region_results(all_proj, all_pa, all_w, R_bins=R_bins)
    return pa_av


def _run_path_B(group_catalog, tracer_positions, tracer_weights, R_bins,
                pimax, n_regions, use_E_ABS):
    """Memory-saver path: rel_angle_regions_binned."""
    pa_av, _, _ = rel_angle_regions_binned(
        group_catalog,
        loc_tracers=tracer_positions,
        tracer_weights=tracer_weights,
        R_bins=R_bins,
        pimax=pimax,
        n_regions=n_regions,
        use_E_ABS=use_E_ABS,
        print_progress=False,
    )
    return pa_av


# ------------------------- the actual tests -------------------------

def test_stick_model_agreement():
    """With E_ABS=1 everywhere (stick model, the most common case), the two
    paths must agree to ATOL. This should pass both before and after the fix
    for the full-shape case.
    """
    group_catalog, tracer_positions, tracer_weights = _get_test_catalog()

    # Ensure no E_ABS is present, so Path B uses its default (E_ABS=1)
    if 'E_ABS' in group_catalog.colnames:
        group_catalog = group_catalog.copy()
        group_catalog.remove_column('E_ABS')

    # Use a modest R-range to keep the default path's memory use bounded.
    R_bins = np.logspace(0, 1.3, 6)   # ~1 to ~20 Mpc/h
    pimax = 30

    pa_A = _run_path_A(group_catalog, tracer_positions, tracer_weights,
                       R_bins, pimax, n_regions=4)
    pa_B = _run_path_B(group_catalog, tracer_positions, tracer_weights,
                       R_bins, pimax, n_regions=4, use_E_ABS=False)

    print(f'  Path A (default):    {np.round(pa_A, 5)}')
    print(f'  Path B (binAverage): {np.round(pa_B, 5)}')
    print(f'  max abs diff:        {np.max(np.abs(pa_A - pa_B)):.2e}')

    assert np.allclose(pa_A, pa_B, atol=ATOL, equal_nan=True), (
        f'Stick-model estimator disagrees between the two paths by more '
        f'than {ATOL}. Max diff: {np.max(np.abs(pa_A - pa_B)):.3e}'
    )


def test_full_shape_agreement():
    """With non-trivial E_ABS, the two paths must compute the SAME
    full-shape estimator <E * cos(2 Phi)>_w.

    This test FAILS on the current rel_angle_regions implementation, which
    silently drops E_ABS, and PASSES once rel_angle_regions is fixed to
    pass abs_e through to calculate_rel_ang_cartesian.
    """
    group_catalog, tracer_positions, tracer_weights = _get_test_catalog()

    # Inject a non-trivial, reproducible E_ABS column
    group_catalog = group_catalog.copy()
    rng = np.random.default_rng(0)
    group_catalog['E_ABS'] = rng.uniform(
        0.1, 0.9, size=len(group_catalog)
    ).astype(np.float32)

    R_bins = np.logspace(0, 1.3, 6)
    pimax = 30

    pa_A = _run_path_A(group_catalog, tracer_positions, tracer_weights,
                       R_bins, pimax, n_regions=4)
    pa_B = _run_path_B(group_catalog, tracer_positions, tracer_weights,
                       R_bins, pimax, n_regions=4, use_E_ABS=True)

    print(f'  Path A (default):    {np.round(pa_A, 5)}')
    print(f'  Path B (binAverage): {np.round(pa_B, 5)}')
    print(f'  max abs diff:        {np.max(np.abs(pa_A - pa_B)):.2e}')

    # Diagnostic: the expected failure mode before the fix is Path A
    # computing roughly <cos(2 Phi)> instead of <E * cos(2 Phi)>,
    # so Path A's values come out larger by ~1/<E> ~ 2x.
    mean_E = float(np.mean(group_catalog['E_ABS']))
    ratio = np.mean(np.abs(pa_A) / np.maximum(np.abs(pa_B), 1e-10))
    hint = ''
    if ratio > 1.5:
        hint = (
            f'\n  DIAGNOSTIC: mean |pa_A| / mean |pa_B| = {ratio:.2f}, '
            f'1/<E_ABS> = {1/mean_E:.2f}. These being close means Path A '
            f'is computing <cos(2 Phi)> while Path B is computing '
            f'<E * cos(2 Phi)>, i.e. the E_ABS-through-rel_angle_regions '
            f'fix has not been applied.'
        )

    assert np.allclose(pa_A, pa_B, atol=ATOL, equal_nan=True), (
        f'Full-shape estimator disagrees between the two paths by more '
        f'than {ATOL}. Max diff: {np.max(np.abs(pa_A - pa_B)):.3e}.{hint}'
    )


def test_estimator_definition():
    """Directly verify that Path B computes sum(w*E*cos(2Phi)) / sum(w),
    i.e. an average of the per-pair quantity E*cos(2Phi) weighted by w,
    NOT weighted by w*E.

    This pins down the estimator definition. If someone "fixes" the
    denominator to include E, this test fails -- serving as a guard.

    We do this by comparing Path B on (E_ABS = const) vs (E_ABS = 1) with
    the same catalog: the first should give const * the second.
    """
    group_catalog, tracer_positions, tracer_weights = _get_test_catalog()

    R_bins = np.logspace(0, 1.3, 6)
    pimax = 30
    const = 0.3

    # Stick-model run
    gc1 = group_catalog.copy()
    if 'E_ABS' in gc1.colnames:
        gc1.remove_column('E_ABS')
    pa_stick = _run_path_B(gc1, tracer_positions, tracer_weights,
                           R_bins, pimax, n_regions=4, use_E_ABS=False)

    # Constant-E_ABS run
    gc2 = group_catalog.copy()
    gc2['E_ABS'] = np.full(len(gc2), const, dtype=np.float32)
    pa_const = _run_path_B(gc2, tracer_positions, tracer_weights,
                           R_bins, pimax, n_regions=4, use_E_ABS=True)

    # If the estimator is sum(w*E*c) / sum(w), these should satisfy
    # pa_const = const * pa_stick exactly.
    # If the estimator were sum(w*E*c) / sum(w*E) instead, pa_const would
    # equal pa_stick, NOT const * pa_stick.
    expected = const * pa_stick
    print(f'  stick model:     {np.round(pa_stick, 5)}')
    print(f'  const E ({const}):    {np.round(pa_const, 5)}')
    print(f'  const * stick:   {np.round(expected, 5)}')
    print(f'  max diff:        {np.max(np.abs(pa_const - expected)):.2e}')

    assert np.allclose(pa_const, expected, atol=1e-6, equal_nan=True), (
        'binAverage path is not computing sum(w*E*cos(2Phi)) / sum(w). '
        f'Got pa_const = {pa_const}, expected {expected} '
        f'(= {const} * stick-model result).'
    )


# ---------- plain-script runner (so pytest is not required) ----------

if __name__ == '__main__':
    failures = []
    for name, fn in [
        ('test_estimator_definition', test_estimator_definition),
        ('test_stick_model_agreement', test_stick_model_agreement),
        ('test_full_shape_agreement', test_full_shape_agreement),
    ]:
        print(f'\n=== {name} ===')
        try:
            fn()
            print(f'PASS')
        except AssertionError as e:
            print(f'FAIL')
            print(f'  {e}')
            failures.append(name)

    print()
    if failures:
        print(f'{len(failures)} test(s) FAILED: {failures}')
        sys.exit(1)
    else:
        print('All tests PASSED')
        sys.exit(0)