"""
Write tests/data/s15_reference.json: the finite-Pi_max Singh, Mandelbaum & More (2015) reference kernels that
tests/test_normalization.py compares the spec-IA E+ and E++ models against.

Source (ground truth): the check-2 normalization validation, IA/parity/validation/norm_check/external/
  external_refs.py  ->  external_refs.npz   (keys S15/{L24,const200}/unit_{gp,pp})
This script copies those saved values and re-runs external_refs.s15_unit (with the same default settings) to
confirm that the script on disk still reproduces them. It refuses to write the file if they differ by > 1e-10.

S15 kernels (annulus-averaged over each R bin, unit amplitude):
  unit_gp = (1/pi^2) int dk_z int dK K^3/(k^2 k_z) P(k) sin(k_z Pi_max) Jbar_2(K)
  unit_pp = (1/pi^2) int dk_z int dK K (K^2/k^2)^2/k_z P(k) sin(k_z Pi_max) (Jbar_0 + Jbar_4)/2   (II extension of S15 Eq. 9)
so that, for b_gal = 1, wbar_p = 0 and tau = 1, the spec-IA templates should be
  E+  (get_rel_e_model / compute_rel_e_model) = -unit_gp / (2 Pi_max)
  E++ (compute_auto_model)                    = +unit_pp / (2 Pi_max)

Run with desiconda python (numpy/scipy/pandas/astropy only; pyccl, mcfit, hankl are stubbed because s15_unit
does not use them):
  /global/common/software/desi/perlmutter/desiconda/20220119-2.0.1/conda/bin/python tests/data/make_s15_reference.py
"""
import datetime, hashlib, importlib, json, os, subprocess, sys, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXT = '/global/homes/c/clamman/IA/parity/validation/norm_check/external/'
OUT = os.path.join(HERE, 's15_reference.json')
S15_SETTINGS = dict(dK=2.5e-4, Kmax=4.0, dkz=2e-4, kzmax=4.0, kzmin=0.0)   # s15_unit defaults (asserted below)


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def mtime(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec='seconds')


def git(*args):
    return subprocess.check_output(['git', '-C', REPO] + list(args)).decode().strip()


# s15_unit needs only numpy/scipy; external_refs imports the external codes at module level
for name in ('pyccl', 'mcfit', 'hankl'):
    stub = types.ModuleType(name)
    stub.__version__, stub.__file__ = 'stub (not used by s15_unit)', '<stub>'
    sys.modules[name] = stub
sys.path.insert(0, EXT)
common = importlib.import_module('common')
ext = importlib.import_module('external_refs')

import inspect
defaults = {k: v.default for k, v in inspect.signature(ext.s15_unit).parameters.items() if k in S15_SETTINGS}
assert defaults == S15_SETTINGS, defaults

saved = np.load(EXT + 'external_refs.npz')
cases = common.cases()
lo, hi, pm_l24 = cases['L24']
assert np.allclose(cases['const200'][2], 200.0)

out = {'cases': {}}
max_dev = 0.0
for cname, (lo_c, hi_c, pm_c) in cases.items():
    entry = {'R_bin_min': lo_c.tolist(), 'R_bin_max': hi_c.tolist(), 'pimax': pm_c.tolist()}
    for kernel, power in (('unit_gp', 1), ('unit_pp', 2)):
        ref = saved[f'S15/{cname}/{kernel}']
        rerun = ext.s15_unit(lo_c, hi_c, pm_c, power=power)
        dev = float(np.max(np.abs(rerun/ref - 1)))
        max_dev = max(max_dev, dev)
        print(f'{cname:9s} {kernel}: rerun/saved - 1 = {dev:.1e}', flush=True)
        entry[kernel] = ref.tolist()
    out['cases'][cname] = entry
if max_dev > 1e-10:
    sys.exit(f'external_refs.s15_unit no longer reproduces external_refs.npz (max deviation {max_dev:.1e}); not writing')

pk = os.path.join(REPO, 'example_catalogs/AbacusSummit_base_c000_z0.800_power_nfft2048.csv')
out['provenance'] = {
    'generated': datetime.datetime.now().isoformat(timespec='seconds'),
    'generator': 'tests/data/make_s15_reference.py',
    'spec_ia_commit_at_generation': git('rev-parse', 'HEAD'),
    'note': ('IA/parity/validation/norm_check is not under version control, so the check-2 generator has no commit; '
             'its files are identified by sha256 and mtime below. The values are the saved check-2 outputs; this '
             'generator re-ran external_refs.s15_unit and required agreement to 1e-10 before writing.'),
    'rerun_max_relative_deviation': max_dev,
    'source_npz': {'path': EXT + 'external_refs.npz', 'sha256': sha256(EXT + 'external_refs.npz'),
                   'mtime': mtime(EXT + 'external_refs.npz'), 'keys': 'S15/{L24,const200}/unit_{gp,pp}'},
    'source_scripts': {name: {'path': EXT + name, 'sha256': sha256(EXT + name), 'mtime': mtime(EXT + name)}
                       for name in ('external_refs.py', 'common.py')},
    'check2_report': EXT + 'report.md',
    'power_spectrum': {'file': 'example_catalogs/AbacusSummit_base_c000_z0.800_power_nfft2048.csv',
                       'sha256': sha256(pk), 'last_commit': git('log', '-1', '--format=%H', '--', pk),
                       'used_as': ('common.P_L24: cubic spline of log10 P vs log10 k through the file (s=0), the same '
                                   'function precompute_kz_integral builds; tabulated on 2,000,001 log-k points over '
                                   '1e-5..10 h/Mpc inside s15_unit. It is zero outside ~[0.003, 3.5] h/Mpc.')},
    'L24_bins': ('R_bin_min/R_bin_max/pimax of the L24 (DR1) LRG multiplet measurement, read by common.l24_bins() '
                 'from ' + common.L24_TABLE + '; pimax = 6 + (2/3) R_mid'),
    's15_settings': S15_SETTINGS,
    's15_convergence': ('check-2: halving dK and dk_z and raising Kmax/kzmax 4 -> 5 changes it by < 1e-6; '
                        'OT20 Eq. 11 agrees to 1e-5'),
    'definitions': {
        'unit_gp': '(1/pi^2) int dk_z int dK K^3/(k^2 k_z) P(k) sin(k_z Pi_max) Jbar_2(K), annulus averaged',
        'unit_pp': '(1/pi^2) int dk_z int dK K (K^2/k^2)^2/k_z P(k) sin(k_z Pi_max) [Jbar_0 + Jbar_4]/2',
        'E+ template (tau=1, b=1, wbar_p=0)': '-unit_gp / (2 pimax)',
        'E++ template (tau=1, wbar_p=0)': '+unit_pp / (2 pimax)',
    },
}
with open(OUT, 'w') as f:
    json.dump(out, f, indent=1)
print('wrote', OUT)
