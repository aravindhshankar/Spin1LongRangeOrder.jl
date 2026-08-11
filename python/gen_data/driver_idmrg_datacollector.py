#!/usr/bin/env python
"""
driver_idmrg_datacollector.py

Computes correlation-function observables for a batch of iDMRG-saved .h5
wavefunctions and dumps ONE pickle per input file into

    <DATA_ROOT>/<REQ_DATA>/corrs/tmp/<basename>.pkl

Designed to be launched as a SLURM array job (see
runner_idmrg_datacollector.slurm). Each array task processes a disjoint,
interleaved subset of the full file list, via
utils.slurmhelpers.ret_arr_for_taskid, so work is balanced even if runtime
per file varies.

To test locally (no SLURM), just run it directly -- it defaults to
task 0 of 1, i.e. all files:

    python driver_idmrg_datacollector.py

Re-running is safe/idempotent: files whose tmp output already exists are
skipped unless OVERWRITE = True below.
"""

import os
import sys
import time
import pickle
import argparse
import traceback

import numpy as np

# adjust this if your utils/ package lives somewhere else relative to this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils import io as io
from utils import obs as obs
from utils import slurmhelpers as sh

# ============================================================
# CONFIG -- edit this section to change what gets processed.
# Nothing below this block should need touching for a normal scan.
# ============================================================

DATA_ROOT = '../../data/iDMRG/'

# Add as many scan directories here as you like. Files across ALL of them
# are pooled and load-balanced together across the SLURM array tasks, so
# a slow scan doesn't starve a fast one.
REQ_DATA_LIST = [
    'scanChi800',
]

DISTRANGE = None   # None -> atomic_action figures it out per-file (0.9*xi)
STOREPSI = False    # keep False unless you really want to pickle the MPS too
OVERWRITE = False   # if False, skip files whose tmp output already exists

# ============================================================


def atomic_action(file, distrange=DISTRANGE, storepsi=STOREPSI):
    psi, metadata = io.load_mps_with_metadata(file)
    model_params = metadata['model_params']
    correlation_length = psi.correlation_length2()
    if not distrange:
        distrange = int(0.9 * correlation_length)
    charge_corr = obs.charge_corr(psi, distrange, connected=True)
    spinz_corr = obs.SzSz_corr(psi, distrange, connected=True)
    spspm_corr = obs.SpSm_corr(psi, distrange)
    electron_corr = obs.electron_corr(psi, distrange)
    _, spin2_corr = obs.op_correlation_idmrg(psi, xs=np.arange(1, distrange+1)) # just to make it the same length 
    row_dict = {
        'dV': model_params['Vpm'] - model_params['Vpp'],
        'mean_Sz': np.mean(psi.expectation_value("Sz")),
        'mean_fill': np.mean(psi.expectation_value("Ntot")),
        'model_params': model_params,
        'diagnostics': metadata['diagnostics'],
        'xi': correlation_length,
        'charge_corr': charge_corr,
        'spinz_corr': spinz_corr,
        'spsm_corr': spspm_corr,
        'electron_corr': electron_corr,
        'spin2_corr': spin2_corr,
    }
    if storepsi:
        row_dict['psi'] = psi
    return row_dict


def gather_jobs():
    """
    Returns a flat list of dicts across every REQ_DATA_LIST entry:
      {'req_data': ..., 'src': <path to .h5 mps file>, 'tmp_out': <path to .pkl output>}
    Also makes sure corrs/tmp/ exists for each req_data.
    """
    jobs = []
    for req_data in REQ_DATA_LIST:
        path_to_req_data = os.path.join(DATA_ROOT, req_data)
        tmp_dir = os.path.join(path_to_req_data, 'corrs', 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)

        h5_files = sorted(
            f for f in os.listdir(path_to_req_data) if f.endswith('.h5')
        )
        for f in h5_files:
            src = os.path.join(path_to_req_data, f)
            basename = os.path.splitext(f)[0]
            tmp_out = os.path.join(tmp_dir, basename + '.pkl')
            jobs.append({'req_data': req_data, 'src': src, 'tmp_out': tmp_out})
    return jobs


def process_job(job):
    src, tmp_out = job['src'], job['tmp_out']

    if not OVERWRITE and os.path.exists(tmp_out):
        print(f"[skip] {tmp_out} already exists")
        return

    t0 = time.time()
    try:
        row_dict = atomic_action(src, distrange=DISTRANGE, storepsi=STOREPSI)
        row_dict['src_file'] = src
    except Exception as e:
        print(f"[FAIL] {src}: {e}")
        traceback.print_exc()
        return

    # write-then-rename so a killed job never leaves a half-written .pkl
    # that combiner.py would choke on
    tmp_out_partial = tmp_out + '.partial'
    with open(tmp_out_partial, 'wb') as fh:
        pickle.dump(row_dict, fh)
    os.rename(tmp_out_partial, tmp_out)

    print(f"[ok] {src} -> {tmp_out} ({time.time() - t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', type=int,
                         default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    parser.add_argument('--task-count', type=int,
                         default=int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)))
    args = parser.parse_args()

    all_jobs = gather_jobs()
    my_jobs = sh.ret_arr_for_taskid(all_jobs, args.task_count, args.task_id)

    print(f"[task {args.task_id}/{args.task_count}] "
          f"{len(my_jobs)} / {len(all_jobs)} files assigned")

    for job in my_jobs:
        process_job(job)


if __name__ == '__main__':
    main()
