import os
from src.consNfextendedHubbard import scan_deltaV
import numpy as np
import logging
from utils.slurmhelpers import ret_arr_for_taskid
logging.basicConfig(level=logging.INFO)

import utils.io as io


t = 1.0
U = 0.1
Vpp = 0.8
dVlist = np.arange(3.0, 4.1, 0.1)
vals = np.concatenate([
    np.arange(3.0, 3.4, 0.1),          # up to 3.3
    np.arange(3.4, 3.6, 0.01),         # 3.40 ... 3.59
    np.arange(3.6, 5.0 + 0.1, 0.1),    # 3.6 ... 5.0
])

vals = np.unique(np.round(vals, 2))
dVlist = vals

# task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
# n_tasks = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))

# arr_for_task  = ret_arr_for_taskid(Vpmlist, n_tasks, task_id)


ROOTDIR = '../data/iDMRG/'
if not os.path.exists(ROOTDIR):
    print("MADE ROOTDIR : ", ROOTDIR)
    os.makedirs(ROOTDIR, exist_ok=True)

# JOBDIR = os.path.join(ROOTDIR, f"t_{t:.2f}_U_{U:.2f}_Vpp_{Vpp:.3f}")
# os.makedirs(JOBDIR, exist_ok=True)

chimax = 20
nsweeps = 1000
max_err = 1e-6

print("STARTING scan_deltaV", flush=True)

scan_deltaV(t, U, Vpp, dVlist, chimax, nsweeps, max_err, saveflag=True, diagnostics=True, ROOTDIR=ROOTDIR)

print("ENDED scan_deltaV", flush=True)