import os
from src.extendedHubbard import run_idmrg
import numpy as np
import logging
from utils.slurmhelpers import ret_arr_for_taskid
logging.basicConfig(level=logging.INFO)

import utils.io as io


t = 1.0
U = 0.1
Vpp = 0.8
dVlist = np.array([2.0, 2.6, 2.9, 3.0, 3.1, 3.05, 3.2, 3.3, 3.4])
Vpmlist = dVlist + Vpp

task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
n_tasks = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))

arr_for_task  = ret_arr_for_taskid(Vpmlist, n_tasks, task_id)

ROOTDIR = '../data/iDMRG/'
if not os.path.exists(ROOTDIR):
    print("MADE ROOTDIR : ", ROOTDIR)
    os.makedirs(ROOTDIR, exist_ok=True)

JOBDIR = os.path.join(ROOTDIR, f"t_{t:.2f}_U_{U:.2f}_Vpp_{Vpp:.3f}")
os.makedirs(JOBDIR, exist_ok=True)

chimax = 100
for Vpmval in arr_for_task:
    savefilename = os.path.join(JOBDIR, f"Vpm_{Vpmval:.4f}_chimax_{chimax}.h5")
    _, psi, _, results = run_idmrg(t=t, U=U, Vpp=Vpp, Vpm=Vpmval, chi_max=chimax, n_sweeps=1000, max_err=1e-5, verbose=True)
    io.save_mps_with_metadata(savefilename, psi, results)
    