import os
# from src.consNfextendedHubbard import scan_deltaV
from src.scanner_versions import scanner_v1
import numpy as np
import logging
from utils.slurmhelpers import ret_arr_for_taskid
logging.basicConfig(level=logging.INFO)

import utils.io as io


# t = 1.0
# U = 0.1
# Vpp = 0.8
# dVlist = np.arange(3.0, 4.1, 0.1)
vals = np.concatenate([
    np.arange(5.9, 4.2, -0.1),          # up to 3.3
    np.arange(4.2, 3.7, -0.05),         # 3.40 ... 3.59
    np.arange(3.7, 3.50 ,-0.01),        # 3.6 ... 5.0
])

vals = np.unique(np.round(vals, 2))[::-1]
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

task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
chilist = [100, 200]
chi_max = chilist[task_id]
n_sweeps = 1200
max_err = (1e-7, 1e-5)

restartditct = {
    100 : 5.9, 
    200 : 5.9,
}
restrtartV = restartditct[chi_max]
scanVlist = dVlist[(dVlist - restrtartV) > 1e-8] 

loadfiledict = {
    200 : 'Vpm6.0cc/Vpm_6.000_chi200.h5',
    100 : 'Vpm6.0cc/Vpm_6.000_chi100.h5'
}
loadfile = os.path.join(ROOTDIR, loadfiledict[chi_max])
# loadfile = os.path.join(ROOTDIR, f'Vpm3.8cc/Vpm_3.800_chi{chi_max}.h5')
psi_init, metadata_init = io.load_mps_with_metadata(loadfile)
print("Loaded initial state from ", loadfile)

print("STARTING scan_deltaV", flush=True)

# scan_deltaV(t, U, Vpp, dVlist, chimax, nsweeps, max_err, saveflag=True, diagnostics=True, ROOTDIR=ROOTDIR)
scanner_v1(dV_values=scanVlist, chi_max=chi_max, hz=-1e-10, psi_init=psi_init,
           n_sweeps=n_sweeps, max_err=max_err, saveflag=True, ROOTDIR=os.path.join(ROOTDIR, 'rev/'))

print("ENDED scan_deltaV", flush=True)