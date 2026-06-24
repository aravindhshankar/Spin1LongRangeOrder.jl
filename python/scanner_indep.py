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
# dVlist = np.array([2.0, 2.6, 2.9, 3.0, 3.1, 3.05, 3.2, 3.3, 3.4])
vals = np.concatenate([
    np.arange(3.0, 3.4, 0.1),          # up to 3.3
    np.arange(3.4, 3.6, 0.01),         # 3.40 ... 3.59
    np.arange(3.6, 5.0 + 0.1, 0.1),    # 3.6 ... 5.0
])

vals = np.unique(np.round(vals, 2))
dVlist = vals
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
    

print("Loading from file : ", reload_psi_path) if verbose else None
psi, metadata = io.load_mps_with_metadata(filename=reload_psi_path)
chi_init = max(psi.chi)
print(f"Success! Loaded psi with bond dimension {chi_init}", flush=True) if verbose else None

REQ_DATA = 'v2fwd/t_1.00_U_0.10_Vpp_0.800_consNf0_25_chimax_100/'
PATH_TO_REQ_DATA = os.path.join(ROOTDIR, REQ_DATA)
savefiletemplate = os.path.join(JOBDIR, f"Vpm_{Vpm:.4f}_chimax_{chi_max}.h5")
file_list = [os.path.join(PATH_TO_REQ_DATA,f) for f in os.listdir(PATH_TO_REQ_DATA) if "chimax_100" in f]