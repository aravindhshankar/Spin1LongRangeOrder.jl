import os
# from src.extendedHubbard import run_idmrg
from src.consNfextendedHubbard import run_idmrg
import numpy as np
import logging
from utils.slurmhelpers import ret_arr_for_taskid, find_converged_psi
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

chimax = 400
# JOBDIR = os.path.join(ROOTDIR, f"t_{t:.2f}_U_{U:.2f}_Vpp_{Vpp:.3f}")
JOBDIR = os.path.join(ROOTDIR, f"v3fwdchi{chimax}/t_{t:.2f}_U_{U:.2f}_Vpp_{Vpp:.3f}_consNf0_25_chimax_{chimax}")
os.makedirs(JOBDIR, exist_ok=True)

mode = 'reload' 
psi = None

if mode == 'reload':
    REQ_DATA_LIST = ['v2fwdchi300/t_1.00_U_0.10_Vpp_0.800_consNf0_25_chimax_300/',
                    #'v2fwdchi200/t_1.00_U_0.10_Vpp_0.800_consNf0_25_chimax_200/',
                    'v2fwd/t_1.00_U_0.10_Vpp_0.800_consNf0_25_chimax_100/',
                    ]
    PATH_TO_REQ_DATA_LIST = [os.path.join(ROOTDIR, REQ_DATA) for REQ_DATA in REQ_DATA_LIST]
    file_list = [[os.path.join(PATH_TO_REQ_DATA,f) for f in os.listdir(PATH_TO_REQ_DATA) ] for PATH_TO_REQ_DATA in PATH_TO_REQ_DATA_LIST]
    for thresh in [0.01, 0.05, 0.1, ]:
        output = find_converged_psi(file_list, 4.9, thresh)
        try : 
            psi , _, bestvpm = output 
            print("best chi =", max(psi.chi))
            print("Closest Vpm : ", bestvpm, flush=True)
            break
        except Exception: 
            print("NO STARTING PSI FOUND at thresh : ", thresh)


for Vpmval in arr_for_task:
    savefilename = os.path.join(JOBDIR, f"Vpm_{Vpmval:.4f}_chimax_{chimax}.h5")
    _, psi, _, results = run_idmrg(t=t, U=U, Vpp=Vpp, Vpm=Vpmval, chi_max=chimax, n_sweeps=1000, max_err=1e-7, verbose=True, psi_init=psi)
    io.save_mps_with_metadata(savefilename, psi, results)