import os
from src.extendedHubbard import scan_deltaV
import numpy as np

t = 1.0
Ulist = [0.1, 0.34, 1.0, 5.0] #whatever
Vpplist = [0.1, 0.2, 0.4, 1.0, 2.6] #whatever

dV_values = np.arange(2,3,0.1) #whatever

task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
scan_deltaV(t=t, U=Ulist[task_id], Vpp=Vpplist[0], dV_values=None, chi_max=150, n_sweeps=8, max_err=1e-5)