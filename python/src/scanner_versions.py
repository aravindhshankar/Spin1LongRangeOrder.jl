import numpy as np
import argparse
import numpy as np
import tenpy

from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg as tenpy_dmrg

import time
import sys
sys.path.append('..')
import utils.io as io
import os 
from .consNfextendedHubbard import HubbardSpinDepV

def scanner_v1(dV_values=None, t=1.0, U=0.1, Vpp=0.8, hz=-1e-14, hx=0.0,
               chi_max=1000, n_sweeps=1000, max_err=1e-7, 
               saveflag=True, psi_init=None, ROOTDIR='../../data/iDMRG/'):
    try : 
        max_E_err, max_S_err = max_err
    except (TypeError, ValueError):
        max_E_err, max_S_err = max_err, max_err

    if not psi_init:
        print("PSI_INIT NOT PROVIDED, making product state with S=0 instead")
        psi = MPS.from_product_state(model.lat.mps_sites(), ["full", "empty", "empty", "empty"] , bc=model.lat.bc_MPS)
    else :
        psi = psi_init
    
    if saveflag: 
        JOBDIR = os.path.join(ROOTDIR, f"scanChi{chi_max}/")
        os.makedirs(JOBDIR, exist_ok=True)

    model_params = dict(
        t=t, U=U, Vpp=Vpp, Vpm=0.0, mu=0.0, hx=hx, hz=hz,
        bc_MPS="infinite",
        L=4, cons_N="N", cons_Sz="None",
    )
    dmrg_params = {
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1.0e-11, 'trunc_cut': None},
        'update_env': 10,
        'start_env': 20,
        'max_E_err': max_E_err,
        'max_S_err': max_S_err,
        # 'mixer': False,
        "mixer": "DensityMatrixMixer",
        "mixer_params": {
            "amplitude": 1e-3,
            "decay": 1.2,
            "disable_after": 10, 
        },
    }
    model = HubbardSpinDepV(model_params)

    engine = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    corr_length = []
    for dV in dV_values:
        Vpm = Vpp + dV
        print('-' * 80)
        print(f'{Vpm=:.4f}')
        print('-' * 80)
        model_params['Vpm'] = Vpm
        model = HubbardSpinDepV(model_params)
        engine.reset_stats() #strictly not needed, but we use a safety precaution
        engine.init_env(model=model)
        engine.run()
        corr_length.append(psi.correlation_length(tol_ev0=1.0e-3))
        print('corr. length', corr_length[-1])
        print('<Sz>', psi.expectation_value('Sz'), flush=True)
        dmrg_params['start_env'] = 0  # (some of) the parameters are read out again
        engine.options['start_env'] = 0
        if saveflag : 
            results = {
                "diagnostics" : engine.sweep_stats, 
                "model_params" : model_params, 
                "dmrg_params" : dmrg_params,
            }
            io.save_mps_with_metadata(os.path.join(JOBDIR, f"Vpm_{Vpm:.3f}_chi{chi_max}.h5"), psi, results)
    print(corr_length)
    print("FINISHED scanner_v1", flush=True)




