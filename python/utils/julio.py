import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import os

def load_correlations(path):

    with h5py.File(path, "r") as f:
        tdict = {
            "szmat":       f["szmat"][()],
            "chargemat":   f["chargemat"][()],
            "spmmat":      f["spmmat"][()],
            "elecupmat":   f["elecupmat"][()],
            "elecdnmat":   f["elecdnmat"][()],
            "sz_expect":   f["sz_expect"][()],
            "ntot_expect": f["ntot_expect"][()],
            "xspin2":      f["xspin2"][()] if "xspin2" in f else None,
            "spin2vec":    f["spin2vec"][()] if "spin2vec" in f else None,
        }
        N = tdict["szmat"].shape[0]
        print(type(tdict["szmat"]), type(tdict["sz_expect"]))
        tdict["szconn"] = tdict["szmat"][N//2, N//2 : ] 
        - tdict["sz_expect"][N//2] * tdict["sz_expect"][N//2: ]
        tdict["chargeconn"] = tdict["chargemat"][N//2, N//2 : ]
        - tdict["ntot_expect"][N//2] * tdict["ntot_expect"][N//2: ]
        
    
    return tdict

def load_correlations_files(paths):
    """Load multiple correlation files into a DataFrame."""

    rows = []

    for path in paths:
        row = load_correlations(path)
        row["filename"] = path
        rows.append(row)

    return pd.DataFrame(rows)



def corr_filename_builder(
    N, t, U, Vpp, dV, prefix="data/Hubbard/", DATAROOT="../../"
):
    """ DATAROOT should be Spin1LongRangeOrder.jl/"""
    Vpm = Vpp + dV
    Npart = N // 2

    prefix = Path(prefix)

    if "ImpPrec" in prefix.parts:
        outdir = Path("data") / "Hubbard" / "corrs_imp"
    else:
        outdir = prefix / "corrs"

    filename = (
        f"N{N}"
        f"_U{U:.3f}"
        f"_Vpp{Vpp:.3f}"
        f"_Vpm{Vpm:.3f}"
        f"_Np{Npart}"
        "_corrs.h5"
    )

    return os.path.join(DATAROOT, os.path.join(outdir, filename))
