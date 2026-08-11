#!/usr/bin/env python
"""
read_corrs.py

Manual (no pd.read_hdf) helper to load a corrs/results.h5 file, written by
combiner.py, into a single pandas DataFrame -- one row per source .h5 file.

    df = read_corrs('DIR/corrs/results.h5')

df columns:
    basename, src_file, dV, mean_Sz, mean_fill, xi,
    model_params, diagnostics,                       (dicts)
    charge_corr, spinz_corr, spsm_corr,
    electron_corr, spin2_corr                         (1D numpy arrays)

df is sorted by dV.

select_dV is a small convenience filter on top of that:

    seldf = select_dV(df)                # no filtering, just df itself
    seldf = select_dV(df, dV=0.3)        # single row (pd.Series) closest to dV=0.3
    seldf = select_dV(df, dV=0.3, atol=0.05)  # DataFrame of rows within 0.05 of dV=0.3

Typical plotting usage:
    df = read_corrs('DIR/corrs/results.h5')
    row = select_dV(df, dV=0.3)
    plt.plot(row['charge_corr'])

    # or loop over everything:
    for _, row in df.iterrows():
        plt.plot(row['charge_corr'], label=f"dV={row['dV']:.3f}")
"""

import json

import h5py
import numpy as np
import pandas as pd

ARRAY_FIELDS = ['charge_corr', 'spinz_corr', 'spsm_corr',
                 'electron_corr', 'spin2_corr']


def read_corrs(path):
    """Read a results.h5 file (written by combiner.py) into a DataFrame,
    one row per source file, sorted by dV."""
    rows = []
    with h5py.File(path, 'r') as f:
        data_grp = f['data']
        for basename in data_grp:
            g = data_grp[basename]
            row = {
                'basename': basename,
                'src_file': g.attrs.get('src_file', ''),
                'dV': float(g.attrs['dV']),
                'mean_Sz': float(g.attrs['mean_Sz']),
                'mean_fill': float(g.attrs['mean_fill']),
                'xi': float(g.attrs['xi']),
                'model_params': json.loads(g.attrs['model_params_json']),
                'diagnostics': json.loads(g.attrs['diagnostics_json']),
            }
            for field in ARRAY_FIELDS:
                row[field] = np.asarray(g[field][:]) if field in g else None
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('dV').reset_index(drop=True)
    return df


def select_dV(df, dV=None, atol=None):
    """
    dV=None            -> return df unchanged (already sorted by dV).
    dV given, atol=None -> return the single closest row as a pd.Series
                            (so seldf['charge_corr'] is one array, ready
                            to plt.plot directly).
    dV given, atol set  -> return all rows within atol of dV, as a
                            DataFrame (reset index).
    """
    if dV is None:
        return df
    diffs = (df['dV'] - dV).abs()
    if atol is not None:
        return df[diffs <= atol].reset_index(drop=True)
    return df.loc[diffs.idxmin()]


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'corrs/results.h5'
    df = read_corrs(path)
    print(df[['basename', 'dV', 'mean_Sz', 'mean_fill', 'xi']])
