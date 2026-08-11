#!/usr/bin/env python
"""
combiner.py

Combines the per-file .pkl outputs of driver_idmrg_datacollector.py
(sitting in <DATA_ROOT>/<REQ_DATA>/corrs/tmp/*.pkl) into a single plain
HDF5 file, written with h5py only (no pandas involved in writing):

    <DATA_ROOT>/<REQ_DATA>/corrs/results.h5

Layout of the resulting file:

    /data/<basename>/charge_corr
    /data/<basename>/spinz_corr
    /data/<basename>/spsm_corr
    /data/<basename>/electron_corr
    /data/<basename>/spin2_corr
    /data/<basename>              <- group; attrs: dV, mean_Sz, mean_fill,
                                      xi, src_file, model_params_json,
                                      diagnostics_json

Usage:
    python combiner.py                        # combine every scan in REQ_DATA_LIST
    python combiner.py --req-data scanChi800   # combine just one scan
    python combiner.py --keep-tmp              # don't delete tmp/*.pkl afterward

To load results.h5 back into a single DataFrame for plotting, use
read_corrs.py (df = read_corrs('DIR/corrs/results.h5')).
"""

import os
import json
import pickle
import argparse

import numpy as np
import h5py

# keep this in sync with driver_idmrg_datacollector.py
DATA_ROOT = '../../data/iDMRG/'
REQ_DATA_LIST = [
    'scanChi800',
]

ARRAY_FIELDS = ['charge_corr', 'spinz_corr', 'spsm_corr',
                 'electron_corr', 'spin2_corr']


def _jsonify(d):
    """Best-effort JSON-encode a possibly-numpy-containing dict, so
    model_params / diagnostics survive as HDF5 attrs."""
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    return json.dumps(d, default=default)


def combine_one(req_data, keep_tmp=False):
    path_to_req_data = os.path.join(DATA_ROOT, req_data)
    corrs_dir = os.path.join(path_to_req_data, 'corrs')
    tmp_dir = os.path.join(corrs_dir, 'tmp')
    out_h5 = os.path.join(corrs_dir, 'results.h5')

    if not os.path.isdir(tmp_dir):
        print(f"[skip] no tmp dir for {req_data} ({tmp_dir})")
        return

    pkl_files = sorted(f for f in os.listdir(tmp_dir) if f.endswith('.pkl'))
    if not pkl_files:
        print(f"[skip] no .pkl files found in {tmp_dir}")
        return

    if os.path.exists(out_h5):
        os.remove(out_h5)

    # write correlation arrays + per-file metadata, plain h5py, single pass
    with h5py.File(out_h5, 'w') as h5f:
        data_grp = h5f.create_group('data')
        for pf in pkl_files:
            basename = os.path.splitext(pf)[0]
            with open(os.path.join(tmp_dir, pf), 'rb') as fh:
                row_dict = pickle.load(fh)

            g = data_grp.create_group(basename)
            for field in ARRAY_FIELDS:
                val = row_dict.get(field, None)
                if val is not None:
                    g.create_dataset(field, data=np.asarray(val))

            g.attrs['dV'] = float(row_dict['dV'])
            g.attrs['mean_Sz'] = float(row_dict['mean_Sz'])
            g.attrs['mean_fill'] = float(row_dict['mean_fill'])
            g.attrs['xi'] = float(row_dict['xi'])
            g.attrs['src_file'] = row_dict.get('src_file', '')
            g.attrs['model_params_json'] = _jsonify(row_dict['model_params'])
            g.attrs['diagnostics_json'] = _jsonify(row_dict['diagnostics'])

    print(f"[ok] {req_data}: combined {len(pkl_files)} files -> {out_h5}")

    if not keep_tmp:
        for pf in pkl_files:
            os.remove(os.path.join(tmp_dir, pf))
        print(f"[cleanup] removed {len(pkl_files)} tmp files from {tmp_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--req-data', default=None,
                         help='combine only this REQ_DATA dir instead of REQ_DATA_LIST')
    parser.add_argument('--keep-tmp', action='store_true',
                         help='do not delete tmp/*.pkl after combining')
    args = parser.parse_args()

    targets = [args.req_data] if args.req_data else REQ_DATA_LIST
    for req_data in targets:
        combine_one(req_data, keep_tmp=args.keep_tmp)


if __name__ == '__main__':
    main()
