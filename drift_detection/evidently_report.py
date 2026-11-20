"""M5: Evidently diagnostic report on frozen reference-fitted embedding PCs.

Existing MMD drift report remains the primary decision input.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def prepare_feature_tables(reference_path, current_path, components=6):
    with np.load(reference_path) as r, np.load(current_path) as c:
        re, ce = np.asarray(r['embeddings'], dtype=float), np.asarray(c['embeddings'], dtype=float)
        rid, cid = str(r['encoder_id']), str(c['encoder_id'])
        rh, ch = set(r['hashes'].tolist()), set(c['hashes'].tolist())
    if rid != cid:
        raise ValueError(f'Encoder mismatch: {rid} vs {cid}')
    if rh & ch:
        raise ValueError('Reference/current contain identical image hashes')
    if re.ndim != 2 or ce.ndim != 2 or re.shape[1] != ce.shape[1]:
        raise ValueError('Incompatible feature matrices')
    if min(len(re), len(ce)) < 5 or not np.isfinite(re).all() or not np.isfinite(ce).all():
        raise ValueError('Need >=5 finite samples in each dataset')
    n = min(components, re.shape[1], len(re) - 1)
    if n < 1:
        raise ValueError('No PCA components available')
    # Fit dimensionality reduction ONLY on reference data; reuse on current.
    mean = re.mean(axis=0)
    x = re - mean
    _, _, v = np.linalg.svd(x, full_matrices=False)
    basis = v[:n].T
    col = [f'pc_{i:02d}' for i in range(1, n + 1)]
    ref_df = pd.DataFrame((re - mean) @ basis, columns=col)
    cur_df = pd.DataFrame((ce - mean) @ basis, columns=col)
    return ref_df, cur_df, {'encoder_id': rid, 'components': n, 'reference_n': len(re), 'current_n': len(ce), 'projection': 'reference_only_pca'}


def generate(reference, current, out_html, out_json=None, components=6):
    from evidently import Report
    from evidently.presets import DataDriftPreset
    ref_df, cur_df, meta = prepare_feature_tables(reference, current, components)
    report = Report([DataDriftPreset()])
    snapshot = report.run(cur_df, ref_df)  # Evidently order: current, reference
    target = Path(out_html)
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(target))
    if out_json:
        j = Path(out_json)
        j.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_json(str(j))
    return {'html': str(target), 'evidently_json': str(out_json) if out_json else None, **meta}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--current', required=True)
    p.add_argument('--out-html', required=True)
    p.add_argument('--out-json')
    p.add_argument('--components', type=int, default=6)
    a = p.parse_args()
    print(json.dumps(generate(a.reference, a.current, a.out_html, a.out_json, a.components), indent=2))
