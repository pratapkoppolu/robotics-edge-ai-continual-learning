"""M5: Multivariate embedding drift using permutation-calibrated RBF MMD."""
import argparse
import json
from pathlib import Path
import numpy as np


def squared_distances(x, y):
    return np.maximum(((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2), 0)


def mmd2_kernel(k, n):
    """Biased MMD^2, valid with n,m small; calibration uses permutations."""
    return float(k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean())


def compare(reference, current, permutations=199, alpha=0.05, seed=42):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(current, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError('Embedding matrices require matching feature dimensions')
    if min(len(a), len(b)) < 5:
        raise ValueError('Need at least 5 reference and current samples')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite embeddings')
    if permutations < 19 or not 0 < alpha < 1:
        raise ValueError('Invalid permutation count or alpha')
    # Normalize with reference-only statistics to avoid using current data to fit representation.
    mu = a.mean(axis=0)
    sd = a.std(axis=0)
    sd[sd < 1e-6] = 1
    a = (a - mu) / sd
    b = (b - mu) / sd
    z = np.vstack([a, b])
    d = squared_distances(z, z)
    positive = d[np.triu_indices(len(z), 1)]
    positive = positive[positive > 1e-12]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    k = np.exp(-d / max(bandwidth, 1e-6))
    n = len(a)
    observed = mmd2_kernel(k, n)
    rng = np.random.default_rng(seed)
    geq = 0
    for _ in range(permutations):
        ix = rng.permutation(len(z))
        kp = k[np.ix_(ix, ix)]
        geq += mmd2_kernel(kp, n) >= observed - 1e-12
    pvalue = (geq + 1) / (permutations + 1)
    return {'reference_n': n, 'current_n': len(b), 'embedding_dimensions': a.shape[1],
            'mmd2': round(max(observed, 0), 8), 'p_value': round(float(pvalue), 6),
            'alpha': alpha, 'drift_detected': bool(pvalue < alpha),
            'permutations': permutations, 'method': 'rbf_mmd_permutation_test',
            'warning': 'Distribution shift alone is not evidence of model failure; investigate task outcomes.'}


def run(ref_path, cur_path, task, out, permutations=199, alpha=0.05, log_mlflow=False):
    with np.load(ref_path) as ref, np.load(cur_path) as cur:
        r_id, c_id = str(ref['encoder_id']), str(cur['encoder_id'])
        if r_id != c_id:
            raise ValueError(f'Encoder mismatch: {r_id} vs {c_id}')
        # An image cannot appear in both groups; hashes catch renamed duplicates.
        shared = set(ref['hashes'].tolist()) & set(cur['hashes'].tolist())
        if shared:
            raise ValueError(f'{len(shared)} identical image(s) in both groups; use disjoint reference/current samples')
        report = compare(ref['embeddings'], cur['embeddings'], permutations, alpha)
        report.update({'task': task, 'encoder_id': r_id, 'reference_features': str(ref_path), 'current_features': str(cur_path)})
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if log_mlflow:
        import mlflow
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        mlflow.set_experiment('robotics_weekly_drift')
        with mlflow.start_run(run_name=f'task_{task}_drift'):
            mlflow.log_params({'task': task, 'encoder': r_id, 'method': report['method'], 'permutations': permutations})
            mlflow.log_metrics({'mmd2': report['mmd2'], 'p_value': report['p_value'], 'drift_detected': int(report['drift_detected'])})
            mlflow.log_artifact(str(output))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--current', required=True)
    p.add_argument('--task', required=True, choices=['A', 'B'])
    p.add_argument('--out', required=True)
    p.add_argument('--permutations', type=int, default=199)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--mlflow', action='store_true')
    a = p.parse_args()
    print(json.dumps(run(a.reference, a.current, a.task, a.out, a.permutations, a.alpha, a.mlflow), indent=2))
