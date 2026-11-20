from pathlib import Path
import numpy as np
import pytest
from drift_detection.evidently_report import prepare_feature_tables


def make(path, emb, hashes, encoder='fixed-v1'):
    np.savez(path, embeddings=np.asarray(emb), hashes=np.asarray(hashes), encoder_id=np.asarray(encoder))


def test_reference_fitted_projection(tmp_path):
    rng = np.random.default_rng(3)
    a, b = tmp_path / 'ref.npz', tmp_path / 'current.npz'
    make(a, rng.normal(size=(12, 8)), [f'a{i}' for i in range(12)])
    make(b, rng.normal(size=(10, 8)) + 2, [f'b{i}' for i in range(10)])
    r, c, meta = prepare_feature_tables(a, b, components=3)
    assert r.shape == (12, 3)
    assert c.shape == (10, 3)
    assert meta['projection'] == 'reference_only_pca'
    assert np.allclose(r.mean().to_numpy(), 0, atol=1e-7)


def test_encoder_mismatch_rejected(tmp_path):
    a, b = tmp_path / 'r.npz', tmp_path / 'c.npz'
    make(a, np.ones((7, 4)), [str(i) for i in range(7)], 'a')
    make(b, np.ones((7, 4)), [f'c{i}' for i in range(7)], 'b')
    with pytest.raises(ValueError, match='Encoder mismatch'):
        prepare_feature_tables(a, b)


def test_duplicate_images_rejected(tmp_path):
    a, b = tmp_path / 'r.npz', tmp_path / 'c.npz'
    make(a, np.ones((7, 4)), [str(i) for i in range(7)])
    make(b, np.ones((7, 4)), ['0'] + [f'c{i}' for i in range(6)])
    with pytest.raises(ValueError, match='identical image'):
        prepare_feature_tables(a, b)
