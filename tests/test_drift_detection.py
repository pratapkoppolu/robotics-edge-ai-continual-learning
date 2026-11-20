import numpy as np
from drift_detection.detect_drift import compare
from drift_detection.retraining_trigger import decide


def test_identical_distribution_not_flagged():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(60, 16)), rng.normal(size=(60, 16))
    r = compare(a, b, permutations=199, seed=15)
    assert not r['drift_detected']


def test_large_shift_is_flagged():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(60, 16)), rng.normal(loc=3, size=(60, 16))
    r = compare(a, b, permutations=199, seed=15)
    assert r['drift_detected']


def test_shift_does_not_automatically_retrain():
    r = decide({'drift_detected': True}, {'n': 100, 'success_rate': .4, 'minimum_success_rate': .8, 'safety_incidents': 0})
    assert r['decision'] == 'human_review_for_retraining' and not r['retrain']
