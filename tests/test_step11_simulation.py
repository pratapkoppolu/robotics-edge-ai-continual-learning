import json
from pathlib import Path
from PIL import Image
import pytest
from continual_learning.simulate_failure import generate


def test_create_shifted_samples(tmp_path):
    src = tmp_path / 'source'
    for split in ('train', 'test'):
        (src / split / 'images').mkdir(parents=True)
        (src / split / 'labels').mkdir(parents=True)
        Image.new('RGB', (24, 24), 'red').save(src / split / 'images' / 'sample_0000.jpg')
        (src / split / 'labels' / 'sample_0000.txt').write_text('0 0.5 0.5 0.5 0.5\n')
    target = tmp_path / 'shifted'
    records = generate(src, target, 1)
    assert len(records['train']) == len(records['test']) == 1
    assert json.loads((target / 'SIMULATION_ONLY.json').read_text())['simulation_only']
    assert (target / 'train' / 'labels' / 'sample_0000.txt').read_text() == '0 0.5 0.5 0.5 0.5\n'
    with pytest.raises(FileExistsError):
        generate(src, target, 1)
