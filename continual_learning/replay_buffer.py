"""Deterministic, class/shift-aware optional historical replay selection."""
import hashlib
import json
import random
from pathlib import Path


def select_replay(records, count, seed=42):
    """records: approved metadata [{'image_id', 'task', 'stratum', 'image_sha256'}]."""
    if count < 1 or count > len(records):
        raise ValueError('Replay count must be in 1..history size')
    ids = [r['image_id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate image_id in history')
    groups = {}
    for r in records:
        for field in ('image_id', 'task', 'stratum', 'image_sha256'):
            if not r.get(field): raise ValueError(f'Missing {field}')
        groups.setdefault((r['task'],r['stratum']), []).append(r)
    rng = random.Random(seed)
    for group in groups.values(): rng.shuffle(group)
    chosen=[]
    keys=sorted(groups)
    while len(chosen)<count:
        for key in keys:
            if groups[key] and len(chosen)<count:
                chosen.append(groups[key].pop())
    return chosen


def manifest_fingerprint(records):
    payload=json.dumps(sorted(records, key=lambda r:r['image_id']), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def make_manifest(history_path, count, seed, out):
    records=json.loads(Path(history_path).read_text(encoding='utf-8'))
    selected=select_replay(records,count,seed)
    doc={'seed':seed,'count':len(selected),'records':selected,'fingerprint':manifest_fingerprint(selected)}
    path=Path(out); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(doc,indent=2),encoding='utf-8')
    return doc
