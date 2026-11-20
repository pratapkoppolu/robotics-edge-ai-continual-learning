"""Synthetic *oracle* targets for pipeline smoke testing ONLY. Not a VLM API.
Creates approved records from existing synthetic GT; do not use as KD evidence.
"""
import argparse
import json
from pathlib import Path
from PIL import Image
from distillation.teacher_adapter import sha256


def generate(task, dataset_root, teacher_root, num_classes=4):
    root = Path(dataset_root); destination = Path(teacher_root)/task/'train'
    destination.mkdir(parents=True, exist_ok=True)
    for image in sorted((root/'train'/'images').glob('*.jpg')):
        record = {'task':task, 'source_model':'SYNTHETIC_GROUND_TRUTH_ORACLE_NOT_VLM',
                  'reviewer':'automated-smoke-test-only', 'approved':True,
                  'image_sha256':sha256(image)}
        if task=='A':
            mask = root/'train'/'masks'/(image.stem+'.png')
            if not mask.is_file(): raise FileNotFoundError(mask)
            record['mask_path'] = str(mask.resolve())
        else:
            label = root/'train'/'labels'/(image.stem+'.txt')
            if not label.is_file(): raise FileNotFoundError(label)
            objects=[]
            for line in label.read_text().splitlines():
                fields=line.split()
                if len(fields)!=5: raise ValueError(f'Invalid YOLO label: {label}')
                cls=int(fields[0]); box=[float(v) for v in fields[1:]]
                if not 0<=cls<num_classes-1: raise ValueError(f'Invalid class {cls}')
                objects.append({'class_id':cls, 'bbox_xywh_normalized':box})
            record['objects']=objects
        (destination/(image.stem+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('Created smoke-only oracle records in', destination)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=('A','B'),required=True)
    p.add_argument('--dataset-root',required=True)
    p.add_argument('--teacher-root',default='datasets/teacher_approved')
    p.add_argument('--num-classes',type=int,default=4)
    a=p.parse_args()
    generate(a.task,a.dataset_root,a.teacher_root,a.num_classes)
